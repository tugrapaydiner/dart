from __future__ import annotations

import copy
import gc
import json
import weakref
from dataclasses import asdict

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.replay import CudaGraphTarget
from dart.models.target import TargetModel
from dart.verification.forward_checks import check_forward_paths
from dart.verification.state_sequences import StateOperation, check_state_sequence


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        pytest.fail("CPU forward check touched CUDA")
    for name in ("init", "Stream", "CUDAGraph", "current_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    torch.set_num_threads(threads)


def models(kv=1):
    torch.manual_seed(61)
    config = LlamaConfig(vocab_size=67, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=kv, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=None, pad_token_id=0, attn_implementation="eager")
    model = LlamaForCausalLM(config).eval()
    return TargetModel(model), copy.deepcopy(model).eval()


def shift_cached(model, monkeypatch, *, full_value=None, cached_value=None):
    forward = model.forward
    def shifted(*args, **kwargs):
        cache = kwargs.get("past_key_values")
        cached = cache is not None and cache.get_seq_length() > 0
        output = forward(*args, **kwargs)
        if full_value is not None:
            output.logits = torch.full_like(output.logits, cached_value if cached else full_value)
        elif cached:
            output.logits = output.logits + 0.125
        return output
    monkeypatch.setattr(model, "forward", shifted)


@pytest.mark.parametrize("kv", [1, 2])
@pytest.mark.parametrize("width", [1, 4, 16])
def test_full_and_cached_paths_agree_at_context_boundary(kv, width):
    target, reference = models(kv)
    report = check_forward_paths(target, reference, [5] * (64 - width), [7] * width)
    assert report.passed, report
    assert report.candidate_incremental.logits_bitwise and report.candidate_incremental.cache_bitwise
    json.dumps(asdict(report), allow_nan=False)


def test_reference_control_distinguishes_otherwise_identical_failures(monkeypatch):
    old_reports, reports = [], []
    for reference_shift in (False, True):
        target, reference = models()
        with monkeypatch.context() as scope:
            shift_cached(target.model, scope)
            if reference_shift:
                shift_cached(reference, scope)
            old_reports.append(check_state_sequence(target, reference, [StateOperation("a", "prefill", (5, 6, 7)),
                                                                         StateOperation("a", "forward", (11, 13))]))
            reports.append(check_forward_paths(target, reference, (5, 6, 7), (11, 13)))
    assert old_reports[0] == old_reports[1] and not old_reports[0].passed
    assert all(not report.passed and not report.candidate_full.passed for report in reports)
    assert not reports[0].candidate_incremental.passed and reports[0].incremental_full.passed
    assert reports[1].candidate_incremental.passed and not reports[1].incremental_full.passed
    assert reports[1].candidate_incremental.logits_bitwise


def test_closeness_to_both_references_does_not_waive_reference_disagreement(monkeypatch):
    target, reference = models()
    shift_cached(target.model, monkeypatch, full_value=0.00015, cached_value=0.00015)
    shift_cached(reference, monkeypatch, full_value=0.0003, cached_value=0.0)
    report = check_forward_paths(target, reference, (5, 6, 7), (11,), atol=0.0002, rtol=0)
    assert report.candidate_full.passed and report.candidate_incremental.passed
    assert not report.incremental_full.passed and not report.passed


@pytest.mark.parametrize("fault", ["nan", "inf", "argmax", "shape", "cache_values", "cache_dtype", "cache_length", "cache_layers", "exception"])
def test_candidate_faults_remain_failed_with_valid_reference_controls(monkeypatch, fault):
    target, reference = models()
    forward = target.forward_block
    def broken(state, tokens):
        logits = forward(state, tokens)
        if fault in ("nan", "inf"):
            logits[0, 0] = float(fault)
        elif fault == "argmax":
            logits[0, (int(logits[0].argmax()) + 1) % logits.shape[-1]] = 100
        elif fault == "shape":
            logits = logits[:-1]
        elif fault == "cache_values":
            state.cache.layers[0].keys.add_(0.125)
        elif fault == "cache_dtype":
            state.cache.layers[0].keys = state.cache.layers[0].keys.half()
        elif fault == "cache_length":
            state.cache.crop(1)
        elif fault == "cache_layers":
            state.cache.layers.append(state.cache.layers[0])
        else:
            raise RuntimeError("candidate failed")
        return logits
    monkeypatch.setattr(target, "forward_block", broken)
    report = check_forward_paths(target, reference, (5, 6, 7), (11, 13))
    assert report.incremental_full.passed
    assert not report.passed and not report.candidate_full.passed and not report.candidate_incremental.passed


def test_equal_nonfinite_logits_still_fail(monkeypatch):
    target, reference = models()
    for model in (target.model, reference):
        shift_cached(model, monkeypatch, full_value=float("inf"), cached_value=float("inf"))
    report = check_forward_paths(target, reference, (5, 6), (11,))
    assert all(not check.passed and check.logits_bitwise for check in (
        report.candidate_full, report.candidate_incremental, report.incremental_full,
    ))


def test_logit_dtype_difference_is_not_hidden_by_conversion(monkeypatch):
    target, reference = models()
    forward = target.forward_block
    monkeypatch.setattr(target, "forward_block", lambda state, tokens: forward(state, tokens).double())
    report = check_forward_paths(target, reference, (5, 6), (11,))
    assert report.incremental_full.passed and not report.passed
    assert not report.candidate_full.passed and not report.candidate_incremental.logits_bitwise


def test_snapshot_ownership_survives_reference_output_reuse(monkeypatch):
    target, reference = models()
    original, previous = reference.forward, []
    def reusing(*args, **kwargs):
        if previous:
            previous[0].logits.fill_(10)
            for layer in previous[0].past_key_values.layers:
                layer.keys.fill_(10)
                layer.values.fill_(10)
        output = original(*args, **kwargs)
        if kwargs.get("past_key_values") is None and kwargs["input_ids"].shape[-1] == 5:
            previous[:] = [output]
        return output
    monkeypatch.setattr(reference, "forward", reusing)
    assert check_forward_paths(target, reference, (5, 6, 7), (11, 13)).passed


@pytest.mark.parametrize("fault", ["exception", "logits_shape", "cache_shape", "cache_length"])
def test_reference_errors_abort_before_candidate(monkeypatch, fault):
    target, reference = models()
    original = reference.forward
    def broken(*args, **kwargs):
        if fault == "exception":
            raise RuntimeError("reference failed")
        output = original(*args, **kwargs)
        if fault == "logits_shape":
            output.logits = output.logits[:, -1:]
        elif fault == "cache_shape":
            output.past_key_values.layers[0].keys = output.past_key_values.layers[0].keys[:, :, :-1]
        else:
            output.past_key_values.get_seq_length = lambda: 100
        return output
    monkeypatch.setattr(reference, "forward", broken)
    monkeypatch.setattr(target, "prefill", lambda *args: pytest.fail("candidate ran after invalid reference"))
    with pytest.raises((RuntimeError, ValueError), match="reference"):
        check_forward_paths(target, reference, (5, 6, 7), (11, 13))


def test_candidate_cancellation_propagates(monkeypatch):
    target, reference = models()
    def interrupted(*args):
        raise KeyboardInterrupt("cancelled")
    monkeypatch.setattr(target, "forward_block", interrupted)
    with pytest.raises(KeyboardInterrupt):
        check_forward_paths(target, reference, (5, 6, 7), (11,))


@pytest.mark.parametrize("prefix,tokens", [([], [1]), ([1], []), ([True], [1]), ([1], [-1]), ([67], [1]),
                                         ([1] * 64, [1]), ([1], [2] * 17), ("123", [1]), ([1], iter([2]))])
def test_invalid_inputs_reject_before_execution(monkeypatch, prefix, tokens):
    target, reference = models()
    monkeypatch.setattr(reference, "forward", lambda *args, **kwargs: pytest.fail("invalid case executed"))
    with pytest.raises(ValueError):
        check_forward_paths(target, reference, prefix, tokens)


@pytest.mark.parametrize("field,value", [("atol", -1), ("atol", True), ("rtol", float("nan")), ("rtol", float("inf"))])
def test_invalid_tolerances_reject_before_execution(monkeypatch, field, value):
    target, reference = models()
    monkeypatch.setattr(reference, "forward", lambda *args, **kwargs: pytest.fail("invalid tolerance executed"))
    with pytest.raises(ValueError):
        check_forward_paths(target, reference, (5,), (7,), **{field: value})


@pytest.mark.parametrize("fault", ["same", "storage", "target_train", "reference_train"])
def test_reference_independence_and_eval_mode_are_required(fault):
    target, reference = models()
    if fault == "same":
        reference = target.model
    elif fault == "storage":
        reference.model.embed_tokens.weight = target.model.model.embed_tokens.weight
    elif fault == "target_train":
        target.model.train()
    else:
        reference.train()
    with pytest.raises(ValueError):
        check_forward_paths(target, reference, (5,), (7,))


@pytest.mark.parametrize("static", [False, True])
def test_live_caches_defaults_and_rng_are_preserved_without_retaining_outputs(monkeypatch, tmp_path, static):
    base, reference = models()
    target = CudaGraphTarget(base, capacity=8, max_block_tokens=3, replay=False) if static else base
    try:
        state, _ = target.prefill([2, 3])
        target.forward_block(state, [17])
        if static:
            live = state.cache.storage.clone()
        else:
            live = state.cache.layers[0].keys.clone()
        prefill, refs = target.prefill, []
        def remember(tokens):
            owned, logits = prefill(tokens)
            refs.append(weakref.ref(owned))
            return owned, logits
        monkeypatch.setattr(target, "prefill", remember)
        monkeypatch.chdir(tmp_path)
        rng = torch.random.get_rng_state().clone()
        defaults = copy.deepcopy(reference.generation_config.to_dict())
        first = check_forward_paths(target, reference, (5, 6, 7), (11, 13))
        second = check_forward_paths(target, reference, (5, 6, 7), (11, 13))
        assert first == second and first.passed
        gc.collect()
        assert all(ref() is None for ref in refs) and not list(tmp_path.iterdir())
        assert reference.generation_config.to_dict() == defaults and torch.equal(rng, torch.random.get_rng_state())
        assert torch.equal(state.cache.storage if static else state.cache.layers[0].keys, live)
    finally:
        if static:
            target.close()
