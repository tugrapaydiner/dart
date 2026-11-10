from __future__ import annotations

import copy
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.hf_static import PersistentStaticControl, _context
from dart.models.replay import CudaGraphTarget
from dart.models.rms_norm import enable_native_rms_norm
from dart.models.target import TargetModel
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import synthetic_model_config


def models(kv=2, attention="eager"):
    torch.manual_seed(17)
    config = LlamaConfig(**{**synthetic_model_config(), "num_key_value_heads": kv, "attn_implementation": attention})
    model = LlamaForCausalLM(config).eval()
    reference = copy.deepcopy(model)
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    target = TargetModel(model)
    target.eos_token_ids = frozenset()
    return target, reference


def expected(reference, prompt, budget):
    return hf_decoder(reference)(GenerationCase("persistent", tuple(prompt), budget))


def cache_refs(control):
    return [weakref.ref(control.cache), *(weakref.ref(tensor) for layer in control.cache.layers
                                        for tensor in (layer.keys, layer.values, layer.cumulative_length))]


def assert_empty(control):
    for layer in control.cache.layers:
        assert int(layer.cumulative_length) == 0
        assert not layer.keys.count_nonzero() and not layer.values.count_nonzero()


@pytest.mark.parametrize("kv", [2, 4])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_reuse_matches_independent_outputs_and_live_cache(kv, attention, monkeypatch):
    target, reference = models(kv, attention)
    prior = target.model._cache = object()
    before = copy.deepcopy(target.model.generation_config)
    live = {}
    monkeypatch.setattr(target.model, "get_compiled_call", lambda *a, **k: pytest.fail("unexpected compilation"))
    with PersistentStaticControl(target, 64) as control:
        refs = cache_refs(control)
        pointers = [ref().data_ptr() for ref in refs[1:]]
        assert control.setup_ms >= 0
        assert control.cache_bytes == 2 * (2 * kv * 64 * 8 * 4 + 8)

        def observe(module, args, kwargs, output):
            assert kwargs["past_key_values"] is output.past_key_values is control.cache
            live["lengths"] = [int(layer.cumulative_length) for layer in control.cache.layers]
            live["kv"] = [(layer.keys[..., :length, :].clone(), layer.values[..., :length, :].clone())
                          for layer, length in zip(control.cache.layers, live["lengths"])]

        handle = target.model.register_forward_hook(observe, with_kwargs=True)
        try:
            for prompt, budget in (([3, 5, 7] * 4, 5), ([0, 5, 0], 7), ([3] * 57, 7), ([3, 5, 7] * 4, 5)):
                with torch.inference_mode():
                    for layer in control.cache.layers:
                        layer.keys.fill_(123)
                        layer.values.fill_(-123)
                output = control.generate(prompt, budget)
                assert output.token_ids == expected(reference, prompt, budget)
                with torch.inference_mode():
                    full = reference(torch.tensor([prompt + output.token_ids[:-1]]), use_cache=True)
                length = len(prompt) + budget - 1
                assert live["lengths"] == [length] * 2
                for pair, layer in zip(live["kv"], full.past_key_values.layers):
                    for actual, wanted in zip(pair, (layer.keys, layer.values)):
                        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
                assert_empty(control)
                assert [ref().data_ptr() for ref in refs[1:]] == pointers
                assert target.model.generation_config == before and target.model._cache is prior
            monkeypatch.setattr(target.model, "generate", lambda *a, **k: pytest.fail("unexpected generation"))
            assert control.generate([3], 0).token_ids == []
            assert_empty(control)
        finally:
            handle.remove()
    gc.collect()
    assert all(ref() is None for ref in refs)
    control.close()
    with pytest.raises(RuntimeError, match="closed"):
        control.generate([3], 1)


@pytest.mark.parametrize("eos_offset", [None, 0, 2])
def test_generation_defaults_and_eos_are_isolated(eos_offset):
    target, reference = models()
    prompt, budget = [3, 5, 7], 7
    wanted = expected(reference, prompt, budget)
    if eos_offset is not None:
        eos = wanted[eos_offset]
        target.eos_token_ids = frozenset([eos])
        wanted = wanted[:wanted.index(eos) + 1]
    config = target.model.generation_config
    config.do_sample, config.num_beams, config.num_return_sequences = True, 3, 2
    config.repetition_penalty, config.min_new_tokens = 2.0, 20
    config.forced_bos_token_id, config.forced_eos_token_id = 1, 2
    config.suppress_tokens, config.cache_implementation = wanted[:1], "static"
    config.return_dict_in_generate, config.output_logits = True, True
    config.output_scores, config.output_hidden_states, config.disable_compile = True, True, False
    config.eos_token_id = list(range(67))
    before = copy.deepcopy(config)
    with PersistentStaticControl(target, 64) as control:
        for _ in range(2):
            assert control.generate(prompt, budget).token_ids == wanted
            assert config == before
            assert_empty(control)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_cpu_native_normalization_uses_the_current_model(attention):
    target, reference = models(attention=attention)
    enable_native_rms_norm(target.model)
    enable_native_rms_norm(reference)
    with PersistentStaticControl(target, 64) as control:
        for prompt in ([3, 5, 7] * 4, [3], [3, 5, 7] * 4):
            assert control.generate(prompt, 7).token_ids == expected(reference, prompt, 7)
            assert_empty(control)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_partial_forward_failure_clears_cache_and_recovers(monkeypatch, error):
    target, reference = models()
    with PersistentStaticControl(target, 64) as control:
        def fail(*args):
            assert [int(layer.cumulative_length) for layer in control.cache.layers] == [3, 3]
            raise error("failed layer")

        handle = target.model.model.layers[-1].register_forward_hook(fail)
        try:
            with pytest.raises(error, match="failed layer"):
                control.generate([3, 5, 7], 5)
        finally:
            handle.remove()
        assert_empty(control)
        assert control.generate([3, 5, 7], 5).token_ids == expected(reference, [3, 5, 7], 5)


def test_reset_failure_closes_and_releases_owned_cache(monkeypatch):
    target, _ = models()
    control = PersistentStaticControl(target, 64)
    refs = cache_refs(control)
    monkeypatch.setattr(type(control.cache), "reset", lambda self: (_ for _ in ()).throw(RuntimeError("reset failed")))
    with pytest.raises(RuntimeError, match="reset failed"):
        control.generate([3], 1)
    gc.collect()
    assert control.closed and all(ref() is None for ref in refs)
    control.close()


def test_reentrant_generation_and_close_cannot_change_busy_cache():
    target, reference = models()
    with PersistentStaticControl(target, 64) as control:
        def inspect(*args):
            for operation in (lambda: control.generate([5], 1), control.close):
                with pytest.raises(RuntimeError, match="busy"):
                    operation()

        handle = target.model.model.layers[-1].register_forward_hook(inspect)
        try:
            assert control.generate([3, 5, 7], 5).token_ids == expected(reference, [3, 5, 7], 5)
        finally:
            handle.remove()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, 65])
def test_invalid_capacity_rejected_before_allocation(monkeypatch, capacity):
    target, _ = models()
    monkeypatch.setattr("benchmarks.hf_static.StaticCache", lambda *a, **k: pytest.fail("unexpected allocation"))
    with pytest.raises(ValueError, match="static_capacity"):
        PersistentStaticControl(target, capacity)


@pytest.mark.parametrize("prompt,budget", [([], 1), ([3], -1), ([3], True), ([3] * 64, 1),
                                         ([-1], 1), ([67], 1), ([True], 1), ([3.0], 1), ("3", 1)])
def test_invalid_request_leaves_cache_untouched(monkeypatch, prompt, budget):
    target, _ = models()
    with PersistentStaticControl(target, 64) as control:
        monkeypatch.setattr(control.cache, "reset", lambda: pytest.fail("unexpected cache mutation"))
        with pytest.raises(ValueError):
            control.generate(prompt, budget)


@pytest.mark.parametrize("change", ["train", "dtype", "geometry", "model", "attention"])
def test_changed_model_rejected_before_reset(monkeypatch, change):
    target, _ = models()
    with PersistentStaticControl(target, 64) as control:
        if change == "train":
            target.model.model.layers[-1].train()
        elif change == "dtype":
            target.model.to(dtype=torch.float16)
        elif change == "geometry":
            target.model.config.num_key_value_heads = 1
        elif change == "model":
            target.model = copy.deepcopy(target.model)
        else:
            target.model.config._attn_implementation = "flash_attention_2"
        monkeypatch.setattr(control.cache, "reset", lambda: pytest.fail("unexpected cache mutation"))
        with pytest.raises(ValueError):
            control.generate([3], 1)


@pytest.mark.parametrize("failure", ["device", "capture", "stream", "thread"])
def test_cuda_ownership_admission_without_device_work(monkeypatch, failure):
    owner = (threading.get_ident() + (failure == "thread"), 7)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1 if failure == "device" else 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: failure == "capture")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(cuda_stream=8 if failure == "stream" else 7))
    with pytest.raises(RuntimeError):
        _context(torch.device("cuda:0"), owner)


def test_live_dart_cache_is_unchanged_and_closed_target_is_rejected():
    base, reference = models()
    target = CudaGraphTarget(base, capacity=64, max_block_tokens=3, replay=False)
    with PersistentStaticControl(target, 64) as control:
        state, _ = target.prefill([11, 13, 17])
        target.forward_block(state, [19])
        original = state.cache.storage.clone()
        assert control.generate([3, 5, 7], 5).token_ids == expected(reference, [3, 5, 7], 5)
        torch.testing.assert_close(state.cache.storage, original, rtol=0, atol=0)
        assert state.seq_len == 4
        target.close()
        for budget in (0, 5):
            with pytest.raises(RuntimeError, match="closed"):
                control.generate([3], budget)
