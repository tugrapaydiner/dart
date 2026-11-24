from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
import torch

from benchmarks.run_bench import build_hf_method
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target, random_prompt
from tests.test_sampling_devices import DEVICES


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length,full", [(1, True), (7, True), (7, False)])
def test_prefill_row_storage(dtype, length, full):
    target = TargetModel(build_tiny_target().model, dtype=dtype)
    outputs = []

    def capture(_model, _args, output):
        outputs.append(output.logits)

    with target.model.register_forward_hook(capture), patch.object(target, "_last_prefill_logits", not full):
        state, logits = target.prefill(random_prompt(603, length))
    raw, = outputs
    assert raw.shape[1] == (length if full else 1) and state.seq_len == length
    last = raw[0, -1]
    assert logits.dtype == torch.float32 and torch.equal(logits, last.float())
    assert logits.untyped_storage().nbytes() == logits.numel() * logits.element_size()
    if raw.dtype == torch.float32:
        assert (logits.data_ptr() == last.data_ptr()) == (raw.shape[1] == 1)


@pytest.mark.parametrize("first_slot", [False, True])
def test_prefill_detaches_a_single_row_view_of_a_larger_buffer(first_slot):
    target = build_tiny_target()
    prompt = [7, 8, 9, 10]
    views = []

    def project_one(_model, _args, output):
        raw = output.logits
        assert raw.shape[1] == len(prompt)
        if first_slot:
            raw[:, :1].copy_(raw[:, -1:])
            output.logits = raw[:, :1]
        else:
            output.logits = raw[:, -1:]
        views.append(output.logits)

    with target.model.register_forward_hook(project_one), patch.object(target, "_last_prefill_logits", False):
        state, logits = target.prefill(prompt)
    view, = views
    assert state.seq_len == len(prompt) and torch.equal(logits, view[0, 0])
    assert view.untyped_storage().nbytes() > logits.numel() * logits.element_size()
    assert logits.untyped_storage().nbytes() == logits.numel() * logits.element_size()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_projection_keeps_hidden_states_cache_and_verification_logits(device, dtype):
    original = build_tiny_target()
    target = TargetModel(original.model, device=device, dtype=dtype, record_hidden=True)
    prompt = random_prompt(601, 7)
    rows = []
    handle = target.model.get_output_embeddings().register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[1]))
    try:
        state, logits = target.prefill(prompt)
        hidden = target.last_hidden_block.clone()
        with patch.object(target, "_last_prefill_logits", False):
            full_state, full_logits = target.prefill(prompt)
        assert hidden.shape == (len(prompt), 64)
        assert torch.equal(hidden, target.last_hidden_block)
        assert state.seq_len == full_state.seq_len == len(prompt)
        torch.testing.assert_close(logits, full_logits, rtol=0.02 if dtype == torch.bfloat16 else 1e-5, atol=0.002 if dtype == torch.bfloat16 else 1e-6)
        assert logits.argmax() == full_logits.argmax()
        block = target.forward_block(state, [18, 19, 20])
        expected_block = target.forward_block(full_state, [18, 19, 20])
        torch.testing.assert_close(block, expected_block, rtol=0, atol=0)
        assert state.seq_len == full_state.seq_len == len(prompt) + 3
        assert target.last_hidden_block.shape == (3, 64)
        assert target.all_logits(prompt).shape == (len(prompt), 211)
        assert target.forward_tree(state, [21], [22, 23, 24], [-1, 0, -1]).shape == (4, 211)
        assert target.last_hidden_block.shape == (4, 64)
        prefill_rows = 1 if torch.device(device).type == "cpu" else len(prompt)
        assert rows == [prefill_rows, len(prompt), 3, 3, len(prompt), 4]
    finally:
        handle.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_restriction_matches_full_hf_generations(device, dtype):
    original = build_tiny_target()
    target = TargetModel(original.model, device=device, dtype=dtype)
    prompt = [7, 8, 9, 7, 8, 9]
    expected = build_hf_method(target, 12)(prompt).token_ids
    assert target.greedy_generate(prompt, 12) == expected
    for drafter in (None, NGramDrafter()):
        with DartEngine(target, drafter) as engine:
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=12, draft_len=4))
        assert output.token_ids == expected


@pytest.mark.parametrize("record_hidden", [False, True])
@pytest.mark.parametrize("wrapper", ["legacy", "kwargs_only", "positional_only"])
def test_models_without_explicit_logit_restriction_use_full_prefill(monkeypatch, record_hidden, wrapper):
    original = build_tiny_target()
    forward = original.model.forward

    def legacy(input_ids, attention_mask, past_key_values, use_cache, output_hidden_states):
        return forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                       use_cache=use_cache, output_hidden_states=output_hidden_states)

    def kwargs_only(**kwargs):
        assert "logits_to_keep" not in kwargs
        return forward(**kwargs)

    def positional_only(logits_to_keep=0, /, **kwargs):
        assert logits_to_keep == 0 and "logits_to_keep" not in kwargs
        return forward(**kwargs)

    wrappers = {"legacy": legacy, "kwargs_only": kwargs_only, "positional_only": positional_only}
    monkeypatch.setattr(original.model, "forward", wrappers[wrapper])
    target = TargetModel(original.model, record_hidden=record_hidden)
    rows = []
    handle = target.model.get_output_embeddings().register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[1]))
    try:
        state, logits = target.prefill([7, 8, 9])
        assert state.seq_len == 3 and logits.shape == (211,)
        assert rows == [3]
        assert (target.last_hidden_block is not None) is record_hidden
    finally:
        handle.remove()


@pytest.mark.parametrize("error", [ValueError, TypeError])
def test_uninspectable_forward_falls_back_to_full_logits(monkeypatch, error):
    original = build_tiny_target()

    def unavailable(call):
        raise error("signature unavailable")

    with monkeypatch.context() as context:
        context.setattr(inspect, "signature", unavailable)
        target = TargetModel(original.model)
    rows = []
    handle = target.model.get_output_embeddings().register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[1]))
    try:
        state, logits = target.prefill([7, 8, 9])
        assert state.seq_len == 3 and logits.shape == (211,)
        assert rows == [3]
    finally:
        handle.remove()


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("length", [1, 17])
@pytest.mark.parametrize("record_hidden", [False, True])
def test_terminal_prompt_matches_cached_scores_without_retaining_keys(attention, length, record_hidden):
    target = build_tiny_target()
    target.model.set_attn_implementation(attention)
    target.record_hidden = record_hidden
    prompt = random_prompt(71, length)
    state, expected = target.prefill(prompt)
    hidden = target.last_hidden_block
    seen = []

    with target.model.register_forward_hook(lambda model, args, out: seen.append(out)):
        tokens = target.greedy_generate(prompt, 1)

    output, = seen
    assert output.past_key_values is None
    torch.testing.assert_close(output.logits[0, -1], expected, rtol=1e-4, atol=1e-5)
    assert tokens == [int(expected.argmax())] and state.seq_len == length
    if record_hidden:
        torch.testing.assert_close(target.last_hidden_block, hidden, rtol=1e-4, atol=1e-5)
    else:
        assert target.last_hidden_block is None


@pytest.mark.parametrize("mode", ["exact", "exact_sampling", "approx"])
def test_terminal_prompt_engine_keeps_first_token_and_request_hooks(monkeypatch, mode):
    target = build_tiny_target()
    target.record_hidden = True
    prompt = [7, 9, 11]
    _, logits = target.prefill(prompt)
    hidden = target.last_hidden_block
    seed = 41
    if mode == "exact_sampling":
        expected = int(torch.multinomial(logits.softmax(-1), 1, generator=torch.Generator().manual_seed(seed)))
    else:
        expected = int(logits.argmax())
    target.eos_token_ids = frozenset([expected])
    drafter = NGramDrafter()
    events = []
    seen = []

    def after_prefill():
        torch.testing.assert_close(target.last_hidden_block, hidden, rtol=1e-4, atol=1e-5)
        events.append("prefill")

    monkeypatch.setattr(drafter, "seed_request", lambda seed: events.append("seed"), raising=False)
    monkeypatch.setattr(drafter, "after_prefill", after_prefill, raising=False)
    monkeypatch.setattr(drafter, "finish_request", lambda: events.append("finish"), raising=False)
    with DartEngine(target, drafter) as engine, target.model.register_forward_hook(
        lambda model, args, out: seen.append(out.past_key_values)
    ):
        result = engine.generate(prompt, DecodeConfig(mode=mode, max_new_tokens=1, seed=seed))
    assert seen == [None] and result.token_ids == [expected]
    assert result.rounds == [] and result.metadata["finished_eos"] and result.metadata["tokens_verified"] == 1
    assert events == ["seed", "prefill", "finish"] and target.last_hidden_block is None


@pytest.mark.parametrize("entry", ["baseline", "engine"])
@pytest.mark.parametrize("case", ["two_tokens", "custom_prefill", "custom_model", "bfloat16"])
def test_terminal_prompt_preserves_cached_generation_for_other_paths(entry, case):
    target = build_tiny_target()
    if case == "bfloat16":
        target = TargetModel(target.model, dtype=torch.bfloat16)
    limit = 2 if case == "two_tokens" else 1
    seen = []
    with patch.object(target, "eos_token_ids", frozenset()), target.model.register_forward_hook(
        lambda model, args, out: seen.append(out.past_key_values)
    ):
        def generate():
            if entry == "baseline":
                return target.greedy_generate([7, 9, 11], limit)
            return DartEngine(target).generate([7, 9, 11], DecodeConfig(max_new_tokens=limit)).token_ids

        if case == "custom_prefill":
            with patch.object(target, "prefill", wraps=target.prefill) as prefill:
                assert len(generate()) == limit
                prefill.assert_called_once()
        elif case == "custom_model":
            with patch.object(target.model, "forward", wraps=target.model.forward):
                assert len(generate()) == limit
        else:
            assert len(generate()) == limit
    assert len(seen) == limit and all(cache is not None for cache in seen)


def test_terminal_prompt_clears_hidden_after_a_failed_forward():
    target = build_tiny_target()
    target.record_hidden = True
    prompt = [7, 9, 11]
    _, logits = target.prefill(prompt)

    def fail(model, args, out):
        if out.past_key_values is None:
            raise RuntimeError("terminal forward failed")

    with target.model.register_forward_hook(fail), pytest.raises(RuntimeError, match="terminal forward failed"):
        target.greedy_generate(prompt, 1)
    assert target.last_hidden_block is None
    assert target.greedy_generate(prompt, 1) == [int(logits.argmax())]


def _first_token(target, prompt, entry):
    if entry == "baseline":
        return target.greedy_generate(prompt, 1)
    with DartEngine(target) as engine:
        return engine.generate(prompt, DecodeConfig(max_new_tokens=1)).token_ids


@pytest.mark.parametrize("entry", ["baseline", "engine"])
@pytest.mark.parametrize("method", ["prefill", "_forward"])
@pytest.mark.parametrize("binding", ["class", "other_target"])
def test_terminal_prompt_honors_target_method(monkeypatch, entry, method, binding):
    target = build_tiny_target()
    prompt = [7, 9, 11]
    _, logits = target.prefill(prompt)
    expected = int(logits.roll(1).argmax())

    if binding == "class":
        original = getattr(TargetModel, method)

        def shift(owner, *args, **kwargs):
            result = original(owner, *args, **kwargs)
            if method == "prefill":
                state, scores = result
                return state, scores.roll(1, -1)
            return result.roll(1, -1)

        monkeypatch.setattr(TargetModel, method, shift)
    else:
        other = TargetModel(target.model)
        run = other._run_model
        monkeypatch.setattr(other, "_run_model", lambda *a, **kw: run(*a, **kw).roll(1, -1))
        monkeypatch.setattr(target, method, getattr(other, method))

    assert expected != int(logits.argmax())
    assert _first_token(target, prompt, entry) == [expected]


@pytest.mark.parametrize("entry", ["baseline", "engine"])
@pytest.mark.parametrize("component", ["lm", "decoder"])
def test_terminal_prompt_honors_model_forward(monkeypatch, entry, component):
    target = build_tiny_target()
    prompt = [7, 9, 11]
    _, expected = target.prefill(prompt)
    module = target.model if component == "lm" else target.model.model
    original = type(module).forward
    calls = []

    def forward(owner, *args, **kwargs):
        calls.append(owner)
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(type(module), "forward", forward)
    seen = []

    def inputs(_module, _args, kwargs):
        seen.append((kwargs["use_cache"], kwargs["attention_mask"] is not None))

    with target.model.register_forward_pre_hook(inputs, with_kwargs=True):
        assert _first_token(target, prompt, entry) == [int(expected.argmax())]
    assert calls == [module]
    assert seen == [(True, True)]
