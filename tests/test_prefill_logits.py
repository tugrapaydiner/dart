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
