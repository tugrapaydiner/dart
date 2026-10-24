from __future__ import annotations

import pytest
import torch

from dart.kv.cache import QuantizedKVCache, QuantSpec
from dart.models.target import TargetState
from tests.conftest import random_prompt


@pytest.mark.parametrize("relative_only", [False, True], ids=["native", "relative_only"])
@pytest.mark.parametrize("length", [0, 3, 6])
def test_crop_keeps_prefix_and_can_continue(tiny_target, monkeypatch, length, relative_only):
    prompt = random_prompt(91, 6)
    state, _ = tiny_target.prefill(prompt)
    if relative_only:
        crop = state.cache.crop

        def remove_tail(count):
            if count > 0:
                raise ValueError("only negative removal counts are supported")
            if count < 0:
                crop(count)

        monkeypatch.setattr(state.cache, "crop", remove_tail)

    tiny_target.crop(state, length)
    assert state.seq_len == length
    actual = tiny_target.forward_block(state, [23])[-1]
    _, expected = tiny_target.prefill(prompt[:length] + [23])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert state.seq_len == length + 1


@pytest.mark.parametrize("length", [-1, 7])
def test_crop_rejects_invalid_length_without_changing_cache(tiny_target, length):
    state, _ = tiny_target.prefill(random_prompt(92, 6))
    with pytest.raises(ValueError, match="cannot crop"):
        tiny_target.crop(state, length)
    assert state.seq_len == 6


@pytest.mark.parametrize("length", [0, 5, 8, 10])
def test_crop_supports_quantized_draft_cache(tiny_target, length):
    cache = QuantizedKVCache(QuantSpec(bits=4, sink_len=2, window_len=3))
    keys = torch.randn(1, 2, 10, 8)
    values = torch.randn_like(keys)
    expected_keys, expected_values = cache.update(keys, values, 0)
    state = TargetState(cache=cache)

    tiny_target.crop(state, length)
    assert state.seq_len == length
    next_keys, next_values = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
    actual_keys, actual_values = cache.update(next_keys, next_values, 0)
    assert state.seq_len == length + 1
    if length < 10:  # A full cache ages one extra token when it grows.
        torch.testing.assert_close(actual_keys[..., :length, :], expected_keys[..., :length, :])
        torch.testing.assert_close(actual_values[..., :length, :], expected_values[..., :length, :])
    torch.testing.assert_close(actual_keys[..., -1:, :], next_keys)
    torch.testing.assert_close(actual_values[..., -1:, :], next_values)
