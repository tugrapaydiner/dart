"""Target-wrapper tests: plumbing must be airtight before anything
else, because the verifiers and baselines are all built on it."""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from tests.conftest import build_tiny_target, random_prompt


@pytest.mark.parametrize("seed,prompt_len,max_new", [(7, 9, 24), (8, 17, 40), (9, 3, 12)])
def test_greedy_baseline_matches_hf_generate(tiny_target, seed, prompt_len, max_new):
    """Our incremental greedy loop == HuggingFace generate(greedy).

    Validates KV cache handling, attention masks, and position bookkeeping
    in TargetModel against the reference HF implementation.
    """
    prompt = random_prompt(seed, prompt_len)
    ours = tiny_target.greedy_generate(prompt, max_new_tokens=max_new)

    input_ids = torch.tensor([prompt])
    hf_out = tiny_target.model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        pad_token_id=0,
    )
    theirs = hf_out[0, len(prompt):].tolist()
    assert ours == theirs


def test_crop_restores_exact_state(tiny_target):
    """KV rollback must restore the same *decisions* as a state that never
    saw the cropped tokens.

    Crop itself is pure slicing (bit-exact); what differs is that state A
    computed KV for [10, 11] inside a 5-token block and state B inside a
    2-token block.  GEMM reduction order varies with block shape, so
    logits agree only to FP noise (measured ~7e-8 on this setup, vs a
    top-2 margin of ~5e-2 - six orders of magnitude of headroom).  The
    contract therefore asserts argmax stability + tight allclose; see
    docs/exactness_contract.md, "Floating-point chunk-shape caveat".
    """
    prompt = random_prompt(11, 8)

    state_a, _ = tiny_target.prefill(prompt)
    tiny_target.forward_block(state_a, [10, 11, 12, 13, 14])
    tiny_target.crop(state_a, len(prompt) + 2)  # keep prompt + [10, 11]
    logits_a = tiny_target.forward_block(state_a, [50])

    state_b, _ = tiny_target.prefill(prompt)
    tiny_target.forward_block(state_b, [10, 11])
    logits_b = tiny_target.forward_block(state_b, [50])

    assert state_a.seq_len == state_b.seq_len
    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)
    assert torch.equal(logits_a.argmax(dim=-1), logits_b.argmax(dim=-1))


def test_prefill_rejects_empty_prompt(tiny_target):
    with pytest.raises(ValueError):
        tiny_target.prefill([])


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@torch.inference_mode()
def test_dense_llama_uses_no_padding_mask(backend):
    target = build_tiny_target()
    target.model.set_attn_implementation(backend)
    target.record_hidden = True
    reference_cache = DynamicCache()
    masks = []

    def record_mask(module, args, kwargs):
        masks.append(kwargs.get("attention_mask"))

    state = None
    for tokens, keep in (([0, 7, 0, 11], None), ([0], None), ([13, 0, 15], None), ([19, 21], 5)):
        if keep is not None:
            target.crop(state, keep)
            reference_cache.crop(keep)
        past = reference_cache.get_seq_length()
        with target.model.register_forward_pre_hook(record_mask, with_kwargs=True):
            if state is None:
                state, actual = target.prefill(tokens)
            else:
                actual = target.forward_block(state, tokens)
        expected = target.model(input_ids=torch.tensor([tokens]), past_key_values=reference_cache,
                                attention_mask=torch.ones((1, past + len(tokens)), dtype=torch.long),
                                use_cache=True, output_hidden_states=True)
        reference_cache = expected.past_key_values
        logits = expected.logits[0] if past else expected.logits[0, -1]
        torch.testing.assert_close(actual, logits, rtol=1e-5, atol=1e-6)
        assert torch.equal(actual.argmax(-1), logits.argmax(-1))
        torch.testing.assert_close(target.last_hidden_block, expected.hidden_states[-1][0], rtol=0, atol=0)
        assert state.seq_len == reference_cache.get_seq_length() == past + len(tokens)
        for layer, reference in zip(state.cache.layers, reference_cache.layers, strict=True):
            torch.testing.assert_close(layer.keys, reference.keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, reference.values, rtol=0, atol=0)
    assert masks == [None] * 4

    with target.model.register_forward_pre_hook(record_mask, with_kwargs=True):
        target.forward_tree(state, [7], [0, 8, 11], [-1, -1, 0])
    assert masks[-1].shape == (1, 1, 4, 11)
    assert masks[-1][0, 0, 3, -2] < 0  # The sibling is hidden.


@pytest.mark.parametrize("kind", ["wrapped_forward", "wrapped_inner_forward", "configured_cache", "cache_subclass"])
def test_other_target_calls_keep_the_explicit_padding_mask(kind, monkeypatch):
    target = build_tiny_target()
    cache = None
    if kind in ("wrapped_forward", "wrapped_inner_forward"):
        model = target.model if kind == "wrapped_forward" else target.model.model
        forward = model.forward
        monkeypatch.setattr(model, "forward", lambda *args, **kwargs: forward(*args, **kwargs))
    elif kind == "configured_cache":
        cache = DynamicCache(config=target.model.config)
    else:
        class CustomCache(DynamicCache):
            pass

        cache = CustomCache()
    masks = []
    with target.model.register_forward_pre_hook(lambda mod, args, kw: masks.append(kw["attention_mask"]), with_kwargs=True):
        state, _ = target.prefill([0, 7], cache=cache)
        target.forward_block(state, [8])
    for mask, length in zip(masks, (2, 3), strict=True):
        torch.testing.assert_close(mask, torch.ones((1, length), dtype=torch.long), rtol=0, atol=0)
