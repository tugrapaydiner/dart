"""Target-wrapper tests: plumbing must be airtight before anything
else, because the verifiers and baselines are all built on it."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import random_prompt


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
