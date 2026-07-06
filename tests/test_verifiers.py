"""Verifier equivalence tests: block verifier == reference verifier, and both commit
only target-greedy tokens.  This is the exactness contract at round level."""

from __future__ import annotations

import pytest
import torch

from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from tests.conftest import EOS_ID, VOCAB, random_prompt


def _fresh_round(target, prompt):
    """Prefill and return (state, pending) per the decode-loop invariant."""
    state, logits = target.prefill(prompt)
    pending = int(torch.argmax(logits).item())
    return state, pending


def _oracle_continuation(target, prompt, n):
    """True greedy continuation (pending is element 0)."""
    return target.greedy_generate(prompt, max_new_tokens=n)


def _draft_cases(target, prompt, k=6):
    """A matrix of draft blocks covering the interesting regimes."""
    oracle = _oracle_continuation(target, prompt, k + 2)
    drafts_after_pending = oracle[1 : 1 + k]  # drafts continue after pending
    g = torch.Generator().manual_seed(99)
    garbage = torch.randint(5, VOCAB, (k,), generator=g).tolist()
    return {
        "empty": [],
        "garbage": garbage,
        "oracle_full": list(drafts_after_pending),
        "oracle_prefix_then_garbage": drafts_after_pending[: k // 2]
        + garbage[: k - k // 2],
        "immediate_mismatch": [(drafts_after_pending[0] + 1) % VOCAB] + garbage[: k - 1],
        "eos_in_drafts": drafts_after_pending[:2] + [EOS_ID] + garbage[: k - 3],
    }


@pytest.mark.parametrize("seed,plen", [(21, 6), (22, 12), (23, 25)])
def test_block_matches_reference(tiny_target, seed, plen):
    prompt = random_prompt(seed, plen)
    for case_name, drafts in _draft_cases(tiny_target, prompt).items():
        state_b, pending_b = _fresh_round(tiny_target, prompt)
        state_r, pending_r = _fresh_round(tiny_target, prompt)
        assert pending_b == pending_r

        res_b = verify_block_greedy(tiny_target, state_b, pending_b, list(drafts))
        res_r = verify_greedy_reference(tiny_target, state_r, pending_r, list(drafts))

        assert res_b == res_r, f"case {case_name!r}: {res_b} != {res_r}"
        assert state_b.seq_len == state_r.seq_len, f"case {case_name!r}: KV length"


@pytest.mark.parametrize("seed,plen", [(31, 8), (32, 16)])
def test_committed_tokens_are_greedy_stream_prefix(tiny_target, seed, plen):
    """Whatever the drafts are, committed tokens must be the next tokens of
    the target's own greedy stream - the round-level exactness property."""
    prompt = random_prompt(seed, plen)
    oracle = _oracle_continuation(tiny_target, prompt, 10)
    for case_name, drafts in _draft_cases(tiny_target, prompt).items():
        state, pending = _fresh_round(tiny_target, prompt)
        res = verify_block_greedy(tiny_target, state, pending, list(drafts))
        expected = oracle[1 : 1 + len(res.committed)]
        assert res.committed == expected, f"case {case_name!r}"


def test_full_oracle_acceptance(tiny_target):
    """A perfect drafter must get everything accepted plus a bonus token."""
    prompt = random_prompt(41, 10)
    k = 5
    oracle = _oracle_continuation(tiny_target, prompt, k + 2)
    if EOS_ID in oracle[: k + 1]:
        pytest.skip("oracle stream hit EOS; covered by EOS tests")
    state, pending = _fresh_round(tiny_target, prompt)
    res = verify_block_greedy(tiny_target, state, pending, oracle[1 : 1 + k])
    assert res.all_accepted and res.n_draft_accepted == k
    assert len(res.committed) == k + 1  # k drafts + bonus token
    assert res.committed == oracle[1 : k + 2]


def test_accepted_eos_mid_block(tiny_target):
    """If the target really wants EOS and the drafter proposes it, the round
    must truncate at EOS with no correction token after it."""
    prompt = random_prompt(51, 9)
    oracle = _oracle_continuation(tiny_target, prompt, 8)
    if len(oracle) < 3:
        pytest.skip("stream too short")

    # Force EOS semantics onto a token the model genuinely emits at step 3.
    forced_eos = oracle[2]
    object.__setattr__(tiny_target, "eos_token_ids", frozenset({forced_eos}))
    try:
        state, pending = _fresh_round(tiny_target, prompt)
        drafts = oracle[1:6]  # contains forced_eos at index 1
        res = verify_block_greedy(tiny_target, state, pending, drafts)
        assert res.finished
        assert res.committed == oracle[1:3]  # ends exactly at forced EOS
        assert res.committed[-1] == forced_eos
    finally:
        object.__setattr__(tiny_target, "eos_token_ids", frozenset({EOS_ID}))
