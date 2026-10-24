"""Exact speculative sampling (docs/design.md §5.3) and approx mode (§2.3).

The sampling guarantee is distributional: speculative_accept_step composed
with residual resampling must emit tokens distributed exactly as the
target's p - tested both analytically-by-Monte-Carlo on the production
code path and end-to-end (temperature -> 0 degenerates to greedy, which
IS trace-comparable).
"""

from __future__ import annotations

import pytest
import torch

from dart.drafters.ar_draft import ARDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.sampling import speculative_accept_step
from tests.conftest import random_prompt
from tests.test_exactness import ConstantDrafter, GarbageDrafter, OracleDrafter


# ------------------------------------------------------------------ #
# The math: emitted distribution == target distribution
# ------------------------------------------------------------------ #
def _mc_emit_distribution(p, q_row, n, seed=0):
    """Monte-Carlo the production accept/resample step; q_row=None means the
    proposal is one-hot at a fixed draft index (delta proposal)."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros_like(p)
    for _ in range(n):
        if q_row is None:
            d = 3  # fixed draft token: proposal distribution is delta_3
        else:
            d = int(torch.multinomial(q_row, 1, generator=g).item())
        tok, _ = speculative_accept_step(p, q_row, d, g)
        counts[tok] += 1
    return counts / n


@pytest.mark.parametrize("seed", [0, 1])
def test_speculative_step_preserves_p_with_onehot_proposal(seed):
    torch.manual_seed(seed)
    p = torch.softmax(torch.randn(7), dim=-1)
    emp = _mc_emit_distribution(p, None, n=40_000, seed=seed)
    assert (emp - p).abs().max().item() < 0.015  # >4 sigma at n=40k


@pytest.mark.parametrize("seed", [2, 3])
def test_speculative_step_preserves_p_with_full_q(seed):
    torch.manual_seed(seed)
    p = torch.softmax(torch.randn(7), dim=-1)
    q = torch.softmax(torch.randn(7) * 1.5, dim=-1)  # deliberately mismatched
    emp = _mc_emit_distribution(p, q, n=40_000, seed=seed)
    assert (emp - p).abs().max().item() < 0.015


def test_speculative_step_p_equals_q_always_accepts():
    p = torch.softmax(torch.arange(5).float(), dim=-1)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        d = int(torch.multinomial(p, 1, generator=g).item())
        tok, accepted = speculative_accept_step(p, p.clone(), d, g)
        assert accepted and tok == d  # min(1, p/p) = 1


def test_speculative_step_never_emits_zero_probability_token():
    """Regression: a draft with q(d)=0 AND p(d)=0 (a lying/buggy drafter)
    must be rejected, never emitted - the emitted distribution must stay
    inside the target's support no matter what the proposal claims."""
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.5, 0.0])
    q = torch.tensor([0.0, 0.0, 1.0])
    for _ in range(100):
        tok, accepted = speculative_accept_step(p, q, 2, g)  # p(2) == q(2) == 0
        assert not accepted
        assert tok in (0, 1)  # resampled from the residual, i.e. from p
    # The q-underflow limit is preserved: q(d)=0 with p(d)>0 still accepts
    # (p/q -> inf), so numerically-underflowed proposals cost nothing.
    p2 = torch.tensor([0.2, 0.8])
    q2 = torch.tensor([1.0, 0.0])
    tok, accepted = speculative_accept_step(p2, q2, 1, g)
    assert accepted and tok == 1


# ------------------------------------------------------------------ #
# Engine integration: exact_sampling mode
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("seed,plen", [(201, 7), (202, 14)])
def test_sampling_at_near_zero_temperature_equals_greedy(tiny_target, seed, plen):
    prompt = random_prompt(seed, plen)
    base = tiny_target.greedy_generate(prompt, max_new_tokens=24)
    engine = DartEngine(tiny_target, GarbageDrafter(), verifier="block")
    out = engine.generate(
        prompt,
        DecodeConfig(
            mode="exact_sampling", max_new_tokens=24, draft_len=4,
            temperature=1e-5, seed=0,
        ),
    )
    assert out.token_ids == base  # T->0: sampling degenerates to greedy
    assert out.metadata["distribution_exact"] is True
    assert out.metadata["exact_greedy_equivalent"] is None
    assert out.metadata["greedy_verified"] is False


def test_sampling_is_seed_deterministic(tiny_target):
    """Reproducibility needs BOTH seeds fixed: the engine's sampling seed
    and the drafter's own RNG (drafts steer which tokens get accepted, so
    a drafter with advancing state changes trajectories run-to-run)."""
    prompt = random_prompt(203, 9)

    def run(cfg_seed):
        engine = DartEngine(tiny_target, GarbageDrafter(seed=1), verifier="block")
        return engine.generate(
            prompt,
            DecodeConfig(
                mode="exact_sampling", max_new_tokens=20, draft_len=4,
                temperature=1.0, seed=cfg_seed,
            ),
        ).token_ids

    a, b, c = run(7), run(7), run(8)
    assert a == b
    assert a != c  # different seed, different trajectory (a.s. on 211 vocab)


def test_sampling_with_self_draft_q_high_acceptance(tiny_target):
    """Self-drafting with matching temperature: q == p up to chunk-shape FP
    noise, so acceptance must be near-perfect."""
    prompt = random_prompt(204, 10)
    drafter = ARDrafter(tiny_target, sample_temperature=1.0, seed=5)
    engine = DartEngine(tiny_target, drafter, verifier="block")
    out = engine.generate(
        prompt,
        DecodeConfig(
            mode="exact_sampling", max_new_tokens=24, draft_len=4,
            temperature=1.0, seed=11,
        ),
    )
    assert len(out.token_ids) <= 24
    assert out.metadata["acceptance_rate"] > 0.9


# ------------------------------------------------------------------ #
# Approx mode (docs/design.md §2.3)
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("seed,plen", [(211, 8), (212, 16)])
def test_approx_with_verify_every_1_is_exact(tiny_target, seed, plen):
    prompt = random_prompt(seed, plen)
    base = tiny_target.greedy_generate(prompt, max_new_tokens=24)
    engine = DartEngine(tiny_target, GarbageDrafter(), verifier="block")
    out = engine.generate(
        prompt, DecodeConfig(mode="approx", verify_every=1, max_new_tokens=24)
    )
    assert out.token_ids == base
    assert out.metadata["exact_greedy_equivalent"] is None
    assert out.metadata["greedy_verified"] is True
    assert out.metadata["unverified_tokens"] == 0
    assert out.metadata["quality_risk"] is False


def test_approx_oracle_skips_stay_on_greedy_stream(tiny_target):
    """Unverified tokens from a perfect drafter ARE the greedy stream: the
    audit must report zero divergence and the output must match baseline."""
    prompt = random_prompt(213, 11)
    base = tiny_target.greedy_generate(prompt, max_new_tokens=24)
    if any(t in tiny_target.eos_token_ids for t in base):
        pytest.skip("EOS in stream")
    engine = DartEngine(tiny_target, OracleDrafter(tiny_target), verifier="block")
    out = engine.generate(
        prompt,
        DecodeConfig(mode="approx", verify_every=2, max_new_tokens=24, draft_len=4),
    )
    assert out.token_ids == base
    assert out.metadata["pending_divergence"] == 0
    assert out.metadata["unverified_tokens"] > 0  # skips actually happened


def test_approx_garbage_diverges_and_reports_risk(tiny_target):
    prompt = random_prompt(214, 12)
    base = tiny_target.greedy_generate(prompt, max_new_tokens=24)
    engine = DartEngine(tiny_target, GarbageDrafter(seed=3), verifier="block")
    out = engine.generate(
        prompt,
        DecodeConfig(mode="approx", verify_every=3, max_new_tokens=24, draft_len=4),
    )
    assert out.metadata["unverified_tokens"] > 0
    assert out.token_ids != base  # garbage committed unverified
    assert out.metadata["pending_divergence"] > 0
    assert out.metadata["exact_greedy_equivalent"] is None
    assert out.metadata["greedy_verified"] is False
    assert out.metadata["quality_risk"] is True
    assert len(out.token_ids) <= 24


def test_approx_unverified_eos_finishes_with_risk_flagged(tiny_target):
    eos = next(iter(tiny_target.eos_token_ids))
    prompt = random_prompt(215, 9)
    engine = DartEngine(tiny_target, ConstantDrafter(), verifier="block")
    engine.drafter.propose = lambda committed, n: [eos] * max(0, n)  # type: ignore
    out = engine.generate(
        prompt,
        DecodeConfig(mode="approx", verify_every=2, max_new_tokens=24, draft_len=4),
    )
    assert out.metadata["finished_eos"] is True
    assert out.metadata["exact_greedy_equivalent"] is None
    assert out.metadata["greedy_verified"] is False
