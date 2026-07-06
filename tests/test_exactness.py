"""End-to-end exactness: DartEngine output must be byte-identical to the
trusted greedy baseline for every drafter, including adversarial ones.

This is the foundational deliverable (docs/design.md §2):
    "DART exact mode produces identical output to target greedy."
"""

from __future__ import annotations

import pytest
import torch

from dart.drafters.base import NullDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig
from tests.conftest import VOCAB, random_prompt


class OracleDrafter:
    """Cheating drafter that proposes the target's true continuation.

    Upper bound on acceptance: everything must be accepted, output exact.
    """

    name = "oracle"

    def __init__(self, target):
        self.target = target

    def propose(self, committed, max_tokens):
        if max_tokens <= 0:
            return []
        # Continuation of the *full* committed sequence (incl. pending).
        return self.target.greedy_generate(committed, max_new_tokens=max_tokens)

    def observe(self, result):
        pass


class GarbageDrafter:
    """Adversarial drafter proposing seeded random tokens.

    Lower bound on acceptance: nearly everything rejected, output must
    STILL be exact - the verifier is what guarantees correctness.
    """

    name = "garbage"

    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)

    def propose(self, committed, max_tokens):
        if max_tokens <= 0:
            return []
        return torch.randint(0, VOCAB, (max_tokens,), generator=self.g).tolist()

    def observe(self, result):
        pass


class ConstantDrafter:
    """Proposes the same token forever; near-zero acceptance."""

    name = "constant"

    def propose(self, committed, max_tokens):
        return [7] * max(0, max_tokens)

    def observe(self, result):
        pass


PROMPTS = [(61, 5), (62, 11), (63, 19), (64, 30), (65, 7)]


def _baseline(target, prompt, n):
    return target.greedy_generate(prompt, max_new_tokens=n)


@pytest.mark.parametrize("verifier", ["block", "reference"])
@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_engine_target_only_is_baseline(tiny_target, verifier, seed, plen):
    prompt = random_prompt(seed, plen)
    n = 32
    engine = DartEngine(tiny_target, NullDrafter(), verifier=verifier)
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=n))
    assert out.token_ids == _baseline(tiny_target, prompt, n)


@pytest.mark.parametrize("drafter_cls", [GarbageDrafter, ConstantDrafter])
@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_engine_exact_under_adversarial_drafts(tiny_target, drafter_cls, seed, plen):
    prompt = random_prompt(seed, plen)
    n = 32
    engine = DartEngine(tiny_target, drafter_cls(), verifier="block")
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=n, draft_len=4))
    base = _baseline(tiny_target, prompt, n)
    assert out.token_ids == base
    assert out.metadata["output_hash"] == output_hash(base)
    assert out.metadata["exact_greedy_equivalent"] is True


@pytest.mark.parametrize("seed,plen", PROMPTS[:3])
def test_engine_exact_with_oracle_drafter(tiny_target, seed, plen):
    prompt = random_prompt(seed, plen)
    n = 24
    base = _baseline(tiny_target, prompt, n)
    if any(t in tiny_target.eos_token_ids for t in base):
        pytest.skip("stream hits EOS; acceptance accounting differs at EOS")
    engine = DartEngine(tiny_target, OracleDrafter(tiny_target), verifier="block")
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=n, draft_len=4))
    assert out.token_ids == base
    # A perfect drafter must achieve full acceptance every round.
    assert out.metadata["acceptance_rate"] == 1.0
    # And materially fewer verify rounds than tokens.
    assert out.metadata["rounds"] < len(base)


def test_engine_respects_token_budget_exactly(tiny_target):
    prompt = random_prompt(71, 9)
    for n in (1, 2, 3, 5):
        engine = DartEngine(tiny_target, GarbageDrafter(), verifier="block")
        out = engine.generate(prompt, DecodeConfig(max_new_tokens=n, draft_len=4))
        assert out.token_ids == _baseline(tiny_target, prompt, n)
        assert len(out.token_ids) <= n


class OverEagerDrafter:
    """Ignores the requested budget and over-returns drafts."""

    name = "overeager"

    def propose(self, committed, max_tokens):
        return [7] * (max_tokens + 50)

    def observe(self, result):
        pass


def test_engine_clamps_over_returning_drafter_to_round_budget(tiny_target):
    """A drafter that over-returns must not push extra tokens into the
    verify pass: proposals are clamped to min(draft_len, remaining budget),
    and output stays exact."""
    prompt = random_prompt(72, 9)
    engine = DartEngine(tiny_target, OverEagerDrafter(), verifier="block")
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=5, draft_len=4))
    assert out.token_ids == _baseline(tiny_target, prompt, 5)
    assert all(rec.n_draft_proposed <= 4 for rec in out.rounds)


def test_engine_stops_at_forced_eos(tiny_target):
    prompt = random_prompt(81, 12)
    base_free = _baseline(tiny_target, prompt, 16)
    if len(base_free) < 4:
        pytest.skip("stream too short for this seed")
    forced_eos = base_free[3]
    saved = tiny_target.eos_token_ids
    tiny_target.eos_token_ids = frozenset({forced_eos})
    try:
        base = _baseline(tiny_target, prompt, 16)
        engine = DartEngine(tiny_target, GarbageDrafter(), verifier="block")
        out = engine.generate(prompt, DecodeConfig(max_new_tokens=16, draft_len=4))
        assert out.token_ids == base
        assert out.token_ids[-1] == forced_eos
        assert out.metadata["finished_eos"] is True
    finally:
        tiny_target.eos_token_ids = saved
