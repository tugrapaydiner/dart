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
    assert out.metadata["exact_greedy_equivalent"] is None
    assert out.metadata["greedy_verified"] is True


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


@pytest.mark.parametrize("mode,verifier,hook", [
    ("exact", "block", "plain"), ("exact", "reference", "plain"), ("approx", "block", "plain"),
    ("exact_sampling", "block", "probabilities"), ("exact", "tree", "plain"),
    ("exact", "tree", "tree"), ("exact", "tree", "capped_tree"),
])
def test_engine_stops_at_the_proposal_budget(tiny_target, monkeypatch, mode, verifier, hook):
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    config = DecodeConfig(mode=mode, max_new_tokens=9, draft_len=5, max_tree_nodes=2,
                          verify_every=2 if mode == "approx" else 1, seed=17)
    prompt = random_prompt(72, 9)
    widths = []

    def finite(_history, width):
        return list(range(7, 7 + width))

    def stream(history, width):
        widths.append(width)
        yield from finite(history, width)
        raise AssertionError("read past the proposal budget")

    def lane(propose):
        drafter = ConstantDrafter()
        if hook == "probabilities":
            def with_probs(history, width):
                rows = torch.full((width, VOCAB), 0.1 / (VOCAB - 1))
                rows[torch.arange(width), torch.arange(7, 7 + width)] = 0.9
                return propose(history, width), rows
            drafter.propose_with_probs = with_probs
        elif hook == "capped_tree":
            def tree(history, width, *, max_nodes):
                assert width <= max_nodes == config.max_tree_nodes
                return propose(history, width)
            drafter.propose_tree = tree
        elif hook == "tree":
            drafter.propose_tree = propose
        else:
            drafter.propose = propose
        return drafter

    with DartEngine(tiny_target, lane(finite), verifier=verifier) as control:
        expected = control.generate(prompt, config)
    with DartEngine(tiny_target, lane(stream), verifier=verifier) as engine:
        output = engine.generate(prompt, config)
    assert widths and all(0 < width <= (2 if verifier == "tree" else 5) for width in widths)
    assert output.token_ids == expected.token_ids
    assert output.metadata["finished_eos"] == expected.metadata["finished_eos"]
    if mode == "exact":
        assert output.token_ids == _baseline(tiny_target, prompt, config.max_new_tokens)
    assert [(r.n_draft_proposed, r.n_draft_accepted, r.n_committed) for r in output.rounds] == [
        (r.n_draft_proposed, r.n_draft_accepted, r.n_committed) for r in expected.rounds]


def test_bounded_proposal_preserves_sampling_rows_and_unread_tokens():
    tokens, rows = iter([7, 8, 9]), torch.eye(3)
    drafter = ConstantDrafter()
    drafter.propose_with_probs = lambda *_: (tokens, rows)
    kept, probabilities = DartEngine._propose(drafter, [5], 2, sampling=True)
    assert kept == [7, 8] and probabilities is rows
    assert next(tokens) == 9


def test_bounded_proposal_respects_list_iteration_and_large_budgets():
    class Drafts(list):
        def __iter__(self):
            return iter([11, 13])

    drafter = ConstantDrafter()
    for source, expected in (([7, 8], [7, 8]), (Drafts([7, 8]), [11, 13]), (iter([17]), [17])):
        drafter.propose = lambda *_, source=source: source
        kept, _ = DartEngine._propose(drafter, [5], 2**65, sampling=False)
        assert kept == expected and kept is not source


def test_error_inside_proposal_prefix_releases_request(tiny_target, monkeypatch):
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    drafter, finished = ConstantDrafter(), []
    drafter.finish_request = lambda: finished.append(True)

    def broken(_history, width):
        assert width >= 2
        yield 7
        raise RuntimeError("draft failed")

    prompt, config = [5, 9, 12], DecodeConfig(max_new_tokens=6, draft_len=2)
    with DartEngine(tiny_target, drafter) as engine:
        with monkeypatch.context() as patch:
            patch.setattr(drafter, "propose", broken)
            with pytest.raises(RuntimeError, match="draft failed"):
                engine.generate(prompt, config)
        assert finished == [True]
        assert engine.generate(prompt, config).token_ids == _baseline(tiny_target, prompt, 6)
    assert finished == [True, True]


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
