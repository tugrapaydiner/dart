from __future__ import annotations

import pytest

from dart.drafters.ar_draft import ARDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.tree_verify import DraftTree
from tests.conftest import build_tiny_target, random_prompt
from tests.test_exactness import OracleDrafter


class ExtraDrafts(OracleDrafter):
    def propose(self, committed, max_tokens):
        assert max_tokens > 0, "No drafts should be requested at zero budget"
        return super().propose(committed, max_tokens + 5)

    def propose_tree(self, committed, max_tokens):
        correct = self.propose(committed, max_tokens)
        wrong = [(token + 1) % self.target.model.config.vocab_size for token in correct]
        return DraftTree.from_branches([wrong, correct])


@pytest.mark.parametrize("verifier", ["block", "reference", "tree"])
@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_rounds_commit_only_the_requested_tokens(verifier, budget):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(320, 8)
    expected = target.greedy_generate(prompt, max_new_tokens=budget)
    with DartEngine(target, ExtraDrafts(target), verifier=verifier) as engine:
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, draft_len=8))
    assert result.token_ids == expected
    assert 1 + sum(step.n_committed for step in result.rounds) == budget
    assert all(step.n_committed <= budget - 1 for step in result.rounds)


@pytest.mark.parametrize("mode", ["exact", "exact_sampling", "approx"])
@pytest.mark.parametrize("budget", [8, 9])
def test_budget_covers_verifier_bonus_tokens(mode, budget):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    with DartEngine(target, ARDrafter(target, seed=3)) as engine:
        result = engine.generate(random_prompt(321, 8), DecodeConfig(
            mode=mode, max_new_tokens=budget, draft_len=3, verify_every=2, seed=5,
        ))
    assert len(result.token_ids) == budget
    assert 1 + sum(step.n_committed for step in result.rounds) == budget


def test_zero_draft_length_never_calls_the_drafter():
    target = build_tiny_target()
    prompt = random_prompt(322, 8)
    expected = target.greedy_generate(prompt, max_new_tokens=5)
    with DartEngine(target, ExtraDrafts(target)) as engine:
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=5, draft_len=0))
    assert result.token_ids == expected
    assert result.metadata["draft_tokens_proposed"] == 0
