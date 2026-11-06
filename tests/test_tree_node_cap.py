from __future__ import annotations

import pytest

from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.tree_verify import DraftTree
from tests.conftest import VOCAB, build_tiny_target, random_prompt


class WideDrafter:
    """Five eight-token branches: forty packed nodes before any cap."""

    name = "wide"
    requires_entropy = False

    def __init__(self, target):
        self.target = target

    def propose(self, committed, max_tokens):
        return []

    def propose_tree(self, committed, max_tokens):
        correct = self.target.greedy_generate(committed, max_tokens)
        branches = [[(token + shift) % VOCAB for token in correct] for shift in (1, 2, 3, 4)] + [correct]
        return DraftTree.from_branches(branches)

    def observe(self, result):
        pass


@pytest.fixture
def target(monkeypatch):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    sizes = []
    forward_tree = TargetModel.forward_tree
    monkeypatch.setattr(TargetModel, "forward_tree",
                        lambda self, state, pending, tokens, parents: sizes.append(len(tokens))
                        or forward_tree(self, state, pending, tokens, parents))
    target.tree_sizes = sizes
    return target


@pytest.mark.parametrize("cap", [15, 7, 1])
def test_engine_packs_at_most_max_tree_nodes(target, cap):
    prompt = random_prompt(51, 6)
    with DartEngine(target, WideDrafter(target), verifier="tree") as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=20, draft_len=8, max_tree_nodes=cap))
    assert output.token_ids == target.greedy_generate(prompt, 20)
    assert target.tree_sizes and max(target.tree_sizes) == cap
    assert all(step.n_draft_proposed <= cap for step in output.rounds)


def test_default_cap_is_fifteen_nodes(target):
    with DartEngine(target, WideDrafter(target), verifier="tree") as engine:
        engine.generate(random_prompt(52, 6), DecodeConfig(max_new_tokens=12, draft_len=8))
    assert max(target.tree_sizes) == 15


def test_zero_cap_runs_target_steps(target):
    prompt = random_prompt(53, 6)
    with DartEngine(target, WideDrafter(target), verifier="tree") as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=6, draft_len=8, max_tree_nodes=0))
    assert output.token_ids == target.greedy_generate(prompt, 6)
    assert target.tree_sizes == [] and all(step.n_draft_proposed == 0 for step in output.rounds)


def test_ngram_trees_respect_the_cap(target):
    prompt = random_prompt(54, 5) * 6
    with DartEngine(target, NGramDrafter(tree_width=3), verifier="tree") as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=16, draft_len=8, max_tree_nodes=10))
    assert output.token_ids == target.greedy_generate(prompt, 16)
    assert all(size <= 10 for size in target.tree_sizes)


def test_limit_keeps_a_valid_packing_prefix():
    tree = DraftTree.from_branches([[1, 2, 3], [1, 4], [5, 6]])
    assert (tree.tokens, tree.parent) == ([1, 2, 3, 4, 5, 6], [-1, 0, 1, 0, -1, 4])
    limited = tree.limit(4)
    assert (limited.tokens, limited.parent, limited.depths) == ([1, 2, 3, 4], [-1, 0, 1, 0], [0, 1, 2, 1])
    assert tree.limit(10) is tree and len(tree.limit(0)) == 0


@pytest.mark.parametrize("value", [-1, 2.0, True, None])
def test_rejects_invalid_node_caps(value):
    with pytest.raises(ValueError, match="max_tree_nodes"):
        DecodeConfig(max_tree_nodes=value)
