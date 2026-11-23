from __future__ import annotations

from itertools import product

import pytest
import torch

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
    forward_block = TargetModel.forward_block

    def record_block(self, state, tokens):
        if len(tokens) > 1:
            sizes.append(len(tokens) - 1)  # Linear trees use causal blocks.
        return forward_block(self, state, tokens)

    monkeypatch.setattr(TargetModel, "forward_block", record_block)
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


class PackedLane:
    name = "packed"
    requires_entropy = False

    def __init__(self, tree):
        self.tree = tree

    def propose_tree(self, committed, max_tokens):
        return self.tree

    def observe(self, result):
        pass


def test_current_tree_depth_at_the_request_context_limit(target):
    target.model.config.max_position_embeddings = 24
    with torch.no_grad():
        target.model.lm_head.weight.zero_()
    prompt = [5] * 20
    expected = target.greedy_generate(prompt, 4)
    parents = [-1] * 9
    tree = DraftTree([0] * 9, parents)
    with DartEngine(target, PackedLane(tree), verifier="tree") as engine:
        for layout in ([-1] * 9, list(range(-1, 8)), [-1] * 9):
            parents[:] = layout
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=4, draft_len=8))
            assert output.token_ids == expected
            assert sum(step.n_committed for step in output.rounds) == 3
            if layout[1] == 0:
                assert output.rounds[0].n_draft_proposed == 2


@pytest.mark.parametrize("edit", ["deepen", "flatten", "append", "returned_depths"])
def test_current_tree_depths_follow_the_parent_list(edit):
    tree = DraftTree([11, 12, 13, 14], [-1, 0, -1, 2])
    if edit == "deepen":
        tree.parent[:] = [-1, 0, 1, 2]
    elif edit == "flatten":
        tree.parent[:] = [-1] * 4
    elif edit == "append":
        tree.tokens.append(15)
        tree.parent.append(3)
    else:
        tree.depths[:] = [9] * 4
    depths = []
    for parent in tree.parent:
        depths.append(0 if parent < 0 else depths[parent] + 1)
    assert tree.depths == depths
    roots = tree.truncate(1)
    assert roots.tokens == [token for token, depth in zip(tree.tokens, depths) if depth == 0]
    assert roots.parent == [-1] * len(roots)


def test_bounded_tree_stops_after_the_last_eligible_slot(target, monkeypatch):
    parents = [p for start in range(0, 768, 3) for p in (-1, start, start + 1)]
    tree = DraftTree([7] * len(parents), parents)
    visited = []

    class Observed(list):
        def __iter__(self):
            for i, value in enumerate(super().__iter__()):
                visited.append(i)
                yield value

        def __getitem__(self, key):
            if isinstance(key, int):
                visited.append(key)
            return super().__getitem__(key)

    tree.parent = Observed(parents)
    sizes = []
    initialize = DraftTree.__post_init__

    def record(candidate):
        sizes.append(len(candidate))
        initialize(candidate)

    monkeypatch.setattr(DraftTree, "__post_init__", record)
    lane = PackedLane(tree)
    config = DecodeConfig(max_new_tokens=4, draft_len=2, max_tree_nodes=5)
    with DartEngine(target, lane, verifier="tree") as engine:
        result, _ = engine._round_drafts(lane, [1, 2], 2, "tree", False, config)
    assert result.parent == [-1, 0, -1, 2, -1]
    assert max(sizes) <= 5
    assert visited and max(visited) <= 6


def test_bounded_tree_matches_independent_parent_walks(target):
    lane = PackedLane(None)
    with DartEngine(target, lane, verifier="tree") as engine:
        for parents in product(*(range(-1, i) for i in range(5))):
            tree = DraftTree(list(range(5)), list(parents))
            lane.tree = tree
            depths = []
            for i in range(5):
                depth, parent = 0, parents[i]
                while parent >= 0:
                    depth, parent = depth + 1, parents[parent]
                depths.append(depth)
            for depth, cap in product((1, 2, 5), (0, 1, 3, 5, 8)):
                config = DecodeConfig(draft_len=depth, max_tree_nodes=cap)
                actual, _ = engine._round_drafts(lane, [7], depth, "tree", False, config)
                keep = [i for i in range(5) if depths[i] < depth][:cap]
                assert actual.tokens == keep
                assert actual.parent == [-1 if parents[i] < 0 else keep.index(parents[i]) for i in keep]
                if len(keep) == 5:
                    assert actual is tree


@pytest.mark.parametrize("depth", [1, 8])
@pytest.mark.parametrize("parents", [[-1, 2, 0], [-1, False, 1], [-1, 0]])
def test_current_tree_invalid_links_are_rejected(depth, parents):
    tree = DraftTree.chain([3, 5, 7])
    tree.parent[:] = parents
    with pytest.raises(ValueError, match="preceding index|equal length"):
        _ = tree.depths
    with pytest.raises(ValueError, match="preceding index|equal length"):
        tree.truncate(depth)


def test_bounded_tree_keeps_subclass_truncation_hooks(target):
    calls = []

    class CustomTree(DraftTree):
        def truncate(self, max_tokens):
            calls.append(max_tokens)
            return DraftTree.chain([11] * max_tokens)

    lane = PackedLane(CustomTree([7], [-1]))
    config = DecodeConfig(draft_len=7, max_tree_nodes=3)
    with DartEngine(target, lane, verifier="tree") as engine:
        tree, _ = engine._round_drafts(lane, [5], 7, "tree", False, config)
    assert calls == [3] and tree.tokens == [11] * 3
