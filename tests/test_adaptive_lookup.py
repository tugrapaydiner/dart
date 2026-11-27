from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from dart.drafters.adaptive_lookup import AdaptiveLookupDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_engine
from dart.runtime.types import DecodeConfig
from dart.verification.tree_verify import DraftTree


@pytest.mark.parametrize("blocks, trees", [
    ({}, {}), ({1: 0}, {}), ({True: 1}, {}), ({1: float("nan")}, {}),
    ({1: 1, 2: float("inf")}, {}), ({1: 1}, {2: 1}), ({1: 1}, {17: 1}),
    ({1: 1}, {3: False}), ([], {}),
])
def test_rejects_invalid_costs(blocks, trees):
    with pytest.raises(ValueError):
        AdaptiveLookupDrafter(blocks, trees)


def test_frequent_roots_respect_prices_and_node_budget():
    costs = {3: 0.5, 4: 0.4}
    drafter = AdaptiveLookupDrafter({1: 2}, costs)
    costs.clear()
    tokens = [3, 7, 2]
    assert drafter.propose_tree(tokens, 1) == DraftTree([2, 7, 3], [-1, -1, -1])
    assert drafter.propose_tree(tokens, 1, max_nodes=2) == DraftTree([2, 7], [-1, -1])
    assert drafter.propose_tree(tokens, 1, max_nodes=1) == DraftTree([], [])
    assert drafter.propose(tokens, 2) == []


def test_uses_recent_branches_and_prices_the_clipped_shape():
    drafter = AdaptiveLookupDrafter({1: 4, 3: 1}, {3: 0.5, 5: 0.1})
    tokens = [1, 8, 4, 1, 7, 3, 1]
    assert drafter.propose_tree(tokens, 2) == DraftTree.from_branches([[7, 3], [8, 4]])
    assert drafter.propose_tree(tokens, 1) == DraftTree.from_branches([[7], [8]])
    assert drafter.propose_tree(tokens, 2, max_nodes=3) == DraftTree.chain([7, 3])
    assert drafter.propose(tokens, 2) == [7, 3]


def test_past_matches_change_whether_a_linear_draft_is_worth_its_cost():
    drafter = AdaptiveLookupDrafter({1: 1, 2: 1.55}, {})
    assert drafter.propose([1, 2] * 13 + [1], 1) == [2]
    assert drafter.propose([1, 3, 1, 4, 1, 5, 1, 6, 1, 2, 1], 1) == []


def test_eos_ends_offers_and_a_later_call_can_change_stop_ids():
    drafter = AdaptiveLookupDrafter({1: 8, 2: 1, 3: 0.3, 4: 0.1}, {}, eos_token_ids={9})
    sequence = [1, 2, 3, 9, 8, 1, 2]
    drafter.after_prefill()
    try:
        assert drafter.propose(sequence, 3) == [3]
        assert drafter.propose_tree(sequence, 3, eos_token_ids=()) == DraftTree.chain([3, 9, 8])
        assert drafter.propose_tree(sequence, 3, eos_token_ids={3}) == DraftTree([], [])
    finally:
        drafter.finish_request()


def test_incremental_history_agrees_with_rebuilding_after_prefix_changes():
    blocks = {i: 1 + i / 10 for i in range(1, 10)}
    trees = {i: 1 + i / 20 for i in range(3, 17)}
    drafter = AdaptiveLookupDrafter(blocks, trees, max_history=24)
    independent = AdaptiveLookupDrafter(blocks, trees, max_history=24)
    prompt = [3, 4, 3, 5, 4, 3, 5, 7, 4, 3, 8, 4, 3, 5]
    drafter.after_prefill()
    try:
        for prefix in [prompt[:i] for i in range(1, len(prompt) + 1)] + [prompt[:4], [7, 5, 7], prompt * 2]:
            assert drafter.propose_tree(prefix, 5, max_nodes=8) == independent.propose_tree(prefix, 5, max_nodes=8)
        assert drafter.propose_tree(prompt * 2, 8) == DraftTree([], [])
        assert drafter._request.history is None
    finally:
        drafter.finish_request()
    assert not hasattr(drafter._request, "history")


def test_request_cleanup_is_independent_across_threads():
    drafter = AdaptiveLookupDrafter({1: 4, 2: 1}, {3: 1})
    barrier = Barrier(2)

    def request(token):
        drafter.after_prefill()
        try:
            drafter.propose([token] * 4, 1)
            barrier.wait(timeout=5)
            return list(drafter._request.history.tokens), drafter.propose([token] * 5, 1)
        finally:
            drafter.finish_request()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(request, [3, 7])) == [([3] * 4, [3]), ([7] * 4, [7])]


@pytest.mark.parametrize("verifier", ["block", "tree"])
def test_engine_uses_the_policy_and_matches_complete_greedy(tiny_target, verifier):
    drafter = AdaptiveLookupDrafter({1: 5, **{i: 1 for i in range(2, 10)}}, {i: 1 for i in range(3, 17)})
    with DartEngine(tiny_target, drafter, verifier=verifier) as engine:
        for prompt in ([3, 5, 7, 3, 5], [7, 8, 9, 7, 8]):
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=6, draft_len=4, max_tree_nodes=5))
            assert result.token_ids == tiny_target.greedy_generate(prompt, 6)
            assert result.metadata["finished_eos"] == (result.token_ids[-1] in tiny_target.eos_token_ids)
            assert all(r.n_draft_proposed <= 5 for r in result.rounds)
            assert not hasattr(drafter._request, "history")


@pytest.mark.parametrize("limit", [0, True, 1025])
def test_rejects_invalid_history_limit(limit):
    with pytest.raises(ValueError, match="max_history"):
        AdaptiveLookupDrafter({1: 1}, {}, max_history=limit)


def test_config_builds_a_priced_lane_for_the_target():
    config = {"target": {"model": "tiny:1234", "device": "cpu"}, "verifier": "tree",
              "lanes": {"adaptive_lookup": {"block_costs": {1: 2, 2: 1}, "tree_costs": {3: 1}}}}
    with build_engine(config) as engine:
        assert isinstance(engine.drafter, AdaptiveLookupDrafter)
        prompt = [5, 7, 9, 7]
        assert engine.generate(prompt, DecodeConfig(max_new_tokens=4)).token_ids == engine.target.greedy_generate(prompt, 4)
    config["lanes"]["adaptive_lookup"] = {}
    with pytest.raises(ValueError, match="one row"):
        build_engine(config)
