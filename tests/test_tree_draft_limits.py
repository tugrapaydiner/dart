from __future__ import annotations

from functools import wraps

import pytest
import torch
import yaml

from benchmarks.run_bench import build_hf_method
from dart.drafters.ar_draft import ARDrafter
from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_lane
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler
from dart.runtime.types import DecodeConfig, DecodeState, VerifyResult
from dart.verification.tree_verify import DraftTree
from tests.conftest import build_tiny_target


@pytest.mark.parametrize("width", [3, 32])
def test_ngram_builds_only_retained_nodes(width, monkeypatch):
    target = build_tiny_target()
    with torch.no_grad():
        target.model.lm_head.weight.zero_()
    suffix = [5, 6, 7, 0]
    prompt = []
    for index in range(width):
        prompt.extend(suffix + [10 + index] + [50] * 7)
    prompt.extend(suffix[:-1])  # The first greedy token completes the suffix.
    config = DecodeConfig(max_new_tokens=10, draft_len=8, max_tree_nodes=15)
    expected = build_hf_method(target, 10)(prompt).token_ids
    sizes = []
    initialize = DraftTree.__post_init__

    def record(tree):
        initialize(tree)
        sizes.append(len(tree))

    monkeypatch.setattr(DraftTree, "__post_init__", record)
    with DartEngine(target, NGramDrafter(tree_width=width), verifier="tree") as engine:
        output = engine.generate(prompt, config)
    assert output.token_ids == expected
    assert output.rounds[0].n_draft_proposed == 15
    assert max(sizes) == 15


@pytest.mark.parametrize(("width", "scheduled", "cap", "nodes"), [
    (1, False, 15, 2), (2, True, 15, 4), (4, False, 15, 8),
    (None, True, 15, 6), (4, True, 3, 3),
])
def test_configured_tree_width_bounds_verified_nodes(tmp_path, width, scheduled, cap, nodes):
    config = {
        "target": {"model": "tiny:1234"}, "verifier": "tree",
        "lanes": {"ngram": {} if width is None else {"tree_width": width}},
        "decode": {"max_new_tokens": 4, "draft_len": 2, "max_tree_nodes": cap},
    }
    if scheduled:
        config["scheduler"] = {"type": "heuristic", "draft_len": 2}
    path = tmp_path / "engine.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    prompt = [token for branch in range(10, 14) for token in (5, 6, 7, 0, branch, 20)] + [5, 6, 7]
    with DartEngine.from_config(str(path)) as engine:
        with torch.no_grad():
            engine.target.model.lm_head.weight.zero_()  # The first token completes the repeated suffix.
        expected = build_hf_method(engine.target, 4)(prompt).token_ids
        output = engine.generate(prompt)
    assert output.token_ids == expected
    assert output.rounds[0].n_draft_proposed == nodes


@pytest.mark.parametrize("width", [0, True, 1.5, float("nan"), float("inf"), "2"])
def test_ngram_config_rejects_invalid_tree_width(width):
    with pytest.raises(ValueError, match="tree_width"):
        build_lane("ngram", {"tree_width": width}, None)


def test_replaced_tree_hook_keeps_its_call_signature():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = NGramDrafter()
    calls = []

    def bounded(committed, depth, *, max_nodes):
        calls.append(max_nodes)
        drafter.propose_tree = legacy
        return DraftTree.chain([])

    @wraps(bounded)
    def legacy(committed, depth):
        calls.append(None)
        return DraftTree.chain([])

    drafter.propose_tree = bounded
    with DartEngine(target, drafter, verifier="tree") as engine:
        output = engine.generate([7, 8], DecodeConfig(max_new_tokens=5, draft_len=2, max_tree_nodes=3))
    assert output.token_ids == target.greedy_generate([7, 8], 5)
    assert calls == [3, None, None]


@pytest.mark.parametrize("cap", [0, 1, 3])
@pytest.mark.parametrize("kind", ["ar", "ngram"])
def test_tree_limit_applies_before_drafting(cap, kind, monkeypatch):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = ARDrafter(build_tiny_target()) if kind == "ar" else NGramDrafter()
    if kind == "ar":
        drafter.dm.eos_token_ids = frozenset()
    method = "propose" if kind == "ar" else "propose_tree"
    propose = getattr(drafter, method)
    calls, forwards = [], []

    def record(committed, budget):
        calls.append(budget)
        return propose(committed, budget)

    monkeypatch.setattr(drafter, method, record)
    handle = (drafter.dm.model.register_forward_pre_hook(lambda *_: forwards.append(1))
              if kind == "ar" else None)
    try:
        with DartEngine(target, drafter, verifier="tree") as engine:
            for prompt in ([5, 6, 7] * 3, [11, 13, 17] * 2):
                calls.clear()
                forwards.clear()
                expected = build_hf_method(target, 12)(prompt).token_ids
                output = engine.generate(prompt, DecodeConfig(max_new_tokens=12, draft_len=8, max_tree_nodes=cap))
                assert output.token_ids == expected
                assert all(step.n_draft_proposed <= cap for step in output.rounds)
                if cap == 0:
                    assert (calls, forwards) == ([], [])
                else:
                    assert calls and calls[0] == cap and all(0 < budget <= cap for budget in calls), calls
                    if kind == "ar":
                        assert len(forwards) == sum(calls)
    finally:
        if handle is not None:
            handle.remove()


def bounded_state(cap, default):
    state = DecodeState(16, 4, float("inf"), 1, "code", True, 0.0, {}, 0.0, 0.0)
    state.max_tree_nodes = cap
    state.default_verifier = default
    return state


@pytest.mark.parametrize("cap,default,menu,expected", [
    (0, "block", ("block", "tree"), [(2, "block"), (4, "block"), (8, "block")]),
    (1, "block", ("block", "tree"), [(2, "block"), (1, "tree"), (4, "block"), (8, "block")]),
    (3, "block", ("block", "tree"), [(2, "block"), (2, "tree"), (4, "block"), (3, "tree"), (8, "block")]),
    (0, "tree", None, []),
    (1, "tree", None, [(1, None)]),
    (3, "tree", None, [(2, None), (3, None)]),
    (0, "block", None, [(2, None), (4, None), (8, None)]),
])
def test_cost_table_explores_each_feasible_tree_budget_once(cap, default, menu, expected):
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1.0),
                                 draft_lengths=(2, 4, 8), verifiers=menu, warmup_rounds=1)
    for size, verifier in [(0, None), *expected]:
        action = planner.choose_action(bounded_state(cap, default))
        assert (action.draft_len, action.verifier) == (size, verifier)
        planner.observe(action.lane, VerifyResult([7] * (size + 1), size, size, True, False), round_ms=10.0)
    assert len(planner.counts) == 1 + len(expected)
    assert all(count == 1 for count in planner.counts.values())
    action = planner.choose_action(bounded_state(cap, default))
    assert (action.draft_len, action.verifier) in (expected or [(0, None)])


@pytest.mark.parametrize("default,menu", [("block", ("block", "tree")), ("tree", ("block", "tree")), ("tree", None)])
def test_engine_passes_tree_limits_to_the_planner(default, menu, monkeypatch):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]),
                                 verifiers=menu, warmup_rounds=1)
    drafter, actions = NGramDrafter(), []
    choose = planner.choose_action

    def record(state, **kwargs):
        action = choose(state, **kwargs)
        actions.append((action.lane, action.draft_len, action.verifier or default))
        return action

    monkeypatch.setattr(planner, "choose_action", record)
    with DartEngine(target, lanes={"ngram": drafter}, scheduler=planner, verifier=default) as engine:
        for cap in (1, 0, 3):
            actions.clear()
            prompt = [5, 6, 7] * 3
            expected = build_hf_method(target, 24)(prompt).token_ids
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=24, max_tree_nodes=cap))
            assert output.token_ids == expected
            assert all(size <= cap for _, size, verifier in actions if verifier == "tree")
            assert sum(planner.counts.values()) == len(output.rounds)
            if menu is not None:
                assert any(size > cap for _, size, verifier in actions if verifier == "block")
            if cap == 0:
                assert all(lane == "target_only" for lane, _, verifier in actions if verifier == "tree")


@pytest.mark.parametrize("cap", [-1, True, 1.5])
def test_invalid_tree_limit_preserves_planner_state(cap):
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]))
    with pytest.raises(ValueError, match="max_tree_nodes"):
        planner.choose_action(bounded_state(cap, "tree"))
    assert planner.decisions == 0 and not planner.counts
