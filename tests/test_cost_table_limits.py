from __future__ import annotations

import pytest

from benchmarks.run_bench import build_hf_method
from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CapacityBuckets, CudaGraphTarget
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler
from dart.runtime.types import DecodeConfig, DecodeState, VerifyResult
from tests.conftest import build_tiny_target, random_prompt


def state(limit, entropy=0.0):
    value = DecodeState(16, 4, float("inf"), 1, "code", True, entropy, {}, 0.0, 0.0)
    value.max_draft_tokens = limit
    return value


def observe(planner, action):
    n = action.draft_len
    planner.observe(action.lane, VerifyResult([7] * (n + 1), n, n, True, False), round_ms=10.0)


@pytest.mark.parametrize("limit,expected", [(4, [0, 2, 4, 4, 4]), (1, [0, 1, 1, 1, 1]), (0, [0] * 5)])
def test_clipped_lengths_share_one_arm(limit, expected):
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1.0),
                                 draft_lengths=(2, 4, 8), warmup_rounds=1)
    widths = []
    for _ in expected:
        action = planner.choose_action(state(limit))
        widths.append(action.draft_len)
        observe(planner, action)
    assert widths == expected
    assert set(planner.counts) == {"target_only", *(f"ngram:{n}" for n in expected if n)}


@pytest.mark.parametrize("lane", ["ngram", "diffusion"])
@pytest.mark.parametrize("verifiers", [None, ("block", "tree")])
def test_gate_then_limit_preserves_distinct_feasible_choices(lane, verifiers):
    planner = CostTableScheduler(SchedulerConfig(available_lanes=[lane], entropy_gate_threshold=1.0),
                                 draft_lengths=(2, 4, 8), verifiers=verifiers, warmup_rounds=1)
    observe(planner, planner.choose_action(state(3, entropy=2.0)))
    for size in (1, 2, 3):
        for verifier in verifiers or (None,):
            action = planner.choose_action(state(3, entropy=2.0))
            assert (action.draft_len, action.canvas_len) == (size, size if lane == "diffusion" else 0)
            assert action.verifier == verifier
            observe(planner, action)
    keys = {f"{lane}:{n}" + (f":{v}" if v is not None else "")
            for n in (1, 2, 3) for v in verifiers or (None,)}
    assert set(planner.counts) == {"target_only", *keys}


def test_budget_changes_keep_costs_for_the_work_that_ran():
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1.0),
                                 draft_lengths=(2, 4, 8), warmup_rounds=1)
    for _ in range(3):
        observe(planner, planner.choose_action(state(4)))
    action = planner.choose_action(state(1))
    assert action.draft_len == 1
    observe(planner, action)
    assert planner.choose_action(state(4)).draft_len == 4
    assert planner.counts == {"target_only": 1, "ngram:2": 1, "ngram:4": 1, "ngram:1": 1}
    assert planner.committed_tokens == {"target_only": 1.0, "ngram:2": 3.0, "ngram:4": 5.0, "ngram:1": 2.0}


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_round_limit_does_not_change_scheduler_state(limit):
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]))
    with pytest.raises(ValueError, match="max_draft_tokens"):
        planner.choose_action(state(limit))
    assert planner.decisions == 0 and not planner.counts


@pytest.mark.parametrize("buckets", [False, True])
def test_engine_names_the_actual_prepared_draft_budget(buckets, monkeypatch):
    base = build_tiny_target()
    base.eos_token_ids = frozenset()
    target = (CapacityBuckets(base, (64, 128), max_block_tokens=5, replay=False) if buckets
              else CudaGraphTarget(base, capacity=128, max_block_tokens=5, replay=False))
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]), draft_lengths=(2, 4, 8),
                                 warmup_rounds=1)
    drafter, choices, budgets = NGramDrafter(), [], []
    choose, propose = planner.choose_action, drafter.propose

    def record_choice(*args, **kwargs):
        action = choose(*args, **kwargs)
        choices.append(action.draft_len)
        return action

    def record_budget(committed, budget):
        budgets.append((choices[-1], budget))
        return propose(committed, budget)

    monkeypatch.setattr(planner, "choose_action", record_choice)
    monkeypatch.setattr(drafter, "propose", record_budget)
    try:
        with DartEngine(target, lanes={"ngram": drafter}, scheduler=planner) as engine:
            for prompt in (random_prompt(4, 5) * 5, [7, 11, 13] * 2):
                choices.clear()
                budgets.clear()
                expected = build_hf_method(base, 40)(prompt).token_ids
                output = engine.generate(prompt, DecodeConfig(max_new_tokens=40))
                assert output.token_ids == expected
                assert choices[:3] == [0, 2, 4]
                assert budgets and all(named == actual <= 4 for named, actual in budgets)
                assert "ngram:8" not in planner.counts
                assert sum(planner.counts.values()) == len(output.rounds)
    finally:
        target.close()
