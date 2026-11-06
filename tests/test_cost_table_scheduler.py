from __future__ import annotations

import pytest

from dart.drafters.ngram import NGramDrafter
from dart.runtime import factory
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler
from dart.runtime.types import DecodeConfig, DecodeState, VerifyResult
from tests.conftest import build_tiny_target, random_prompt


def state():
    return DecodeState(16, 4, float("inf"), 1, "code", True, 0.0, {}, 0.0, 0.0)


def result(committed):
    return VerifyResult(committed=[7] * committed, n_draft_proposed=max(0, committed - 1),
                        n_draft_accepted=max(0, committed - 1), all_accepted=True, finished=False)


def play(scheduler, timings, rounds):
    """Run choose/observe cycles with fixed (round ms, committed tokens) per arm."""
    chosen = []
    for _ in range(rounds):
        action = scheduler.choose_action(state())
        key = "target_only" if action.lane == "target_only" else f"{action.lane}:{action.draft_len}"
        chosen.append(key)
        ms, committed = timings[key]
        scheduler.observe(action.lane, result(committed), round_ms=ms)
    return chosen


def scheduler(**options):
    config = SchedulerConfig(available_lanes=["ngram", "target_only"], ewma_alpha=1.0)
    return CostTableScheduler(config, draft_lengths=(2, 4, 8), warmup_rounds=1, retry_interval=50, **options)


TIMINGS = {"target_only": (10.0, 1), "ngram:2": (12.0, 2), "ngram:4": (14.0, 4), "ngram:8": (30.0, 5)}


def test_tries_each_arm_then_uses_the_cheapest_time_per_token():
    planner = scheduler()
    chosen = play(planner, TIMINGS, 10)
    assert chosen[:4] == ["target_only", "ngram:2", "ngram:4", "ngram:8"]
    assert chosen[4:] == ["ngram:4"] * 6  # 3.5 ms per token beats 6, 6 and 10.
    assert planner.counts == {"target_only": 1, "ngram:2": 1, "ngram:4": 7, "ngram:8": 1}


def test_keeps_target_steps_when_no_arm_saves_enough_time():
    slow = {"target_only": (10.0, 1), "ngram:2": (19.5, 2), "ngram:4": (39.0, 4), "ngram:8": (79.0, 8)}
    chosen = play(scheduler(min_time_saving=0.05), slow, 8)
    assert chosen[4:] == ["target_only"] * 4


def test_refreshes_stale_arms_and_the_baseline():
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1.0), draft_lengths=(2, 4),
                                 warmup_rounds=1, retry_interval=4)
    timings = {"target_only": (10.0, 1), "ngram:2": (12.0, 2), "ngram:4": (14.0, 4)}
    chosen = play(planner, timings, 9)
    assert set(chosen) == set(timings)
    assert chosen.count("ngram:4") >= 4 and chosen.count("target_only") >= 2


def test_observation_follows_the_lane_that_actually_ran():
    planner = scheduler()
    play(planner, TIMINGS, 2)
    action = planner.choose_action(state())
    assert (action.lane, action.draft_len) == ("ngram", 4)
    planner.observe("target_only", result(1), round_ms=11.0)  # The engine fell back to a target step.
    assert planner.counts["target_only"] == 2 and "ngram:4" not in planner.counts


def test_kv_budget_and_allowed_lanes_limit_arms():
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram", "ar_draft"], ewma_alpha=1.0,
                                                 max_gpu_kv_gb=0.5), draft_lengths=(2,), warmup_rounds=1)
    assert planner.choose_action(state()).lane == "target_only"
    planner.observe("target_only", result(1), round_ms=10.0)
    first = planner.choose_action(state(), allowed={"ar_draft"})
    assert (first.lane, first.draft_len) == ("ar_draft", 2)
    planner.observe("ar_draft", result(2), round_ms=4.0, kv_gb=0.9)
    assert planner.choose_action(state(), allowed={"ar_draft"}).lane == "target_only"


def test_reset_clears_request_state():
    planner = scheduler()
    play(planner, TIMINGS, 5)
    planner.reset()
    assert not planner.counts and not planner.round_ms and planner.decisions == 0
    assert planner.choose_action(state()).lane == "target_only"


@pytest.mark.parametrize("lengths", [(), (0,), (2, 2), (2.0,), [4], (65,), "24"])
def test_rejects_invalid_draft_length_menus(lengths):
    with pytest.raises(ValueError, match="draft_lengths"):
        CostTableScheduler(SchedulerConfig(), draft_lengths=lengths)


def test_engine_with_cost_table_matches_greedy_and_plays_every_arm():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(41, 12) * 3  # Repetition gives the n-gram lane accepted drafts.
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]), draft_lengths=(2, 4, 8),
                                 warmup_rounds=1, retry_interval=6)
    with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=planner) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=40))
    assert output.token_ids == target.greedy_generate(prompt, 40)
    assert set(planner.counts) == {"target_only", "ngram:2", "ngram:4", "ngram:8"}
    assert sum(planner.counts.values()) == len(output.rounds)
    assert all(step.n_draft_proposed <= 8 for step in output.rounds)


def test_factory_builds_the_cost_table_scheduler():
    engine = factory.build_engine({"target": {"model": "tiny:0"}, "lanes": {"ngram": {}},
                                   "scheduler": {"type": "cost_table", "draft_lengths": [2, 6], "warmup_rounds": 1}})
    with engine:
        assert isinstance(engine.scheduler, CostTableScheduler)
        assert engine.scheduler.draft_lengths == (2, 6) and engine.scheduler.warmup_rounds == 1
        engine.generate([5, 6, 7, 5, 6, 7], DecodeConfig(max_new_tokens=8))
