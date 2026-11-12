from __future__ import annotations

import pytest

from benchmarks.run_bench import build_hf_method
from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_engine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
from dart.runtime.types import DecodeConfig, VerifyResult
from tests.conftest import build_tiny_target
from tests.test_scheduler_state import decode_state


@pytest.fixture(params=[TimeScheduler, CostTableScheduler])
def scheduler_type(request):
    return request.param


def planner(scheduler_type, limit, **options):
    return scheduler_type(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1, max_gpu_kv_gb=0.5),
                          warmup_rounds=1, retry_interval=4, max_probe_rounds=limit, **options)


def observe(scheduler, action, milliseconds=10, committed=1, **options):
    drafted = min(action.draft_len, committed - 1)
    scheduler.observe(action.lane, VerifyResult([7] * committed, action.draft_len, drafted,
                                               drafted == action.draft_len, False),
                      round_ms=milliseconds, **options)


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_slow_drafts_stop_at_the_probe_limit(scheduler_type, limit):
    scheduler = planner(scheduler_type, limit)
    lanes = []
    for _ in range(24):
        action = scheduler.choose_action(decode_state())
        lanes.append(action.lane)
        observe(scheduler, action, 10 if action.lane == "target_only" else 30)
    assert lanes.count("ngram") == limit
    assert scheduler.probe_rounds == limit
    assert lanes[-4:] == ["target_only"] * 4


def test_exhaustion_keeps_the_measured_winner_and_refreshes_the_baseline(scheduler_type):
    scheduler = planner(scheduler_type, 1)
    first = scheduler.choose_action(decode_state())
    assert first.lane == "target_only"
    observe(scheduler, first)
    for _ in range(3):
        action = scheduler.choose_action(decode_state())
        assert action.lane == "ngram"
        observe(scheduler, action, 8, 2)
    assert scheduler.probe_rounds == 1
    refresh = scheduler.choose_action(decode_state())
    assert refresh.lane == "target_only"
    observe(scheduler, refresh, 1)
    assert scheduler.choose_action(decode_state()).lane == "target_only"
    assert scheduler.probe_rounds == 1


def test_shortest_retry_interval_leaves_room_for_the_winner(scheduler_type):
    scheduler = scheduler_type(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1),
                               warmup_rounds=1, retry_interval=1, max_probe_rounds=1)
    lanes = []
    for _ in range(10):
        action = scheduler.choose_action(decode_state())
        lanes.append(action.lane)
        observe(scheduler, action, 10 if action.lane == "target_only" else 8,
                1 if action.lane == "target_only" else 2)
    assert lanes == ["target_only", "ngram"] * 5
    assert scheduler.probe_rounds == 1


@pytest.mark.parametrize("fallback", [False, True])
def test_failed_or_fallback_probes_still_spend_the_budget(scheduler_type, fallback):
    scheduler = planner(scheduler_type, 1)
    observe(scheduler, scheduler.choose_action(decode_state()))
    assert scheduler.choose_action(decode_state()).lane == "ngram"
    if fallback:
        scheduler.observe("target_only", VerifyResult([7], 0, 0, True, False), round_ms=10)
    for _ in range(8):
        action = scheduler.choose_action(decode_state())
        assert action.lane == "target_only"
        observe(scheduler, action)
    assert scheduler.probe_rounds == 1
    assert set(scheduler.counts) == {"target_only"}


def test_restrictions_do_not_spend_or_bypass_the_budget(scheduler_type):
    scheduler = planner(scheduler_type, 1)
    for _ in range(6):
        action = scheduler.choose_action(decode_state(), allowed=[])
        assert action.lane == "target_only"
        observe(scheduler, action)
    assert scheduler.probe_rounds == 0
    action = scheduler.choose_action(decode_state())
    assert action.lane == "ngram"
    observe(scheduler, action, 1, 2, kv_gb=1)
    for _ in range(8):
        action = scheduler.choose_action(decode_state())
        assert action.lane == "target_only"
        observe(scheduler, action)
    assert scheduler.probe_rounds == 1


def test_unplayed_clipped_arms_and_verifiers_stay_out_after_exhaustion():
    scheduler = planner(CostTableScheduler, 2, draft_lengths=(2, 4), verifiers=("block", "tree"))
    state = decode_state()
    state.max_tree_nodes = 1
    observe(scheduler, scheduler.choose_action(state))
    for expected in ((2, "block"), (1, "tree")):
        action = scheduler.choose_action(state)
        assert (action.draft_len, action.verifier) == expected
        observe(scheduler, action, 4 if action.verifier == "tree" else 20, 2)
    for _ in range(8):
        action = scheduler.choose_action(state)
        assert action.lane == "target_only" or (action.draft_len, action.verifier) == (1, "tree")
        observe(scheduler, action, 10 if action.lane == "target_only" else 4, 1 if action.lane == "target_only" else 2)
    state.max_draft_tokens = 1
    assert scheduler.choose_action(state).verifier == "tree"
    state.max_tree_nodes = 0
    assert scheduler.choose_action(state).lane == "target_only"
    assert set(scheduler.counts) == {"target_only", "ngram:2:block", "ngram:1:tree"}
    assert scheduler.probe_rounds == 2


def test_reset_restores_exploration(scheduler_type):
    scheduler = planner(scheduler_type, 1)
    for _ in range(2):
        observe(scheduler, scheduler.choose_action(decode_state()))
    assert scheduler.probe_rounds == 1
    scheduler.reset()
    assert scheduler.probe_rounds == 0 and not scheduler.counts
    observe(scheduler, scheduler.choose_action(decode_state()))
    assert scheduler.choose_action(decode_state()).lane == "ngram"
    assert scheduler.probe_rounds == 1


@pytest.mark.parametrize("limit", [-1, True, 1.5, "2"])
def test_invalid_probe_limits_are_rejected(scheduler_type, limit):
    with pytest.raises(ValueError, match="max_probe_rounds"):
        planner(scheduler_type, limit)


def test_no_limit_keeps_retrying_slow_options(scheduler_type):
    scheduler = planner(scheduler_type, None)
    for _ in range(24):
        action = scheduler.choose_action(decode_state())
        observe(scheduler, action, 10 if action.lane == "target_only" else 30)
    assert scheduler.probe_rounds > 2


@pytest.mark.parametrize("kind", ["time", "cost_table"])
def test_factory_passes_the_probe_limit(kind):
    with build_engine({"target": {"model": "tiny:0"}, "lanes": {"ngram": {}},
                       "scheduler": {"type": kind, "max_probe_rounds": 2}}) as engine:
        assert engine.scheduler.max_probe_rounds == 2


@pytest.mark.parametrize("limit", [0, 2])
def test_bounded_probes_preserve_exact_outputs_and_reset_between_requests(scheduler_type, limit):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    scheduler = planner(scheduler_type, limit)
    with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
        for prompt in ([7, 8, 9] * 3, [21, 22, 23] * 2):
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=16))
            assert output.token_ids == build_hf_method(target, 16)(prompt).token_ids
            assert 0 <= scheduler.probe_rounds <= limit
            assert bool(scheduler.probe_rounds) == bool(limit)
            assert sum(scheduler.counts.values()) == len(output.rounds)
            assert output.rounds[0].lane == "target_only"


@pytest.mark.parametrize("eos_position", [0, 4])
def test_probe_limit_preserves_eos_and_single_token_requests(scheduler_type, eos_position):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = [7, 8, 9]
    continuation = target.greedy_generate(prompt, 8)
    eos = continuation[eos_position]
    target.eos_token_ids = frozenset({eos})
    expected = continuation[:continuation.index(eos) + 1]
    scheduler = planner(scheduler_type, 1)
    with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
        assert output.token_ids == build_hf_method(target, 8)(prompt).token_ids == expected
        assert output.metadata["finished_eos"]
        assert scheduler.probe_rounds <= 1
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=1))
        assert output.token_ids == continuation[:1]
        assert not output.rounds and not scheduler.counts and scheduler.probe_rounds == 0
