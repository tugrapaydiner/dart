from __future__ import annotations

import pytest
import torch

from benchmarks.run_bench import build_hf_method
from dart.drafters.base import NullDrafter
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime import engine as engine_module
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_engine
from dart.runtime.metrics import DecodeMetrics
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import TimeScheduler
from dart.runtime.types import DecodeConfig, VerifyResult
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES
from tests.test_scheduler_state import decode_state


def play(scheduler, milliseconds=10.0, committed=1, proposed=0, accepted=0, allowed=None, kv_gb=0.0):
    lane = scheduler.choose_action(decode_state(), allowed=allowed).lane
    scheduler.observe(lane, VerifyResult([7] * committed, proposed, accepted, accepted == proposed, False),
                      round_ms=milliseconds, kv_gb=kv_gb)
    return lane


def test_selection_uses_time_per_token_instead_of_acceptance():
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["ngram", "ar_draft"], ewma_alpha=1), retry_interval=20)
    assert [play(scheduler), play(scheduler)] == ["target_only", "target_only"]
    assert play(scheduler, 6, 2, 4, 1) == "ngram"
    assert play(scheduler, 60, 5, 4, 4) == "ar_draft"
    assert scheduler.acceptance_ewma == {"ngram": 0.25, "ar_draft": 1}
    assert scheduler.choose_action(decode_state()).lane == "ngram"
    assert scheduler.choose_action(decode_state(), allowed=[]).lane == "target_only"
    assert scheduler.choose_action(decode_state(), allowed=["ar_draft"]).lane == "target_only"


def test_slow_lane_is_retried_and_can_recover():
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1),
                              warmup_rounds=1, retry_interval=4)
    assert play(scheduler) == "target_only"
    assert play(scheduler, 20, 1, 4, 0) == "ngram"
    assert [play(scheduler) for _ in range(3)] == ["target_only"] * 3
    assert play(scheduler, 8, 4, 4, 3) == "ngram"
    assert play(scheduler, 8, 4, 4, 3) == "ngram"


def test_periodic_target_probe_can_retire_a_previously_fast_lane():
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1),
                              warmup_rounds=1, retry_interval=4)
    assert play(scheduler) == "target_only"
    assert [play(scheduler, 8, 4, 4, 3) for _ in range(3)] == ["ngram"] * 3
    assert play(scheduler, 1) == "target_only"
    assert scheduler.choose_action(decode_state()).lane == "target_only"


def test_memory_limit_is_not_relaxed_by_periodic_retries():
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["ngram"], max_gpu_kv_gb=0.5),
                              warmup_rounds=1, retry_interval=2)
    assert play(scheduler) == "target_only"
    assert play(scheduler, 1, 5, 4, 4, kv_gb=1) == "ngram"
    assert [play(scheduler) for _ in range(8)] == ["target_only"] * 8
    scheduler.reset()
    assert all(not getattr(scheduler, name) for name in
               ("round_ms", "committed_tokens", "counts", "last_played", "kv_gb", "acceptance_ewma"))
    assert scheduler.decisions == 0
    assert play(scheduler) == "target_only"
    assert play(scheduler) == "ngram"


@pytest.mark.parametrize("cost,expected", [(9.49, "ngram"), (9.5, "target_only"), (10, "target_only")])
def test_minimum_time_saving_margin(cost, expected):
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["ngram"]), warmup_rounds=1)
    play(scheduler)
    play(scheduler, cost)
    assert scheduler.choose_action(decode_state()).lane == expected


@pytest.mark.parametrize("options", [
    {"retry_interval": 0}, {"retry_interval": True}, {"warmup_rounds": 1.5},
    {"min_time_saving": -0.1}, {"min_time_saving": 1}, {"min_time_saving": float("nan")},
    {"min_time_saving": True}, {"min_time_saving": None},
])
def test_invalid_policy_options_are_rejected(options):
    with pytest.raises(ValueError):
        TimeScheduler(SchedulerConfig(), **options)


@pytest.mark.parametrize("options", [
    {"ewma_alpha": 0}, {"ewma_alpha": 1.1}, {"ewma_alpha": float("nan")},
    {"max_gpu_kv_gb": -1}, {"max_gpu_kv_gb": float("nan")},
    {"ewma_alpha": True}, {"max_gpu_kv_gb": None},
])
def test_invalid_cost_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        TimeScheduler(SchedulerConfig(**options))


@pytest.mark.parametrize("timing", [
    {"round_ms": 0}, {"round_ms": -1}, {"round_ms": float("inf")},
    {"round_ms": float("nan")}, {"round_ms": 1, "kv_gb": -1},
    {"round_ms": True}, {"round_ms": None}, {"round_ms": 1, "kv_gb": float("nan")},
])
def test_invalid_measurements_do_not_change_policy_state(timing):
    scheduler = TimeScheduler(SchedulerConfig())
    with pytest.raises(ValueError):
        scheduler.observe("target_only", VerifyResult([7], 0, 0, True, False), **timing)
    assert not scheduler.counts and not scheduler.round_ms


def test_round_cost_includes_orchestration_and_pending_device_work(monkeypatch):
    clock, pending = [0.0], [0.0]
    monkeypatch.setattr(engine_module.time, "perf_counter", lambda: clock[0])

    def timed(call, milliseconds, queued=0):
        def run(*args, **kwargs):
            result = call(*args, **kwargs)
            clock[0] += milliseconds / 1000
            pending[0] += queued / 1000
            return result
        return run

    def synchronize():
        clock[0] += pending[0]
        pending[0] = 0

    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["target_only"]))
    null, other = NullDrafter(), NullDrafter()
    other.after_verify = lambda result: None
    monkeypatch.setattr(null, "observe", timed(null.observe, 7))
    monkeypatch.setattr(other, "after_verify", timed(other.after_verify, 11))
    monkeypatch.setattr(DecodeMetrics, "record_round", timed(DecodeMetrics.record_round, 13))
    with DartEngine(target, lanes={"target_only": null, "ngram": other}, scheduler=scheduler) as engine:
        monkeypatch.setattr(engine, "_synchronize_round", synchronize)
        monkeypatch.setattr(engine, "_pick_lane", timed(engine._pick_lane, 2))
        monkeypatch.setattr(engine, "_propose", timed(engine._propose, 3))
        monkeypatch.setattr(engine, "verify", timed(engine.verify, 5, queued=19))
        monkeypatch.setattr(engine.trace, "log_round", timed(engine.trace.log_round, 17))
        output = engine.generate([7, 8, 9], DecodeConfig(max_new_tokens=2))
    assert len(output.rounds) == 1
    assert scheduler.round_ms["target_only"] == pytest.approx(77)
    assert scheduler.counts == {"target_only": 1}
    assert pending[0] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_time_scheduler_matches_hf_and_resets_between_requests(device, dtype):
    target = TargetModel(build_tiny_target().model, device=device, dtype=dtype)
    target.eos_token_ids = frozenset()
    lanes = {"ngram": NGramDrafter(), "target_only": NullDrafter()}
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=list(lanes)), retry_interval=4)
    with DartEngine(target, lanes=lanes, scheduler=scheduler) as engine:
        for prompt in ([7, 8, 9] * 3, [21, 22, 23] * 2):
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=16))
            assert output.token_ids == build_hf_method(target, 16)(prompt).token_ids
            assert [row.lane for row in output.rounds[:3]] == ["target_only", "target_only", "ngram"]
            assert sum(scheduler.counts.values()) == len(output.rounds)
            assert all(cost > 0 for cost in scheduler.round_ms.values())


@pytest.mark.parametrize("options", [{"mode": "approx"}, {"mode": "exact_sampling"}, {"full_kv_residency": "cpu_pinned"}])
def test_time_scheduler_rejects_unmeasured_execution_modes(options):
    target = build_tiny_target()
    with DartEngine(target, lanes={}, scheduler=TimeScheduler(SchedulerConfig())) as engine:
        with pytest.raises(ValueError, match="time scheduling requires"):
            engine.generate([7], DecodeConfig(max_new_tokens=4, **options))


def test_factory_builds_time_policy_with_explicit_options():
    with build_engine({
        "target": {"model": "tiny:1234"}, "lanes": {"ngram": {}},
        "scheduler": {"type": "time", "retry_interval": 7, "warmup_rounds": 3,
                      "min_time_saving": 0.1, "ewma_alpha": 0.4},
    }) as engine:
        assert isinstance(engine.scheduler, TimeScheduler)
        assert engine.scheduler.retry_interval == 7
        assert engine.scheduler.warmup_rounds == 3
        assert engine.scheduler.min_time_saving == 0.1
        assert engine.scheduler.config.ewma_alpha == 0.4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("eos_position", [0, 3])
def test_time_scheduler_preserves_eos_and_short_requests(device, eos_position):
    target = TargetModel(build_tiny_target().model, device=device)
    prompt = [7, 8, 9]
    target.eos_token_ids = frozenset()
    continuation = target.greedy_generate(prompt, 8)
    eos = continuation[eos_position]
    expected_length = continuation.index(eos) + 1
    target.eos_token_ids = frozenset({eos})
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["ngram"]), warmup_rounds=1)
    with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
        assert output.token_ids == build_hf_method(target, 8)(prompt).token_ids == continuation[:expected_length]
        assert output.metadata["finished_eos"]
        assert sum(scheduler.counts.values()) == len(output.rounds)
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=1))
        assert output.token_ids == continuation[:1]
        assert not output.rounds and not scheduler.counts


def test_existing_scheduler_does_not_add_round_barriers(monkeypatch):
    with build_engine({"target": {"model": "tiny:1234"}, "lanes": {"ngram": {}},
                       "scheduler": {"type": "heuristic"}}) as engine:
        def unexpected_barrier():
            pytest.fail("the existing scheduler added a device barrier")
        monkeypatch.setattr(engine, "_synchronize_round", unexpected_barrier)
        engine.generate([7, 8, 9], DecodeConfig(max_new_tokens=6))
