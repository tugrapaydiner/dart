from __future__ import annotations

import pytest

from benchmarks.run_bench import build_hf_method
from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import BanditScheduler, HeuristicScheduler, SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
from dart.runtime.types import DecodeConfig, VerifyResult
from tests.test_scheduler_state import decode_state


@pytest.fixture(params=[HeuristicScheduler, BanditScheduler, TimeScheduler, CostTableScheduler])
def scheduler(request):
    options = {"warmup_rounds": 1, "max_probe_rounds": 1} if issubclass(request.param, TimeScheduler) else {}
    return request.param(SchedulerConfig(available_lanes=["ngram"], draft_len=2), **options)


def seed_history(scheduler):
    timing = {"round_ms": 10} if isinstance(scheduler, TimeScheduler) else {"draft_ms": 10}
    for _ in range(2):
        action = scheduler.choose_action(decode_state())
        scheduler.observe(action.lane, VerifyResult([7], action.draft_len, 0, False, False), **timing)
    assert scheduler.acceptance_ewma


def assert_empty(scheduler):
    assert not scheduler.acceptance_ewma
    if isinstance(scheduler, BanditScheduler):
        assert not scheduler.counts and not scheduler.reward_sums and not scheduler.kv_gb
        assert scheduler._last_lane is None
    if isinstance(scheduler, TimeScheduler):
        assert not scheduler.counts and not scheduler.round_ms and not scheduler.last_played
        assert not scheduler.committed_tokens and not scheduler.kv_gb
        assert scheduler.decisions == scheduler.probe_rounds == 0


def test_zero_output_request_clears_history_without_prefill(tiny_target, scheduler, monkeypatch):
    seed_history(scheduler)

    def unexpected(*args, **kwargs):
        pytest.fail("zero-output request performed model or draft work")

    drafter = NGramDrafter()
    monkeypatch.setattr(tiny_target, "prefill", unexpected)
    monkeypatch.setattr(drafter, "propose", unexpected)
    with DartEngine(tiny_target, lanes={"ngram": drafter}, scheduler=scheduler) as engine:
        output = engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=0))
    assert output.token_ids == [] and output.rounds == []
    assert not output.metadata["finished_eos"]
    assert_empty(scheduler)


@pytest.mark.parametrize("stage", ["prefill", "seed"])
def test_failure_before_first_round_does_not_keep_previous_history(tiny_target, scheduler, monkeypatch, stage):
    seed_history(scheduler)
    failure = RuntimeError("request failed before decoding")

    def fail(*args, **kwargs):
        raise failure

    drafter = NGramDrafter()
    with DartEngine(tiny_target, lanes={"ngram": drafter}, scheduler=scheduler) as engine:
        with monkeypatch.context() as patch:
            patch.setattr(tiny_target if stage == "prefill" else engine,
                          "prefill" if stage == "prefill" else "_seed_drafters", fail)
            with pytest.raises(RuntimeError) as caught:
                engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4, seed=12))
        assert caught.value is failure
        assert_empty(scheduler)
        assert tiny_target.last_hidden_block is None
        monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
        expected = tiny_target.greedy_generate([5, 6, 7], 4)
        output = engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4, seed=12))
        assert output.token_ids == expected


def test_rejected_request_settings_preserve_previous_history(tiny_target, scheduler):
    seed_history(scheduler)
    saved = dict(scheduler.acceptance_ewma)
    with DartEngine(tiny_target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
        with pytest.raises(ValueError):
            engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=-1))
    assert scheduler.acceptance_ewma == saved


def test_zero_output_between_real_requests_starts_fresh(tiny_target, scheduler, monkeypatch):
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    drafter = NGramDrafter()
    monkeypatch.setattr(drafter, "propose", lambda committed, budget: [5] * budget)
    with DartEngine(tiny_target, lanes={"ngram": drafter}, scheduler=scheduler) as engine:
        for prompt in ([5, 6, 7] * 2, [8, 9, 10] * 2):
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
            assert output.token_ids == build_hf_method(tiny_target, 8)(prompt).token_ids
            assert scheduler.acceptance_ewma
            assert engine.generate(prompt, DecodeConfig(max_new_tokens=0)).token_ids == []
            assert_empty(scheduler)
