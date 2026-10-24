from __future__ import annotations

import pytest

from dart.drafters.base import NullDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import BanditScheduler, HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeConfig, DecodeState, VerifyResult
from tests.conftest import build_tiny_target, random_prompt


def decode_state():
    return DecodeState(
        context_len=8, generated_len=1, gpu_free_gb=4.0, batch_size=1,
        prompt_kind="chat", exact_required=True, entropy_ewma=0.0,
        acceptance_ewma={}, kv_transfer_ms=0.0, target_step_ms=0.0,
    )


@pytest.mark.parametrize("scheduler_type", [HeuristicScheduler, BanditScheduler])
def test_reset_allows_a_rejected_lane_to_run_again(scheduler_type):
    scheduler = scheduler_type(SchedulerConfig(
        available_lanes=["ngram", "target_only"], max_rejection_rate=0.5, draft_len=3,
    ))
    scheduler.observe("ngram", VerifyResult([5], 3, 0, False, False))
    assert scheduler.choose_action(decode_state()).lane == "target_only"
    scheduler.reset()
    action = scheduler.choose_action(decode_state())
    assert action.lane == "ngram"
    assert action.draft_len == 3


def test_bandit_reset_rechecks_memory_constraints():
    scheduler = BanditScheduler(SchedulerConfig(
        available_lanes=["compressed_kv", "target_only"], max_gpu_kv_gb=1.0,
    ))
    scheduler.observe("compressed_kv", VerifyResult([5, 6], 1, 1, True, False), kv_gb=2.0)
    assert scheduler.choose_action(decode_state()).lane == "target_only"
    scheduler.reset()
    assert scheduler.choose_action(decode_state()).lane == "compressed_kv"


class WrongDrafter:
    name = "ngram"

    def __init__(self, target):
        self.target = target

    def propose(self, committed, max_tokens):
        correct = self.target.greedy_generate(committed, max_new_tokens=1)[0]
        return [(correct + 1) % self.target.model.config.vocab_size] * max_tokens

    def observe(self, result):
        pass


@pytest.mark.parametrize("scheduler_type", [HeuristicScheduler, BanditScheduler])
def test_new_requests_retry_lanes_benched_by_previous_prompts(scheduler_type):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    lanes = {"ngram": WrongDrafter(target), "target_only": NullDrafter()}
    scheduler = scheduler_type(SchedulerConfig(
        available_lanes=list(lanes), max_rejection_rate=0.5, draft_len=2,
    ))
    with DartEngine(target, lanes=lanes, scheduler=scheduler) as engine:
        for seed in (311, 312):
            prompt = random_prompt(seed, 8)
            expected = target.greedy_generate(prompt, max_new_tokens=8)
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
            assert result.token_ids == expected
            assert result.rounds[0].lane == "ngram"
            assert len(result.rounds) > 1
            assert all(step.lane == "target_only" for step in result.rounds[1:])
