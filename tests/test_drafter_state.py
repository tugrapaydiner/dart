from __future__ import annotations

import pytest

from dart.drafters.ar_draft import ARDrafter
from dart.drafters.base import NullDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeAction, DecodeConfig, VerifyResult
from tests.conftest import VOCAB, build_tiny_target, random_prompt
from tests.test_lanes_exactness import _diffusion_drafter


@pytest.mark.parametrize("next_context", ["continue", "restart", "other_prompt", "lane_gap"])
def test_repair_only_reuses_drafts_for_their_original_context(next_context):
    drafter = _diffusion_drafter("reconcile")
    prompt = random_prompt(340, 12)
    drafts = drafter.propose(prompt, 8)
    correction = (drafts[0] + 1) % VOCAB
    drafter.observe(VerifyResult([correction], 8, 0, False, False))
    contexts = {
        "continue": prompt + [correction],
        "restart": prompt,
        "other_prompt": random_prompt(341, 13),
        "lane_gap": prompt + [correction, 10],
    }
    drafter.propose(contexts[next_context], 8)
    assert drafter.stats["warm_rounds"] == int(next_context == "continue")


class AlternatingScheduler(HeuristicScheduler):
    def __init__(self, lane):
        super().__init__(SchedulerConfig())
        self.lane = lane
        self.round = 0

    def reset(self):
        super().reset()
        self.round = 0

    def choose_action(self, state, allowed=None):
        lane = self.lane if self.round % 2 == 0 else "target_only"
        self.round += 1
        return DecodeAction(lane, 3, 3, 16, 1, 4, True)


@pytest.mark.parametrize("lane", ["ar_draft", "diffusion"])
def test_lane_switches_and_repeated_requests_keep_complete_outputs(lane):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = ARDrafter(target) if lane == "ar_draft" else _diffusion_drafter("reconcile")
    lanes = {lane: drafter, "target_only": NullDrafter()}
    with DartEngine(target, lanes=lanes, scheduler=AlternatingScheduler(lane)) as engine:
        for seed in (342, 343, 342):
            prompt = random_prompt(seed, 8)
            expected = target.greedy_generate(prompt, max_new_tokens=16)
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=16))
            assert result.token_ids == expected
            assert {r.lane for r in result.rounds} == {lane, "target_only"}
    if lane == "diffusion":
        assert drafter.stats["warm_rounds"] == 0
