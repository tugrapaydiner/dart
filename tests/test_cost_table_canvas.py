from __future__ import annotations

import pytest

from benchmarks.run_bench import build_hf_method
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
from dart.runtime.types import DecodeConfig, DecodeState, VerifyResult
from tests.conftest import build_tiny_target


def state(entropy=0.0):
    return DecodeState(16, 4, float("inf"), 1, "code", True, entropy, {}, 0.0, 0.0)


def observe(planner, action):
    planner.observe(action.lane, VerifyResult([7], 0, 0, True, False), round_ms=10.0)


@pytest.mark.parametrize("verifiers", [None, ("block", "tree")])
@pytest.mark.parametrize("entropy,expected", [(0.0, [2, 4, 8]), (2.0, [1, 2, 4])])
def test_canvas_budget_follows_each_cost_table_arm(verifiers, entropy, expected):
    config = SchedulerConfig(available_lanes=["diffusion"], canvas_len=32, entropy_gate_threshold=1.0)
    planner = CostTableScheduler(config, draft_lengths=(2, 4, 8), verifiers=verifiers, warmup_rounds=1)
    baseline = planner.choose_action(state(entropy))
    assert baseline.lane == "target_only" and baseline.canvas_len == 0
    observe(planner, baseline)
    widths = []
    for width in expected:
        for verifier in verifiers or (None,):
            action = planner.choose_action(state(entropy))
            assert action.lane == "diffusion" and action.verifier == verifier
            assert action.draft_len == width
            widths.append(action.canvas_len)
            observe(planner, action)
    assert widths == [width for width in expected for _ in verifiers or (None,)]


@pytest.mark.parametrize("entropy,expected", [(0.0, 32), (2.0, 16)])
def test_lane_only_time_scheduler_keeps_its_configured_canvas(entropy, expected):
    config = SchedulerConfig(available_lanes=["diffusion"], draft_len=4, canvas_len=32,
                             entropy_gate_threshold=1.0)
    planner = TimeScheduler(config, warmup_rounds=1)
    observe(planner, planner.choose_action(state(entropy)))
    assert planner.choose_action(state(entropy)).canvas_len == expected


@pytest.mark.parametrize("verifier", ["block", "tree"])
def test_engine_runs_distinct_canvas_widths_and_preserves_complete_outputs(verifier):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    model = DiffusionDrafterModel(DiffusionDrafterConfig(
        vocab_size=target.model.config.vocab_size, hidden_size=16, num_layers=1, num_heads=2,
        ctx_window=8, max_canvas=32,
    ))
    drafter = DiffusionDrafter(model, canvas_len=32, denoise_steps=1, adaptive_stop=False)
    planner = CostTableScheduler(SchedulerConfig(available_lanes=["diffusion"], canvas_len=32),
                                 draft_lengths=(2, 4, 8), verifiers=(verifier,), warmup_rounds=1)
    widths = []
    hook = model.register_forward_pre_hook(lambda module, args: widths.append(args[2].shape[1]))
    try:
        with DartEngine(target, lanes={"diffusion": drafter}, scheduler=planner) as engine:
            for prompt in ([7, 11, 13], [17, 19, 23]):
                expected = build_hf_method(target, 24)(prompt).token_ids
                widths.clear()
                output = engine.generate(prompt, DecodeConfig(max_new_tokens=24))
                assert output.token_ids == expected and len(output.token_ids) == 24
                assert widths[:3] == [2, 4, 8]
                assert {f"diffusion:{size}:{verifier}" for size in (2, 4, 8)} <= set(planner.counts)
                assert drafter._last_canvas is None and drafter._context == []
    finally:
        hook.remove()
