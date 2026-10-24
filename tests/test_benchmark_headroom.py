from __future__ import annotations

import json
import sys

import pytest

from benchmarks import run_bench
from benchmarks.headroom import PrecomputedDrafter, draft_budget, measure_headroom
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


def test_precomputed_drafter_checks_prefix_and_owns_its_tokens():
    prompt, continuation = [7, 8], [9, 10, 11]
    drafter = PrecomputedDrafter(prompt, continuation)
    continuation[1] = 99
    assert drafter.propose([7, 8, 9], 1) == [10]
    assert drafter.propose([7, 8, 9, 10, 11], 8) == []
    with pytest.raises(ValueError, match="committed prefix"):
        drafter.propose([7, 8, 6], 4)


@pytest.mark.parametrize("perfect_ms", [4, 12])
def test_draft_budget_uses_fastest_baseline_and_preserves_negative_room(perfect_ms):
    samples = [
        {"method": name, "repeat": repeat, "latency_ms": latency * noise}
        for repeat, noise in enumerate([1, 1.1, 0.9])
        for name, latency in {"hf_greedy": 15, "target_only": 12, "target_greedy": 10, "perfect_draft": perfect_ms}.items()
    ]
    result = draft_budget(samples, [2, 2])
    assert result["fastest_baseline"] == "target_greedy"
    assert result["extra_draft_ms_per_request"] == pytest.approx(10 - perfect_ms)
    assert result["extra_draft_ms_per_round"] == pytest.approx((10 - perfect_ms) / 2)
    assert result["ci95_perfect_speedup"] == pytest.approx([10 / perfect_ms] * 2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("early_eos", [False, True])
def test_headroom_replays_full_matches_including_eos(device, early_eos):
    target = TargetModel(build_tiny_target().model, device=device)
    prompts = [[7, 8, 9], [9, 8, 7]]
    if early_eos:
        target.eos_token_ids = frozenset(target.greedy_generate(p, 1)[0] for p in prompts)
    cfg = DecodeConfig(max_new_tokens=8, draft_len=4)
    with DartEngine(target) as engine:
        report = measure_headroom({
            "hf_greedy": run_bench.build_hf_method(target, 8),
            "target_greedy": run_bench.build_target_method(target, 8),
            "target_only": lambda p: engine.generate(p, cfg),
        }, prompts, target, cfg, warmup=1, repeats=2)
    assert "not deployable" in report["scope"]
    assert len(report["timing"]["samples"]) == 16
    for index in range(2):
        assert len({r["output_hash"] for r in report["timing"]["samples"] if r["prompt_index"] == index}) == 1
    if early_eos:
        assert report["budget"]["mean_proposal_rounds"] == 0
        assert report["budget"]["extra_draft_ms_per_round"] is None
    else:
        assert report["budget"]["mean_proposal_rounds"] > 0


def test_command_keeps_perfect_draft_out_of_deployable_results(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--prompts", "1", "--max-new", "8", "--warmup", "1",
        "--repeats", "2", "--headroom", "--out", str(tmp_path), "--tag", "room",
    ])
    run_bench.main()
    report = json.loads((tmp_path / "room.json").read_text())
    assert "perfect_draft" not in report["timing"]["summary"]
    assert "perfect_draft" not in report["lanes"]
    assert "perfect_draft" in report["headroom"]["timing"]["summary"]
