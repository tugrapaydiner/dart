from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import profiling, run_bench
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


def test_phase_timers_wait_for_device_and_restore_engine(monkeypatch):
    clock, pending = [0.0], [0.0]
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: clock[0])

    def synchronize(device):
        clock[0] += pending[0]
        pending[0] = 0.0

    def enqueue(seconds):
        pending[0] += seconds

    class Model(torch.nn.Module):
        def forward(self, seconds):
            enqueue(seconds)

    monkeypatch.setattr(profiling, "synchronize", synchronize)
    target = SimpleNamespace(device="cuda", model=Model())
    engine = SimpleNamespace(target=target, verifier_name="block", _propose=lambda: enqueue(0.03), verify=lambda: target.model(0.02))
    original = dict(vars(engine))

    def method(prompt):
        clock[0] += 0.005
        target.model(0.01)
        engine._propose()
        engine.verify()
        clock[0] += 0.004
        return [7]

    report = profiling.profile_methods({"dart": method}, [[1]], target, {"dart": engine}, reference="dart", repeats=1)
    row = report["samples"][0]
    for key, expected in {"request_ms": 69, "setup_ms": 5, "prefill_ms": 10, "decode_ms": 54, "draft_ms": 30, "verify_ms": 20}.items():
        assert row[key] == pytest.approx(expected)
    assert vars(engine) == original
    assert not target.model._forward_hooks
    assert not target.model._forward_pre_hooks


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("early_eos", [False, True])
def test_real_phases_preserve_complete_output_and_do_not_overlap(device, early_eos):
    target = TargetModel(build_tiny_target().model, device=device)
    prompt = [7, 8, 9, 7, 8, 9]
    if early_eos:
        target.eos_token_ids = frozenset(target.greedy_generate(prompt, 1))
    cfg = DecodeConfig(max_new_tokens=5, draft_len=2)
    with DartEngine(target, NGramDrafter()) as engine:
        report = profiling.profile_methods({
            "hf_greedy": run_bench.build_hf_method(target, 5),
            "ngram": lambda p: engine.generate(p, cfg),
        }, [prompt], target, {"ngram": engine}, repeats=1)
        assert "_propose" not in vars(engine)
        assert engine.generate(prompt, cfg).token_ids == target.greedy_generate(prompt, 5)
    assert len({row["output_hash"] for row in report["samples"]}) == 1
    for row in report["samples"]:
        assert row["request_ms"] == pytest.approx(row["setup_ms"] + row["prefill_ms"] + row["decode_ms"])
        assert row["prefill_ms"] > 0
        if row["method"] == "ngram":
            assert row["draft_ms"] + row["verify_ms"] <= row["decode_ms"]
            assert (row["verify_ms"] == 0) is early_eos
        else:
            assert row["verify_ms"] is None


def test_failed_profile_removes_hooks_and_wrappers():
    target = build_tiny_target()
    engine = DartEngine(target)
    original = dict(vars(engine))

    def fail(prompt):
        target.prefill(prompt)
        raise RuntimeError("failed request")

    with pytest.raises(RuntimeError, match="failed request"):
        profiling.profile_methods({"hf_greedy": run_bench.build_hf_method(target, 2), "bad": fail}, [[7, 8]], target, {"bad": engine})
    assert vars(engine) == original
    assert not target.model._forward_hooks
    assert not target.model._forward_pre_hooks


def test_command_keeps_phase_diagnostics_out_of_timed_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--prompts", "1", "--max-new", "3", "--warmup", "1",
        "--repeats", "2", "--profile", "--out", str(tmp_path), "--tag", "phases",
    ])
    run_bench.main()
    data = json.loads((tmp_path / "phases.json").read_text())
    assert len(data["timing"]["samples"]) == 8
    assert len(data["profile"]["samples"]) == 8
    assert data["profile"]["summary"]["ngram"]["verify_ms"] > 0
    assert "prefill_ms" not in data["timing"]["samples"][0]
