from __future__ import annotations

import pytest
import torch

from benchmarks import timing
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


def test_warmup_is_excluded_and_method_order_rotates(monkeypatch):
    clock = [0.0]
    calls = {"hf_greedy": 0, "dart": 0}
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])

    def method(name):
        def run(prompt):
            calls[name] += 1
            clock[0] += 1.0 if calls[name] <= 2 else 0.01
            return [7, 8]
        return run

    report = timing.measure_methods(
        {name: method(name) for name in calls}, [[1]], "cpu", warmup=1, repeats=4,
    )
    assert len(report["samples"]) == 8
    assert all(sample["latency_ms"] == pytest.approx(10) for sample in report["samples"])
    assert [sample["method"] for sample in report["samples"][:4]] == ["hf_greedy", "dart", "dart", "hf_greedy"]
    assert report["summary"]["dart"]["mean_ms"] == pytest.approx(10)


def test_gpu_timing_waits_for_pending_work(monkeypatch):
    clock, pending = [0.0], [0.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])

    def synchronize(device):
        clock[0] += pending[0]
        pending[0] = 0.0

    def run(prompt):
        pending[0] += 0.02
        return [7]

    monkeypatch.setattr(timing.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(timing.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(timing.torch.cuda, "max_memory_allocated", lambda device: 123)
    report = timing.measure_methods({"hf_greedy": run}, [[1]], "cuda", warmup=1, repeats=2)
    assert all(sample["latency_ms"] == pytest.approx(20) for sample in report["samples"])
    assert report["summary"]["hf_greedy"]["peak_gpu_bytes"] == 123


def test_progress_overhead_is_excluded_from_timing(monkeypatch):
    clock, messages, calls = [0.0], [], []
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])

    def run(prompt):
        calls.append(tuple(prompt))
        clock[0] += 0.01
        return [7, 8]

    def progress(message):
        messages.append(message)
        clock[0] += 5.0

    report = timing.measure_methods(
        {"hf_greedy": run, "dart": run}, [[1], [2]], "cpu", warmup=1, repeats=2, progress=progress,
    )
    assert len(calls) == 18
    assert messages[:2] == ["Reference outputs: 1/2 prompts", "Reference outputs: 2/2 prompts"]
    assert messages[3] == "Preflight checks: 2/2 prompts"
    assert messages[5] == "Warm-up 1/1: 2/2 prompts"
    assert messages[-1] == "Timed 8/8 samples"
    assert all(sample["latency_ms"] == pytest.approx(10) for sample in report["samples"])


@pytest.mark.parametrize("wrong", [[7], [7, 9]])
def test_mismatched_outputs_fail_before_timing(wrong):
    with pytest.raises(ValueError, match="complete token mismatch"):
        timing.measure_methods({"hf_greedy": lambda p: [7, 8], "wrong": lambda p: wrong}, [[1]], "cpu")


def test_timed_runs_also_check_complete_outputs():
    calls = [0]

    def unstable(prompt):
        calls[0] += 1
        return [7, 8] if calls[0] <= 2 else [7]

    with pytest.raises(ValueError, match="unstable: complete token mismatch"):
        timing.measure_methods({"hf_greedy": lambda p: [7, 8], "unstable": unstable}, [[1]], "cpu", warmup=1, repeats=2)


def test_reference_output_is_frozen_before_other_methods_run():
    shared = [7, 8]

    def mutate(prompt):
        shared[-1] = 9
        return shared

    with pytest.raises(ValueError, match="complete token mismatch"):
        timing.measure_methods({"hf_greedy": lambda p: shared, "mutate": mutate}, [[1]], "cpu", warmup=1, repeats=2)


def test_paired_speedup_interval_keeps_shared_timing_noise():
    samples = []
    for repeat, delay in enumerate([10, 20, 30, 40]):
        for name, scale in (("hf_greedy", 2), ("dart", 1)):
            samples.append({"method": name, "repeat": repeat, "latency_ms": delay * scale,
                            "tokens": 10, "peak_gpu_bytes": 0})
    summary = timing.summarize_samples(samples, "hf_greedy")
    assert summary["dart"]["mean_ms"] == 25
    assert summary["dart"]["speedup"] == 2
    assert summary["dart"]["ci95_speedup"] == [2, 2]
    assert summary["dart"]["ci95_ms"][0] < 25 < summary["dart"]["ci95_ms"][1]


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_invalid_timings_cannot_produce_a_speedup(latency):
    samples = [{"method": "hf_greedy", "repeat": i, "latency_ms": latency,
                "tokens": 1, "peak_gpu_bytes": 0} for i in range(2)]
    with pytest.raises(ValueError, match="positive finite"):
        timing.summarize_samples(samples, "hf_greedy")


def test_incomplete_repeats_cannot_produce_a_speedup():
    samples = [{"method": "hf_greedy", "repeat": i, "latency_ms": 10,
                "tokens": 1, "peak_gpu_bytes": 0} for i in range(2)]
    samples.append({**samples[0], "method": "dart"})
    with pytest.raises(ValueError, match="same number of samples"):
        timing.summarize_samples(samples, "hf_greedy")


@pytest.mark.parametrize("warmup,repeats", [(0, 2), (1, 1), (1.5, 2), (1, True)])
def test_invalid_repeat_settings_fail_before_generation(warmup, repeats):
    def run(prompt):
        pytest.fail("Invalid settings reached the model")

    with pytest.raises(ValueError, match="warmup >= 1"):
        timing.measure_methods({"hf_greedy": run}, [[1]], "cpu", warmup=warmup, repeats=repeats)


@pytest.mark.parametrize("device", DEVICES)
def test_real_model_methods_produce_complete_timed_samples(device):
    target = TargetModel(build_tiny_target().model, device=device)
    prompt = [7, 8, 9, 7, 8, 9]
    config = DecodeConfig(max_new_tokens=5, draft_len=2)

    def hf(ids):
        output = target.model.generate(torch.tensor([ids], device=device), max_new_tokens=5, do_sample=False, pad_token_id=0)
        return output[0, len(ids):].tolist()

    with DartEngine(target) as base, DartEngine(target, NGramDrafter()) as ngram:
        report = timing.measure_methods({
            "hf_greedy": hf, "target_only": lambda p: base.generate(p, config),
            "ngram": lambda p: ngram.generate(p, config),
        }, [prompt], device, warmup=1, repeats=2)
    assert len(report["samples"]) == 6
    assert len({sample["output_hash"] for sample in report["samples"]}) == 1
    assert all(row["mean_ms"] > 0 for row in report["summary"].values())
    assert (report["summary"]["ngram"]["peak_gpu_bytes"] > 0) == device.startswith("cuda")
