from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks import timing

CASE_KEYS = {"case_speedup_geomean", "case_ci95_speedup"}


def grid(ratios, repeats=2, base=10.0, jitter=0.0):
    """Samples where the reference is ratios[i] times slower than dart on prompt i."""
    rng = np.random.default_rng(5)
    samples = []
    for repeat in range(repeats):
        for index, ratio in enumerate(ratios):
            noise = 1 + jitter * rng.standard_normal()
            for name, latency in (("hf_greedy", base * ratio * noise), ("dart", base * noise)):
                samples.append({"method": name, "repeat": repeat, "prompt_index": index, "latency_ms": latency,
                                "tokens": 8, "peak_gpu_bytes": 0})
    return samples


def test_case_statistics_follow_per_prompt_speedups():
    summary = timing.summarize_samples(grid([1, 2, 4, 8]), "hf_greedy")["dart"]
    assert summary["case_speedup_geomean"] == pytest.approx(math.sqrt(8))
    low, high = summary["case_ci95_speedup"]
    assert 1 <= low < summary["speedup"] < high <= 8
    assert summary["ci95_speedup"] == pytest.approx([summary["speedup"]] * 2)  # Repeats alone show no noise.
    reference = timing.summarize_samples(grid([1, 2, 4, 8]), "hf_greedy")["hf_greedy"]
    assert reference["case_speedup_geomean"] == 1 and reference["case_ci95_speedup"] == [1, 1]


def test_case_statistics_leave_repeat_statistics_unchanged():
    samples = grid([1.5, 2, 3], repeats=4, jitter=0.2)
    summary = timing.summarize_samples(samples, "hf_greedy", seed=11)
    for row in summary.values():
        assert set(row) - CASE_KEYS == {"mean_ms", "median_ms", "ci95_ms", "tokens_per_second", "peak_gpu_bytes",
                                        "speedup", "ci95_speedup"}
    repeat_only = [{key: value for key, value in sample.items() if key != "prompt_index"} for sample in samples]
    old = timing.summarize_samples(repeat_only, "hf_greedy", seed=11)
    for name in summary:
        assert {key: summary[name][key] for key in old[name] if key not in CASE_KEYS} == \
               {key: value for key, value in old[name].items() if key not in CASE_KEYS}


@pytest.mark.parametrize("drop", ["prompt_index", "one_case"])
def test_case_statistics_need_a_complete_prompt_grid(drop):
    samples = grid([2, 3])
    if drop == "prompt_index":
        samples = [{key: value for key, value in sample.items() if key != "prompt_index"} for sample in samples]
    else:
        samples = [sample for sample in samples if not (sample["method"] == "dart" and sample["prompt_index"] == 1)]
        samples += [{**sample, "prompt_index": 0} for sample in samples
                    if sample["method"] == "dart" and sample["prompt_index"] == 0]
    row = timing.summarize_samples(samples, "hf_greedy")["dart"]
    assert row["case_speedup_geomean"] is None and row["case_ci95_speedup"] is None


def test_single_case_reports_a_ratio_but_no_case_interval():
    row = timing.summarize_samples(grid([3]), "hf_greedy")["dart"]
    assert row["case_speedup_geomean"] == pytest.approx(3) and row["case_ci95_speedup"] is None


def test_measure_methods_reports_case_statistics():
    calls = {"hf_greedy": 0, "dart": 0}

    def method(name):
        def run(prompt):
            calls[name] += 1
            return [1, 2]
        return run

    report = timing.measure_methods({"hf_greedy": method("hf_greedy"), "dart": method("dart")}, [[1], [2], [3]],
                                    "cpu", warmup=1, repeats=2)
    assert CASE_KEYS <= set(report["summary"]["dart"])
    assert "case" in report["confidence_method"]
