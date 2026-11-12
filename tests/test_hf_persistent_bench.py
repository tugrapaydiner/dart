from __future__ import annotations

import gc
import json
import weakref
from dataclasses import replace

import pytest

from benchmarks import corpus_bench
from dart import cli
from tests.test_cli_policy import policy_model as policy_model
from tests.test_corpus_bench import build_engine
from tests.test_corpus_eval import records


@pytest.mark.parametrize("profile", [False, True])
def test_one_cache_across_budgets_counts_in_every_comparison(monkeypatch, profile):
    measure, created, refs = corpus_bench.measure_methods, [], []
    control = corpus_bench.PersistentStaticControl

    class Observed(control):
        def __init__(self, *args):
            super().__init__(*args)
            created.append(self)
            refs.extend([weakref.ref(self.cache), weakref.ref(self.cache.layers[0].keys)])

    def timed(methods, *args, **kwargs):
        result = measure(methods, *args, **kwargs)
        for sample in result["samples"]:
            sample["latency_ms"] = 1.0 if sample["method"] == corpus_bench.PERSISTENT_CONTROL else 2.0
        return result

    monkeypatch.setattr(corpus_bench, "PersistentStaticControl", Observed)
    monkeypatch.setattr(corpus_bench, "measure_methods", timed)
    saved = []
    with build_engine() as engine:
        report = corpus_bench.benchmark_corpus(engine, records(), hf_static_capacity=16, hf_persistent_capacity=16,
                                               warmup=1, repeats=2, profile=profile, on_timing=saved.append)
        assert engine.target.record_hidden is True
    metadata = report[corpus_bench.PERSISTENT_CONTROL]
    assert metadata == saved[0][corpus_bench.PERSISTENT_CONTROL]
    assert metadata["cache_count"] == 1 and metadata["cache_bytes"] > 0 and metadata["preallocation_ms"] >= 0
    assert "reset before and after" in metadata["allocation_scope"]
    assert "all methods" in report["memory_scope"]
    assert report["hf_static_greedy"]["cache_lifetime"] == "request"
    assert corpus_bench.PERSISTENT_CONTROL in report["baseline_methods"]
    assert not report["record_hidden_by_method"][corpus_bench.PERSISTENT_CONTROL]
    for summary in [report["summary"], *report["by_workload"].values(), *report["by_budget"].values()]:
        assert summary["fastest_baseline"] == corpus_bench.PERSISTENT_CONTROL
        assert summary["vs_fastest"]["dart"]["speedup"] == 0.5
    assert len(created) == 1 and created[0].closed
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("capacity", [0, True, 1.5, 4, 1025])
def test_all_requests_admitted_before_correctness_or_allocation(monkeypatch, capacity):
    monkeypatch.setattr(corpus_bench, "evaluate_corpus", lambda *a, **k: pytest.fail("unexpected generation"))
    monkeypatch.setattr("benchmarks.hf_static.StaticCache", lambda *a, **k: pytest.fail("unexpected allocation"))
    with build_engine() as engine, pytest.raises(ValueError, match="capacity"):
        corpus_bench.benchmark_corpus(engine, records(), hf_persistent_capacity=capacity)


@pytest.mark.parametrize("phase", ["preflight", "warmup", "timing", "callback", "profile"])
def test_all_failure_paths_close_the_shared_cache(monkeypatch, phase):
    original, instances = corpus_bench.PersistentStaticControl, []

    class Observed(original):
        def __init__(self, *args):
            super().__init__(*args)
            instances.append(self)
            self.calls = 0

        def generate(self, *args):
            self.calls += 1
            output = super().generate(*args)
            if self.calls == {"preflight": 1, "warmup": 3, "timing": 5}.get(phase):
                return replace(output, token_ids=output.token_ids[:-1])
            return output

    def fail(*args):
        raise RuntimeError("failed phase")

    monkeypatch.setattr(corpus_bench, "PersistentStaticControl", Observed)
    if phase == "profile":
        monkeypatch.setattr(corpus_bench, "profile_methods", fail)
    with build_engine() as engine, pytest.raises((ValueError, RuntimeError), match="mismatch|failed phase"):
        corpus_bench.benchmark_corpus(engine, records(), hf_persistent_capacity=16, warmup=1, repeats=2,
                                      on_timing=fail if phase == "callback" else None, profile=phase == "profile")
    assert len(instances) == 1 and instances[0].closed and instances[0].cache is None


def test_cli_runs_both_static_controls(policy_model, monkeypatch, tmp_path):
    directory, built, _ = policy_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    output = tmp_path / "persistent.json"
    assert cli.main(["bench", "--model", str(directory), "--hf-static-capacity", "16", "--hf-persistent-capacity", "16",
                     "--skip-profile", "--warmup", "1", "--repeats", "2", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    partial = json.loads(output.with_suffix(".timing.json").read_text())
    assert len(built) == 1 and len(report["timing"]["samples"]) == 28
    assert report[corpus_bench.PERSISTENT_CONTROL] == partial[corpus_bench.PERSISTENT_CONTROL]
    assert report[corpus_bench.PERSISTENT_CONTROL]["capacity"] == 16


def test_invalid_cli_capacity_fails_before_loading(policy_model, capsys):
    directory, built, _ = policy_model
    with pytest.raises(SystemExit) as error:
        cli.main(["bench", "--model", str(directory), "--hf-persistent-capacity", "0"])
    assert error.value.code == 2 and not built
    assert "hf-persistent-capacity must be positive" in capsys.readouterr().err
