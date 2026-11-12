from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks import corpus_bench
from dart import cli
from tests.test_cli_policy import policy_model as policy_model
from tests.test_corpus_bench import build_engine
from tests.test_corpus_eval import records


@pytest.mark.parametrize("profile", [False, True])
def test_static_control_matches_outputs_and_can_be_the_fastest_in_every_group(monkeypatch, profile):
    measure = corpus_bench.measure_methods

    def timed(*args, **kwargs):
        result = measure(*args, **kwargs)
        for sample in result["samples"]:
            sample["latency_ms"] = 1.0 if sample["method"] == "hf_static_greedy" else 2.0
        return result

    monkeypatch.setattr(corpus_bench, "measure_methods", timed)
    saved = []
    with build_engine() as engine:
        report = corpus_bench.benchmark_corpus(engine, records(), hf_static_capacity=16, warmup=1,
                                               repeats=2, profile=profile, on_timing=saved.append)
        assert engine.target.record_hidden is True
    assert report["hf_static_greedy"] == saved[0]["hf_static_greedy"] == {
        "capacity": 16, "cache_implementation": "static", "compile": False,
        "cache_lifetime": "request", "allocation_scope": "included in request timing",
    }
    assert "hf_static_greedy" in report["baseline_methods"]
    assert report["record_hidden_by_method"]["hf_static_greedy"] is False
    samples = report["timing"]["samples"]
    assert len(samples) == 24
    if profile:
        samples += report["profile"]["samples"]
    for sample in samples:
        expected = report["correctness"]["outputs"][sample["prompt_index"]]
        assert sample["output_hash"] == expected["output_hash"]
        assert sample["tokens"] == expected["generated_tokens"]
    for summary in [report["summary"], *report["by_workload"].values(), *report["by_budget"].values()]:
        assert summary["fastest_baseline"] == "hf_static_greedy"
        assert summary["vs_fastest"]["dart"]["speedup"] == 0.5


@pytest.mark.parametrize("capacity", [0, True, 1.5, 4, 1025])
def test_static_benchmark_admits_all_requests_before_generation(monkeypatch, capacity):
    monkeypatch.setattr(corpus_bench, "evaluate_corpus", lambda *a, **k: pytest.fail("unexpected generation"))
    with build_engine() as engine, pytest.raises(ValueError, match="static_capacity"):
        corpus_bench.benchmark_corpus(engine, records(), hf_static_capacity=capacity)


@pytest.mark.parametrize("failure_call", [1, 3, 5], ids=["preflight", "warmup", "timing"])
def test_static_mismatch_stops_before_saving(monkeypatch, failure_call):
    measure, saved = corpus_bench.measure_methods, []

    def corrupt(methods, *args, **kwargs):
        original, calls = methods["hf_static_greedy"], 0

        def changed(prompt):
            nonlocal calls
            calls += 1
            output = original(prompt)
            return replace(output, token_ids=output.token_ids[:-1]) if calls == failure_call else output

        methods["hf_static_greedy"] = changed
        return measure(methods, *args, **kwargs)

    monkeypatch.setattr(corpus_bench, "measure_methods", corrupt)
    with build_engine() as engine, pytest.raises(ValueError, match="hf_static_greedy: complete token mismatch"):
        corpus_bench.benchmark_corpus(engine, records(), hf_static_capacity=16, profile=False,
                                      warmup=1, repeats=2, on_timing=saved.append)
    assert not saved


def test_cli_runs_static_and_policy_controls_on_the_same_model(policy_model, monkeypatch, tmp_path):
    directory, built, _ = policy_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    output = tmp_path / "static.json"
    assert cli.main(["bench", "--model", str(directory), "--hf-static-capacity", "16", "--compare-policies",
                     "--skip-profile", "--warmup", "1", "--repeats", "2", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    partial = json.loads(output.with_suffix(".timing.json").read_text())
    assert len(built) == 1 and len(report["timing"]["samples"]) == 52
    assert report["hf_static_greedy"] == partial["hf_static_greedy"]
    assert report["hf_static_greedy"]["capacity"] == 16
    assert "hf_static_greedy" in report["baseline_methods"] and len(report["policy_controls"]) == 7


def test_invalid_cli_capacity_fails_before_loading(policy_model, capsys):
    directory, built, _ = policy_model
    with pytest.raises(SystemExit) as error:
        cli.main(["bench", "--model", str(directory), "--hf-static-capacity", "0"])
    assert error.value.code == 2 and not built
    assert "hf-static-capacity must be positive" in capsys.readouterr().err
