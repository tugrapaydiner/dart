import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from benchmarks import execution_protocol as execution
from benchmarks import feedback_analysis as analysis
from benchmarks import feedback_bench as bench
from benchmarks import feedback_figures as figures
from benchmarks import feedback_protocol as protocol
from dart.runtime.engine import DartEngine


@pytest.fixture
def feedback_study(study, monkeypatch, tmp_path):
    monkeypatch.setattr(protocol, "encode_prompt", lambda *args: [7, 8, 9])
    monkeypatch.setattr(protocol, "tokenizer_hash", lambda _: "a" * 64)
    monkeypatch.setattr(protocol, "version", lambda _: "1.0")
    monkeypatch.setattr(protocol.feedback_workloads, "load_frozen", lambda *args: protocol.records("test"))
    monkeypatch.setattr(protocol.verification_workloads, "load_frozen", lambda *args: protocol.records("validation"))
    sources = execution.source_hashes()
    sources.update(dict.fromkeys(("benchmarks/feedback_protocol.py", "benchmarks/feedback_bench.py",
                                  "benchmarks/feedback_analysis.py", "dart/verification/block_verify.py"), "d" * 64))
    monkeypatch.setattr(execution, "source_hashes", lambda: dict(sources))

    def build(split="test", *, candidate=(50, 50, 50), repetitive_only=False):
        _, status, templates = study(split)
        plan = protocol.make_protocol(SimpleNamespace(eos_token_id=2), 1024, tmp_path, split=split)
        status.update(protocol_hash=plan["protocol_hash"], timeout_seconds=plan["timeout_seconds"])
        reports = []
        cases = {case["id"]: case for case in plan["cases"]}

        def output(latency):
            return {"token_ids": [7, 8, 2], "output_hash": execution.digest([7, 8, 2]),
                    "latency_ms": latency, "eos": True, "early_eos": True}

        for session, template in enumerate(templates):
            report = copy.deepcopy(template)
            report.update(protocol_hash=plan["protocol_hash"], memory_scope=plan["memory_scope"])
            report["setup_ms"].pop("static")
            report.update(references={name: output(100) for name in cases},
                          checks=[{"case": name, "match": True, **output(candidate[session])} for name in cases],
                          warmups=[], samples=[])
            for call in execution.schedule(plan, session):
                latency = {"entropy_on4": 100, "entropy_off4": candidate[session], "entropy_off2": 65,
                           "replay_greedy": 80, "replay_hf_greedy": 90, "replay_hf_lookup": 95, "time_scheduler": 75}[call["method"]]
                if repetitive_only and call["method"] == "entropy_off4":
                    latency = 10 if cases[call["case"]]["workload"] == "repetition" else 80
                key = "samples" if call["phase"] == "timed" else "warmups"
                report[key].append({**call, "match": True, **output(latency)})
            reports.append(report)
        return plan, status, reports

    return build


def test_every_final_case_is_timed_and_both_gates_are_required(feedback_study):
    plan, status, reports = feedback_study()
    result = analysis.analyze(plan, status, reports)
    assert len(plan["cases"]) == len(plan["timed_ids"]) == result["unique_cases"] == 48
    assert result["confirmation_passed"]
    assert len(result["nonrepetitive"]["cases"]) == 36
    assert result["full_output_matches"] == 4176
    assert result["aggregate"]["mean_latency_ms"]["entropy_off4"] == 50
    assert result["nonrepetitive"]["candidate_comparisons"]["replay_greedy"]["session_speed_ratios"] == [1.6] * 3
    assert result["savings_concentration"]["repetition_fraction"] == pytest.approx(0.25)


def test_repetition_cannot_hide_nonrepetitive_parity(feedback_study):
    result = analysis.analyze(*feedback_study(repetitive_only=True))
    assert result["gates"]["entropy_latency_reduction"]
    assert not result["gates"]["nonrepetitive_latency_reduction"]
    assert not result["gates"]["nonrepetitive_interval"]
    assert result["savings_concentration"]["repetition_fraction"] == pytest.approx(1)
    assert not result["confirmation_passed"]


def test_validation_cannot_be_reported_as_confirmation(feedback_study):
    result = analysis.analyze(*feedback_study("validation"))
    assert result["unique_cases"] == 16
    assert not result["confirmation_passed"]
    assert result["nonrepetitive"]["candidate_comparisons"]["replay_greedy"]["session_ratio_ci95"] is None


def test_a_losing_session_remains_visible(feedback_study):
    result = analysis.analyze(*feedback_study(candidate=(40, 40, 110)))
    assert not result["gates"]["entropy_each_session"]
    assert not result["gates"]["nonrepetitive_each_session"]
    assert not result["confirmation_passed"]


@pytest.mark.parametrize("field", ["minimum_mean_latency_reduction", "minimum_nonrepetitive_latency_reduction",
                                   "maximum_nonrepetitive_workload_regression", "maximum_peak_allocated_bytes"])
def test_rehashing_cannot_change_acceptance_targets(feedback_study, field):
    plan, _, _ = feedback_study()
    plan["gates"][field] = 0.01
    plan["protocol_hash"] = execution.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    with pytest.raises(ValueError, match="settings or acceptance gates"):
        protocol.validate_archive(plan)


@pytest.mark.parametrize("changed", ["sample", "case", "method", "source", "model", "reference"])
def test_incomplete_or_changed_evidence_is_rejected(feedback_study, changed):
    plan, status, reports = feedback_study()
    if changed == "sample":
        reports[0]["samples"].pop()
    elif changed == "case":
        plan["cases"].pop()
    elif changed == "method":
        plan["methods"].remove("replay_hf_lookup")
    elif changed == "source":
        plan["source_hashes"].pop("benchmarks/feedback_bench.py")
    elif changed == "model":
        plan["model_hashes"].pop("model.safetensors")
    else:
        reports[1]["references"][plan["cases"][0]["id"]]["token_ids"] = [7, 2]
    plan["protocol_hash"] = execution.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    with pytest.raises(ValueError):
        analysis.analyze(plan, status, reports)


def test_live_reader_rejects_source_drift(feedback_study, monkeypatch, tmp_path):
    plan, _, _ = feedback_study()
    path = tmp_path / "plan.json"
    execution.write_protocol(path, plan)
    assert protocol.read_protocol(path) == plan
    monkeypatch.setattr(execution, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="source changed"):
        protocol.read_protocol(path)


def test_methods_share_the_target_and_keep_budget_and_feedback_settings(monkeypatch):
    from benchmarks import run_bench

    graph = object()

    def generate(engine, prompt, config):
        assert engine.target is graph
        return config.max_new_tokens, config.draft_len, engine.drafter.requires_entropy, engine.scheduler

    def baseline(target, budget, **kwargs):
        assert target is graph
        return lambda prompt: (budget, kwargs)

    monkeypatch.setattr(DartEngine, "generate", generate)
    monkeypatch.setattr(run_bench, "build_hf_method", baseline)
    monkeypatch.setattr(run_bench, "build_target_method", baseline)
    with ExitStack() as stack:
        methods = bench.build_methods(graph, [16, 64], stack)
        for budget, calls in methods.items():
            assert set(calls) == set(protocol.METHODS)
            assert calls["entropy_on4"]([7]) == (budget, 4, True, None)
            assert calls["entropy_off4"]([7]) == (budget, 4, False, None)
            assert calls["entropy_off2"]([7]) == (budget, 2, False, None)
            assert calls["replay_greedy"]([7]) == calls["replay_hf_greedy"]([7]) == (budget, {})
            assert calls["replay_hf_lookup"]([7]) == (budget, {"prompt_lookup_num_tokens": 4})
            assert calls["time_scheduler"]([7])[-1].requires_round_timing
            assert not calls["time_scheduler"]([7])[-1].requires_entropy


def write_study(path, value):
    path.mkdir()
    plan, status, reports = value
    for name, data in [("protocol", plan), ("status", status), *[(f"session-{i}", row) for i, row in enumerate(reports)]]:
        (path / (name + ".json")).write_text(json.dumps(data), encoding="utf-8")


def test_figures_recompute_raw_results_and_retain_every_loss(feedback_study, tmp_path):
    value = feedback_study(candidate=(70, 80, 90))
    for report in value[2]:
        for row in report["samples"]:
            if row["method"] == "entropy_off4" and "instruction" in row["case"]:
                row["latency_ms"] *= 2
    source, output, previews = [tmp_path / name for name in ("raw", "figures", "png")]
    write_study(source, value)
    (source / "analysis.json").write_text('{"confirmation_passed": true, "speedup": 999}')
    figures.main(["--run", str(source), "--out", str(output), "--previews", str(previews)])
    audit = json.loads((output / "figure-data.json").read_text())
    data = audit["data"]
    assert data["aggregate"]["mean_latency_ms"]["entropy_off4"] == 100
    assert len(data["cases"]) == 48 and len(data["nonrepetitive"]["cases"]) == 36
    assert len(data["slower_than_greedy"]) == 12 and not data["confirmation_passed"]
    assert set(audit["figures_sha256"]) == set(figures.NAMES)
    for name, digest in audit["figures_sha256"].items():
        raw = (output / (name + ".svg")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        assert ET.fromstring(raw).attrib["width"] == "982"
        assert (previews / (name + ".png")).read_bytes().startswith(b"\x89PNG")
    assert b"#ebcece" in (output / "cases.svg").read_bytes()
    assert data["artifact_hashes"]["session-0.json"] == hashlib.sha256((source / "session-0.json").read_bytes()).hexdigest()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(SystemExit):
        figures.main(["--run", str(source), "--out", str(output)])
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("kind", ["pilot", "missing", "extra", "mismatch"])
def test_figures_reject_invalid_or_single_session_evidence(feedback_study, tmp_path, kind):
    value = feedback_study("validation" if kind == "pilot" else "test")
    if kind == "missing":
        value[2][0]["samples"].pop()
    elif kind == "mismatch":
        value[2][0]["samples"][0]["token_ids"] = [7, 2]
    source, output = tmp_path / "raw", tmp_path / "figures"
    write_study(source, value)
    if kind == "extra":
        (source / "session-3.json").write_text("{}")
    with pytest.raises(SystemExit):
        figures.main(["--run", str(source), "--out", str(output)])
    assert not output.exists()
