from __future__ import annotations

import copy
import json
import math

import pytest

from benchmarks import execution_analysis as analysis
from benchmarks import execution_protocol as protocol


def rehash(plan, status, reports):
    plan["protocol_hash"] = protocol.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    status["protocol_hash"] = plan["protocol_hash"]
    for report in reports:
        report["protocol_hash"] = plan["protocol_hash"]


def test_complete_study_has_all_comparisons_and_only_an_evaluation_verdict(study):
    result = analysis.analyze(*study())
    assert result["evaluation_passed"] is True and all(result["evaluation_gates"].values())
    assert result["release_complete"] is False
    assert len(result["cases"]) == 12 and len(result["workloads"]) == 4 and len(result["sessions"]) == 3
    assert set(result["output_budgets"]) == {"16", "32", "64"}
    assert result["correctness"]["unique_cases"] == 48
    assert result["correctness"]["candidate_checks"] == 72
    assert result["correctness"]["warmup_outputs"] == 216
    assert result["correctness"]["timed_outputs"] == 648
    assert len(result["correctness"]["early_eos_cases"]) == 48
    comparison = result["aggregate"]["candidate_comparisons"]["current_dart"]
    assert comparison["mean_latency_reduction"] == 0.5
    assert comparison["session_ratio_ci95"] == pytest.approx([2, 2])
    assert all(not cases for cases in result["losses"].values())


def test_session_interval_uses_three_log_ratios_and_a_separate_arithmetic_estimator():
    result = analysis.paired_ratio([200, 200, 200], [50, 100, 200])
    assert result["arithmetic_mean_speed_ratio"] == pytest.approx(12 / 7)
    assert result["geometric_session_ratio"] == pytest.approx(2)
    lower, upper = result["session_ratio_ci95"]
    assert lower < 1 < upper and lower * upper == pytest.approx(4)
    margin = math.log(upper / 2)
    critical = margin * math.sqrt(3) / math.log(2)
    assert 0.5 + critical / (2 * math.sqrt(critical**2 + 2)) == pytest.approx(0.975, abs=1e-11)


def test_repeats_do_not_become_independent_sessions(study):
    plan, status, reports = study(candidate=(40, 50, 60))
    before = analysis.analyze(plan, status, reports)
    for report in reports:
        for row in report["samples"]:
            if row["method"] == "replay_dart":
                row["latency_ms"] += (-20, 0, 20)[row["repeat"]]
    after = analysis.analyze(plan, status, reports)
    assert before["aggregate"] == after["aggregate"]


@pytest.mark.parametrize("control,candidate", [([], []), ([1, 2], [1, 2]), ([1], [1, 2]), ([1], [0]),
                                               ([float("inf")], [1]), ([1], [True])])
def test_invalid_pairs_are_rejected(control, candidate):
    with pytest.raises(ValueError):
        analysis.paired_ratio(control, candidate)


def test_validation_pilot_cannot_claim_promotion_or_repeatability(study):
    result = analysis.analyze(*study("validation"))
    assert result["evaluation_passed"] is result["evaluation_gates"] is None
    assert result["aggregate"]["candidate_comparisons"]["current_dart"]["session_ratio_ci95"] is None
    assert result["correctness"]["unique_cases"] == 4


@pytest.mark.parametrize("change", ["gate", "type", "setting", "control", "case", "prompt_hash", "context", "source", "model", "package", "unknown"])
def test_changed_protocol_is_rejected_even_after_rehashing(study, change):
    plan, status, reports = study()
    if change == "gate":
        plan["gates"]["minimum_mean_latency_reduction"] = 0.01
    elif change == "type":
        plan["sessions"] = 3.0
    elif change == "setting":
        plan["settings"]["native_rms_norm"] = False
    elif change == "control":
        plan["control_commit"] = "0" * 40
    elif change == "case":
        plan["cases"][0]["workload"] = "repetition"
    elif change == "prompt_hash":
        plan["cases"][0]["prompt_ids"][0] = 1
    elif change == "context":
        plan["cases"][0]["prompt_ids"] *= 1024
        plan["cases"][0]["prompt_hash"] = protocol.digest(plan["cases"][0]["prompt_ids"])
    elif change == "source":
        plan["source_hashes"].pop("dart/models/replay.py")
    elif change == "model":
        plan["model_hashes"].pop("model.safetensors")
    elif change == "package":
        plan["packages"].pop("torch")
    else:
        plan["unknown"] = "setting"
    rehash(plan, status, reports)
    with pytest.raises(ValueError):
        analysis.analyze(plan, status, reports)


@pytest.mark.parametrize("change", ["status", "missing", "extra", "order", "duplicate", "tokens", "hash", "nan", "boolean",
                                    "process", "identity", "timeout", "memory", "setup", "environment", "early_stop"])
def test_partial_or_inconsistent_evidence_cannot_pass(study, change):
    plan, status, reports = study()
    if change == "status":
        status["status"] = "failed"
    elif change == "missing":
        reports.pop()
    elif change == "extra":
        reports.append(copy.deepcopy(reports[-1]))
    elif change == "order":
        reports.reverse()
    elif change == "duplicate":
        reports[0]["samples"][1] = copy.deepcopy(reports[0]["samples"][0])
    elif change == "tokens":
        reports[0]["checks"][0]["token_ids"].pop()
    elif change == "hash":
        reports[0]["samples"][0]["output_hash"] = "changed"
    elif change == "nan":
        reports[0]["samples"][0]["latency_ms"] = float("nan")
    elif change == "boolean":
        reports[0]["samples"][0]["latency_ms"] = True
    elif change == "process":
        status["processes"][1]["worker_pid"] = 999
    elif change == "identity":
        reports[1].update(pid=reports[0]["pid"], started_at=reports[0]["started_at"])
        status["processes"][1].update(pid=reports[0]["pid"], worker_pid=reports[0]["pid"])
    elif change == "timeout":
        status["elapsed_seconds"] = plan["timeout_seconds"] + 1
    elif change == "memory":
        reports[0]["peak_allocated_bytes"] = plan["gates"]["maximum_peak_allocated_bytes"] + 1
    elif change == "setup":
        reports[0]["setup_ms"].pop("replay")
    elif change == "environment":
        reports[1]["environment"]["device"] = "different"
    else:
        for report in reports:
            for row in [*report["references"].values(), *report["checks"], *report["warmups"], *report["samples"]]:
                row.update(token_ids=[7, 8], output_hash=protocol.digest([7, 8]), eos=False, early_eos=False)
    with pytest.raises(ValueError):
        analysis.analyze(plan, status, reports)


def test_consistent_but_changed_reference_in_another_process_fails(study):
    plan, status, reports = study()
    report = reports[1]
    for row in [*report["references"].values(), *report["checks"], *report["warmups"], *report["samples"]]:
        row.update(token_ids=[9, 8, 2], output_hash=protocol.digest([9, 8, 2]))
    with pytest.raises(ValueError, match="between processes"):
        analysis.analyze(plan, status, reports)


def test_os_pid_reuse_and_venv_redirectors_are_allowed(study):
    plan, status, reports = study()
    for report, process in zip(reports, status["processes"]):
        report.update(pid=999, parent_pid=process["pid"])
        process["worker_pid"] = 999
    assert analysis.analyze(plan, status, reports)["evaluation_passed"] is True


def test_slow_workload_and_strong_baseline_losses_are_not_hidden(study):
    plan, status, reports = study()
    for report in reports:
        for row in report["samples"]:
            if "-code-" in row["case"] and row["method"] == "replay_dart":
                row["latency_ms"] = 106
    result = analysis.analyze(plan, status, reports)
    assert result["evaluation_gates"]["mean_latency_reduction"] is True
    assert result["evaluation_gates"]["workload_regression"] is False
    assert result["evaluation_passed"] is False
    assert len(result["losses"]["current_dart"]) == len(result["losses"]["replay_greedy"]) == 3


def test_fastest_applicable_control_sets_the_aggregate_gate(study):
    plan, status, reports = study()
    for report in reports:
        for row in report["samples"]:
            if row["method"] == "replay_hf_lookup":
                row["latency_ms"] = 49
    result = analysis.analyze(plan, status, reports)
    assert result["aggregate"]["fastest_matched_baseline"] == "replay_hf_lookup"
    assert result["evaluation_gates"]["beat_fastest_matched_baseline"] is False
    assert result["evaluation_passed"] is False


def test_one_losing_session_and_uncertainty_can_fail_despite_a_mean_win(study):
    result = analysis.analyze(*study(candidate=(20, 20, 110)))
    assert result["evaluation_gates"]["mean_latency_reduction"] is True
    assert result["evaluation_gates"]["each_session_improves"] is False
    assert result["evaluation_gates"]["session_ratio_ci95"] is False
    assert result["evaluation_passed"] is False


def test_exact_latency_threshold_and_parameter_limit(study):
    plan, status, reports = study(candidate=(90, 90, 90))
    result = analysis.analyze(plan, status, reports)
    assert result["evaluation_gates"]["mean_latency_reduction"] is True
    for report in reports:
        report["extra_learned_parameters"] = 5_000_001
    assert analysis.analyze(plan, status, reports)["evaluation_gates"]["extra_learned_parameters"] is False


def write_study(path, data):
    plan, status, reports = data
    files = {"protocol.json": plan, "status.json": status,
             **{f"session-{index}.json": report for index, report in enumerate(reports)}}
    for name, value in files.items():
        (path / name).write_text(json.dumps(value), encoding="utf-8")


def test_archived_cli_needs_no_local_model_or_matching_current_sources(study, monkeypatch, tmp_path):
    write_study(tmp_path, study())
    monkeypatch.setattr(protocol, "source_hashes", lambda: pytest.fail("must not compare current sources"))
    monkeypatch.setattr(protocol, "model_hashes", lambda model: pytest.fail("must not load model files"))
    out = tmp_path / "analysis.json"
    monkeypatch.setattr("sys.argv", ["execution_analysis", "--run", str(tmp_path), "--out", str(out)])
    analysis.main()
    result = json.loads(out.read_text())
    assert result["evaluation_passed"] is True
    assert set(result["evidence_sha256"]) == {"protocol.json", "status.json", "session-0.json", "session-1.json", "session-2.json"}
    with pytest.raises(SystemExit):
        analysis.main()


@pytest.mark.parametrize("extra", [False, True])
def test_directory_requires_exact_session_files(study, tmp_path, extra):
    write_study(tmp_path, study())
    if extra:
        (tmp_path / "session-3.json").write_text("{}")
    else:
        (tmp_path / "session-2.json").unlink()
    with pytest.raises(ValueError, match="missing or extra"):
        analysis.analyze_directory(tmp_path)
