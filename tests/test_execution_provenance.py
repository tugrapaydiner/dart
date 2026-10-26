import json

import pytest

from benchmarks import execution_analysis as analysis
from benchmarks import execution_protocol as protocol
from benchmarks.execution_bench import read_protocol


def rehash(plan, status, reports):
    plan["protocol_hash"] = protocol.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    status["protocol_hash"] = plan["protocol_hash"]
    for report in reports:
        report["protocol_hash"] = plan["protocol_hash"]


def legacy(study):
    plan, status, reports = study()
    plan.update(schema="dart-execution-protocol-v1", control_commit=protocol.CONTROL_COMMIT,
                control_scope="optimized eager DART path retained from the control commit")
    rehash(plan, status, reports)
    return plan, status, reports


def test_new_runs_identify_the_control_by_their_actual_source(study):
    plan, status, reports = study()
    assert plan["schema"] == "dart-execution-protocol-v2"
    assert plan["control_commit"] is None
    result = analysis.analyze(plan, status, reports)
    assert result["provenance"]["control_commit"] is None
    assert result["provenance"]["source_hashes"] == plan["source_hashes"]
    assert result["provenance"]["control_scope"] == plan["control_scope"]
    assert result["provenance"]["protocol_schema"] == plan["schema"]


def test_archived_runs_keep_their_original_control_provenance(study):
    plan, status, reports = legacy(study)
    result = analysis.analyze(plan, status, reports)
    assert result["evaluation_passed"]
    assert result["provenance"]["control_commit"] == protocol.CONTROL_COMMIT
    assert "protocol_schema" not in result["provenance"]
    assert "control_scope" not in result["provenance"]


def test_old_protocols_cannot_run_with_the_new_executor(study, tmp_path, monkeypatch):
    plan, _, _ = legacy(study)
    monkeypatch.setattr("benchmarks.execution_bench.source_hashes", lambda: plan["source_hashes"])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported execution protocol"):
        read_protocol(path)


@pytest.mark.parametrize("field,value", [
    ("control_commit", protocol.CONTROL_COMMIT),
    ("control_scope", "optimized eager DART path retained from the control commit"),
])
def test_new_runs_cannot_claim_the_historical_control_after_rehashing(study, field, value):
    plan, status, reports = study()
    plan[field] = value
    rehash(plan, status, reports)
    with pytest.raises(ValueError, match="settings"):
        analysis.analyze(plan, status, reports)


@pytest.mark.parametrize("schema", ["unknown", None, 2])
def test_unknown_protocol_versions_fail_archive_validation(study, schema):
    plan, status, reports = study()
    plan["schema"] = schema
    rehash(plan, status, reports)
    with pytest.raises(ValueError):
        analysis.analyze(plan, status, reports)


def test_old_control_scope_cannot_be_relabelled(study):
    plan, status, reports = legacy(study)
    plan["control_scope"] = "different baseline"
    rehash(plan, status, reports)
    with pytest.raises(ValueError, match="settings"):
        analysis.analyze(plan, status, reports)
