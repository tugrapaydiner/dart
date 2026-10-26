from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from benchmarks import execution_figures as figures


def write_study(path, value):
    plan, status, reports = value
    path.mkdir()
    for name, data in [("protocol", plan), ("status", status), *[(f"session-{i}", row) for i, row in enumerate(reports)]]:
        (path / (name + ".json")).write_text(json.dumps(data), encoding="utf-8")


def test_public_figures_recompute_raw_results_and_keep_losses(study, tmp_path):
    value = study(candidate=(70, 80, 90))
    for report in value[2]:
        for row in report["samples"]:
            if row["method"] == "replay_dart" and "structured" in row["case"]:
                row["latency_ms"] *= 2
    source, output, previews = (tmp_path / name for name in ("raw", "figures", "png"))
    write_study(source, value)
    (source / "analysis.json").write_text('{"evaluation_passed": true, "speedup": 999}')
    figures.main(["--run", str(source), "--out", str(output), "--previews", str(previews)])
    audit = json.loads((output / "figure-data.json").read_text())
    data = audit["data"]
    assert data["aggregate"]["mean_latency_ms"]["replay_dart"] == 100
    assert data["aggregate"]["mean_latency_ms"]["replay_greedy"] == 80
    assert data["evaluation_passed"] is False and data["release_complete"] is False
    assert len(data["cases"]) == 12 and len(data["workloads"]) == 4
    assert len(data["aggregate"]["session_mean_latency_ms"]) == 3
    assert set(audit["figures_sha256"]) == set(figures.NAMES)
    for name, digest in audit["figures_sha256"].items():
        raw = (output / (name + ".svg")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        assert ET.fromstring(raw).attrib["width"] == "982"
        assert (previews / (name + ".png")).read_bytes().startswith(b"\x89PNG")
        if name in ("cases", "workloads"):
            assert b"#ebcece" in raw and b"Structured" in raw
            assert b"three session log ratios" in raw
        else:
            assert b"not confidence intervals" in raw
    assert audit["data"]["evidence_sha256"]["session-0.json"] == hashlib.sha256((source / "session-0.json").read_bytes()).hexdigest()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit):
        figures.main(["--run", str(source), "--out", str(output)])
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("kind", ["pilot", "incomplete", "mismatch"])
def test_invalid_or_single_session_evidence_cannot_produce_final_figures(study, tmp_path, kind):
    value = study("validation" if kind == "pilot" else "test")
    if kind == "incomplete":
        value[2][0]["samples"].pop()
    elif kind == "mismatch":
        value[2][0]["samples"][0]["token_ids"][0] = 9
    source, output = tmp_path / "raw", tmp_path / "figures"
    write_study(source, value)
    with pytest.raises(SystemExit):
        figures.main(["--run", str(source), "--out", str(output)])
    assert not output.exists()
