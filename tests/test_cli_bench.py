from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dart import cli
from tests.test_cli_check import local_model as local_model
from tests.test_corpus_eval import records


def test_benchmark_command_uses_validation_and_records_full_comparisons(local_model, tmp_path, monkeypatch, capsys):
    directory, built = local_model
    rows = records()
    extra = [replace(row, id="z-" + row.id, source="z-" + row.source, text="z" + row.text) for row in rows]
    held_out = replace(rows[0], id="held-out", source="held-out", family="held-out", split="test", text="zzz")
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: [held_out, *extra, *reversed(rows)])
    output = tmp_path / "bench.json"
    assert cli.main(["bench", "--model", str(directory), "--warmup", "1", "--repeats", "2", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["split"] == "validation" and report["command"] == "bench"
    assert report["selection"] == {"rule": "first sorted IDs per workload", "per_workload": 1,
                                    "available_prompts": 4, "selected_prompts": 2}
    assert report["correctness"]["summary"]["exact_match_rate"] == 1
    assert [row["id"] for row in report["correctness"]["outputs"]] == ["a", "b"]
    assert len(report["timing"]["samples"]) == 16 and len(report["profile"]["samples"]) == 16
    assert report["timing"]["warmup"] == 1 and report["timing"]["repeats"] == 2
    assert report["preset"]["target"]["local_files_only"] is True and len(built) == 1
    assert report["environment"]["platform"]
    console = capsys.readouterr().out
    assert "DART vs " in console and "draft acceptance" in console
    assert report["correctness"]["outputs"][0]["output_hash"] in console


@pytest.mark.parametrize("option,value", [
    ("--warmup", "0"), ("--warmup", "11"), ("--repeats", "1"), ("--repeats", "101"),
    ("--per-workload", "0"), ("--per-workload", "33"), ("--seed", "-1"),
])
def test_benchmark_work_limits_fail_before_model_loading(local_model, tmp_path, option, value):
    directory, built = local_model
    output = tmp_path / "bench.json"
    with pytest.raises(SystemExit) as error:
        cli.main(["bench", "--model", str(directory), option, value, "--out", str(output)])
    assert error.value.code == 2 and not built and not output.exists()


def test_benchmark_failure_does_not_write_a_success_report(local_model, tmp_path, monkeypatch):
    directory, _ = local_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())

    def mismatch(*args, **kwargs):
        raise ValueError("complete token mismatch")

    monkeypatch.setattr("benchmarks.corpus_bench.measure_methods", mismatch)
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as error:
        cli.main(["bench", "--model", str(directory), "--out", str(output)])
    assert error.value.code == 2 and not output.exists()
