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
    assert len(report["timing"]["samples"]) == 20 and len(report["profile"]["samples"]) == 20
    assert report["hf_prompt_lookup"] == {"prompt_lookup_num_tokens": 4, "max_matching_ngram_size": 4}
    assert report["timing"]["warmup"] == 1 and report["timing"]["repeats"] == 2
    assert report["preset"]["target"]["local_files_only"] is True and len(built) == 1
    assert report["environment"]["platform"]
    saved = json.loads(output.with_suffix(".timing.json").read_text())
    assert saved["status"] == "timing_complete" and "profile" not in saved
    for key in ("correctness", "timing", "preset", "target", "corpus", "environment", "selection", "hf_prompt_lookup"):
        assert saved[key] == report[key]
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
    assert not output.with_suffix(".timing.json").exists()


def test_completed_timings_survive_interrupted_profiling(local_model, tmp_path, monkeypatch):
    directory, _ = local_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    output = tmp_path / "nested" / "bench.json"

    def interrupt(*args, **kwargs):
        saved = json.loads(output.with_suffix(".timing.json").read_text())
        assert saved["status"] == "timing_complete" and "profile" not in saved
        assert len(saved["timing"]["samples"]) == 20
        assert saved["correctness"]["summary"]["exact_match_rate"] == 1
        for sample in saved["timing"]["samples"]:
            assert sample["output_hash"] == saved["correctness"]["outputs"][sample["prompt_index"]]["output_hash"]
        raise KeyboardInterrupt

    monkeypatch.setattr("benchmarks.corpus_bench.profile_methods", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["bench", "--model", str(directory), "--warmup", "1", "--repeats", "2", "--out", str(output)])
    assert not output.exists()
    assert output.with_suffix(".timing.json").exists()


def test_existing_timing_file_is_preserved_before_loading(local_model, tmp_path):
    directory, built = local_model
    output = tmp_path / "bench.json"
    saved = output.with_suffix(".timing.json")
    saved.write_text("keep previous timings", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["bench", "--model", str(directory), "--out", str(output)])
    assert error.value.code == 2 and not built and not output.exists()
    assert saved.read_text() == "keep previous timings"


def test_command_can_skip_profiles_without_skipping_timings(local_model, tmp_path, monkeypatch, capsys):
    directory, _ = local_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    monkeypatch.setattr("benchmarks.corpus_bench.profile_methods", lambda *a, **k: pytest.fail("unexpected profile"))
    output = tmp_path / "timings.json"
    assert cli.main(["bench", "--model", str(directory), "--warmup", "1", "--repeats", "2",
                     "--skip-profile", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["profile"] is None and report["profiling_requested"] is False
    assert len(report["timing"]["samples"]) == 20
    assert "profiling was skipped" in capsys.readouterr().err
