from __future__ import annotations

import json
import runpy
import sys

import pytest
import torch

from dart import cli
from dart.runtime import engine as engine_module
from dart.verification import synthetic
from dart.verification.differential import (
    CheckFailure,
    CheckReport,
    CheckResult,
    GenerationCase,
)


@pytest.fixture
def forbid_model_loading(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("loaded a checkpoint or contacted the model hub")

    monkeypatch.setattr("dart.runtime.factory.build_engine", fail)
    monkeypatch.setattr(synthetic.LlamaForCausalLM, "from_pretrained", fail)


def test_module_cli_runs_actual_paths_without_writing_and_restores_cpu_state(tmp_path, monkeypatch, capsys, forbid_model_loading):
    monkeypatch.chdir(tmp_path)
    threads, rng = torch.get_num_threads(), torch.random.get_rng_state().clone()
    monkeypatch.setattr(sys, "argv", ["dart", "check", "--synthetic"])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("dart", run_name="__main__")
    assert error.value.code == 0
    assert capsys.readouterr().out == "Synthetic CPU check: 18/18 comparisons passed (seed 17).\n"
    assert torch.get_num_threads() == threads and torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []


def test_opt_in_report_contains_reproducible_cases_and_complete_outputs(tmp_path, forbid_model_loading):
    path = tmp_path / "nested" / "synthetic.json"
    assert cli.main(["check", "--synthetic", "--seed", "29", "--out", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == "dart-synthetic-check-v1" and report["status"] == "passed"
    assert report["summary"] == {"cases": 6, "comparisons": 18, "passed": 18, "failed": 0}
    assert report["model"]["config"] == synthetic.synthetic_model_config()
    assert report["model"]["device"] == "cpu" and report["model"]["threads"] == 1
    assert all(report["versions"].values())
    assert report["reproduce"] == ["dart", "check", "--synthetic", "--seed", "29"]
    assert all(row["actual"] == row["expected"] and row["failure"] is None for row in report["results"])
    cases = {row["case"]["name"]: GenerationCase(**row["case"]) for row in report["results"]}
    assert tuple(cases.values()) == synthetic.synthetic_cases(29)
    replay = tmp_path / "repeated.json"
    assert cli.main([*report["reproduce"][1:], "--out", str(replay)]) == 0
    assert json.loads(replay.read_text(encoding="utf-8")) == report


def test_real_verifier_fault_returns_failure_and_saves_failed_report(tmp_path, monkeypatch, capsys, forbid_model_loading):
    original = engine_module._VERIFIERS["block"]

    def wrong_token(*args, **kwargs):
        result = original(*args, **kwargs)
        result.committed[-1] = (result.committed[-1] + 1) % synthetic.VOCAB_SIZE
        return result

    monkeypatch.setitem(engine_module._VERIFIERS, "block", wrong_token)
    output = tmp_path / "failed.json"
    assert cli.main(["check", "--synthetic", "--out", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["summary"]["failed"] > 0
    failures = [row for row in report["results"] if row["failure"]]
    assert {row["backend"] for row in failures} == {"greedy", "block"}
    assert all(row["actual"] != row["expected"] for row in failures)
    assert "Failed:" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [
    [], ["--synthetic", "--model", "unused"], ["--synthetic", "--device", "cuda"],
    ["--synthetic", "--corpus", "unused"], ["--synthetic", "--split", "test"],
    ["--synthetic", "--native-rms-norm"], ["--synthetic", "--norm-control"],
    ["--synthetic", "--seed", "-1"], ["--synthetic", "--seed", str(2**63)],
])
def test_invalid_options_fail_before_any_model_is_built(arguments, monkeypatch):
    monkeypatch.setattr(synthetic, "run_synthetic_check", lambda *args: pytest.fail("ran invalid check"))
    with pytest.raises(SystemExit) as error:
        cli.main(["check", *arguments])
    assert error.value.code == 2


def test_existing_report_is_not_overwritten_or_checked(tmp_path, monkeypatch):
    output = tmp_path / "keep.json"
    output.write_text("keep this", encoding="utf-8")
    monkeypatch.setattr(synthetic, "run_synthetic_check", lambda *args: pytest.fail("ran before output validation"))
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--synthetic", "--out", str(output)])
    assert error.value.code == 2 and output.read_text(encoding="utf-8") == "keep this"


def test_reference_error_restores_threads_and_writes_nothing(tmp_path, monkeypatch):
    def fail(seed):
        assert torch.get_num_threads() == 1
        raise RuntimeError("reference failed")

    monkeypatch.setattr(synthetic, "run_synthetic_check", fail)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="reference failed"):
        cli.main(["check", "--synthetic", "--out", str(tmp_path / "failed.json")])
    assert torch.get_num_threads() == threads and list(tmp_path.iterdir()) == []


def test_candidate_exception_is_reported_without_a_traceback(tmp_path, monkeypatch, capsys):
    case = GenerationCase("case", (5,), 1)
    result = CheckResult(case, "block", (7,), None, CheckFailure("exception", detail="RuntimeError: synthetic"))
    monkeypatch.setattr(synthetic, "run_synthetic_check", lambda seed: CheckReport((result,)))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["check", "--synthetic"]) == 1
    assert "block / case: exception" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_output_created_during_check_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "raced.json"
    case = GenerationCase("case", (5,), 0)

    def race(seed):
        output.write_text("other writer", encoding="utf-8")
        return CheckReport((CheckResult(case, "block", (), (), None),))

    monkeypatch.setattr(synthetic, "run_synthetic_check", race)
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--synthetic", "--out", str(output)])
    assert error.value.code == 2 and output.read_text(encoding="utf-8") == "other writer"


def test_config_snapshots_do_not_mutate_later_checks():
    config = synthetic.synthetic_model_config()
    config["vocab_size"] = 999
    assert synthetic.synthetic_model_config()["vocab_size"] == synthetic.VOCAB_SIZE
