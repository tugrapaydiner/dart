from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from dart import cli
from dart.models.target import TargetModel
from dart.verification import synthetic
from dart.verification.reduction import case_size, failure_key
from dart.verification.state_sequences import StateOperation
from dart.verification.synthetic_replay import (
    MAX_REPORT_BYTES,
    load_state_replay,
    reduce_state_check,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(synthetic.LlamaForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("loaded a checkpoint"))
    yield
    torch.set_num_threads(previous)


def record(operations=None):
    if operations is None:
        operations = (StateOperation("a", "prefill", (5, 6)), StateOperation("a", "forward", (7,)))
    return {
        "schema": "dart-synthetic-state-check-v1", "seed": 29,
        "model": {"architecture": "LlamaForCausalLM", "config": synthetic.synthetic_model_config(),
                  "device": "cpu", "dtype": "float32", "threads": 1},
        "steps": [{"operation": asdict(operation)} for operation in operations],
    }


def write_report(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_real_rollback_failure_is_reduced_replayed_and_removed_by_fix(tmp_path, monkeypatch, capsys):
    path = tmp_path / "reduced.json"
    with monkeypatch.context() as fault:
        fault.setattr(TargetModel, "crop", lambda *a: None)
        assert cli.main(["check", "--synthetic", "--state", "--reduce", "24", "--out", str(path)]) == 1
        data = json.loads(path.read_text(encoding="utf-8"))
        reduction = data["reduction"]
        assert reduction["reproduced"] and reduction["failure"] == reduction["final_failure"]
        assert 2 <= reduction["evaluations"] <= 24 and reduction["accepted"] > 0
        replay = load_state_replay(path)
        assert replay.seed == 17 and replay.selection == "reduced"
        assert case_size(replay.operations) < case_size(synthetic.synthetic_state_operations())
        checked = synthetic.run_synthetic_state_check(replay.seed, operations=replay.operations)
        assert asdict(failure_key(checked)) == reduction["failure"]
        replay_path = tmp_path / "replayed.json"
        assert cli.main([*data["reproduce"][1:], "--out", str(replay_path)]) == 1
        repeated = json.loads(replay_path.read_text(encoding="utf-8"))
        assert len(repeated["steps"]) == len(replay.operations)
        assert repeated["replay"] == {"source_sha256": replay.source_sha256, "selection": "reduced"}
    assert cli.main(data["reproduce"][1:]) == 0
    assert "same failure reproduced" in capsys.readouterr().out


def test_evaluation_cap_counts_original_and_final_with_fresh_targets(monkeypatch):
    models, seen = synthetic._synthetic_models, []

    def fresh(seed):
        pair = models(seed)
        seen.append(pair[0])
        return pair

    monkeypatch.setattr(synthetic, "_synthetic_models", fresh)
    monkeypatch.setattr(TargetModel, "crop", lambda *a: None)
    report = synthetic.run_synthetic_state_check()
    result = reduce_state_check(report, 17, max_evaluations=2)
    assert result.evaluations == 2 and result.candidates == 0 and result.reproduced
    assert result.reduced == result.original
    assert len(seen) == 2 and seen[0] is not seen[1]


def test_final_replay_detects_disappearing_failure_and_loader_uses_original(tmp_path, monkeypatch):
    with monkeypatch.context() as fault:
        fault.setattr(TargetModel, "crop", lambda *a: None)
        report = synthetic.run_synthetic_state_check()
    result = reduce_state_check(report, 17, max_evaluations=2)
    assert not result.reproduced and result.final_failure is None
    data = record(result.original)
    data["reduction"] = {**asdict(result), "reproduced": result.reproduced}
    loaded = load_state_replay(write_report(tmp_path / "unstable.json", data))
    assert loaded.selection == "original" and loaded.operations == result.original


def test_passing_check_does_not_start_reduction_or_write_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["check", "--synthetic", "--state", "--reduce"]) == 0
    assert "No failing case to reduce." in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_replay_uses_saved_operations_and_seed_instead_of_regenerating(tmp_path, monkeypatch):
    path = write_report(tmp_path / "case.json", record())
    monkeypatch.setattr(synthetic, "synthetic_state_operations", lambda *a: pytest.fail("regenerated the original case"))
    rng, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
    output = tmp_path / "result.json"
    assert cli.main(["check", "--synthetic", "--state", "--replay", str(path), "--out", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["seed"] == 29 and data["summary"]["steps"] == 2
    assert data["steps"][0]["operation"]["tokens"] == [5, 6]
    assert torch.equal(rng, torch.random.get_rng_state()) and torch.get_num_threads() == threads


@pytest.mark.parametrize("change", [
    {"schema": "dart-synthetic-check-v1"}, {"seed": True}, {"seed": 2**63}, {"model": {}},
    {"steps": []}, {"steps": [None]}, {"steps": [{"operation": {"request": "a", "action": "unknown"}}]},
    {"steps": [{"operation": {"request": "a", "action": "prefill", "tokens": [67]}}]},
    {"steps": [{"operation": {"request": "a", "action": "prefill", "tokens": [5] * 66}}]},
    {"steps": [{"operation": {"request": "a", "action": "prefill", "tokens": [True]}}]},
    {"steps": [{"operation": {"request": "a", "action": "release"}}]},
    {"reduction": {"reproduced": "true"}}, {"reduction": {"reproduced": True, "reduced": []}},
])
def test_invalid_replay_is_rejected_before_model_creation(tmp_path, monkeypatch, change):
    path = write_report(tmp_path / "invalid.json", {**record(), **change})
    monkeypatch.setattr(synthetic, "_synthetic_models", lambda *a: pytest.fail("built a model"))
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--synthetic", "--state", "--replay", str(path)])
    assert error.value.code == 2


@pytest.mark.parametrize("raw", [b"not json", b"[" * 2000, b" " * (MAX_REPORT_BYTES + 1)],
                         ids=["invalid-json", "deep-json", "oversized"])
def test_unreadable_and_oversized_reports_are_rejected(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_state_replay(path)


@pytest.mark.parametrize("args", [
    ["--synthetic", "--reduce"], ["--model", "unused", "--reduce"],
    ["--synthetic", "--replay", "unused"], ["--model", "unused", "--replay", "unused"],
    ["--synthetic", "--replay", ""],
    ["--synthetic", "--state", "--reduce", "1"], ["--synthetic", "--state", "--reduce", "33"],
    ["--synthetic", "--state", "--replay", "unused", "--seed", "17"],
])
def test_invalid_cli_combinations_fail_before_execution(monkeypatch, args):
    monkeypatch.setattr(synthetic, "_synthetic_models", lambda *a: pytest.fail("built a model"))
    monkeypatch.setattr(cli, "_run", lambda *a: pytest.fail("loaded a local model"))
    with pytest.raises(SystemExit) as error:
        cli.main(["check", *args])
    assert error.value.code == 2


def test_replay_report_cannot_be_overwritten(tmp_path):
    path = write_report(tmp_path / "case.json", record())
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--synthetic", "--state", "--replay", str(path), "--out", str(path)])
    assert error.value.code == 2 and path.read_bytes() == before


def test_reference_error_during_reduction_propagates_and_restores_threads(tmp_path, monkeypatch):
    check, calls = synthetic.run_synthetic_state_check, []
    monkeypatch.setattr(TargetModel, "crop", lambda *a: None)

    def fail_later(seed, **kwargs):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("reference failed")
        return check(seed, **kwargs)

    monkeypatch.setattr(synthetic, "run_synthetic_state_check", fail_later)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="reference failed"):
        cli.main(["check", "--synthetic", "--state", "--reduce", "2", "--out", str(tmp_path / "failed.json")])
    assert torch.get_num_threads() == threads and list(tmp_path.iterdir()) == []
