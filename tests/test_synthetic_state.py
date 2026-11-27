from __future__ import annotations

import json
import runpy
import sys
from functools import wraps

import pytest
import torch

from dart import cli
from dart.models.target import TargetModel
from dart.verification import synthetic
from dart.verification.state_sequences import StateOperation


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def fail(*args, **kwargs):
        pytest.fail("loaded a checkpoint or used the shared greedy reference")

    monkeypatch.setattr(synthetic.LlamaForCausalLM, "from_pretrained", fail)
    monkeypatch.setattr(TargetModel, "greedy_generate", fail)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("seed", [17, 29, 101])
def test_seeded_operations_check_all_live_states_and_recover_from_errors(seed):
    report = synthetic.run_synthetic_state_check(seed)
    operations = synthetic.synthetic_state_operations(seed)
    assert report.passed, report
    assert tuple(step.operation for step in report.steps) == operations
    assert len(report.steps) == 26
    assert report.steps[2].checked_requests == ("a", "b")
    assert report.steps[9].checked_requests == ("b",)
    assert report.steps[-1].checked_requests == ()
    assert sum(operation.expect_error is not None for operation in operations) == 5
    assert {operation.action for operation in operations} == {"prefill", "forward", "block", "tree", "crop", "release"}


def test_operation_seed_is_repeatable_and_changes_tokens_without_touching_rng():
    rng = torch.random.get_rng_state().clone()
    first = synthetic.synthetic_state_operations(17)
    assert first == synthetic.synthetic_state_operations(17)
    assert first != synthetic.synthetic_state_operations(29)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert max(len(operation.tokens) for operation in first) == synthetic.CONTEXT_LENGTH + 1


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**63])
def test_invalid_seed_fails_before_building_a_model(seed, monkeypatch):
    monkeypatch.setattr(synthetic, "_synthetic_models", lambda *a: pytest.fail("built a model"))
    with pytest.raises(ValueError, match="seed"):
        synthetic.run_synthetic_state_check(seed)


def test_missing_rollback_is_detected_and_restoring_crop_removes_failure(monkeypatch):
    with monkeypatch.context() as fault:
        fault.setattr(TargetModel, "crop", lambda *a: None)
        report = synthetic.run_synthetic_state_check()
    assert not report.passed
    assert any(item.failure.kind == "cache_length" for step in report.steps for item in step.failures)
    assert synthetic.run_synthetic_state_check().passed


def test_idle_request_corruption_reports_victim_and_actor(monkeypatch):
    prefill, forward, saved = TargetModel.prefill, TargetModel.forward_block, []

    def remember(self, tokens, *args, **kwargs):
        state, logits = prefill(self, tokens, *args, **kwargs)
        saved.append(state)
        return state, logits

    def corrupt(self, state, tokens):
        logits = forward(self, state, tokens)
        if len(saved) == 2 and state is saved[0]:
            saved[1].cache.layers[0].keys.add_(0.1)
        return logits

    monkeypatch.setattr(TargetModel, "prefill", remember)
    monkeypatch.setattr(TargetModel, "forward_block", corrupt)
    report = synthetic.run_synthetic_state_check()
    step = report.steps[2]
    assert step.operation.request == "a"
    assert {(item.request, item.failure.kind) for item in step.failures} == {("b", "cache_values")}


def test_tree_mask_fault_is_detected_and_restoring_it_removes_failure(monkeypatch):
    forward_tree, injected = TargetModel.forward_tree, []

    def unmasked(self, *args, **kwargs):
        forward = self.model.forward

        @wraps(forward)
        def wrong_mask(*args, **kwargs):
            mask = kwargs.get("attention_mask")
            if mask is not None and mask.ndim == 4:
                kwargs["attention_mask"] = torch.zeros_like(mask)
                injected.append(True)
            return forward(*args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(self.model, "forward", wrong_mask)
            return forward_tree(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(TargetModel, "forward_tree", unmasked)
        report = synthetic.run_synthetic_state_check()
    assert injected and not report.passed
    assert any(item.failure.kind in {"committed", "cache_values"}
               for step in report.steps for item in step.failures)
    assert synthetic.run_synthetic_state_check().passed


def test_cli_module_runs_without_files_and_restores_rng_and_threads(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    threads, rng = torch.get_num_threads(), torch.random.get_rng_state().clone()
    monkeypatch.setattr(sys, "argv", ["dart", "check", "--synthetic", "--state"])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("dart", run_name="__main__")
    assert error.value.code == 0
    assert capsys.readouterr().out == "Synthetic CPU state check: 26/26 steps passed (seed 17).\n"
    assert torch.get_num_threads() == threads and torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []


def test_saved_report_preserves_operations_and_replays_exactly(tmp_path):
    path = tmp_path / "state.json"
    assert cli.main(["check", "--synthetic", "--state", "--seed", "29", "--out", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == "dart-synthetic-state-check-v1" and report["status"] == "passed"
    assert report["summary"]["steps"] == report["summary"]["passed"] == 26
    assert report["summary"]["failed"] == 0 and report["summary"]["cache_checks"] > 26
    assert tuple(StateOperation(**step["operation"]) for step in report["steps"]) == synthetic.synthetic_state_operations(29)
    assert report["model"]["config"] == synthetic.synthetic_model_config()
    assert all(report["versions"].values())
    replay = tmp_path / "replay.json"
    assert cli.main([*report["reproduce"][1:], "--out", str(replay)]) == 0
    assert json.loads(replay.read_text(encoding="utf-8")) == report


def test_fault_returns_failure_and_writes_diagnostics(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(TargetModel, "crop", lambda *a: None)
    path = tmp_path / "failed.json"
    assert cli.main(["check", "--synthetic", "--state", "--out", str(path)]) == 1
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["summary"]["failed"] > 0
    assert any(item["failure"]["kind"] == "cache_length" for step in report["steps"] for item in step["failures"])
    assert "Failed: step" in capsys.readouterr().out


def test_state_option_rejects_local_model_before_loading(monkeypatch):
    monkeypatch.setattr(cli, "_run", lambda *a: pytest.fail("loaded a local model"))
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--model", "unused", "--state"])
    assert error.value.code == 2


def test_existing_report_is_preserved_before_execution(tmp_path, monkeypatch):
    path = tmp_path / "keep.json"
    path.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(synthetic, "run_synthetic_state_check", lambda *a: pytest.fail("ran before output validation"))
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--synthetic", "--state", "--out", str(path)])
    assert error.value.code == 2 and path.read_text(encoding="utf-8") == "keep"


def test_reference_failure_restores_threads_and_writes_nothing(tmp_path, monkeypatch):
    def fail(seed):
        assert torch.get_num_threads() == 1
        raise RuntimeError("reference failed")

    monkeypatch.setattr(synthetic, "run_synthetic_state_check", fail)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="reference failed"):
        cli.main(["check", "--synthetic", "--state", "--out", str(tmp_path / "failed.json")])
    assert torch.get_num_threads() == threads and list(tmp_path.iterdir()) == []
