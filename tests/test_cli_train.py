from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dart import cli
from dart.cli_train import add_arguments
from dart.diffusion.compact_training import read_checkpoint
from dart.models.target import TargetModel
from tests.test_compact_training import training_inputs
from tests.test_sampling_devices import DEVICES


@pytest.fixture
def model_path(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


@pytest.mark.parametrize("device", DEVICES)
def test_training_command_resumes_exactly_with_selected_defaults(device, model_path, tmp_path, monkeypatch):
    target, traces = training_inputs(tmp_path, device)
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return target

    monkeypatch.setattr(TargetModel, "from_pretrained", load)
    selected = "cuda" if device.startswith("cuda") else device
    common = ["train", "--model", str(model_path), "--traces", str(traces.directory),
              "--device", selected, "--batch-size", "1", "--save-every", "1"]
    full, part, resumed = (tmp_path / name for name in ("full", "part", "resumed"))
    for output, steps, extra in ((full, 4, []), (part, 2, []), (resumed, 4, ["--resume", str(part / "latest.pt")])):
        assert cli.main([*common, "--out", str(output), "--steps", str(steps), *extra]) == 0
    expected, actual = (read_checkpoint(path / "latest.pt") for path in (full, resumed))
    assert actual["step"] == expected["step"] == 4
    assert actual["last_loss"] == expected["last_loss"]
    assert actual["metadata"] == expected["metadata"]
    for name, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=0)
    logs = {path.name: [json.loads(line) for line in (path / "losses.jsonl").read_text().splitlines()]
            for path in (full, part, resumed)}
    assert [(row["step"], row["loss"]) for row in logs["full"]] == [
        (row["step"], row["loss"]) for row in logs["part"] + logs["resumed"]]
    assert read_checkpoint(full / "initial.pt")["step"] == 0
    assert not (resumed / "initial.pt").exists()
    report = json.loads((resumed / "report.json").read_text())
    assert report["steps"] == 4 and report["learned_parameters"] <= 5_000_000
    assert report["model_config"]["embedding_init_std"] == 0.02
    assert report["training_config"]["distill_weight"] == 0.5
    assert report["training_config"]["distill_temperature"] == 2.0
    assert report["memory_scope"].endswith("excludes initialization")
    assert calls == [(str(model_path.resolve()), {"device": selected, "dtype": torch.float32,
                     "attn_implementation": "eager", "local_files_only": True, "trust_remote_code": False})] * 3


@pytest.mark.parametrize("options", [
    ["--steps", "0"], ["--save-every", "0"], ["--threads", "0"], ["--batch-size", "0"],
    ["--canvas-len", "9"], ["--learning-rate", "nan"], ["--full-mask-fraction", "2"],
    ["--distill-weight", "2"], ["--embedding-init-std", "0"], ["--resume", "missing.pt"],
    ["--failure-masking", "observed"], [],
])
def test_training_command_rejects_invalid_inputs_before_loading(options, model_path, tmp_path, monkeypatch):
    _, traces = training_inputs(tmp_path)
    trace_path = traces.directory if options else tmp_path / "missing-traces"
    monkeypatch.setattr(TargetModel, "from_pretrained", lambda *a, **kw: pytest.fail("loaded before validating inputs"))
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        cli.main(["train", "--model", str(model_path), "--traces", str(trace_path),
                  "--out", str(output), "--steps", "1", *options])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("case", ["changed_options", "completed_steps", "existing_output"])
def test_training_resume_errors_preserve_the_prior_run(case, model_path, tmp_path, monkeypatch):
    target, traces = training_inputs(tmp_path)
    monkeypatch.setattr(TargetModel, "from_pretrained", lambda *a, **kw: target)
    original = tmp_path / "original"
    common = ["train", "--model", str(model_path), "--traces", str(traces.directory), "--batch-size", "1"]
    assert cli.main([*common, "--out", str(original), "--steps", "1"]) == 0
    checkpoint = original / "latest.pt"
    before = {path.name: path.read_bytes() for path in original.iterdir()}
    output = original if case == "existing_output" else tmp_path / "retry"
    extra = ["--learning-rate", "0.01"] if case == "changed_options" else []
    steps = "1" if case == "completed_steps" else "2"
    with pytest.raises(SystemExit) as error:
        cli.main([*common, "--out", str(output), "--steps", steps, "--resume", str(checkpoint), *extra])
    assert error.value.code == 2
    assert {path.name: path.read_bytes() for path in original.iterdir()} == before
    if output != original:
        assert not output.exists()


def test_standalone_script_keeps_original_defaults_and_dependency_free_help():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--model", "local", "--traces", "traces", "--out", "run", "--steps", "1"])
    assert args.distill_weight == 0.0 and args.distill_temperature == 1.0 and args.embedding_init_std == 1.0
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", str(root / "scripts/train_compact.py"), "--help"],
                            cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout
