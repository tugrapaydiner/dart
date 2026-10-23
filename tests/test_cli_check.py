from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dart import cli
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import GenerationOutput
from tests.conftest import build_tiny_target
from tests.test_corpus_eval import Tokenizer as DecodeTokenizer
from tests.test_corpus_eval import records
from tests.test_corpus_storage import Tokenizer as StoredTokenizer
from tests.test_sampling_devices import DEVICES


class Tokenizer(DecodeTokenizer, StoredTokenizer):
    pass


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    built = []

    def build(config):
        model = build_tiny_target().model
        model.set_attn_implementation(config["target"]["attn_implementation"])
        target = TargetModel(model, tokenizer=Tokenizer(), device=config["target"]["device"])
        engine = DartEngine(target, NGramDrafter())
        built.append((config, engine))
        return engine

    monkeypatch.setattr("dart.runtime.factory.build_engine", build)
    return directory, built


@pytest.mark.parametrize("device", DEVICES)
def test_check_command_defaults_to_validation_and_records_complete_matches(local_model, tmp_path, monkeypatch, device):
    directory, built = local_model
    seen = []
    rows = records()
    held_out = replace(rows[0], id="held-out", source="held-out", family="held-out", split="test", text="zzz")

    def pilot(seed):
        seen.append(seed)
        return [held_out, *reversed(rows)]

    monkeypatch.setattr("dart.data.pilot.build_pilot", pilot)
    output = tmp_path / "report.json"
    device = "cuda" if device.startswith("cuda") else device
    assert cli.main(["check", "--model", str(directory), "--device", device, "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["split"] == "validation" and report["summary"]["prompts"] == 2
    assert report["summary"]["exact_match_rate"] == 1
    assert set(report["by_budget"]) == {"3", "7"}
    assert [row["id"] for row in report["outputs"]] == ["a", "b"]
    assert report["target"]["device"] == device
    assert report["target"]["attention"] == ("sdpa" if device == "cpu" else "eager")
    assert len(report["target"]["weights_sha256"]) == 64
    assert report["preset"]["target"]["local_files_only"] is True
    assert report["environment"]["dart"]
    assert seen == [17] and len(built) == 1


def test_check_command_reads_an_explicit_versioned_split(local_model, tmp_path, monkeypatch):
    directory, _ = local_model
    seen = []
    selected = [replace(row, split="test") for row in records()]

    def read(path, split, tokenizer):
        seen.append((path, split))
        return selected, {"version": "fixture", "corpus_hash": "a" * 64}

    monkeypatch.setattr("dart.data.storage.read_split", read)
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: pytest.fail("built-in data replaced the supplied corpus"))
    output = tmp_path / "test-report.json"
    assert cli.main(["check", "--model", str(directory), "--corpus", "fixture-corpus", "--split", "test", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert seen == [("fixture-corpus", "test")]
    assert report["split"] == "test" and report["corpus"]["version"] == "fixture"


def test_check_command_rejects_mismatches_without_writing_a_success_report(local_model, tmp_path, monkeypatch, capsys):
    directory, _ = local_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    monkeypatch.setattr(DartEngine, "generate", lambda *args, **kwargs: GenerationOutput([], None, {}))
    output = tmp_path / "bad.json"
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--model", str(directory), "--out", str(output)])
    assert error.value.code == 2
    assert "complete token mismatch" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("case", ["missing_model", "existing_output"])
def test_invalid_paths_fail_before_loading_and_preserve_existing_files(local_model, tmp_path, case):
    directory, built = local_model
    output = tmp_path / "report.json"
    if case == "missing_model":
        directory = tmp_path / "missing"
    else:
        output.write_text("keep this", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["check", "--model", str(directory), "--out", str(output)])
    assert error.value.code == 2 and not built
    if case == "existing_output":
        assert output.read_text() == "keep this"


def test_module_help_works_without_third_party_packages():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", "-m", "dart", "check", "--help"],
                            cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout and "--split" in result.stdout
