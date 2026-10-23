from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dart import cli
from dart.data.pilot import VERSION, build_pilot
from dart.data.storage import read_split
from dart.data.teacher import TraceDataset
from dart.models.target import TargetModel
from tests.test_corpus_storage import Tokenizer
from tests.test_sampling_devices import DEVICES
from tests.test_teacher import setup_corpus


@pytest.fixture
def model_path(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def test_corpus_command_builds_versioned_splits_without_loading_weights(model_path, tmp_path, monkeypatch, capsys):
    calls = []

    def loader(kind, result):
        def load(path, **kwargs):
            calls.append((kind, path, kwargs))
            return result
        return load

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", loader("tokenizer", Tokenizer()))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", loader("config", SimpleNamespace(max_position_embeddings=1024)))
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **kw: pytest.fail("loaded model weights"))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["corpus", "--model", str(model_path), "--train-per-family", "1",
                     "--validation-per-family", "1", "--test-per-family", "1"]) == 0
    output = tmp_path / "runs/corpus/pilot-v1"
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == VERSION and Path(report["out"]) == Path("runs/corpus/pilot-v1")
    expected = build_pilot(17, 1, 1, 1)
    for split, count in (("train", 12), ("validation", 8), ("test", 8)):
        rows, manifest = read_split(output, split, Tokenizer())
        assert len(rows) == count and report["splits"][split]["records"] == count
        assert rows == sorted((row for row in expected if row.split == split), key=lambda row: row.id)
        assert report["corpus_hash"] == manifest["corpus_hash"]
        assert report["tokenizer_hash"] == manifest["tokenizer_hash"]
    assert calls == [(kind, str(model_path.resolve()), {"local_files_only": True, "trust_remote_code": False})
                     for kind in ("tokenizer", "config")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("split", ["train", "validation"])
def test_collect_command_writes_readable_bounded_traces(model_path, tmp_path, monkeypatch, capsys, device, split):
    target, corpus, output = setup_corpus(tmp_path)
    target = TargetModel(target.model, tokenizer=target.tokenizer, device=device, record_hidden=True)
    target.eos_token_ids = frozenset()
    (corpus / "test.jsonl").unlink()  # Collection must not read held-out prompts.
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return target

    monkeypatch.setattr(TargetModel, "from_pretrained", load)
    selected = "cuda" if device.startswith("cuda") else device
    assert cli.main(["collect", "--model", str(model_path), "--corpus", str(corpus), "--split", split,
                     "--out", str(output), "--device", selected, "--max-storage-mib", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    traces = TraceDataset(output, max_bytes=1024 ** 2)
    assert len(traces) == (2 if split == "train" else 1) == report["records"]
    assert traces.manifest["split"] == split and report["shard_bytes"] == traces.manifest["shard_bytes"]
    assert sum(path.stat().st_size for path in output.iterdir()) <= 1024 ** 2
    assert report["target"]["attention"] == "eager" and report["target"]["dtype"] == "torch.float32"
    for trace in traces:
        assert trace["continuation_ids"].tolist() == target.greedy_generate(trace["prompt_ids"].tolist(), 4)
        assert trace["hidden"].shape == (4, target.model.config.hidden_size)
        assert trace["hidden"].dtype == torch.float16 and torch.isfinite(trace["hidden"]).all()
    assert calls == [(str(model_path.resolve()), {"device": selected, "dtype": torch.float32,
                     "attn_implementation": "eager", "record_hidden": True,
                     "local_files_only": True, "trust_remote_code": False})]
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize("command,options", [
    ("corpus", ["--train-per-family", "0"]),
    ("corpus", ["--validation-per-family", "10000"]),
    ("corpus", ["--corpus-version", " "]),
    ("collect", ["--max-storage-mib", "0"]),
    ("collect", ["--split", "test"]),
    ("collect", []),
])
def test_invalid_data_inputs_fail_before_loading(command, options, model_path, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("loaded before validating inputs")

    monkeypatch.setattr(TargetModel, "from_pretrained", fail)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fail)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", fail)
    output = tmp_path / "output"
    extra = ["--corpus", str(tmp_path / "missing")] if command == "collect" else []
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--model", str(model_path), "--out", str(output), *extra, *options])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("command", ["corpus", "collect"])
@pytest.mark.parametrize("case", ["missing_model", "output_file", "nonempty_output"])
def test_data_paths_fail_before_loading_and_keep_existing_files(command, case, model_path, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("loaded before validating paths")

    monkeypatch.setattr(TargetModel, "from_pretrained", fail)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fail)
    output = tmp_path / "output"
    kept = None
    if case == "missing_model":
        model_path = tmp_path / "missing"
    elif case == "output_file":
        kept = output
    else:
        output.mkdir()
        kept = output / "existing.txt"
    if kept is not None:
        kept.write_text("keep this", encoding="utf-8")
    extra = ["--corpus", str(tmp_path / "missing")] if command == "collect" else []
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--model", str(model_path), "--out", str(output), *extra])
    assert error.value.code == 2
    if kept is not None:
        assert kept.read_text() == "keep this"
