from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save

from dart.data.storage import write_corpus
from dart.data.teacher import (
    MAX_SHARD_BYTES,
    TraceDataset,
    collect_labels,
    target_fingerprint,
    write_traces,
)
from dart.models.target import TargetModel
from tests.conftest import build_tiny_target, random_prompt
from tests.test_corpus_storage import Tokenizer, records
from tests.test_sampling_devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [0, 1, 6])
@pytest.mark.parametrize("stop_early", [False, True])
def test_teacher_rows_match_greedy_tokens_and_predictive_features(device, budget, stop_early):
    target = TargetModel(build_tiny_target().model, device=device, record_hidden=True)
    target.eos_token_ids = frozenset()
    prompt = random_prompt(370, 8)
    expected = target.greedy_generate(prompt, budget)
    if stop_early and expected:
        eos = expected[min(2, len(expected) - 1)]
        target.eos_token_ids = frozenset([eos])
        expected = expected[:expected.index(eos) + 1]
    calls = []
    hook = target.model.register_forward_hook(lambda *args: calls.append(1))
    try:
        trace = collect_labels(target, prompt, budget)
    finally:
        hook.remove()
    assert trace["continuation_ids"].tolist() == expected
    assert len(calls) == len(expected)
    assert trace["prompt_ids"].tolist() == prompt
    assert trace["hidden"].shape == (len(expected), 64)
    assert trace["hidden"].dtype == torch.float16
    assert all(value.device.type == "cpu" and not value.requires_grad for value in trace.values())
    for index in range(len(expected)):
        target.prefill(prompt + expected[:index])
        torch.testing.assert_close(trace["hidden"][index].float(), target.last_hidden_block[-1].float().cpu(), rtol=1e-3, atol=1e-3)


def test_teacher_limits_and_missing_features_fail_before_generation(monkeypatch):
    target = build_tiny_target()
    monkeypatch.setattr(target, "prefill", lambda *a: pytest.fail("forward happened before validation"))
    with pytest.raises(ValueError, match="record_hidden"):
        collect_labels(target, [7], 1)
    target.record_hidden = True
    with pytest.raises(ValueError, match="context limit"):
        collect_labels(target, [7] * 1024, 1)
    target.model.config.max_position_embeddings = 100_000
    with pytest.raises(ValueError, match="shard limit"):
        collect_labels(target, [7], 100_000 - 1)


def test_target_fingerprint_covers_weights_and_decode_settings():
    target = build_tiny_target()
    before = target_fingerprint(target)
    assert target_fingerprint(target) == before
    target.model.config._name_or_path = "a different local directory"
    assert target_fingerprint(target) == before
    with torch.no_grad():
        next(target.model.parameters()).view(-1)[0] += 1
    assert target_fingerprint(target)["weights_sha256"] != before["weights_sha256"]
    target.eos_token_ids = frozenset([3])
    assert target_fingerprint(target)["eos_token_ids"] == [3]


def setup_corpus(tmp_path):
    target = build_tiny_target()
    target.record_hidden = True
    target.eos_token_ids = frozenset()
    target.tokenizer = Tokenizer()
    rows = [replace(row, max_new_tokens=4) for row in records()]
    rows.append(replace(rows[0], id="another-train", text="A different training prompt."))
    corpus = tmp_path / "corpus"
    write_corpus(corpus, rows, target.tokenizer, 1024, "test", 17)
    return target, corpus, tmp_path / "traces"


def test_trace_shards_roundtrip_lazily_without_reading_heldout_prompts(tmp_path):
    target, corpus, output = setup_corpus(tmp_path)
    (corpus / "test.jsonl").unlink()
    progress = []
    manifest = write_traces(target, corpus, "train", output, progress=lambda done, total: progress.append((done, total)))
    dataset = TraceDataset(output)
    assert dataset.manifest == manifest and len(dataset) == 2
    assert progress == [(1, 2), (2, 2)]
    for trace in dataset:
        assert trace["continuation_ids"].tolist() == target.greedy_generate([0, 1, 2], 4)
        assert set(trace) == {"prompt_ids", "continuation_ids", "hidden"}
    (output / manifest["shards"][1]["file"]).unlink()
    assert dataset[0]["hidden"].shape == (4, 64)
    with pytest.raises(FileNotFoundError):
        dataset[1]


def test_collection_caps_leave_no_complete_manifest_and_preserve_existing_files(tmp_path):
    target, corpus, output = setup_corpus(tmp_path)
    baseline = write_traces(target, corpus, "train", output)
    cap = sum(path.stat().st_size for path in output.iterdir())
    with pytest.raises(FileExistsError, match="not empty"):
        write_traces(target, corpus, "train", output)
    with pytest.raises(ValueError, match="storage cap"):
        write_traces(target, corpus, "train", tmp_path / "small", max_bytes=cap - 1)
    partial = tmp_path / "small"
    assert not (partial / "manifest.json").exists()
    assert sum(path.stat().st_size for path in partial.iterdir()) <= cap - 1
    with pytest.raises(ValueError, match="storage"):
        TraceDataset(output, max_bytes=cap - 1)
    assert TraceDataset(output, max_bytes=cap).manifest == baseline
    with pytest.raises(ValueError, match="train or validation"):
        write_traces(target, corpus, "test", tmp_path / "heldout")
    assert not (tmp_path / "heldout").exists()


@pytest.mark.parametrize("change", ["path", "bytes", "hash", "shape", "token", "nan", "eos"])
def test_invalid_shards_are_rejected_before_training(tmp_path, change):
    target, corpus, output = setup_corpus(tmp_path)
    manifest = write_traces(target, corpus, "train", output)
    entry = manifest["shards"][0]
    if change == "path":
        entry["file"] = "../outside.safetensors"
    elif change == "bytes":
        entry["bytes"] = MAX_SHARD_BYTES + 1
    elif change == "hash":
        entry["sha256"] = "0" * 64
    else:
        tensors = TraceDataset(output)[0]
        if change == "shape":
            tensors["hidden"] = tensors["hidden"][:, :-1].contiguous()
        elif change == "token":
            tensors["continuation_ids"][0] = -1
        elif change == "nan":
            tensors["hidden"][0, 0] = float("nan")
        else:
            manifest["target"]["eos_token_ids"] = [int(tensors["continuation_ids"][0])]
        data = save(tensors)
        (output / entry["file"]).write_bytes(data)
        manifest["shard_bytes"] += len(data) - entry["bytes"]
        entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        TraceDataset(output)[0]
