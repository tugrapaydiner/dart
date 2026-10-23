from __future__ import annotations

import hashlib
import json

import pytest

from dart.data.failures import MAX_FILE_BYTES, FailureDataset, write_failures
from dart.drafters.compact_diffusion import CompactDiffusionModel
from dart.runtime.types import GenerationOutput
from tests.test_compact_training import MODEL, training_inputs
from tests.test_sampling_devices import DEVICES


def inputs(tmp_path, device="cpu"):
    target, traces = training_inputs(tmp_path, device)
    target.tokenizer.decode = lambda ids, **kwargs: " ".join(map(str, ids))
    return target, CompactDiffusionModel(target, MODEL), traces


@pytest.mark.parametrize("device", DEVICES)
def test_actual_proposals_are_bounded_reproducible_and_aligned_with_teacher_labels(tmp_path, device):
    target, model, traces = inputs(tmp_path, device)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    progress = []
    manifest = write_failures(target, model, traces, a, per_prompt=1, progress=lambda *args: progress.append(args))
    repeated = write_failures(target, model, traces, b, per_prompt=1, indices=list(range(len(traces)))[::-1])
    assert manifest == repeated and a.read_bytes() == b.read_bytes()
    dataset = FailureDataset(a, traces)
    assert dataset.sha256 == hashlib.sha256(a.read_bytes()).hexdigest()
    assert 1 <= len(dataset) <= len(traces)
    assert progress == [(1, 2), (2, 2)]
    assert a.stat().st_size < MAX_FILE_BYTES
    assert manifest["settings"] == {"canvas_len": 4, "denoise_steps": 1, "reconcile": "discard", "adaptive_stop": False}
    for row in dataset.records:
        trace = traces[row["trace_index"]]
        offset = row["offset"]
        labels = trace["continuation_ids"][offset:offset + len(row["proposal_ids"])].tolist()
        assert row["failed_positions"] == [i for i, (token, label) in enumerate(zip(row["proposal_ids"], labels)) if token != label]
        assert offset >= 1 and trace["hidden"][offset - 1].numel() == target.model.config.hidden_size
    assert all(row["retained_windows"] <= 1 for row in manifest["prompts"])


def test_failure_reservoir_observes_actual_verified_rounds(tmp_path, monkeypatch):
    target, model, traces = inputs(tmp_path)
    trace = traces[0]
    prompt, continuation = trace["prompt_ids"].tolist(), trace["continuation_ids"].tolist()
    calls = []

    def wrong(self, committed, budget):
        offset = len(committed) - len(prompt)
        assert committed == prompt + continuation[:offset]
        calls.append((offset, budget))
        return [(token + 1) % target.model.config.vocab_size for token in continuation[offset:offset + budget]]

    monkeypatch.setattr("dart.data.failures.CompactDiffusionDrafter.propose", wrong)
    manifest = write_failures(target, model, traces, tmp_path / "failures.json", indices=[0], per_prompt=1)
    assert calls == [(1, 2), (2, 1)]  # The final token uses the target alone.
    assert manifest["prompts"] == [{"trace_index": 0, "failing_windows": 2, "retained_windows": 1}]
    row = manifest["records"][0]
    assert row["failed_positions"] == list(range(len(row["proposal_ids"])))


@pytest.mark.parametrize("corruption", ["trace_hash", "offset", "tokens", "positions", "duplicate", "capacity", "budget", "lookahead", "reservoir", "summary"])
def test_corrupt_failure_data_is_rejected(tmp_path, corruption):
    target, model, traces = inputs(tmp_path)
    path = tmp_path / "failures.json"
    manifest = write_failures(target, model, traces, path)
    row = manifest["records"][0]
    if corruption == "trace_hash":
        manifest["traces_sha256"] = "0" * 64
    elif corruption == "offset":
        row["offset"] = 0
    elif corruption == "tokens":
        row["proposal_ids"][0] = target.model.config.vocab_size
    elif corruption == "positions":
        row["failed_positions"] = []
    elif corruption == "duplicate":
        manifest["records"].append(dict(row))
    elif corruption == "capacity":
        manifest["settings"]["canvas_len"] = 65
    elif corruption == "reservoir":
        manifest["per_prompt"] = 0
    elif corruption == "summary":
        manifest["prompts"][0]["retained_windows"] += 1
    elif corruption == "lookahead":
        row["proposal_ids"] = traces[row["trace_index"]]["continuation_ids"][row["offset"]:].tolist()
        row["proposal_ids"][0] = (row["proposal_ids"][0] + 1) % target.model.config.vocab_size
        row["failed_positions"] = [0]
    else:
        row["proposal_ids"] += [0] * 4
        manifest["settings"]["canvas_len"] = 8
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        FailureDataset(path, traces)


def test_wrong_outputs_and_storage_overflow_never_publish_a_dataset(tmp_path, monkeypatch):
    target, model, traces = inputs(tmp_path)
    path = tmp_path / "failures.json"
    with pytest.raises(ValueError, match="storage cap"):
        write_failures(target, model, traces, path, max_bytes=1)
    assert not path.exists()
    monkeypatch.setattr("dart.data.failures.DartEngine.generate", lambda *a, **k: GenerationOutput([], "", {}))
    with pytest.raises(ValueError, match="complete output"):
        write_failures(target, model, traces, path)
    assert not path.exists()


def test_existing_outputs_invalid_indices_and_target_changes_fail_before_generation(tmp_path, monkeypatch):
    target, model, traces = inputs(tmp_path)
    path = tmp_path / "failures.json"
    monkeypatch.setattr("dart.data.failures.DartEngine.generate", lambda *a, **k: pytest.fail("invalid request reached generation"))
    for indices in ([], [0, 0], [-1], [True], [len(traces)]):
        with pytest.raises(ValueError, match="indices"):
            write_failures(target, model, traces, path, indices=indices)
    original = target.eos_token_ids
    target.eos_token_ids = frozenset([0])
    with pytest.raises(ValueError, match="target settings"):
        write_failures(target, model, traces, path)
    target.eos_token_ids = original
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_failures(target, model, traces, path)
    assert path.read_text(encoding="utf-8") == "keep"


def test_reader_rejects_oversized_files_before_parsing(tmp_path):
    _, _, traces = inputs(tmp_path)
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="input limit"):
        FailureDataset(path, traces)
