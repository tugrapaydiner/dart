from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from scripts import collect_failures, train_compact

from dart.data.failures import FailureDataset, write_failures
from dart.diffusion.compact_training import (
    CompactTrainer,
    CompactTrainingConfig,
    load_compact_checkpoint,
    read_checkpoint,
)
from dart.drafters.compact_diffusion import CompactDiffusionModel
from dart.models.target import TargetModel
from tests.test_compact_training import MODEL, TRAIN, training_inputs
from tests.test_sampling_devices import DEVICES


def failure_inputs(tmp_path, device="cpu"):
    target, traces = training_inputs(tmp_path, device)
    target.tokenizer.decode = lambda ids, **kwargs: " ".join(map(str, ids))
    trace = traces[0]
    prompt, continuation = trace["prompt_ids"].tolist(), trace["continuation_ids"].tolist()

    def propose(self, committed, budget):
        offset = len(committed) - len(prompt)
        assert committed == prompt + continuation[:offset]
        tokens = continuation[offset:offset + budget]
        tokens[0] = (tokens[0] + 1) % target.model.config.vocab_size
        return tokens

    path = tmp_path / "failures.json"
    with patch("dart.data.failures.CompactDiffusionDrafter.propose", propose):
        write_failures(target, CompactDiffusionModel(target, MODEL), traces, path, canvas_len=3)
    return target, traces, FailureDataset(path, traces)


@pytest.mark.parametrize("full_mask_fraction", [0.0, 1.0])
def test_replayed_masks_have_matched_draws_counts_and_correct_feature_alignment(tmp_path, monkeypatch, full_mask_fraction):
    target, traces, failures = failure_inputs(tmp_path)
    cfg = replace(TRAIN, distill_weight=0.5, full_mask_fraction=full_mask_fraction)
    observed = CompactTrainer(target, traces, MODEL, replace(cfg, failure_masking="observed"), failures=failures)
    control = CompactTrainer(target, traces, MODEL, replace(cfg, failure_masking="random_matched"), failures=failures)
    trace = {
        "prompt_ids": torch.tensor([7, 8]), "continuation_ids": torch.tensor([10, 11, 12, 13]),
        "hidden": torch.arange(4, dtype=torch.float16).unsqueeze(1).expand(4, 64),
    }
    for trainer in (observed, control):
        monkeypatch.setattr(trainer, "_trace", lambda index: trace)
    for key, value in observed.model.state_dict().items():
        assert torch.equal(value, control.model.state_dict()[key])
    different = False
    for _ in range(20):
        a, b = observed.sample_batch(), control.sample_batch()
        assert torch.equal(observed.generator.get_state(), control.generator.get_state())
        for key in a.keys() - {"canvas_masked"}:
            assert torch.equal(a[key], b[key])
        assert torch.equal((a["canvas_masked"] & a["valid"]).sum(1), (b["canvas_masked"] & b["valid"]).sum(1))
        different |= not torch.equal(a["canvas_masked"], b["canvas_masked"])
        for i in range(cfg.batch_size):
            hidden_row = int(a["target_hidden"][i, 0])
            valid = a["valid"][i]
            labels = a["canvas_tokens"][i, valid].tolist()
            assert labels == trace["continuation_ids"][hidden_row + 1:hidden_row + 1 + len(labels)].tolist()
            assert a["context_ids"][i, ~a["context_pad_mask"][i]].tolist() == [7, 8] + trace["continuation_ids"][:hidden_row + 1].tolist()
            assert a["teacher_hidden"][i, valid, 0].tolist() == list(range(hidden_row + 1, hidden_row + 1 + len(labels)))
            assert bool(a["canvas_masked"][i, ~valid].all()) and bool((a["teacher_hidden"][i, ~valid] == 0).all())
            expected_mask = [True] * len(labels) if full_mask_fraction else [True] + [False] * (len(labels) - 1)
            assert a["canvas_masked"][i, valid].tolist() == expected_mask
    assert different is (full_mask_fraction == 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["observed", "random_matched"])
def test_failure_training_resumes_exactly_and_inference_needs_no_failure_file(tmp_path, device, dtype, mode):
    target, traces, failures = failure_inputs(tmp_path, device)
    cfg = replace(TRAIN, failure_masking=mode, dtype=dtype, distill_weight=0.5, full_mask_fraction=0.25)
    full = CompactTrainer(target, traces, MODEL, cfg, failures=failures)
    expected = [full.train_step() for _ in range(4)]
    split = CompactTrainer(target, traces, MODEL, cfg, failures=failures)
    actual = [split.train_step() for _ in range(2)]
    path = tmp_path / "resume.pt"
    split.save(path)
    resumed = CompactTrainer(target, traces, MODEL, cfg, failures=failures)
    resumed.resume(path)
    actual.extend(resumed.train_step() for _ in range(2))
    assert actual == expected
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    resumed.save(path)
    (tmp_path / "failures.json").unlink()
    loaded, data = load_compact_checkpoint(path, target)
    assert data["metadata"]["failure_data_sha256"] == failures.sha256
    assert data["training_config"]["failure_masking"] == mode
    assert next(loaded.parameters()).dtype == getattr(torch, dtype)
    batch = resumed.sample_batch()
    batch.pop("valid")
    batch.pop("teacher_hidden")
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**batch), full.model.eval()(**batch), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["mode", "data"])
def test_resume_rejects_changed_failure_mode_or_manifest(tmp_path, change):
    target, traces, failures = failure_inputs(tmp_path)
    cfg = replace(TRAIN, failure_masking="observed")
    original = CompactTrainer(target, traces, MODEL, cfg, failures=failures)
    original.train_step()
    checkpoint = tmp_path / "checkpoint.pt"
    original.save(checkpoint)
    if change == "mode":
        cfg = replace(cfg, failure_masking="random_matched")
    else:
        path = tmp_path / "changed.json"
        manifest = dict(failures.manifest)
        manifest["seed"] += 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
        failures = FailureDataset(path, traces)
    changed = CompactTrainer(target, traces, MODEL, cfg, failures=failures)
    before = {key: value.clone() for key, value in changed.model.state_dict().items()}
    with pytest.raises(ValueError, match="do not match"):
        changed.resume(checkpoint)
    for key, value in before.items():
        assert torch.equal(value, changed.model.state_dict()[key])


def test_legacy_random_mask_training_keeps_exact_resume(tmp_path):
    target, traces = training_inputs(tmp_path)
    original = CompactTrainer(target, traces, MODEL, TRAIN)
    original.train_step()
    path = tmp_path / "legacy.pt"
    original.save(path)
    data = read_checkpoint(path)
    del data["training_config"]["failure_masking"]
    assert "failure_data_sha256" not in data["metadata"]
    torch.save(data, path)
    expected = [original.train_step() for _ in range(2)]
    resumed = CompactTrainer(target, traces, MODEL, TRAIN)
    resumed.resume(path)
    assert [resumed.train_step() for _ in range(2)] == expected
    for key, value in original.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    load_compact_checkpoint(path, target)


def test_failure_options_require_matching_data_and_canvas(tmp_path):
    target, traces, failures = failure_inputs(tmp_path)
    with pytest.raises(ValueError, match="masking"):
        CompactTrainingConfig(failure_masking="unknown")
    cfg = replace(TRAIN, failure_masking="observed")
    with pytest.raises(ValueError, match="explicit mode"):
        CompactTrainer(target, traces, MODEL, cfg)
    with pytest.raises(ValueError, match="explicit mode"):
        CompactTrainer(target, traces, MODEL, TRAIN, failures=failures)
    with pytest.raises(ValueError, match="training canvas"):
        CompactTrainer(target, traces, MODEL, replace(cfg, canvas_len=1), failures=failures)
    traces.manifest["corpus_hash"] = "changed"
    with pytest.raises(ValueError, match="match the training traces"):
        CompactTrainer(target, traces, MODEL, cfg, failures=failures)


def test_collection_training_and_resume_commands_preserve_failure_provenance(tmp_path, monkeypatch):
    target, traces = training_inputs(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    target.tokenizer.decode = lambda ids, **kwargs: " ".join(map(str, ids))
    proposer = CompactTrainer(target, traces, MODEL, TRAIN)
    proposer.train_step()
    checkpoint = tmp_path / "proposer.pt"
    proposer.save(checkpoint)
    monkeypatch.setattr(collect_failures.TargetModel, "from_pretrained", lambda *a, **k: target)
    path = tmp_path / "failures.json"
    monkeypatch.setattr(sys, "argv", [
        "collect_failures", "--model", str(tmp_path), "--checkpoint", str(checkpoint),
        "--traces", str(traces.directory), "--out", str(path), "--per-workload", "1", "--canvas-len", "3",
    ])
    collect_failures.main()
    failures = FailureDataset(path, traces)
    assert len(failures.manifest["selected_indices"]) == 1
    args = ["train_compact", "--model", str(tmp_path), "--traces", str(traces.directory),
            "--failures", str(path), "--failure-masking", "observed", "--full-mask-fraction", "0",
            "--batch-size", "2", "--canvas-len", "3"]
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", args + ["--out", str(first), "--steps", "2"])
    train_compact.main()
    monkeypatch.setattr(sys, "argv", args + ["--out", str(resumed), "--steps", "3", "--resume", str(first / "latest.pt")])
    train_compact.main()
    report = json.loads((resumed / "report.json").read_text(encoding="utf-8"))
    assert report["steps"] == 3 and report["training_config"]["failure_masking"] == "observed"
    assert report["training_config"]["full_mask_fraction"] == 0
    assert report["metadata"]["failure_data_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_checkpoint(resumed / "latest.pt")["step"] == 3


@pytest.mark.parametrize("options", [
    ["--failure-masking", "observed"], ["--failures", "unused.json"],
    ["--failures", "missing.json", "--failure-masking", "random_matched"],
    ["--full-mask-fraction", "nan"], ["--full-mask-fraction", "1.1"],
])
def test_invalid_training_options_fail_before_loading(tmp_path, monkeypatch, options):
    _, traces = training_inputs(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(TargetModel, "from_pretrained", lambda *a, **k: pytest.fail("loaded before validation"))
    monkeypatch.setattr(sys, "argv", ["train_compact", "--model", str(tmp_path), "--traces", str(traces.directory),
                                   "--out", str(tmp_path / "out"), "--steps", "1"] + options)
    with pytest.raises(SystemExit) as error:
        train_compact.main()
    assert error.value.code == 2
