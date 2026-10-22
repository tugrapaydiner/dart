from __future__ import annotations

import zipfile
from dataclasses import replace

import pytest
import torch

from dart.data.teacher import TraceDataset, target_fingerprint, write_traces
from dart.diffusion.compact_training import (
    CompactTrainer,
    CompactTrainingConfig,
    load_compact_checkpoint,
    read_checkpoint,
)
from dart.drafters.compact_diffusion import CompactDrafterConfig
from dart.models.target import TargetModel
from tests.test_sampling_devices import DEVICES
from tests.test_teacher import setup_corpus

MODEL = CompactDrafterConfig(hidden_size=48, num_layers=1, num_heads=4, ctx_window=8, max_canvas=4, dropout=0.1)
TRAIN = CompactTrainingConfig(batch_size=2, canvas_len=3, cache_shards=1)


def training_inputs(tmp_path, device="cpu"):
    original, corpus, output = setup_corpus(tmp_path)
    target = TargetModel(original.model, original.tokenizer, device=device, record_hidden=True)
    target.eos_token_ids = frozenset()
    write_traces(target, corpus, "train", output)
    return target, TraceDataset(output)


@pytest.mark.parametrize("device", DEVICES)
def test_resumed_training_matches_uninterrupted_updates(device, tmp_path):
    target, dataset = training_inputs(tmp_path, device)
    before = target_fingerprint(target)
    full = CompactTrainer(target, dataset, MODEL, TRAIN)
    expected_losses = [full.train_step() for _ in range(6)]
    split = CompactTrainer(target, dataset, MODEL, TRAIN)
    losses = [split.train_step() for _ in range(3)]
    checkpoint = tmp_path / "resume.pt"
    split.save(checkpoint)
    resumed = CompactTrainer(target, dataset, MODEL, TRAIN)
    resumed.resume(checkpoint)
    assert resumed.step == 3
    losses.extend(resumed.train_step() for _ in range(3))
    assert losses == expected_losses
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert target_fingerprint(target) == before
    assert all(p.grad is None for p in target.model.parameters())
    assert len(resumed.cache) <= TRAIN.cache_shards


@pytest.mark.parametrize("full_mask_fraction", [0.0, 1.0])
def test_batches_start_after_the_pending_token_and_never_supervise_padding(tmp_path, monkeypatch, full_mask_fraction):
    target, dataset = training_inputs(tmp_path)
    trainer = CompactTrainer(target, dataset, MODEL, replace(TRAIN, full_mask_fraction=full_mask_fraction))
    trace = {
        "prompt_ids": torch.tensor([7, 8]), "continuation_ids": torch.tensor([10, 11, 12, 13]),
        "hidden": torch.arange(4, dtype=torch.float16).unsqueeze(1).expand(4, 64),
    }
    monkeypatch.setattr(trainer, "_trace", lambda index: trace)
    saw_padding = False
    for _ in range(12):
        batch = trainer.sample_batch()
        for row in range(TRAIN.batch_size):
            offset = int(batch["target_hidden"][row, 0])
            context = batch["context_ids"][row][~batch["context_pad_mask"][row]].tolist()
            assert context == [7, 8] + [10, 11, 12, 13][:offset + 1]
            valid = batch["valid"][row]
            labels = batch["canvas_tokens"][row][valid].tolist()
            assert labels == [10, 11, 12, 13][offset + 1:offset + 1 + int(valid.sum())]
            assert (batch["canvas_masked"][row] & valid).any()
            assert batch["canvas_masked"][row][~valid].all()
            if full_mask_fraction == 1:
                assert batch["canvas_masked"][row].all()
            saw_padding |= bool((~valid).any())
    assert saw_padding


def test_checkpoint_loads_for_inference_without_teacher_files(tmp_path):
    target, dataset = training_inputs(tmp_path)
    trainer = CompactTrainer(target, dataset, MODEL, replace(TRAIN, dtype="bfloat16"))
    trainer.train_step()
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    (dataset.directory / "manifest.json").unlink()
    loaded, data = load_compact_checkpoint(path, target)
    assert data["step"] == 1
    assert sum(t.numel() for t in data["model"].values()) == MODEL.parameter_count(64)
    assert next(loaded.parameters()).dtype == torch.bfloat16
    assert target.model.get_input_embeddings().weight.dtype == torch.float32
    batch = trainer.sample_batch()
    batch.pop("valid")
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**batch), trainer.model.eval()(**batch), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["options", "data"])
def test_resume_rejects_changed_inputs_before_mutating_the_model(tmp_path, change):
    target, dataset = training_inputs(tmp_path)
    original = CompactTrainer(target, dataset, MODEL, TRAIN)
    original.train_step()
    path = tmp_path / "checkpoint.pt"
    original.save(path)
    if change == "data":
        dataset.manifest["corpus_hash"] = "changed corpus"
    config = replace(TRAIN, learning_rate=0.01) if change == "options" else TRAIN
    changed = CompactTrainer(target, dataset, MODEL, config)
    before = {k: v.clone() for k, v in changed.model.state_dict().items()}
    with pytest.raises(ValueError, match="do not match"):
        changed.resume(path)
    for key, value in before.items():
        assert torch.equal(changed.model.state_dict()[key], value)
    with torch.no_grad():
        next(target.model.parameters()).view(-1)[0] += 1
    with pytest.raises(ValueError, match="do not match"):
        load_compact_checkpoint(path, target)
    with pytest.raises(ValueError, match="do not match"):
        CompactTrainer(target, dataset, MODEL, TRAIN)


def test_validation_traces_cannot_be_used_for_optimizer_updates(tmp_path):
    target, dataset = training_inputs(tmp_path)
    dataset.manifest["split"] = "validation"
    with pytest.raises(ValueError, match="training traces"):
        CompactTrainer(target, dataset, MODEL, TRAIN)


def test_failed_checkpoint_write_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    target, dataset = training_inputs(tmp_path)
    trainer = CompactTrainer(target, dataset, MODEL, TRAIN)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    before = path.read_bytes()

    def fail_save(payload, output):
        output.write(b"partial checkpoint")
        raise OSError("disk write failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk write failed"):
        trainer.save(path)
    assert path.read_bytes() == before
    assert not path.with_suffix(".pt.tmp").exists()
    assert read_checkpoint(path)["step"] == 0


def test_compressed_checkpoint_size_is_checked_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "compressed.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", b"x" * 2048)
    assert path.stat().st_size < 1024
    monkeypatch.setattr("dart.diffusion.compact_training.MAX_CHECKPOINT_BYTES", 1024)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded an oversized archive"))
    with pytest.raises(ValueError, match="expanded"):
        read_checkpoint(path)


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"batch_size": 0}, {"learning_rate": float("nan")}, {"learning_rate": 0}, {"dtype": "float16"}, {"full_mask_fraction": 2}])
def test_invalid_training_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CompactTrainingConfig(**kwargs)
