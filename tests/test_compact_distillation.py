from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from dart.data.teacher import target_fingerprint
from dart.diffusion.compact_training import (
    CompactTrainer,
    CompactTrainingConfig,
    load_compact_checkpoint,
    read_checkpoint,
)
from tests.test_compact_training import MODEL, TRAIN, training_inputs
from tests.test_sampling_devices import DEVICES


def test_future_features_align_with_labels_without_changing_batch_draws(tmp_path, monkeypatch):
    target, dataset = training_inputs(tmp_path)
    plain = CompactTrainer(target, dataset, MODEL, TRAIN)
    distill = CompactTrainer(target, dataset, MODEL, replace(TRAIN, distill_weight=0.5))
    trace = {
        "prompt_ids": torch.tensor([7, 8]), "continuation_ids": torch.tensor([10, 11, 12, 13]),
        "hidden": torch.arange(4, dtype=torch.float16).unsqueeze(1).expand(4, 64),
    }
    for trainer in (plain, distill):
        monkeypatch.setattr(trainer, "_trace", lambda index: trace)
    for key, value in plain.model.state_dict().items():
        assert torch.equal(value, distill.model.state_dict()[key])
    saw_padding = False
    for _ in range(12):
        expected, actual = plain.sample_batch(), distill.sample_batch()
        assert "teacher_hidden" not in expected
        for key, value in expected.items():
            assert torch.equal(value, actual[key])
        for row in range(TRAIN.batch_size):
            offset = int(actual["target_hidden"][row, 0])
            valid = actual["valid"][row]
            future = actual["teacher_hidden"][row]
            assert future[valid, 0].tolist() == list(range(offset + 1, offset + 1 + int(valid.sum())))
            assert (future[~valid] == 0).all()
            assert not future.requires_grad
            saw_padding |= bool((~valid).any())
    assert saw_padding


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight", [0.25, 1.0])
def test_distillation_matches_masked_kl_loss_and_gradients(device, weight, tmp_path, monkeypatch):
    target, dataset = training_inputs(tmp_path, device)
    cfg = replace(TRAIN, distill_weight=weight, distill_temperature=2.0)
    trainer = CompactTrainer(target, dataset, replace(MODEL, dropout=0), cfg)
    before = target_fingerprint(target)
    batch = {
        "context_ids": torch.arange(16, device=device).reshape(2, 8) + 7,
        "context_pad_mask": torch.zeros(2, 8, dtype=torch.bool, device=device),
        "canvas_tokens": torch.tensor([[10, 11, 0], [20, 21, 22]], device=device),
        "canvas_masked": torch.tensor([[True, False, True], [True, True, False]], device=device),
        "target_hidden": torch.randn(2, 64, device=device),
        "valid": torch.tensor([[True, True, False], [True, True, True]], device=device),
        "teacher_hidden": torch.randn(2, 3, 64, dtype=torch.float16, device=device),
    }
    supervised = batch["valid"] & batch["canvas_masked"]
    batch["teacher_hidden"][~supervised] = float("nan")  # Unused rows must not affect the loss.
    batch["teacher_hidden"].requires_grad_(True)
    inputs = {key: value for key, value in batch.items() if key not in ("valid", "teacher_hidden")}
    logits = trainer.model(**inputs)[supervised]
    with torch.no_grad():
        head = target.model.get_output_embeddings()
        teacher = head(batch["teacher_hidden"][supervised].to(dtype=head.weight.dtype)).float()
        teacher_log = F.log_softmax(teacher / 2, dim=-1)
    student_log = F.log_softmax(logits / 2, dim=-1)
    kl = (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean() * 4
    expected = (1 - weight) * F.cross_entropy(logits, batch["canvas_tokens"][supervised]) + weight * kl
    expected.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.grad_clip)
    expected_gradients = {key: value.grad.clone() for key, value in trainer.model.named_parameters()}
    monkeypatch.setattr(trainer, "sample_batch", lambda: dict(batch))
    monkeypatch.setattr(target.model, "forward", lambda *a, **k: pytest.fail("distillation ran the target"))
    assert trainer.train_step() == pytest.approx(float(expected.detach()), abs=2e-6)
    for key, value in trainer.model.named_parameters():
        torch.testing.assert_close(value.grad, expected_gradients[key], rtol=1e-5, atol=1e-6)
    assert batch["teacher_hidden"].grad is None
    assert all(p.grad is None for p in target.model.parameters())
    assert target_fingerprint(target) == before


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_distilled_checkpoint_resumes_exactly_and_loads_for_inference(device, dtype, tmp_path):
    target, dataset = training_inputs(tmp_path, device)
    cfg = replace(TRAIN, dtype=dtype, distill_weight=0.5, distill_temperature=2.0)
    full = CompactTrainer(target, dataset, MODEL, cfg)
    expected = [full.train_step() for _ in range(4)]
    split = CompactTrainer(target, dataset, MODEL, cfg)
    actual = [split.train_step() for _ in range(2)]
    path = tmp_path / "distill.pt"
    split.save(path)
    resumed = CompactTrainer(target, dataset, MODEL, cfg)
    resumed.resume(path)
    actual.extend(resumed.train_step() for _ in range(2))
    assert actual == expected
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    resumed.save(path)
    loaded, data = load_compact_checkpoint(path, target)
    assert next(loaded.parameters()).dtype == getattr(torch, dtype)
    assert data["training_config"]["distill_weight"] == 0.5
    assert data["training_config"]["distill_temperature"] == 2.0
    batch = resumed.sample_batch()
    batch.pop("valid")
    batch.pop("teacher_hidden")
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**batch), full.model.eval()(**batch), rtol=0, atol=0)


def test_token_label_checkpoints_from_before_distillation_still_resume(tmp_path):
    target, dataset = training_inputs(tmp_path)
    original = CompactTrainer(target, dataset, MODEL, TRAIN)
    original.train_step()
    path = tmp_path / "legacy.pt"
    original.save(path)
    data = read_checkpoint(path)
    del data["training_config"]["distill_weight"]
    del data["training_config"]["distill_temperature"]
    torch.save(data, path)
    expected = [original.train_step() for _ in range(2)]
    resumed = CompactTrainer(target, dataset, MODEL, TRAIN)
    resumed.resume(path)
    assert [resumed.train_step() for _ in range(2)] == expected
    for key, value in original.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    loaded, _ = load_compact_checkpoint(path, target)
    assert not loaded.training


@pytest.mark.parametrize("change", [{"distill_weight": 0.75}, {"distill_temperature": 3.0}])
def test_resume_rejects_changed_distillation_settings(tmp_path, change):
    target, dataset = training_inputs(tmp_path)
    cfg = replace(TRAIN, distill_weight=0.5, distill_temperature=2.0)
    original = CompactTrainer(target, dataset, MODEL, cfg)
    original.train_step()
    path = tmp_path / "distill.pt"
    original.save(path)
    changed = CompactTrainer(target, dataset, MODEL, replace(cfg, **change))
    before = {key: value.clone() for key, value in changed.model.state_dict().items()}
    with pytest.raises(ValueError, match="do not match"):
        changed.resume(path)
    for key, value in before.items():
        assert torch.equal(value, changed.model.state_dict()[key])


@pytest.mark.parametrize("options", [
    {"distill_weight": -0.1}, {"distill_weight": 1.1}, {"distill_weight": float("nan")},
    {"distill_weight": True}, {"distill_temperature": 0},
    {"distill_temperature": float("nan")}, {"distill_temperature": True},
])
def test_invalid_distillation_settings_are_rejected(options):
    with pytest.raises(ValueError):
        CompactTrainingConfig(**options)
