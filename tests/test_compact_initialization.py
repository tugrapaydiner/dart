from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dart.data.teacher import target_fingerprint
from dart.diffusion.compact_training import (
    CompactTrainer,
    load_compact_checkpoint,
    read_checkpoint,
)
from dart.drafters.compact_diffusion import CompactDiffusionModel, CompactDrafterConfig
from tests.test_compact_training import MODEL, TRAIN, training_inputs
from tests.test_sampling_devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_embedding_scale_preserves_other_weights_and_random_draws(device, tmp_path):
    target, _ = training_inputs(tmp_path, device)
    before = target_fingerprint(target)
    torch.manual_seed(17)
    original = CompactDiffusionModel(target, MODEL)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    torch.manual_seed(17)
    scaled = CompactDiffusionModel(target, replace(MODEL, embedding_init_std=0.02))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for key, value in original.state_dict().items():
        expected = value * 0.02 if key in ("pos_emb.weight", "type_emb.weight") else value
        torch.testing.assert_close(scaled.state_dict()[key], expected, rtol=0, atol=0)
    assert target_fingerprint(target) == before
    assert sum(p.numel() for p in scaled.parameters()) == MODEL.parameter_count(64)


@pytest.mark.parametrize("device", DEVICES)
def test_scaled_embedding_training_resumes_with_identical_weights(device, tmp_path):
    target, dataset = training_inputs(tmp_path, device)
    model_cfg = replace(MODEL, embedding_init_std=0.02)
    cfg = replace(TRAIN, distill_weight=0.5, distill_temperature=2.0)
    full = CompactTrainer(target, dataset, model_cfg, cfg)
    expected = [full.train_step() for _ in range(4)]
    split = CompactTrainer(target, dataset, model_cfg, cfg)
    actual = [split.train_step() for _ in range(2)]
    path = tmp_path / "scaled.pt"
    split.save(path)
    resumed = CompactTrainer(target, dataset, model_cfg, cfg)
    resumed.resume(path)
    actual.extend(resumed.train_step() for _ in range(2))
    assert actual == expected
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    loaded, _ = load_compact_checkpoint(path, target)
    assert loaded.cfg.embedding_init_std == 0.02
    for key, value in split.model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_checkpoints_without_an_embedding_scale_keep_the_original_default(tmp_path):
    target, dataset = training_inputs(tmp_path)
    full = CompactTrainer(target, dataset, MODEL, TRAIN)
    full.train_step()
    path = tmp_path / "legacy.pt"
    full.save(path)
    data = read_checkpoint(path)
    del data["model_config"]["embedding_init_std"]
    torch.save(data, path)
    expected = full.train_step()
    resumed = CompactTrainer(target, dataset, MODEL, TRAIN)
    resumed.resume(path)
    assert resumed.train_step() == expected
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    loaded, _ = load_compact_checkpoint(path, target)
    assert loaded.cfg.embedding_init_std == 1.0
    for key, value in data["model"].items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_resume_rejects_a_changed_embedding_scale(tmp_path):
    target, dataset = training_inputs(tmp_path)
    original = CompactTrainer(target, dataset, MODEL, TRAIN)
    original.train_step()
    path = tmp_path / "original.pt"
    original.save(path)
    scaled = CompactTrainer(target, dataset, replace(MODEL, embedding_init_std=0.02), TRAIN)
    before = {key: value.clone() for key, value in scaled.model.state_dict().items()}
    with pytest.raises(ValueError, match="do not match"):
        scaled.resume(path)
    for key, value in before.items():
        assert torch.equal(value, scaled.model.state_dict()[key])


@pytest.mark.parametrize("value", [0, -0.02, float("nan"), float("inf"), True])
def test_invalid_embedding_scales_are_rejected(value):
    with pytest.raises(ValueError, match="embedding_init_std"):
        CompactDrafterConfig(embedding_init_std=value)
