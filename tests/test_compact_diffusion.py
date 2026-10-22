from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dart.drafters.compact_diffusion import CompactDiffusionModel, CompactDrafterConfig
from dart.models.target import TargetModel
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


def inputs(device, hidden_size=64):
    return (
        torch.tensor([[0, 0, 7, 8], [0, 9, 10, 11]], device=device),
        torch.tensor([[True, True, False, False], [True, False, False, False]], device=device),
        torch.tensor([[12, 13, 14], [15, 16, 17]], device=device),
        torch.tensor([[True, False, True], [False, True, True]], device=device),
        torch.randn(2, hidden_size, device=device),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_updates_only_compact_weights(device, dtype):
    target = TargetModel(build_tiny_target().model, device=device)
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=2, num_heads=4)).to(dtype=dtype)
    before = {name: parameter.detach().clone() for name, parameter in target.model.named_parameters()}
    batch = inputs(device)
    logits = model(*batch)
    assert logits.shape == (2, 3, target.model.config.vocab_size)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits[batch[3]], batch[2][batch[3]])
    loss.backward()
    assert model.feature_proj.weight.grad is not None and model.feature_proj.weight.grad.abs().sum() > 0
    old = model.output_proj.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=0.001).step()
    assert not torch.equal(model.output_proj.weight, old)
    for name, parameter in target.model.named_parameters():
        assert parameter.grad is None and torch.equal(parameter, before[name])
    assert next(target.model.parameters()).dtype == torch.float32


def test_masked_values_and_left_padding_do_not_leak_into_predictions():
    target = build_tiny_target()
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=2, num_heads=4)).eval()
    context, pad, canvas, masked, hidden = inputs("cpu")
    reference = model(context, pad, canvas, masked, hidden)
    changed_context, changed_canvas = context.clone(), canvas.clone()
    changed_context[pad] = 9999
    changed_canvas[masked] = 9999
    torch.testing.assert_close(model(changed_context, pad, changed_canvas, masked, hidden), reference)
    short = model(context[:1, -2:], pad[:1, -2:], canvas[:1], masked[:1], hidden[:1])
    torch.testing.assert_close(short, reference[:1], rtol=1e-5, atol=1e-5)
    assert not torch.equal(model(context, pad, canvas, masked, hidden + 1), reference)


def test_inference_features_can_train_without_a_target_forward(monkeypatch):
    target = build_tiny_target()
    target.record_hidden = True
    with torch.inference_mode():
        target.prefill([7, 8, 9])
        hidden = target.last_hidden_block[-1:].expand(2, -1)
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=1, num_heads=4))
    monkeypatch.setattr(target.model, "forward", lambda *a, **k: pytest.fail("Drafter called the target model"))
    context, pad, canvas, masked, _ = inputs("cpu")
    model(context, pad, canvas, masked, hidden).sum().backward()
    assert model.feature_proj.weight.grad is not None


def test_checkpoint_and_deepcopy_do_not_copy_target_weights():
    target = build_tiny_target()
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=1, num_heads=4))
    copied = copy.deepcopy(model)
    assert copied._target() is target
    assert sum(value.numel() for value in model.state_dict().values()) == model.cfg.parameter_count(64)
    assert not ({id(p) for p in model.parameters()} & {id(p) for p in target.model.parameters()})
    copied.to(dtype=torch.bfloat16).train()
    assert target.model.training is False
    assert target.model.get_input_embeddings().weight.dtype == torch.float32


def test_supra_shape_stays_under_five_million_learned_parameters():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32_000, 512, device="meta")
            self.output = nn.Linear(512, 32_000, bias=False, device="meta")
            self.requires_grad_(False)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.output

    class Target:
        model = Model()

    target = Target()
    model = CompactDiffusionModel(target)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == model.cfg.parameter_count(512)
    assert 2_000_000 <= count <= 5_000_000
    with pytest.raises(ValueError, match="5M"):
        CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=1024, num_heads=8))


@pytest.mark.parametrize("kwargs", [{"num_layers": 0}, {"num_heads": True}, {"hidden_size": 191}, {"dropout": float("nan")}])
def test_invalid_compact_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CompactDrafterConfig(**kwargs)
