from __future__ import annotations

import pytest
import torch

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.runtime.types import VerifyResult
from tests.test_sampling_devices import DEVICES


def lane(device, monkeypatch, mode="adaptive"):
    model = DiffusionDrafterModel(DiffusionDrafterConfig(
        vocab_size=32, hidden_size=16, num_layers=1, num_heads=4,
        ctx_window=4, max_canvas=4,
    ))
    drafter = DiffusionDrafter(model, device=device, canvas_len=4, denoise_steps=2,
                             adaptive_stop=False, reconcile=ReconcilePolicy(mode=mode, repair_steps=1))

    def predict(context, canvas):
        logits = torch.zeros(canvas.length, 32, device=device)
        logits[:, 7] = 100
        return logits

    monkeypatch.setattr(drafter, "_model_fn", predict)
    return drafter


def reject(drafter, accepted=0):
    result = VerifyResult([7] * accepted + [9], 4, accepted, False, False)
    drafter.observe(result)
    return result.committed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trim", [False, True])
def test_adaptive_restarts_if_no_anchor_survives_filtering_or_trimming(device, trim, monkeypatch):
    drafter = lane(device, monkeypatch)
    context = [1, 2]
    drafter.propose(context, 4)
    with torch.inference_mode():
        drafter._last_canvas.confidence[:] = torch.tensor([0.1, 0.1, 0.1, 1.0 if trim else 0.1], device=device)
    context += reject(drafter)
    before = drafter.stats["diffusion_forwards"]
    output = drafter.propose(context, 2 if trim else 4)
    assert len(output) == (2 if trim else 4)
    assert drafter.stats["diffusion_forwards"] - before == 2
    assert drafter.stats["restart_no_anchors"] == 1
    assert drafter.stats["warm_rounds"] == 0 and not drafter._used_warm


@pytest.mark.parametrize("device", DEVICES)
def test_failed_repair_restarts_once_then_can_retry_and_recover(device, monkeypatch):
    drafter = lane(device, monkeypatch)
    context = [1, 2]
    drafter.propose(context, 4)
    context += reject(drafter)
    drafter.propose(context, 4)
    assert drafter.stats["warm_rounds"] == 1
    assert drafter.stats["diffusion_forwards"] == 3
    context += reject(drafter)
    assert drafter._warm is None and drafter.stats["restart_failed_repair"] == 1
    drafter.propose(context, 4)
    assert not drafter._used_warm and drafter.stats["diffusion_forwards"] == 5
    context += reject(drafter)
    drafter.propose(context, 4)
    assert drafter._used_warm and drafter.stats["warm_rounds"] == 2
    context += reject(drafter, accepted=1)
    assert drafter._warm is not None and drafter.stats["warm_rounds_recovered"] == 1
    drafter.propose(context, 4)
    assert drafter._used_warm and drafter.stats["restart_failed_repair"] == 1
    drafter.reset()
    assert not drafter._used_warm and drafter._warm is None and drafter._last_canvas is None
    before = drafter.stats["diffusion_forwards"]
    drafter.propose([3, 4], 4)
    assert not drafter._used_warm and drafter.stats["diffusion_forwards"] - before == 2


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_reuse_keeps_its_original_budget_after_an_empty_failed_repair(device, monkeypatch):
    drafter = lane(device, monkeypatch, mode="reconcile")
    context = [1, 2]
    for _ in range(3):
        drafter.propose(context, 4)
        with torch.inference_mode():
            drafter._last_canvas.confidence.zero_()
        context += reject(drafter)
    assert drafter.stats["warm_rounds"] == 2 and drafter.stats["diffusion_forwards"] == 4
    assert drafter.stats["restart_no_anchors"] == drafter.stats["restart_failed_repair"] == 0
