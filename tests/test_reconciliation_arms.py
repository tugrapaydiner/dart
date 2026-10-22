from __future__ import annotations

import pytest
import torch

from benchmarks.reconciliation import build_reconciliation_arms
from benchmarks.run_bench import build_hf_method
from dart.drafters.compact_diffusion import CompactDiffusionModel, CompactDrafterConfig
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target, random_prompt
from tests.test_sampling_devices import DEVICES


def compact_model(device="cpu"):
    target = TargetModel(build_tiny_target().model, device=device, record_hidden=True)
    target.eos_token_ids = frozenset()
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=1, num_heads=4))
    return target, model


def test_arms_share_weights_and_separate_reuse_from_the_pass_budget():
    target, model = compact_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    arms = build_reconciliation_arms(model, full_steps=3, repair_steps=1, keep_confidence_above=0.6, max_keep_fraction=0.5, seed=12)
    expected = {
        "fresh_one_pass": ("discard", 1, 1),
        "fresh_multi_pass": ("discard", 3, 3),
        "reuse_equal_budget": ("reconcile", 3, 3),
        "reuse_reduced_budget": ("reconcile", 3, 1),
    }
    assert set(arms) == set(expected)
    for name, drafter in arms.items():
        assert drafter.model is model and drafter.target is target
        assert (drafter.policy.mode, drafter.denoise_steps, drafter.policy.repair_steps) == expected[name]
        assert drafter.policy.keep_confidence_above == 0.6
        assert drafter.policy.max_keep_fraction == 0.5
        assert drafter.adaptive_stop is False
        assert drafter.gen.initial_seed() == 12
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize("device", DEVICES)
def test_shared_model_arms_match_hf_and_count_actual_forwards(device):
    target, model = compact_model(device)
    arms = build_reconciliation_arms(model)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    try:
        for prompt_seed, budget in ((410, 1), (411, 9), (410, 9)):
            prompt = random_prompt(prompt_seed, 8)
            expected = build_hf_method(target, budget, record_hidden=False)(prompt).token_ids
            for drafter in arms.values():
                calls.clear()
                before = drafter.stats["diffusion_forwards"]
                with DartEngine(target, drafter) as engine:
                    output = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, draft_len=4, canvas_len=4))
                assert output.token_ids == expected
                assert drafter.stats["diffusion_forwards"] - before == len(calls)
                assert bool(calls) == (budget > 1)
    finally:
        hook.remove()


@pytest.mark.parametrize("options", [
    {"canvas_len": 0}, {"canvas_len": True}, {"canvas_len": 9},
    {"full_steps": 1}, {"full_steps": 2.0}, {"repair_steps": 0}, {"repair_steps": 2},
    {"keep_confidence_above": float("nan")}, {"keep_confidence_above": 1.1},
    {"max_keep_fraction": -0.1}, {"max_keep_fraction": True}, {"seed": True},
])
def test_invalid_arm_settings_fail_before_drafter_construction(options, monkeypatch):
    target, model = compact_model()
    monkeypatch.setattr("benchmarks.reconciliation.CompactDiffusionDrafter", lambda *a, **k: pytest.fail("constructed a drafter before validation"))
    with pytest.raises(ValueError):
        build_reconciliation_arms(model, **options)
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert model._target() is target
