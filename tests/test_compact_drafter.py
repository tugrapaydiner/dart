from __future__ import annotations

import pytest
import torch

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.base import NullDrafter
from dart.drafters.compact_diffusion import (
    CompactDiffusionDrafter,
    CompactDiffusionModel,
    CompactDrafterConfig,
)
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.block_verify import verify_block_greedy
from tests.conftest import VOCAB, build_tiny_target, random_prompt
from tests.test_drafter_state import AlternatingScheduler
from tests.test_sampling_devices import DEVICES


def build_lane(device="cpu", mode="reconcile"):
    target = TargetModel(build_tiny_target().model, device=device, record_hidden=True)
    target.eos_token_ids = frozenset()
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=1, num_heads=4))
    drafter = CompactDiffusionDrafter(model, reconcile=ReconcilePolicy(mode=mode, repair_steps=1))
    return target, drafter


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["discard", "reconcile", "adaptive"])
@pytest.mark.parametrize("verifier", ["block", "tree"])
def test_compact_lane_preserves_complete_outputs_budgets_and_eos(device, mode, verifier):
    target, drafter = build_lane(device, mode)
    prompt = random_prompt(360, 10)
    expected = target.greedy_generate(prompt, 14)
    calls = []
    hook = target.model.register_forward_hook(lambda *args: calls.append(1))
    try:
        with DartEngine(target, drafter, verifier=verifier) as engine:
            for budget in (0, 1, 2, 14):
                calls.clear()
                result = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, canvas_len=4))
                assert result.token_ids == expected[:budget]
                assert len(calls) == int(budget > 0) + len(result.rounds)
                if budget == 0:
                    assert result.rounds == []
            target.eos_token_ids = frozenset([expected[3]])
            stopped = target.greedy_generate(prompt, 14)
            assert engine.generate(prompt, DecodeConfig(max_new_tokens=14)).token_ids == stopped
            assert stopped[-1] == expected[3]
    finally:
        hook.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["reconcile", "adaptive"])
def test_features_refresh_after_lane_switches_and_new_requests(device, mode, monkeypatch):
    target, drafter = build_lane(device, mode)
    seen = []
    original = drafter.propose

    def propose(committed, max_tokens):
        if max_tokens > 0:
            seen.append((list(committed), drafter._hidden.clone()))
        return original(committed, max_tokens)

    monkeypatch.setattr(drafter, "propose", propose)
    lanes = {"diffusion": drafter, "target_only": NullDrafter()}
    with DartEngine(target, lanes=lanes, scheduler=AlternatingScheduler("diffusion")) as engine:
        for seed in (361, 362, 361):
            prompt = random_prompt(seed, 8)
            expected = target.greedy_generate(prompt, 14)
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=14))
            assert result.token_ids == expected
            assert {row.lane for row in result.rounds} == {"diffusion", "target_only"}
    assert drafter.stats["warm_rounds"] == 0
    assert len(seen) >= 6
    for committed, hidden in seen:
        target.prefill(committed[:-1])
        torch.testing.assert_close(hidden, target.last_hidden_block[-1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_repair_uses_verified_features_and_clears_on_prefill(accepted):
    target, drafter = build_lane()
    prompt = random_prompt(363, 9)
    continuation = target.greedy_generate(prompt, 5)
    state, _ = target.prefill(prompt)
    drafter.after_prefill()
    committed = prompt + continuation[:1]
    drafts = drafter.propose(committed, 3)
    drafts[:accepted] = continuation[1:1 + accepted]
    if accepted < 3:
        drafts[accepted] = (continuation[1 + accepted] + 1) % VOCAB
    result = verify_block_greedy(target, state, continuation[:1], drafts)
    assert result.n_draft_accepted == accepted
    drafter.observe(result)
    hidden = drafter._hidden.clone()
    target.prefill((committed + result.committed)[:-1])
    torch.testing.assert_close(hidden, target.last_hidden_block[-1], atol=1e-5, rtol=1e-5)
    assert (drafter._warm is not None) is (accepted < 2)
    if accepted < 2:
        drafter.propose(committed + result.committed, 3)
        assert drafter.stats["warm_rounds"] == 1
    target.prefill(prompt)
    drafter.after_prefill()
    assert drafter._warm is None and drafter._last_canvas is None and drafter._context == []


def test_hidden_features_are_required_before_drafting():
    target = build_tiny_target()
    model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=48, num_layers=1, num_heads=4))
    with pytest.raises(ValueError, match="record_hidden"):
        CompactDiffusionDrafter(model)
    target.record_hidden = True
    drafter = CompactDiffusionDrafter(model)
    with pytest.raises(RuntimeError, match="after_prefill"):
        drafter.propose([7, 8], 4)
    with pytest.raises(RuntimeError, match="unavailable"):
        drafter.after_prefill()
