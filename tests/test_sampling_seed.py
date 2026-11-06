from __future__ import annotations

import pytest
import torch

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.ar_draft import ARDrafter
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import VOCAB, build_tiny_target, random_prompt


def ar_drafter(target):
    return ARDrafter(build_tiny_target(99), seed=3)


def diffusion_drafter(target):
    torch.manual_seed(5)
    model = DiffusionDrafterModel(DiffusionDrafterConfig(VOCAB, hidden_size=32, num_layers=1, num_heads=4,
                                                         ctx_window=16, max_canvas=8))
    return DiffusionDrafter(model, canvas_len=4, denoise_steps=2, reconcile=ReconcilePolicy(mode="reconcile"))


def sample(engine, prompt, seed):
    config = DecodeConfig(mode="exact_sampling", max_new_tokens=24, draft_len=3, canvas_len=4, seed=seed)
    result = engine.generate(prompt, config)
    return result.token_ids, [(step.n_draft_proposed, step.n_draft_accepted) for step in result.rounds]


@pytest.mark.parametrize("make_drafter", [ar_drafter, diffusion_drafter], ids=["ar", "diffusion"])
def test_explicit_seed_repeats_sampled_requests(make_drafter):
    target = build_tiny_target()
    prompt = random_prompt(11, 6)
    with DartEngine(target, make_drafter(target)) as engine:
        first = sample(engine, prompt, seed=7)
        sample(engine, random_prompt(12, 5), seed=8)
        assert sample(engine, prompt, seed=7) == first
    with DartEngine(target, make_drafter(target)) as fresh:
        assert sample(fresh, prompt, seed=7) == first


def test_unseeded_requests_continue_the_drafter_stream():
    target = build_tiny_target()
    drafter = ar_drafter(target)
    with DartEngine(target, drafter) as engine:
        sample(engine, random_prompt(11, 6), seed=None)
        state = drafter.gen.get_state().clone()
        sample(engine, random_prompt(11, 6), seed=None)
        assert not torch.equal(drafter.gen.get_state(), state)


@pytest.mark.parametrize("prompt, budget", [([5, 6], 0), ([5, VOCAB], 4)], ids=["zero-output", "rejected"])
def test_requests_without_execution_leave_the_drafter_stream(prompt, budget):
    target = build_tiny_target()
    drafter = ar_drafter(target)
    state = drafter.gen.get_state().clone()
    with DartEngine(target, drafter) as engine:
        try:
            engine.generate(prompt, DecodeConfig(mode="exact_sampling", max_new_tokens=budget, seed=7))
        except ValueError:
            pass
    assert torch.equal(drafter.gen.get_state(), state)
