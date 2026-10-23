from __future__ import annotations

import pytest
import torch

from dart.drafters.ar_draft import ARDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.sampling import speculative_accept_step
from tests.conftest import build_tiny_target, random_prompt
from tests.test_exactness import ConstantDrafter

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
DEVICES = ["cpu", pytest.param("cuda:0", marks=CUDA_REQUIRED)]


@pytest.fixture(params=DEVICES)
def sampling_target(request):
    target = build_tiny_target()
    target = TargetModel(target.model, device=request.param)
    target.eos_token_ids = frozenset()  # Exercise every round up to the budget.
    return target


@pytest.mark.parametrize("lane", ["target_only", "constant", "ar"])
def test_sampling_runs_and_repeats_on_each_device(sampling_target, lane):
    prompt = random_prompt(301, 8)
    config = DecodeConfig(
        mode="exact_sampling", max_new_tokens=12, draft_len=3, seed=7,
    )

    def run():
        if lane == "ar":
            drafter = ARDrafter(sampling_target, seed=11)
        elif lane == "constant":
            drafter = ConstantDrafter()
        else:
            drafter = None
        with DartEngine(sampling_target, drafter) as engine:
            return engine.generate(prompt, config)

    first, second = run(), run()
    assert first.token_ids == second.token_ids
    assert len(first.token_ids) == config.max_new_tokens
    assert first.rounds
    assert first.metadata["distribution_exact"] is True
    assert first.metadata["exact_greedy_equivalent"] is None
    assert first.metadata["greedy_verified"] is False
    if lane == "ar":
        assert first.metadata["acceptance_rate"] > 0.9


def test_ar_proposals_repeat_with_the_same_seed(sampling_target):
    prompt = random_prompt(302, 8)
    first = ARDrafter(sampling_target, seed=13).propose_with_probs(prompt, 4)
    second = ARDrafter(sampling_target, seed=13).propose_with_probs(prompt, 4)
    assert first[0] == second[0]
    assert len(first[0]) == 4
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[1].sum(-1), torch.ones_like(first[1][:, 0]))


@pytest.mark.parametrize("target_device", DEVICES)
@pytest.mark.parametrize("draft_device", DEVICES)
def test_acceptance_and_resampling_across_devices(target_device, draft_device):
    generator = torch.Generator(device=target_device).manual_seed(17)
    p = torch.tensor([0.0, 1.0], device=target_device)
    q = torch.tensor([1.0, 0.0], device=draft_device)
    token, accepted = speculative_accept_step(p, q, 0, generator)
    assert (token, accepted) == (1, False)
    token, accepted = speculative_accept_step(p, p.to(draft_device), 1, generator)
    assert (token, accepted) == (1, True)


@pytest.mark.parametrize("target_device", DEVICES)
@pytest.mark.parametrize("draft_device", DEVICES)
def test_sampling_with_a_separate_draft_device(target_device, draft_device):
    target = TargetModel(build_tiny_target().model, device=target_device)
    draft = TargetModel(build_tiny_target(seed=42).model, device=draft_device)
    target.eos_token_ids = draft.eos_token_ids = frozenset()
    with DartEngine(target, ARDrafter(draft, seed=19)) as engine:
        output = engine.generate(
            random_prompt(303, 8),
            DecodeConfig(mode="exact_sampling", max_new_tokens=12, draft_len=3, seed=23),
        )
    assert len(output.token_ids) == 12
    assert output.rounds
    assert output.metadata["distribution_exact"] is True


@pytest.mark.parametrize("target_device,proposal_device", [
    ("cpu", None),
    ("cpu", "cpu"),
    pytest.param("cuda:0", None, marks=CUDA_REQUIRED),
    pytest.param("cuda:0", "cuda:0", marks=CUDA_REQUIRED),
    pytest.param("cuda:0", "cpu", marks=CUDA_REQUIRED),
])
def test_sampled_distribution_matches_target_on_each_device(target_device, proposal_device):
    expected = torch.tensor([0.0, 0.15, 0.35, 0.5])
    p = expected.to(target_device)
    q = None
    n = 12_000
    if proposal_device is None:
        drafts = [2] * n
        expected_acceptance = float(expected[2])
    else:
        q = torch.tensor([0.45, 0.4, 0.1, 0.05], device=proposal_device)
        drafts = torch.multinomial(
            q.cpu(), n, replacement=True, generator=torch.Generator().manual_seed(11),
        ).tolist()
        expected_acceptance = float(torch.minimum(expected, q.cpu()).sum())
    generator = torch.Generator(device=target_device).manual_seed(29)
    counts = [0] * len(expected)
    accepted = 0
    for draft in drafts:
        token, keep = speculative_accept_step(p, q, draft, generator)
        counts[token] += 1
        accepted += keep
    empirical = torch.tensor(counts) / n
    assert counts[0] == 0
    assert float((empirical - expected).abs().max()) < 0.025
    assert abs(accepted / n - expected_acceptance) < 0.025
    print(f"{target_device}, q={proposal_device}: p={empirical.tolist()}, acceptance={accepted / n:.4f}")
