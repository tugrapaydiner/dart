from __future__ import annotations

from importlib.metadata import version

import pytest
from packaging.version import Version

from dart.drafters.ar_draft import ARDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target, random_prompt

static_only = pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                                 reason="static execution requires transformers >= 5.16.1")


def short_eager_draft():
    draft = build_tiny_target(99)
    draft.model.config.max_position_embeddings = 16
    return draft


def short_static_draft():
    from dart.models.replay import CudaGraphTarget

    return CudaGraphTarget(build_tiny_target(99), capacity=16, max_block_tokens=2, replay=False)


DRAFTS = [short_eager_draft, pytest.param(short_static_draft, marks=static_only)]


@pytest.fixture
def target():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    return target


@pytest.mark.parametrize("make_draft", DRAFTS, ids=["eager", "static"])
@pytest.mark.parametrize("mode", ["exact", "exact_sampling"])
def test_draft_model_context_limits_proposals(target, make_draft, mode):
    draft = make_draft()
    assert draft.context_limit == 16
    prompt = random_prompt(21, 9)
    config = DecodeConfig(mode=mode, max_new_tokens=20, draft_len=4, seed=3)
    with DartEngine(target, ARDrafter(draft)) as engine:
        result = engine.generate(prompt, config)
    if mode == "exact":
        assert result.token_ids == target.greedy_generate(prompt, 20)
    assert len(result.token_ids) == 20
    assert result.rounds[0].n_draft_proposed > 0
    committed = len(prompt) + 1  # Prefill already committed the first token.
    for step in result.rounds:
        assert step.n_draft_proposed <= max(0, draft.context_limit - committed + 1)
        committed += step.n_committed
    assert result.rounds[-1].n_draft_proposed == 0


@pytest.mark.parametrize("make_draft", DRAFTS, ids=["eager", "static"])
def test_prompt_beyond_draft_context_falls_back_to_target_steps(target, make_draft):
    prompt = random_prompt(22, 20)
    with DartEngine(target, ARDrafter(make_draft())) as engine:
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=6, draft_len=4))
    assert result.token_ids == target.greedy_generate(prompt, 6)
    assert all(step.n_draft_proposed == 0 for step in result.rounds)


@static_only
def test_static_draft_catches_up_on_blocks_longer_than_it_prepared(target):
    draft = short_static_draft()
    drafter = ARDrafter(draft)
    prompt = random_prompt(23, 4)
    assert drafter.propose(prompt, 2)
    committed = prompt + random_prompt(24, 4)
    drafts = drafter.propose(committed, 2)
    assert drafts == ARDrafter(short_eager_draft()).propose(committed, 2)
