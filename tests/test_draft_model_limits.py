from __future__ import annotations

from contextlib import closing
from importlib.metadata import version
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version

from dart.drafters.ar_draft import ARDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.state_checks import _cache_failures, _reference_forward
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


@static_only
@pytest.mark.parametrize("sampling", [False, True])
@pytest.mark.parametrize("prefix_len,extra,changed,chunks", [
    (6, 2, False, [2]), (3, 3, False, [2, 1]), (6, 4, False, [2, 2]),
    (6, 5, False, []), (2, 3, False, []), (6, 3, True, []),
])
@torch.inference_mode()
def test_static_draft_syncs_short_extensions(prefix_len, extra, changed, chunks, sampling):
    prefix = random_prompt(46, prefix_len)
    context = prefix + random_prompt(47, extra)
    reference = short_eager_draft().model
    if changed:
        context[0] = (context[0] + 1) % reference.config.vocab_size
    wanted = _reference_forward(reference, context)
    with closing(short_static_draft()) as target:
        drafter = ARDrafter(target, seed=29)
        drafter.propose(prefix, 1)
        with patch.object(target, "prefill", wraps=target.prefill) as prefill, patch.object(
            target, "forward_block", wraps=target.forward_block
        ) as block:
            if sampling:
                tokens, probs = drafter.propose_with_probs(context, 1)
                row = wanted.logits[0, -1].softmax(-1)
                torch.testing.assert_close(probs[0], row, atol=2e-6, rtol=2e-5)
                expected = torch.multinomial(row, 1, generator=torch.Generator().manual_seed(29)).item()
            else:
                tokens = drafter.propose(context, 1)
                expected = wanted.logits[0, -1].argmax().item()
        assert tokens == [expected]
        assert [len(call.args[1]) for call in block.call_args_list] == chunks
        assert [call.args[0] for call in prefill.call_args_list] == ([] if chunks else [context])
        assert drafter._processed == context
        assert not _cache_failures(drafter._state, wanted.past_key_values, len(context), atol=2e-6, rtol=2e-5)
        following = target.forward_block(drafter._state, tokens)[-1]
        native = _reference_forward(reference, context + tokens)
        torch.testing.assert_close(following, native.logits[0, -1], atol=2e-6, rtol=2e-5)


@static_only
@pytest.mark.parametrize("fail_at", [1, 2])
def test_static_draft_discards_a_failed_append(fail_at):
    prefix = random_prompt(48, 6)
    context = prefix + random_prompt(49, 3)
    with closing(short_static_draft()) as target:
        drafter = ARDrafter(target)
        drafter.propose(prefix, 1)
        forward = target.forward_block
        calls = []

        def interrupt(state, tokens):
            logits = forward(state, tokens)
            calls.append(len(tokens))
            if len(calls) == fail_at:
                raise RuntimeError("interrupted after cache write")
            return logits

        with patch.object(target, "forward_block", interrupt), pytest.raises(RuntimeError, match="after cache write"):
            drafter.propose(context, 1)
        assert calls == [2, 1][:fail_at]
        assert drafter._state is None and drafter._processed == []
        assert drafter.propose(context, 2) == ARDrafter(short_eager_draft()).propose(context, 2)
