from __future__ import annotations

import pytest
import torch

from dart.drafters.ar_draft import ARDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.tree_verify import DraftTree
from tests.conftest import build_tiny_target, random_prompt


@pytest.fixture
def target():
    model = build_tiny_target()
    model.model.config.max_position_embeddings = 16
    model.eos_token_ids = frozenset()
    return model


def fail_forward(*args, **kwargs):
    pytest.fail("Invalid input reached the model")


@pytest.mark.parametrize("entry", ["engine", "baseline"])
@pytest.mark.parametrize("prompt_len,budget", [(8, 9), (16, 1), (17, 0)])
def test_generation_rejects_context_overflow(target, monkeypatch, entry, prompt_len, budget):
    monkeypatch.setattr(target.model, "forward", fail_forward)
    prompt = random_prompt(330, prompt_len)
    with pytest.raises(ValueError, match="context limit"):
        if entry == "baseline":
            target.greedy_generate(prompt, max_new_tokens=budget)
        else:
            with DartEngine(target) as engine:
                engine.generate(prompt, DecodeConfig(max_new_tokens=budget))


@pytest.mark.parametrize("verifier", ["block", "reference", "tree"])
def test_generation_at_context_limit_matches_baseline(target, verifier):
    prompt = random_prompt(331, 8)
    expected = target.greedy_generate(torch.tensor([prompt]), max_new_tokens=8)
    with DartEngine(target, ARDrafter(target), verifier=verifier) as engine:
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=8, draft_len=32))
    assert result.token_ids == expected
    assert len(result.token_ids) == 8


def test_prefill_rejects_context_overflow(target, monkeypatch):
    monkeypatch.setattr(target.model, "forward", fail_forward)
    with pytest.raises(ValueError, match="context limit"):
        target.prefill(random_prompt(332, 17))


def test_block_rejects_context_overflow_without_changing_cache(target, monkeypatch):
    state, _ = target.prefill(random_prompt(333, 8))
    monkeypatch.setattr(target.model, "forward", fail_forward)
    with pytest.raises(ValueError, match="context limit"):
        target.forward_block(state, [10] * 9)
    assert state.seq_len == 8


def test_tree_rejects_context_overflow_without_changing_cache(target, monkeypatch):
    state, _ = target.prefill(random_prompt(334, 8))
    tree = DraftTree.chain([10] * 8)
    monkeypatch.setattr(target.model, "forward", fail_forward)
    with pytest.raises(ValueError, match="context limit"):
        target.forward_tree(state, [11], tree.tokens, tree.parent)
    assert state.seq_len == 8


def test_tree_context_limit_uses_branch_depth(target):
    state, _ = target.prefill(random_prompt(335, 8))
    tree = DraftTree.from_branches([[i, 20, 21] for i in range(10, 15)])
    logits = target.forward_tree(state, [11], tree.tokens, tree.parent)
    assert logits.shape[0] == 1 + len(tree)
    assert state.seq_len == 24  # Packed branches share positions.


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_baseline_rejects_invalid_budget_before_prefill(target, monkeypatch, budget):
    monkeypatch.setattr(target.model, "forward", fail_forward)
    with pytest.raises(ValueError, match="max_new_tokens"):
        target.greedy_generate([10], max_new_tokens=budget)
