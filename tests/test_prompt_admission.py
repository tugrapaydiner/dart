from __future__ import annotations

import pytest
import torch

from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import VOCAB, build_tiny_target, random_prompt

BAD_PROMPTS = [[5, VOCAB], [VOCAB + 40, 7], [-1, 5], torch.tensor([5, VOCAB])]


def count_forwards(target, monkeypatch):
    calls = []
    forward = target.model.forward
    monkeypatch.setattr(target.model, "forward", lambda *args, **kwargs: calls.append(1) or forward(*args, **kwargs))
    return calls


@pytest.mark.parametrize("prompt", BAD_PROMPTS, ids=["vocab", "far", "negative", "tensor"])
@pytest.mark.parametrize("budget", [0, 1, 6])
def test_engine_rejects_out_of_vocab_prompt_before_execution(prompt, budget, monkeypatch):
    target = build_tiny_target()
    calls = count_forwards(target, monkeypatch)
    with DartEngine(target, NGramDrafter()) as engine:
        with pytest.raises(ValueError, match="outside the model vocabulary"):
            engine.generate(prompt, DecodeConfig(max_new_tokens=budget))
        assert calls == []
        good = random_prompt(5, 8)
        assert engine.generate(good, DecodeConfig(max_new_tokens=6)).token_ids == target.greedy_generate(good, 6)


@pytest.mark.parametrize("prompt", BAD_PROMPTS, ids=["vocab", "far", "negative", "tensor"])
def test_greedy_generate_rejects_out_of_vocab_prompt(prompt, monkeypatch):
    target = build_tiny_target()
    calls = count_forwards(target, monkeypatch)
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        target.greedy_generate(prompt, max_new_tokens=4)
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        target.prefill(prompt)
    assert calls == []


def test_prompt_at_vocab_edge_is_accepted():
    target = build_tiny_target()
    prompt = [0, VOCAB - 1, 7]
    with DartEngine(target) as engine:
        assert engine.generate(prompt, DecodeConfig(max_new_tokens=3)).token_ids == target.greedy_generate(prompt, 3)
