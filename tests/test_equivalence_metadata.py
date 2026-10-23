from __future__ import annotations

import pytest

from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target


@pytest.mark.parametrize("mode", ["exact", "exact_sampling", "approx"])
@pytest.mark.parametrize("early_eos", [False, True])
def test_engine_distinguishes_verification_from_reference_comparison(monkeypatch, mode, early_eos):
    target = build_tiny_target()
    target.eos_token_ids = frozenset(range(target.model.config.vocab_size)) if early_eos else frozenset()
    prompt = [7, 8, 9] * 2
    expected = target.greedy_generate(prompt, 8)

    def unexpected_reference(*args, **kwargs):
        pytest.fail("the engine ran an extra reference generation")

    monkeypatch.setattr(target, "greedy_generate", unexpected_reference)
    monkeypatch.setattr(target.model, "generate", unexpected_reference)
    with DartEngine(target, NGramDrafter()) as engine:
        output = engine.generate(prompt, DecodeConfig(mode=mode, max_new_tokens=8, verify_every=1))
    assert output.metadata["exact_greedy_equivalent"] is None
    assert output.metadata["greedy_verified"] is (mode != "exact_sampling")
    assert output.metadata["finished_eos"] is early_eos
    if mode != "exact_sampling":
        assert output.token_ids == expected
