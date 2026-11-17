from __future__ import annotations

import pytest
import torch

from dart.drafters.ar_draft import ARDrafter
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.kv.cache import QuantSpec
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    target.eos_token_ids = frozenset()
    reference.generation_config.eos_token_id = None
    reference.config.eos_token_id = None
    yield target, reference
    torch.set_num_threads(previous)


@torch.inference_mode()
def reference_proposals(model, prompt, budget, sampling, generator):
    tokens, rows = [], []
    if not budget:
        return tokens, rows
    result = model(input_ids=torch.tensor([prompt]), use_cache=True)
    for index in range(budget):
        logits = result.logits[0, -1]
        if sampling:
            row = torch.softmax(logits.float() / 0.7, -1)
            token = int(torch.multinomial(row, 1, generator=generator).item())
            rows.append(row)
        else:
            token = int(logits.argmax().item())
        tokens.append(token)
        if index + 1 < budget:
            result = model(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values,
                           use_cache=True)
    return tokens, rows


def propose(drafter, prompt, budget, sampling):
    if sampling:
        return drafter.propose_with_probs(prompt, budget)
    return drafter.propose(prompt, budget), None


def count_forwards(target, monkeypatch):
    calls = []
    original = target.model.forward

    def forward(*args, **kwargs):
        calls.append(kwargs['input_ids'].shape[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(target.model, 'forward', forward)
    return calls


@pytest.mark.parametrize('sampling', [False, True])
@pytest.mark.parametrize('budget', [0, 1, 4])
def test_drafting_skips_the_unused_final_forward(models, monkeypatch, sampling, budget):
    target, reference = models
    prompt = [5, 6, 7]
    generator = torch.Generator().manual_seed(23)
    expected, rows = reference_proposals(reference, prompt, budget, sampling, generator)
    drafter = ARDrafter(target, sample_temperature=0.7, seed=23)
    calls = count_forwards(target, monkeypatch)
    actual, probabilities = propose(drafter, prompt, budget, sampling)
    assert actual == expected
    assert calls == ([len(prompt)] + [1] * (budget - 1) if budget else [])
    assert torch.equal(drafter.gen.get_state(), generator.get_state())
    if sampling and budget:
        torch.testing.assert_close(probabilities, torch.stack(rows), atol=2e-6, rtol=2e-5)
    else:
        assert probabilities is None
    if budget:
        assert drafter._state.seq_len == len(prompt) and drafter._processed == prompt
        with torch.inference_mode():
            cache = reference(input_ids=torch.tensor([prompt]), use_cache=True).past_key_values
        for actual_layer, expected_layer in zip(drafter._state.cache.layers, cache.layers, strict=True):
            torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=2e-6, rtol=2e-5)
    else:
        assert drafter._state is None and drafter._processed == []


@pytest.mark.parametrize('sampling', [False, True])
@pytest.mark.parametrize('eos_offset', [1, 2, 3, 4])
def test_draft_eos_never_runs_an_extra_forward(models, monkeypatch, sampling, eos_offset):
    target, _ = models
    prompt = [5, 6, 7]
    drafter = ARDrafter(target, sample_temperature=0.7, seed=23)
    tokens, probabilities = propose(drafter, prompt, 4, sampling)
    eos = tokens[eos_offset - 1]
    first_eos = tokens.index(eos) + 1
    assert first_eos == eos_offset
    drafter.reset()
    drafter.gen.manual_seed(23)
    target.eos_token_ids = frozenset({eos})
    calls = count_forwards(target, monkeypatch)
    actual, rows = propose(drafter, prompt, 4, sampling)
    assert actual == tokens[:first_eos]
    assert calls == [len(prompt)] + [1] * (first_eos - 1)
    assert drafter._state.seq_len == len(prompt)
    if sampling:
        torch.testing.assert_close(rows, probabilities[:first_eos], atol=0, rtol=0)


@pytest.mark.parametrize('sampling', [False, True])
def test_repeat_extension_and_rejected_drafts_keep_reference_proposals(models, sampling):
    target, reference = models
    drafter = ARDrafter(target, sample_temperature=0.7, seed=23)
    generator = torch.Generator().manual_seed(23)
    prompt = [5, 6, 7]
    for context in [prompt, prompt, prompt + [13], prompt + [13, 17, 19], [29, 31, 37]]:
        expected, rows = reference_proposals(reference, context, 3, sampling, generator)
        actual, probabilities = propose(drafter, context, 3, sampling)
        assert actual == expected
        assert drafter._state.seq_len == len(context) and drafter._processed == context
        if sampling:
            torch.testing.assert_close(probabilities, torch.stack(rows), atol=2e-6, rtol=2e-5)
    assert torch.equal(drafter.gen.get_state(), generator.get_state())


@pytest.mark.parametrize('budget', [1, 4])
def test_compressed_drafting_also_skips_the_unused_forward(models, monkeypatch, budget):
    target, _ = models
    drafter = CompressedKVDrafter(target, QuantSpec(bits=4, sink_len=1, window_len=2))
    prompt = [5, 6, 7, 8, 9, 10]
    calls = count_forwards(target, monkeypatch)
    assert len(drafter.propose(prompt, budget)) == budget
    assert calls == [len(prompt)] + [1] * (budget - 1)
    assert drafter._state.seq_len == len(prompt)


@pytest.mark.parametrize('compressed', [False, True])
def test_generation_after_shortened_drafting_matches_independent_hf(models, compressed):
    target, reference = models
    drafter = (CompressedKVDrafter(target, QuantSpec(bits=4, sink_len=1, window_len=2))
               if compressed else ARDrafter(target))
    with DartEngine(target, drafter) as engine:
        for prompt in ([5, 6, 7, 8], [13, 17, 19], [5, 6, 7, 8]):
            expected = hf_decoder(reference)(GenerationCase('draft', tuple(prompt), 9))
            assert engine.generate(prompt, DecodeConfig(max_new_tokens=9, draft_len=3)).token_ids == expected
