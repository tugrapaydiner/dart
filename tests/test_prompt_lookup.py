from __future__ import annotations

import copy

import pytest

from benchmarks.run_bench import build_hf_method
from dart.models.target import TargetModel
from tests.conftest import build_tiny_target, random_prompt
from tests.test_sampling_devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_lookup_uses_drafts_and_preserves_model_settings(device):
    target = TargetModel(build_tiny_target().model, device=device, record_hidden=True)
    target.model.lm_head.weight.zero_()
    target.model.generation_config.repetition_penalty = 3.0
    before = copy.deepcopy(target.model.generation_config)
    forwards = []

    def record(module, args, kwargs, output):
        assert output.hidden_states is None
        forwards.append(kwargs["input_ids"].shape[1])

    hook = target.model.register_forward_hook(record, with_kwargs=True)
    try:
        greedy = build_hf_method(target, 12, record_hidden=False)([0] * 12)
        greedy_calls = len(forwards)
        forwards.clear()
        lookup = build_hf_method(target, 12, record_hidden=False, prompt_lookup_num_tokens=4)
        first = lookup([0] * 12)
        lookup_calls = len(forwards)
        second = lookup([0] * 12)
    finally:
        hook.remove()
    assert greedy.token_ids == first.token_ids == second.token_ids == [0] * 12
    assert 0 < lookup_calls < greedy_calls == 12
    assert target.model.generation_config == before
    assert target.record_hidden is True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("draft_len", [1, 4])
@pytest.mark.parametrize("budget", [1, 3, 8])
def test_lookup_matches_complete_greedy_output(device, draft_len, budget):
    target = TargetModel(build_tiny_target().model, device=device)
    prompt = random_prompt(370, 12)
    expected = build_hf_method(target, budget)(prompt).token_ids
    actual = build_hf_method(target, budget, prompt_lookup_num_tokens=draft_len)(prompt)
    assert actual.token_ids == expected


@pytest.mark.parametrize("device", DEVICES)
def test_lookup_stops_at_eos_in_a_repeated_prompt(device):
    target = TargetModel(build_tiny_target().model, device=device)
    target.model.lm_head.weight.zero_()
    target.eos_token_ids = frozenset([0, 2])
    prompt = [5, 0] * 6 + [5]
    expected = build_hf_method(target, 8)(prompt).token_ids
    actual = build_hf_method(target, 8, prompt_lookup_num_tokens=4)(prompt)
    assert actual.token_ids == expected == [0]


def test_lookup_rejects_a_prompt_ending_in_eos():
    target = build_tiny_target()
    with pytest.raises(ValueError, match="prompt that does not end in EOS"):
        build_hf_method(target, 8, prompt_lookup_num_tokens=4)([7, 2])


@pytest.mark.parametrize("device", DEVICES)
def test_lookup_keeps_verification_inside_context_limit(device):
    target = TargetModel(build_tiny_target().model, device=device)
    target.model.lm_head.weight.zero_()
    target.model.config.max_position_embeddings = 24
    lengths = []

    def record(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        cached = cache.get_seq_length() if cache is not None else 0
        lengths.append(cached + kwargs["input_ids"].shape[-1])

    hook = target.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        result = build_hf_method(target, 8, prompt_lookup_num_tokens=8)([0] * 9)
    finally:
        hook.remove()
    assert result.token_ids == [0] * 8
    assert lengths and max(lengths) <= 24


def test_lookup_rejects_insufficient_lookahead_before_model_execution(monkeypatch):
    target = build_tiny_target()
    target.model.config.max_position_embeddings = 16

    def unexpected(*args, **kwargs):
        pytest.fail("model ran outside the supported context budget")

    monkeypatch.setattr(target.model, "generate", unexpected)
    with pytest.raises(ValueError, match="context limit"):
        build_hf_method(target, 4, prompt_lookup_num_tokens=8)([0] * 12)


def test_one_token_lookup_needs_no_extra_lookahead():
    target = build_tiny_target()
    target.model.config.max_position_embeddings = 16
    prompt = [0] * 15
    expected = build_hf_method(target, 1)(prompt).token_ids
    assert build_hf_method(target, 1, prompt_lookup_num_tokens=8)(prompt).token_ids == expected


@pytest.mark.parametrize("name", ["prompt_lookup_num_tokens", "max_matching_ngram_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_lookup_rejects_invalid_settings(name, value):
    options = {"prompt_lookup_num_tokens": 4, name: value}
    with pytest.raises(ValueError, match=name):
        build_hf_method(build_tiny_target(), 8, **options)
