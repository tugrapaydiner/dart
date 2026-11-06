from __future__ import annotations

import copy

import pytest
import torch

from dart.models.replay import ReplayCache
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_checks import RoundCase, _cache_pairs, check_verifier_state
from dart.verification.state_sequences import StateOperation as Op
from dart.verification.state_sequences import check_state_sequence
from tests.test_replay_predictions import execution as execution

cpu_static = pytest.mark.parametrize("execution", ["cpu_static"], indirect=True)


def round_case(reference, rejection=None):
    prefix = (5, 6, 7)
    oracle = hf_decoder(reference)(GenerationCase("oracle", prefix, 6))
    drafts = list(oracle[1:5])
    if rejection is not None:
        drafts[rejection] = (drafts[rejection] + 1) % reference.config.vocab_size
    return RoundCase("replay", prefix, oracle[0], tuple(drafts), (-1, 0, 1, 2))


@pytest.mark.parametrize("capacity", [4, 8, 16])
def test_reader_exposes_only_the_valid_prefix_without_copying(capacity):
    storage = torch.arange(2 * 2 * 1 * 3 * capacity * 4).reshape(2, 2, 1, 3, capacity, 4).float()
    cache = ReplayCache(storage, 3, object())
    saved = storage.clone()
    pairs = _cache_pairs(cache)
    assert len(pairs) == 2
    for layer, pair in enumerate(pairs):
        for kind, tensor in enumerate(pair):
            assert tensor.shape == (1, 3, 3, 4)
            assert tensor.data_ptr() == storage[layer, kind].data_ptr()
            torch.testing.assert_close(tensor, storage[layer, kind, ..., :3, :], atol=0, rtol=0)
    cache.crop(-3)
    assert all(key.shape[-2] == value.shape[-2] == 0 for key, value in _cache_pairs(cache))
    torch.testing.assert_close(storage, saved, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [None, (2, 2, 1, 8, 4), (2, 3, 1, 3, 8, 4),
                                  (2, 2, 2, 3, 8, 4), (0, 2, 1, 3, 8, 4),
                                  (2, 2, 1, 0, 8, 4), (2, 2, 1, 3, 0, 4), (2, 2, 1, 3, 8, 0)])
def test_reader_rejects_malformed_replay_storage(shape):
    cache = ReplayCache(None if shape is None else torch.empty(shape), 0, object())
    with pytest.raises(ValueError, match="storage"):
        _cache_pairs(cache)


@pytest.mark.parametrize("length", [-1, 9, True, 3.0])
def test_reader_rejects_invalid_prefix_length(length):
    cache = ReplayCache(torch.empty(2, 2, 1, 3, 8, 4), length, object())
    with pytest.raises(ValueError, match="length"):
        _cache_pairs(cache)


@cpu_static
@pytest.mark.parametrize("rejection", [0, 1, 2, 3, None])
def test_real_static_rounds_match_independent_hf_cache(execution, rejection):
    base, target = execution
    reference = copy.deepcopy(base.model)
    case = round_case(reference, rejection)
    result = check_verifier_state(target, reference, case, verifier="block")
    assert result.passed, result.failures
    assert result.actual == result.expected
    assert result.cache_length == result.expected_cache_length


@cpu_static
def test_real_static_interleaving_crop_to_empty_and_reuse(execution):
    base, target = execution
    reference = copy.deepcopy(base.model)
    operations = [Op("a", "prefill", (5, 6, 7)), Op("b", "prefill", (17, 19)),
                  Op("a", "forward", (11, 12)), Op("b", "forward", (23,)),
                  Op("a", "crop", length=0), Op("b", "block", (29, 31)),
                  Op("a", "forward", (13,)), Op("a", "block", (17, 19)),
                  Op("b", "crop", length=1), Op("b", "forward", (23, 29)),
                  Op("a", "release"), Op("a", "prefill", (31,)), Op("a", "forward", (32,))]
    report = check_state_sequence(target, reference, operations)
    assert report.passed, report
    assert len(report.steps) == len(operations)


@cpu_static
def test_unused_storage_does_not_cause_false_failures(execution, monkeypatch):
    base, target = execution
    reference = copy.deepcopy(base.model)
    forward = target.forward_block

    def poison_tail(state, tokens):
        output = forward(state, tokens)
        state.cache.storage[..., state.seq_len:, :].fill_(float("nan"))
        return output

    monkeypatch.setattr(target, "forward_block", poison_tail)
    result = check_verifier_state(target, reference, round_case(reference, 0), verifier="block")
    assert result.passed, result.failures


@cpu_static
def test_corrupted_active_cache_is_detected_with_matching_output(execution, monkeypatch):
    base, target = execution
    reference = copy.deepcopy(base.model)
    forward = target.forward_block

    def corrupt(state, tokens):
        output = forward(state, tokens)
        state.cache.storage[0, 0, ..., 0, :].add_(0.25)
        return output

    monkeypatch.setattr(target, "forward_block", corrupt)
    result = check_verifier_state(target, reference, round_case(reference), verifier="block")
    assert result.actual == result.expected and not result.passed
    assert "cache_values" in {failure.kind for failure in result.failures}


@cpu_static
def test_idle_replay_request_corruption_is_detected(execution, monkeypatch):
    base, target = execution
    reference = copy.deepcopy(base.model)
    prefill, forward = target.prefill, target.forward_block
    states = {}

    def remember(tokens):
        state, logits = prefill(tokens)
        states[tokens[0]] = state
        return state, logits

    def leak(state, tokens):
        output = forward(state, tokens)
        if state is states[17]:
            states[5].cache.storage[0, 1, ..., 0, :].add_(0.25)
        return output

    monkeypatch.setattr(target, "prefill", remember)
    monkeypatch.setattr(target, "forward_block", leak)
    report = check_state_sequence(target, reference, [Op("a", "prefill", (5, 6, 7)),
                                                     Op("b", "prefill", (17, 19)),
                                                     Op("a", "forward", (11,)), Op("b", "forward", (23,))])
    assert all(step.passed for step in report.steps[:-1])
    assert {(item.request, item.failure.kind) for item in report.steps[-1].failures} == {("a", "cache_values")}


@cpu_static
def test_malformed_replay_storage_is_a_check_failure(execution, monkeypatch):
    base, target = execution
    reference = copy.deepcopy(base.model)
    forward = target.forward_block

    def malformed(state, tokens):
        output = forward(state, tokens)
        state.cache.storage = state.cache.storage[:, :1]
        return output

    monkeypatch.setattr(target, "forward_block", malformed)
    report = check_state_sequence(target, reference, [Op("a", "prefill", (5, 6)), Op("a", "forward", (7,))])
    assert not report.passed
    assert any(item.failure.kind == "cache_error" and "storage" in item.failure.detail
               for item in report.steps[-1].failures)
