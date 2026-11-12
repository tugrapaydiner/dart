from __future__ import annotations

import pytest
import torch

from dart.models.target import TargetModel
from dart.verification.state_checks import (
    _cache_failures,
    _cache_pairs,
    _reference_forward,
)
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        target, reference = _synthetic_models(17)
        target.record_hidden = True
        yield target, reference
    finally:
        torch.set_num_threads(previous)


def unexpected(*args, **kwargs):
    pytest.fail('invalid tokens reached the model')


@pytest.mark.parametrize('tokens', [
    [-1], [67], [2**80], (5, 67), [[5, 67]], [67.9],
    torch.tensor([-1]), torch.tensor([[5, 67]]), torch.tensor([float('nan')]),
])
@pytest.mark.parametrize('entry', ['block', 'all_logits'])
def test_bad_decode_ids_reject_before_model_and_preserve_request(models, monkeypatch, tokens, entry):
    target, reference = models
    state, _ = target.prefill([5, 6, 7])
    hidden = target.last_hidden_block
    before = [(k.clone(), v.clone()) for k, v in _cache_pairs(state.cache)]
    with monkeypatch.context() as patch:
        patch.setattr(target.model, 'forward', unexpected)
        with pytest.raises(ValueError, match='outside the model vocabulary'):
            if entry == 'block':
                target.forward_block(state, tokens)
            else:
                target.all_logits(tokens)
    assert not state._failed and state.seq_len == 3
    assert target.last_hidden_block is hidden
    for got, expected in zip(_cache_pairs(state.cache), before):
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    actual = target.forward_block(state, [0, 66])
    expected = _reference_forward(reference, [5, 6, 7, 0, 66])
    torch.testing.assert_close(actual, expected.logits[0, -2:].float(), atol=2e-6, rtol=2e-5)
    assert torch.equal(actual.argmax(-1), expected.logits[0, -2:].argmax(-1))
    assert not _cache_failures(state, expected.past_key_values, 5, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('bad', [-1, 67, 2**80])
@pytest.mark.parametrize('position', ['pending', 'draft'])
def test_tree_bad_ids_reject_before_model_and_allow_reuse(models, monkeypatch, bad, position):
    target, reference = models
    state, _ = target.prefill([5, 6, 7])
    hidden = target.last_hidden_block
    pending = [bad] if position == 'pending' else [11]
    drafts = [bad, 13] if position == 'draft' else [11, 13]
    with monkeypatch.context() as patch:
        patch.setattr(target.model, 'forward', unexpected)
        with pytest.raises(ValueError, match='outside the model vocabulary'):
            target.forward_tree(state, pending, drafts, [-1, -1])
    assert not state._failed and state.seq_len == 3 and target.last_hidden_block is hidden
    expected = _reference_forward(reference, [5, 6, 7])
    assert not _cache_failures(state, expected.past_key_values, 3, atol=2e-6, rtol=2e-5)
    actual = target.forward_block(state, [17])
    wanted = _reference_forward(reference, [5, 6, 7, 17])
    torch.testing.assert_close(actual, wanted.logits[0, -1:].float(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('tokens', [[], [[]], torch.empty(0, dtype=torch.long)])
@pytest.mark.parametrize('entry', ['block', 'all_logits'])
def test_empty_decode_blocks_reject_before_model(models, monkeypatch, tokens, entry):
    target, _ = models
    state, _ = target.prefill([5, 6, 7])
    monkeypatch.setattr(target.model, 'forward', unexpected)
    with pytest.raises(ValueError, match='at least one token'):
        if entry == 'block':
            target.forward_block(state, tokens)
        else:
            target.all_logits(tokens)
    assert not state._failed and state.seq_len == 3


@pytest.mark.parametrize('tokens', [[-1], [67], (0, 67), [2**80]])
def test_python_ids_reject_before_allocation(models, monkeypatch, tokens):
    target, _ = models
    monkeypatch.setattr(torch, 'tensor', lambda *a, **k: pytest.fail('allocated invalid token IDs'))
    with pytest.raises(ValueError, match='outside the model vocabulary'):
        target._ids(tokens)


@pytest.mark.parametrize('tokens', [[0], [0, 66], (0, 5, 66)])
def test_valid_integer_lists_do_not_need_tensor_range_checks(models, monkeypatch, tokens):
    target, _ = models
    monkeypatch.setattr(torch, 'aminmax', lambda *a, **k: pytest.fail('reduced Python integer IDs'))
    actual = target._ids(tokens)
    assert actual.dtype == torch.long and actual.device.type == 'cpu'
    assert actual.tolist() == [list(tokens)]


@pytest.mark.parametrize('tokens, expected', [
    ([0.9, 66.9], [[0, 66]]), ([True, False], [[1, 0]]), ([[0, 66]], [[0, 66]]),
    (torch.tensor([0, 66]), [[0, 66]]), (torch.tensor([[0.9, 66.9]]), [[0, 66]]),
])
def test_supported_inputs_keep_long_conversion_and_device(models, tokens, expected):
    target, _ = models
    with torch.device('meta'):
        actual = target._ids(tokens)
    assert actual.tolist() == expected and actual.dtype == torch.long and actual.device.type == 'cpu'


@pytest.mark.parametrize('tokens', [[[1], [2]], [[[1]]], torch.tensor(5)])
def test_bad_shapes_reject_before_model(models, monkeypatch, tokens):
    target, _ = models
    state, _ = target.prefill([5, 6, 7])
    monkeypatch.setattr(target.model, 'forward', unexpected)
    with pytest.raises(ValueError, match='expected'):
        target.forward_block(state, tokens)
    assert not state._failed and state.seq_len == 3


def test_tensor_validation_precedes_device_transfer(models, monkeypatch):
    target, _ = models
    original = torch.Tensor.to

    def no_transfer(tensor, *args, **kwargs):
        if 'device' in kwargs:
            pytest.fail('invalid tokens reached device transfer')
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', no_transfer)
    with pytest.raises(ValueError, match='outside the model vocabulary'):
        TargetModel._ids(target, torch.tensor([67]))
