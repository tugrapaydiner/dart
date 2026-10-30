from __future__ import annotations

import pytest
import torch

from dart.verification.state_checks import _cache_pairs, _reference_forward
from dart.verification.synthetic import _synthetic_models
from tests.test_replay_predictions import execution as execution


class Cancelled(BaseException):
    pass


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield _synthetic_models(17)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('operation', ['crop', 'contiguous', 'gather'])
@pytest.mark.parametrize('error_type', [RuntimeError, Cancelled])
def test_partial_cache_mutation_blocks_reuse_but_other_requests_recover(models, monkeypatch, operation, error_type):
    target, reference = models
    target.record_hidden = True
    state, _ = target.prefill([5, 6, 7, 8, 9])
    other, _ = target.prefill([13, 17, 19])
    error = error_type('interrupted cache update')

    with monkeypatch.context() as fault:
        if operation == 'gather':
            select, calls = torch.Tensor.index_select, []

            def fail_later(tensor, *args, **kwargs):
                calls.append(True)
                if len(calls) == 3:
                    raise error
                return select(tensor, *args, **kwargs)

            fault.setattr(torch.Tensor, 'index_select', fail_later)
        else:
            second_key = _cache_pairs(state.cache)[1][0]
            getitem = torch.Tensor.__getitem__

            def fail_later(tensor, index):
                if tensor is second_key:
                    raise error
                return getitem(tensor, index)

            fault.setattr(torch.Tensor, '__getitem__', fail_later)
        with pytest.raises(error_type) as caught:
            if operation == 'crop':
                target.crop(state, 2)
            else:
                target.gather_tree_tail(state, 1, [2] if operation == 'gather' else [0])
        assert caught.value is error
    assert [key.shape[-2] for key, value in _cache_pairs(state.cache)] == [2, 5]
    before = [(k.clone(), v.clone()) for k, v in _cache_pairs(state.cache)]
    with monkeypatch.context() as guard:
        guard.setattr(target.model, 'forward', lambda *a, **k: pytest.fail('damaged cache reached model execution'))
        for reuse in (
            lambda: target.forward_block(state, [23]),
            lambda: target.forward_tree(state, [23], [29], [-1]),
            lambda: target.crop(state, 0),
            lambda: target.gather_tree_tail(state, 1, []),
        ):
            with pytest.raises(RuntimeError, match='cache is invalid'):
                reuse()
    assert target.last_hidden_block is None
    for pair, wanted in zip(_cache_pairs(state.cache), before):
        for got, old in zip(pair, wanted):
            torch.testing.assert_close(got, old, atol=0, rtol=0)
    expected = _reference_forward(reference, (13, 17, 19, 31)).logits[0, -1]
    torch.testing.assert_close(target.forward_block(other, [31])[-1], expected, atol=2e-6, rtol=2e-5)
    fresh, _ = target.prefill([5, 6, 7])
    expected = _reference_forward(reference, (5, 6, 7, 37)).logits[0, -1]
    torch.testing.assert_close(target.forward_block(fresh, [37])[-1], expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('operation,args', [
    ('crop', (-1,)), ('crop', (4,)), ('crop', (1.5,)), ('crop', (True,)),
    ('gather', (-1, [])), ('gather', (4, [])), ('gather', (1.5, [])), ('gather', (True, [])),
    ('gather', (1, [-1])), ('gather', (1, [2])), ('gather', (1, [1.0])), ('gather', (1, [True])),
    ('gather', (1, None)), ('gather', (1, '')),
])
def test_bad_mutation_arguments_are_rejected_without_changing_live_cache(models, monkeypatch, operation, args):
    target, reference = models
    target.record_hidden = True
    state, _ = target.prefill([5, 6, 7])
    before = [(k.clone(), v.clone()) for k, v in _cache_pairs(state.cache)]
    hidden = target.last_hidden_block
    with monkeypatch.context() as guard:
        guard.setattr(torch, 'tensor', lambda *a, **k: pytest.fail('bad arguments reached index allocation'))
        with pytest.raises(ValueError):
            if operation == 'crop':
                target.crop(state, *args)
            else:
                target.gather_tree_tail(state, *args)
    assert target.last_hidden_block is hidden
    for pair, wanted in zip(_cache_pairs(state.cache), before):
        for got, old in zip(pair, wanted):
            torch.testing.assert_close(got, old, atol=0, rtol=0)
    expected = _reference_forward(reference, (5, 6, 7, 11)).logits[0, -1]
    torch.testing.assert_close(target.forward_block(state, [11])[-1], expected, atol=2e-6, rtol=2e-5)


def test_index_allocation_failure_leaves_the_unmodified_cache_usable(models, monkeypatch):
    target, reference = models
    state, _ = target.prefill([5, 6, 7])
    error = RuntimeError('index allocation failed')

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as fault:
        fault.setattr(torch, 'tensor', fail)
        with pytest.raises(RuntimeError) as caught:
            target.gather_tree_tail(state, 1, [1])
        assert caught.value is error
    assert state.seq_len == 3
    expected = _reference_forward(reference, (5, 6, 7, 11)).logits[0, -1]
    torch.testing.assert_close(target.forward_block(state, [11])[-1], expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('execution', ['cpu_static'], indirect=True)
def test_failed_static_crop_cannot_be_reused_for_predictions(execution, monkeypatch):
    base, target = execution
    state, _ = target.prefill([5, 6, 7])
    target.forward_predictions(state, [11])
    crop = state.cache.crop
    error = RuntimeError('crop interrupted')

    def fail_after_crop(amount):
        crop(amount)
        raise error

    with monkeypatch.context() as fault:
        fault.setattr(state.cache, 'crop', fail_after_crop)
        with pytest.raises(RuntimeError) as caught:
            target.crop(state, 2)
        assert caught.value is error
    with pytest.raises(RuntimeError, match='cache is invalid'):
        target.forward_predictions(state, [13])
    assert target.greedy_generate([17, 19, 23], 4) == base.greedy_generate([17, 19, 23], 4)
