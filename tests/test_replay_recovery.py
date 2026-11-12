import pytest
import torch

from dart.verification.state_checks import _cache_pairs
from tests.test_replay_prefix import forward
from tests.test_replay_prefix import pair as pair


class Cancelled(BaseException):
    pass


def check_cache(state, reference):
    assert state.seq_len == reference.seq_len
    actual_pairs, expected_pairs = _cache_pairs(state.cache), _cache_pairs(reference.cache)
    assert len(actual_pairs) == len(expected_pairs)
    for actual, expected in zip(actual_pairs, expected_pairs):
        for got, wanted in zip(actual, expected):
            torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-5)


def observe_resets(monkeypatch, target):
    resets = []
    original = torch.Tensor.zero_

    def observed(tensor, *args, **kwargs):
        if tensor is target._workspace.storage:
            resets.append(True)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'zero_', observed)
    return resets


def inject_failure(monkeypatch, target, state, mode, error_type, region):
    block = target._greedy_blocks[4] if mode == 'chunk' else target._blocks[1]
    run = block.run

    def fail():
        run()
        start = 32 if region == 'unused' else 0
        target._workspace.storage[..., start:, :].fill_(float('nan'))
        raise error_type('interrupted replay')

    with monkeypatch.context() as fault:
        fault.setattr(block, 'run', fail)
        with pytest.raises(error_type, match='interrupted replay'):
            if mode == 'chunk':
                target.forward_greedy(state, 13, 4)
            elif mode == 'predictions':
                target.forward_predictions(state, [13])
            else:
                target.forward_block(state, [13])


@pytest.mark.parametrize('mode', ['logits', 'predictions', 'chunk'])
@pytest.mark.parametrize('other_request', [False, True])
@pytest.mark.parametrize('region', ['unused', 'whole'])
@pytest.mark.parametrize('error_type', [RuntimeError, Cancelled])
def test_failed_round_cannot_poison_next_request(pair, monkeypatch, mode, other_request, region, error_type):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    forward(base, target, ar, a, 'predictions', 11)
    b, _ = target.prefill([2, 4, 6, 8, 10, 12])
    br, _ = base.prefill([2, 4, 6, 8, 10, 12])
    resets = observe_resets(monkeypatch, target)
    inject_failure(monkeypatch, target, a, mode, error_type, region)
    assert not a._failed and a.seq_len == 4
    check_cache(a, ar)
    check_cache(b, br)
    assert not resets
    actual, expected = (b, br) if other_request else (a, ar)
    token = forward(base, target, expected, actual, '4' if mode == 'chunk' else mode, 17)
    forward(base, target, expected, actual, 'logits', token)
    check_cache(a, ar)
    check_cache(b, br)
    assert resets == [True]


@pytest.mark.parametrize('error_type', [RuntimeError, Cancelled])
def test_failed_reset_stops_model_work_and_is_retried(pair, monkeypatch, error_type):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    forward(base, target, ar, a, 'predictions', 11)
    inject_failure(monkeypatch, target, a, 'predictions', RuntimeError, 'whole')
    calls, zero = [], torch.Tensor.zero_

    def fail(tensor, *args, **kwargs):
        if tensor is target._workspace.storage:
            calls.append(True)
            raise error_type('reset interrupted')
        return zero(tensor, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(torch.Tensor, 'zero_', fail)
        fault.setattr(target._blocks[1], 'run', lambda: pytest.fail('model ran after failed reset'))
        with pytest.raises(error_type, match='reset interrupted'):
            target.forward_predictions(a, [17])
    assert calls == [True] and a.seq_len == 4 and not a._failed
    resets = observe_resets(monkeypatch, target)
    forward(base, target, ar, a, 'predictions', 17)
    check_cache(a, ar)
    assert resets == [True]


def test_invalid_and_zero_budget_calls_do_not_reset_dirty_workspace(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    forward(base, target, ar, a, 'predictions', 11)
    inject_failure(monkeypatch, target, a, 'logits', RuntimeError, 'unused')
    resets = observe_resets(monkeypatch, target)
    with pytest.raises(ValueError, match='vocabulary'):
        target.forward_predictions(a, [97])
    with pytest.raises(ValueError, match='not prepared'):
        target.forward_greedy(a, 17, 3)
    assert target.greedy_generate([19, 23], 0) == []
    assert not resets and a.seq_len == 4
    forward(base, target, ar, a, 'logits', 17)
    check_cache(a, ar)
    assert resets == [True]


def test_successful_calls_and_validation_errors_never_clear_workspace(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    b, _ = target.prefill([2, 4])
    br, _ = base.prefill([2, 4])
    resets = observe_resets(monkeypatch, target)
    token = forward(base, target, ar, a, 'predictions', 11)
    with pytest.raises(ValueError, match='not prepared'):
        target.forward_greedy(a, token, 3)
    forward(base, target, br, b, '4', 17)
    forward(base, target, ar, a, 'logits', token)
    target.crop(a, 1)
    base.crop(ar, 1)
    forward(base, target, ar, a, '2', 19)
    check_cache(a, ar)
    check_cache(b, br)
    assert not resets
