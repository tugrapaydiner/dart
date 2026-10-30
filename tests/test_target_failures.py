from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from dart.models.target import TargetState
from dart.verification.state_checks import _cache_pairs, _reference_forward
from dart.verification.synthetic import _synthetic_models
from tests.test_replay_predictions import execution as execution


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield _synthetic_models(17)
    torch.set_num_threads(previous)


def fail_in_second_layer(target, state, entry, error):
    def fail(module, args):
        raise error

    hook = target.model.model.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(type(error)) as caught:
            if entry == "block":
                target.forward_block(state, [23, 29])
            else:
                target.forward_tree(state, [23], [29, 31], [-1, -1])
        assert caught.value is error
    finally:
        hook.remove()


@pytest.mark.parametrize("entry", ["block", "tree"])
@pytest.mark.parametrize("reuse", ["block", "tree", "crop", "gather"])
def test_failed_cache_is_rejected_before_any_later_mutation(models, monkeypatch, entry, reuse):
    target, _ = models
    state, _ = target.prefill([5, 6, 7])
    fail_in_second_layer(target, state, entry, RuntimeError("injected layer failure"))
    lengths = [key.shape[-2] for key, value in _cache_pairs(state.cache)]
    assert lengths[0] > lengths[1] == 3
    before = [(key.clone(), value.clone()) for key, value in _cache_pairs(state.cache)]
    monkeypatch.setattr(target.model, "forward", lambda *a, **k: pytest.fail("reused a failed request cache"))
    with pytest.raises(RuntimeError, match="failed forward"):
        if reuse == "block":
            target.forward_block(state, [37])
        elif reuse == "tree":
            target.forward_tree(state, [37], [41], [-1])
        elif reuse == "crop":
            target.crop(state, 0)
        else:
            target.gather_tree_tail(state, 1, [])
    for actual, expected in zip(_cache_pairs(state.cache), before):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("entry", ["block", "tree"])
def test_failure_is_local_to_one_request_and_new_prefill_recovers(models, entry):
    target, reference = models
    state, _ = target.prefill([5, 6, 7])
    other, _ = target.prefill([13, 17, 19])
    fail_in_second_layer(target, state, entry, RuntimeError("injected layer failure"))
    actual = target.forward_block(other, [31])
    expected = _reference_forward(reference, (13, 17, 19, 31)).logits[0, -1:].float()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    fresh, logits = target.prefill([5, 6, 7])
    assert fresh.seq_len == 3
    assert logits.argmax().item() == _reference_forward(reference, (5, 6, 7)).logits[0, -1].argmax().item()
    actual = target.forward_block(fresh, [37])
    expected = _reference_forward(reference, (5, 6, 7, 37)).logits[0, -1:].float()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("entry", ["block", "tree"])
def test_admission_errors_leave_request_usable(models, entry):
    target, reference = models
    state, _ = target.prefill([5, 6, 7])
    with pytest.raises(ValueError):
        if entry == "block":
            target.forward_block(state, [11] * 64)
        else:
            target.forward_tree(state, [11], [13], [0])
    assert state.seq_len == 3
    actual = target.forward_block(state, [17])
    expected = _reference_forward(reference, (5, 6, 7, 17)).logits[0, -1:].float()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_cancellation_invalidates_partial_request_and_keeps_original_exception(models):
    class Cancelled(BaseException):
        pass

    target, _ = models
    state, _ = target.prefill([5, 6, 7])
    fail_in_second_layer(target, state, "block", Cancelled("cancelled"))
    with pytest.raises(RuntimeError, match="failed forward"):
        target.crop(state, 0)


def test_failed_prefill_clears_stale_hidden_output_and_allows_new_request(models):
    target, _ = models
    target.record_hidden = True
    target.prefill([5, 6, 7])
    assert target.last_hidden_block is not None

    def fail(module, args):
        raise RuntimeError("injected layer failure")

    hook = target.model.model.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            target.prefill([17, 19])
    finally:
        hook.remove()
    assert target.last_hidden_block is None
    state, _ = target.prefill([23, 29])
    assert state.seq_len == 2 and target.last_hidden_block.shape[0] == 2


@pytest.mark.parametrize("execution", ["cpu_static"], indirect=True)
def test_replay_cannot_reuse_state_after_eager_prefill_fails(execution):
    _, target = execution
    state = TargetState(DynamicCache())

    def fail(module, args):
        raise RuntimeError("injected norm failure")

    hook = target.model.model.norm.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected norm failure"):
            target.forward_block(state, [5, 6, 7])
    finally:
        hook.remove()
    assert state.seq_len == 3
    with pytest.raises(RuntimeError, match="failed forward"):
        target.forward_predictions(state, [11])
    fresh, _ = target.prefill([5, 6, 7])
    assert len(target.forward_predictions(fresh, [11])) == 1
