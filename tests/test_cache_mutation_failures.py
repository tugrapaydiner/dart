from __future__ import annotations

import pytest
import torch

from dart.models import target as target_module
from dart.verification import block_verify
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from dart.verification.state_checks import _cache_pairs, _reference_forward
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree, verify_tree_greedy
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
            def fail_later(*args, **kwargs):
                raise error

            fault.setattr(state.cache.layers[1], 'crop', fail_later)  # Patching Tensor.__getitem__ leaks a type slot.
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


@pytest.mark.parametrize("stage", ["buffer", "tensor", "transfer"])
def test_index_allocation_failure_leaves_the_unmodified_cache_usable(models, monkeypatch, stage):
    target, reference = models
    state, _ = target.prefill([5, 6, 7])
    error = RuntimeError('index allocation failed')

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as fault:
        if stage == "buffer":
            fault.setattr(target_module, "array", fail)
        elif stage == "tensor":
            fault.setattr(torch, "frombuffer", fail)
        else:
            fault.setattr(torch.Tensor, "to", fail)
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


def _greedy_round(target, state, method, pending):
    if method == "reference":
        return verify_greedy_reference(target, state, pending, [17, 19])
    if method == "block":
        return verify_block_greedy(target, state, pending, [17, 19])
    first = next(iter(target.eos_token_ids)) if method == "pruned_tree" else 17
    return verify_tree_greedy(target, state, pending, DraftTree([first, 19], [-1, -1]))


@pytest.mark.parametrize("method", ["reference", "block", "tree", "pruned_tree"])
@pytest.mark.parametrize("error_type", [RuntimeError, Cancelled])
def test_failed_greedy_round_retries_from_the_original_prefix(models, monkeypatch, method, error_type):
    target, reference = models
    target.record_hidden = True
    prompt, pending = [5, 6, 7], [11, 13]
    state, _ = target.prefill(prompt)
    control, _ = target.prefill(prompt)
    error = error_type("decision interrupted")
    argmax = torch.argmax

    def fail(scores, *args, **kwargs):
        if method == "reference" and state.seq_len == len(prompt) + 1:
            return argmax(scores, *args, **kwargs)
        raise error

    with monkeypatch.context() as fault:
        fault.setattr(torch, "argmax", fail)
        with pytest.raises(error_type) as caught:
            _greedy_round(target, state, method, pending)
    assert caught.value is error
    assert state.seq_len == len(prompt) and not state._failed
    assert target.last_hidden_block is None
    for pair, original in zip(_cache_pairs(state.cache), _cache_pairs(control.cache)):
        for row, expected in zip(pair, original):
            assert torch.equal(row, expected)

    actual = _greedy_round(target, state, method, pending)
    assert actual == _greedy_round(target, control, method, pending)
    assert actual.committed[0] == int(_reference_forward(reference, (*prompt, *pending)).logits[0, -1].argmax())


@pytest.mark.parametrize("error_type", [MemoryError, Cancelled])
def test_failed_greedy_round_rolls_back_packed_siblings(models, monkeypatch, error_type):
    target, reference = models
    target.eos_token_ids = frozenset()
    prompt, pending = [5, 6, 7], 11
    correct = int(_reference_forward(reference, (*prompt, pending)).logits[0, -1].argmax())
    tree = DraftTree([(correct + 1) % target.model.config.vocab_size, correct], [-1, -1])
    state, _ = target.prefill(prompt)
    error = error_type("path index unavailable")

    def fail(*args):
        raise error

    with monkeypatch.context() as fault:
        fault.setattr(target_module, "array", fail)
        with pytest.raises(error_type) as caught:
            verify_tree_greedy(target, state, pending, tree)
    assert caught.value is error and state.seq_len == len(prompt) and not state._failed
    assert verify_tree_greedy(target, state, pending, tree).committed[0] == correct


@pytest.mark.parametrize("method", ["block", "tree"])
@pytest.mark.parametrize("cleanup", ["raises", "leaves_extra_rows"])
def test_failed_greedy_round_invalidates_an_unrestored_cache(models, monkeypatch, method, cleanup):
    target, _ = models
    target.record_hidden = True
    state, _ = target.prefill([5, 6, 7])
    error = Cancelled("decision interrupted")

    def interrupted(*args, **kwargs):
        raise error

    def bad_crop(*args):
        if cleanup == "raises":
            raise RuntimeError("rollback failed")

    with monkeypatch.context() as fault:
        fault.setattr(torch, "argmax", interrupted)
        fault.setattr(target, "crop", bad_crop)
        with pytest.raises(Cancelled) as caught:
            _greedy_round(target, state, method, [11])
    assert caught.value is error and state._failed
    assert target.last_hidden_block is None
    with pytest.raises(RuntimeError, match="cache is invalid"):
        target.forward_block(state, [13])


@pytest.mark.parametrize("method", ["reference", "block", "tree"])
def test_failed_greedy_round_leaves_admission_errors_untouched(models, monkeypatch, method):
    target, _ = models
    target.record_hidden = True
    state, _ = target.prefill([5, 6, 7])
    hidden = target.last_hidden_block
    crops = []
    monkeypatch.setattr(target, "crop", lambda *args: crops.append(args))
    with pytest.raises(ValueError, match="vocabulary"):
        _greedy_round(target, state, method, [target.model.config.vocab_size])
    assert state.seq_len == 3 and not state._failed and target.last_hidden_block is hidden
    assert crops == []


@pytest.mark.parametrize("limit", ["vocabulary", "context"])
def test_failed_greedy_round_restores_tokens_before_a_late_input_error(models, monkeypatch, limit):
    target, _ = models
    target.record_hidden = True
    state, _ = target.prefill([5, 6, 7])
    pending = [11, target.model.config.vocab_size] if limit == "vocabulary" else [11, 13]
    if limit == "context":
        monkeypatch.setattr(target.model.config, "max_position_embeddings", 4)
    with pytest.raises(ValueError, match=limit):
        verify_greedy_reference(target, state, pending, [])
    assert state.seq_len == 3 and not state._failed
    assert target.last_hidden_block is None


@pytest.mark.parametrize("execution", ["cpu_static"], indirect=True)
@pytest.mark.parametrize("overridden", [False, True])
def test_failed_greedy_round_preserves_a_replay_request(execution, monkeypatch, overridden):
    base, target = execution
    state, _ = target.prefill([5, 6, 7])
    target.forward_predictions(state, [11])
    error = Cancelled("interrupted decision")

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as fault:
        if overridden:
            predictions = target.forward_predictions
            fault.setattr(target, "forward_predictions", lambda *args: predictions(*args))
        fault.setattr(block_verify, "_block_result", fail)
        with pytest.raises(Cancelled) as caught:
            verify_block_greedy(target, state, 13, [17, 19], compute_entropy=False)
    assert caught.value is error and state.seq_len == 4 and not state._failed
    _, expected = base.prefill([5, 6, 7, 11, 13])
    torch.testing.assert_close(target.forward_block(state, [13])[-1], expected, atol=2e-6, rtol=2e-5)
