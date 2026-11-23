import copy
import gc
import weakref

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.replay import CudaGraphTarget, ReplayCache
from dart.models.target import TargetModel, TargetState


@pytest.fixture
def pair(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def unexpected(*args, **kwargs):
        pytest.fail('CPU execution called CUDA')

    for name in ('current_stream', 'Stream', 'CUDAGraph', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    torch.manual_seed(731)
    config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         eos_token_id=None, pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config)
    base = TargetModel(copy.deepcopy(model), device='cpu')
    target = CudaGraphTarget(TargetModel(model, device='cpu'), capacity=64, max_block_tokens=4,
                             replay=False, capture_predictions=True, greedy_chunk_sizes=(2, 4))
    yield base, target
    target.close()
    torch.set_num_threads(threads)


def observe_restores(monkeypatch, target):
    copies = []
    original = torch.Tensor.copy_

    def observed(destination, source, *args, **kwargs):
        if destination.ndim == 6 and destination.data_ptr() == target._workspace.storage.data_ptr():
            copies.append(destination.shape[-2])
        return original(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'copy_', observed)
    return copies


def check_cache(state, reference):
    assert state.seq_len == reference.seq_len
    storage = state.cache.storage
    for i, layer in enumerate(reference.cache.layers):
        for kind, expected in enumerate((layer.keys, layer.values)):
            torch.testing.assert_close(storage[i, kind, ..., :state.seq_len, :], expected, atol=2e-6, rtol=2e-5)


def forward(base, target, expected, actual, mode, token):
    count = int(mode) if mode in {'2', '4'} else 1
    if mode == 'logits':
        result = target.forward_block(actual, [token])
        wanted = base.forward_block(expected, [token])
        torch.testing.assert_close(result, wanted, atol=2e-6, rtol=2e-5)
        return int(result[-1].argmax())
    result = target.forward_predictions(actual, [token]) if mode == 'predictions' else target.forward_greedy(actual, token, count)
    wanted = []
    for _ in range(count):
        token = int(base.forward_block(expected, [token])[-1].argmax())
        wanted.append(token)
    assert result == wanted
    return result[-1]


@pytest.mark.parametrize('mode', ['logits', 'predictions', '2', '4'])
def test_consecutive_calls_restore_prefix_once(pair, monkeypatch, mode):
    base, target = pair
    a, logits = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    copies = observe_restores(monkeypatch, target)
    token = int(logits.argmax())
    for _ in range(4):
        token = forward(base, target, ar, a, mode, token)
    check_cache(a, ar)
    assert copies == [3]


@pytest.mark.parametrize('length', [0, 2, 4, 6])
def test_crop_and_mixed_calls_reuse_only_valid_prefix(pair, monkeypatch, length):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    copies = observe_restores(monkeypatch, target)
    for tokens in ([9, 11, 13], [17, 19]):
        torch.testing.assert_close(target.forward_block(a, tokens), base.forward_block(ar, tokens), atol=2e-6, rtol=2e-5)
    target.crop(a, length)
    base.crop(ar, length)
    token = forward(base, target, ar, a, 'predictions', 23)
    forward(base, target, ar, a, '2', token)
    check_cache(a, ar)
    assert copies == [3]


def test_interleaving_restores_other_request_then_reuses(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    b, _ = target.prefill([2, 4])
    br, _ = base.prefill([2, 4])
    copies = observe_restores(monkeypatch, target)
    for actual, expected, mode, token in ((a, ar, 'logits', 11), (a, ar, '2', 13),
                                          (b, br, '4', 17), (b, br, 'logits', 19),
                                          (a, ar, 'predictions', 23), (a, ar, '2', 29)):
        forward(base, target, expected, actual, mode, token)
    check_cache(a, ar)
    check_cache(b, br)
    assert copies == [3, 2, 6]


@pytest.mark.parametrize('access', ['direct', 'shallow_copy', 'replacement', 'external'])
def test_mutable_storage_alias_always_restores(pair, monkeypatch, access):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    forward(base, target, ar, a, 'logits', 11)
    if access == 'shallow_copy':
        storage = copy.copy(a.cache).storage
    else:
        storage = a.cache.storage
    if access in {'replacement', 'external'}:
        storage = storage.clone()
        if access == 'replacement':
            a.cache.storage = storage
        else:
            a.cache = ReplayCache(storage, a.seq_len, a.cache.owner)
    copies = observe_restores(monkeypatch, target)
    with torch.inference_mode():
        for token in (13, 17, 19):
            storage[0, 1, ..., 0, :].add_(0.25)  # Keep and edit the old alias between calls.
            ar.cache.layers[0].values[..., 0, :].add_(0.25)
            forward(base, target, ar, a, 'logits', token)
    check_cache(a, ar)
    assert copies == [4, 5, 6]


@pytest.mark.parametrize('mode', ['logits', '4'])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_failed_execution_invalidates_workspace_reuse(pair, monkeypatch, mode, error_type):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    copies = observe_restores(monkeypatch, target)
    forward(base, target, ar, a, 'logits', 11)
    block = target._blocks[1] if mode == 'logits' else target._greedy_blocks[4]
    original = block.run

    def fail():
        target._workspace.storage[..., :a.seq_len, :].fill_(float('nan'))
        raise error_type('synthetic execution failure')

    with monkeypatch.context() as patch:
        patch.setattr(block, 'run', fail)
        with pytest.raises(error_type, match='synthetic'):
            if mode == 'logits':
                target.forward_block(a, [13])
            else:
                target.forward_greedy(a, 13, 4)
    assert block.run == original and not a._failed and a.seq_len == 4
    forward(base, target, ar, a, mode, 13)
    check_cache(a, ar)
    assert copies == [3, 4]


def test_failed_restore_invalidates_previous_resident_request(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    b, _ = target.prefill([2, 4])
    forward(base, target, ar, a, 'logits', 11)
    original = torch.Tensor.copy_

    def fail(destination, source, *args, **kwargs):
        if destination.ndim == 6 and destination.data_ptr() == target._workspace.storage.data_ptr():
            destination.zero_()
            raise RuntimeError('synthetic copy failure')
        return original(destination, source, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'copy_', fail)
        with pytest.raises(RuntimeError, match='synthetic copy'):
            target.forward_predictions(b, [13])
    forward(base, target, ar, a, 'logits', 17)
    check_cache(a, ar)
    assert b.seq_len == 2 and not b._failed


def test_residency_does_not_retain_request_cache(pair):
    _, target = pair
    a, _ = target.prefill([3, 5, 7])
    target.forward_predictions(a, [11])
    cache = weakref.ref(a.cache)
    del a
    gc.collect()
    assert cache() is None
    b, _ = target.prefill([2, 4])
    assert len(target.forward_predictions(b, [13])) == 1


def test_external_storage_is_not_assumed_to_match_resident_prefix(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    forward(base, target, ar, a, 'logits', 11)
    b = TargetState(ReplayCache(a.cache.storage.clone(), a.seq_len, a.cache.owner))
    copies = observe_restores(monkeypatch, target)
    forward(base, target, ar, b, '2', 13)
    forward(base, target, ar, b, 'logits', 17)
    check_cache(b, ar)
    assert copies == [4, 6]


@pytest.mark.parametrize('keep, path', [(3, [0, 3]), (0, [6, 1, 6]), (4, [3] * 6), (3, [0, 1]), (0, [])])
@pytest.mark.parametrize('exposed', [False, True])
def test_tree_tail_continues_from_selected_rows(pair, keep, path, exposed):
    base, target = pair
    a, _ = target.prefill([3, 5, 7, 9])
    ar, _ = base.prefill([3, 5, 7, 9])
    target.forward_block(a, [11, 13, 17, 19])
    base.forward_block(ar, [11, 13, 17, 19])
    alias = a.cache.storage if exposed else None
    base.gather_tree_tail(ar, keep, path)
    target.gather_tree_tail(a, keep, path)
    assert a.seq_len == ar.seq_len == keep + len(path)
    if exposed and a.seq_len:
        with torch.inference_mode():
            alias[0, 1, ..., 0, :].add_(0.5)
            ar.cache.layers[0].values[..., 0, :].add_(0.5)
    forward(base, target, ar, a, 'logits', 23)  # Read the compacted rows before exposing the cache.
    check_cache(a, ar)


@pytest.mark.parametrize('after_replay', [False, True])
def test_tree_tail_rejects_capacity_growth_without_changing_state(pair, after_replay):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    if after_replay:
        forward(base, target, ar, a, 'predictions', 11)
    with pytest.raises(ValueError, match='capacity|context'):
        target.gather_tree_tail(a, 1, [1] * target.capacity)
    assert a.seq_len == ar.seq_len
    forward(base, target, ar, a, 'logits', 13)
    check_cache(a, ar)


def test_tree_tail_rejects_foreign_and_closed_targets(pair):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    ar, _ = base.prefill([3, 5, 7])
    forward(base, target, ar, a, 'logits', 11)
    other = CudaGraphTarget(base, capacity=64, replay=False)
    try:
        with pytest.raises(ValueError, match='different replay target'):
            other.gather_tree_tail(a, 1, [2])
        other.close()
        with pytest.raises(RuntimeError, match='closed'):
            other.gather_tree_tail(a, 1, [2])
    finally:
        other.close()
    forward(base, target, ar, a, 'logits', 17)
    check_cache(a, ar)


def test_tree_tail_interleaved_requests_keep_their_own_rows(pair):
    base, target = pair
    states = []
    for prompt in ([3, 5, 7], [17, 11, 13, 19]):
        a, _ = target.prefill(prompt)
        ar, _ = base.prefill(prompt)
        target.forward_block(a, [5, 7, 11, 13])
        base.forward_block(ar, [5, 7, 11, 13])
        states.append((a, ar))
    for (a, ar), path in zip(states, ([4, 1], [2, 0, 3])):
        base.gather_tree_tail(ar, 2, path)
        target.gather_tree_tail(a, 2, path)
    for a, ar in reversed(states):
        forward(base, target, ar, a, '2', 23)
        check_cache(a, ar)


def test_tree_tail_selects_only_the_requested_rows(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3] * 48)
    ar, _ = base.prefill([3] * 48)
    target.forward_block(a, [5, 7, 11, 13])
    base.forward_block(ar, [5, 7, 11, 13])
    base.gather_tree_tail(ar, 48, [3, 0, 3])
    selects = []
    select = torch.Tensor.index_select

    def observe(tensor, dim, index):
        selects.append((tensor.shape[dim], index.tolist()))
        return select(tensor, dim, index)

    with monkeypatch.context() as spy:
        spy.setattr(torch.Tensor, 'index_select', observe)
        target.gather_tree_tail(a, 48, [3, 0, 3])
    assert selects == [(4, [3, 0, 3])]
    forward(base, target, ar, a, 'logits', 23)
    check_cache(a, ar)


def test_tree_tail_partial_write_invalidates_only_its_request(pair, monkeypatch):
    base, target = pair
    a, _ = target.prefill([3, 5, 7])
    target.forward_block(a, [11, 13, 17])
    other, _ = target.prefill([17, 19])
    ref, _ = base.prefill([17, 19])
    forward(base, target, ref, other, 'logits', 23)
    pointer = a.cache.storage.untyped_storage().data_ptr()
    copy_ = torch.Tensor.copy_

    def fail(destination, source, *args, **kwargs):
        if destination.untyped_storage().data_ptr() == pointer:
            destination.zero_()
            raise RuntimeError('partial tail write')
        return copy_(destination, source, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(torch.Tensor, 'copy_', fail)
        with pytest.raises(RuntimeError, match='partial tail write'):
            target.gather_tree_tail(a, 3, [2, 0])
    with pytest.raises(RuntimeError, match='invalid'):
        target.forward_block(a, [23])
    forward(base, target, ref, other, 'logits', 29)
    check_cache(other, ref)
