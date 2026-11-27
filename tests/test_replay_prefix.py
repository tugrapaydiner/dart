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


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('bad_format, mode', [('half', 'logits'), ('integer', '2'), ('meta', 'predictions'),
                                              ('sparse', 'logits')])
def test_incompatible_kv_preserves_the_request(pair, packed, bad_format, mode):
    base, target = pair
    state, _ = target.prefill([3, 5, 7])
    ref, _ = base.prefill([3, 5, 7])
    if packed:
        forward(base, target, ref, state, 'logits', 11)
    cache = state.cache
    length = state.seq_len
    convert = {'half': lambda t: t.half(), 'integer': lambda t: t.to(torch.int8),
               'meta': lambda t: t.to('meta'), 'sparse': lambda t: t.to_sparse()}[bad_format]
    if packed:
        saved = cache.storage
        cache.storage = convert(saved)
    else:
        saved = [(layer.keys, layer.values) for layer in cache.layers]
        for layer, (keys, values) in zip(cache.layers, saved):
            layer.keys, layer.values = convert(keys), convert(values)

    other, _ = target.prefill([2, 4])
    target.forward_predictions(other, [13])
    workspace = target._workspace.storage.clone()
    with pytest.raises(ValueError, match='cache storage does not match'):
        forward(base, target, ref, state, mode, 17)
    assert state.cache is cache and state.seq_len == length and not state._failed
    assert target._resident_cache() is other.cache and not target._workspace_dirty
    assert torch.equal(target._workspace.storage, workspace)

    if packed:
        cache.storage = saved
    else:
        for layer, (keys, values) in zip(cache.layers, saved):
            layer.keys, layer.values = keys, values
    forward(base, target, ref, state, mode, 17)
    check_cache(state, ref)


@pytest.mark.parametrize('mode, axis', [('logits', -2), ('predictions', -2), ('2', -2),
                                        ('logits', -1), ('gather', 0)])
def test_broadcast_snapshot_rejected_before_writing(pair, mode, axis):
    base, target = pair
    state, _ = target.prefill([3, 5, 7])
    ref, _ = base.prefill([3, 5, 7])
    forward(base, target, ref, state, 'predictions', 11)
    saved = state.cache.storage
    broadcast = saved.narrow(axis, 0, 1).expand_as(saved)
    state.cache.storage = broadcast
    before = broadcast.contiguous().view(torch.uint8).clone()
    other, _ = target.prefill([2, 4])
    target.forward_predictions(other, [13])
    workspace = target._workspace.storage.clone()

    def step():
        if mode == 'gather':
            target.gather_tree_tail(state, 1, [2, 0])
        else:
            forward(base, target, ref, state, mode, 17)

    with pytest.raises(ValueError, match='broadcast'):
        step()
    assert state.cache.storage is broadcast and state.seq_len == 4 and not state._failed
    assert torch.equal(broadcast.contiguous().view(torch.uint8), before)
    assert target._resident_cache() is other.cache and not target._workspace_dirty
    assert torch.equal(target._workspace.storage.view(torch.uint8), workspace.view(torch.uint8))

    state.cache.storage = saved
    if mode == 'gather':
        base.gather_tree_tail(ref, 1, [2, 0])
        step()
        forward(base, target, ref, state, 'logits', 17)
    else:
        step()
    check_cache(state, ref)


def test_writable_strided_snapshot_preserves_external_updates(pair):
    base, target = pair
    state, _ = target.prefill([3, 5, 7])
    ref, _ = base.prefill([3, 5, 7])
    forward(base, target, ref, state, 'predictions', 11)
    with torch.inference_mode():
        saved = state.cache.storage
        backing = saved.new_zeros((*saved.shape[:-1], 2 * saved.size(-1)))
        view = backing[..., 1::2]
        strides = list(view.stride())
        strides[2] = 0  # Batch size is one.
        view = view.as_strided(view.shape, strides)
        view.copy_(saved)
        state.cache.storage = view
        view[0, 1, ..., 0, :].add_(0.25)
        ref.cache.layers[0].values[..., 0, :].add_(0.25)
    forward(base, target, ref, state, '2', 17)
    check_cache(state, ref)
    assert state.cache.storage is view and backing[..., ::2].count_nonzero() == 0


@pytest.mark.parametrize('invalid', ['heads', 'mixed_layout'])
def test_incompatible_dynamic_cache_fails_before_snapshot_allocation(pair, monkeypatch, invalid):
    _, target = pair
    state, _ = target.prefill([3, 5, 7])
    if invalid == 'heads':
        for layer in state.cache.layers:
            layer.keys = layer.keys.repeat(1, 2, 1, 1)
            layer.values = layer.values.repeat(1, 2, 1, 1)
    else:
        state.cache.layers[-1].values = state.cache.layers[-1].values.to_sparse()
    new_empty = torch.Tensor.new_empty

    def allocate(tensor, shape, *args, **kwargs):
        if len(shape) == 6:
            pytest.fail('allocated a full snapshot for an incompatible cache')
        return new_empty(tensor, shape, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'new_empty', allocate)
    with pytest.raises(ValueError, match='cache storage does not match|matching shapes'):
        target.forward_predictions(state, [11])
    assert state.seq_len == 3 and not state._failed


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


@pytest.mark.parametrize('path', [[2, 0], [2]])
def test_tree_tail_partial_write_invalidates_only_its_request(pair, monkeypatch, path):
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
            target.gather_tree_tail(a, 3, path)
    with pytest.raises(RuntimeError, match='invalid'):
        target.forward_block(a, [23])
    forward(base, target, ref, other, 'logits', 29)
    check_cache(other, ref)


def workspace_writes(monkeypatch, target):
    workspace = target._workspace.storage
    address = workspace.untyped_storage().data_ptr()
    writes = []
    copy_ = torch.Tensor.copy_

    def record(destination, source, *args, **kwargs):
        if destination.ndim == 6 and destination.untyped_storage().data_ptr() == address:
            start = (destination.storage_offset() - workspace.storage_offset()) // workspace.stride(-2)
            writes.append((start, destination.size(-2)))
        return copy_(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'copy_', record)
    return writes


@pytest.mark.parametrize('mode, regather, crop_to, expected', [
    ('logits', False, None, [(48, 3)]), ('predictions', False, None, [(48, 3)]),
    ('2', True, None, [(48, 3)]), ('4', True, 50, [(48, 2)]),
    ('logits', False, 49, [(48, 1)]), ('predictions', True, 48, []), ('2', False, 47, []),
])
def test_selected_tail_resumes_at_the_shared_prefix(pair, monkeypatch, mode, regather, crop_to, expected):
    base, target = pair
    state, _ = target.prefill([3] * 48)
    ref, _ = base.prefill([3] * 48)
    target.forward_block(state, [5, 7, 11, 13])
    base.forward_block(ref, [5, 7, 11, 13])
    writes = workspace_writes(monkeypatch, target)
    for model, request in ((target, state), (base, ref)):
        model.gather_tree_tail(request, 48, [3, 0, 2])
        if regather:
            model.gather_tree_tail(request, 49, [1, 0])
        if crop_to is not None:
            model.crop(request, crop_to)
    assert writes == []
    token = forward(base, target, ref, state, mode, 23)
    forward(base, target, ref, state, 'predictions', token)
    check_cache(state, ref)
    assert writes == expected


@pytest.mark.parametrize('interruption', ['switch', 'alias', 'shallow'])
def test_selected_tail_restores_a_changed_or_exposed_prefix(pair, monkeypatch, interruption):
    base, target = pair
    state, _ = target.prefill([3, 5, 7])
    ref, _ = base.prefill([3, 5, 7])
    target.forward_block(state, [9, 11, 13])
    base.forward_block(ref, [9, 11, 13])
    target.gather_tree_tail(state, 3, [2, 0])
    base.gather_tree_tail(ref, 3, [2, 0])
    if interruption == 'switch':
        other, _ = target.prefill([19, 23])
        target.forward_predictions(other, [29])
    else:
        alias = state.cache if interruption == 'alias' else copy.copy(state.cache)
        with torch.inference_mode():
            alias.storage[0, 1, ..., 0, :].add_(0.25)
            ref.cache.layers[0].values[..., 0, :].add_(0.25)
    writes = workspace_writes(monkeypatch, target)
    forward(base, target, ref, state, 'predictions', 31)
    check_cache(state, ref)
    assert writes == [(0, 5)]


def test_selected_tail_failed_restore_reloads_the_request(pair, monkeypatch):
    base, target = pair
    state, _ = target.prefill([3, 5, 7])
    ref, _ = base.prefill([3, 5, 7])
    target.forward_block(state, [11, 13, 17])
    base.forward_block(ref, [11, 13, 17])
    target.gather_tree_tail(state, 3, [2, 0])
    base.gather_tree_tail(ref, 3, [2, 0])
    workspace = target._workspace.storage
    copy_ = torch.Tensor.copy_

    def fail(destination, source, *args, **kwargs):
        if destination.ndim == 6 and destination.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr():
            workspace.fill_(float('nan'))
            raise RuntimeError('interrupted prefix restore')
        return copy_(destination, source, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(torch.Tensor, 'copy_', fail)
        with pytest.raises(RuntimeError, match='interrupted prefix restore'):
            target.forward_predictions(state, [23])
    assert state.seq_len == 5
    writes = workspace_writes(monkeypatch, target)
    forward(base, target, ref, state, 'logits', 23)
    check_cache(state, ref)
    assert writes == [(0, 5)]


@pytest.mark.parametrize("path,strided,direct", [
    ([2], False, True), ([2, 3], False, True), ((4, 5, 6), False, True),
    ([1, 2], False, False), ([3, 2], False, False), ([3, 3], False, False),
    ([2, 3], True, False),
])
@torch.inference_mode()
def test_disjoint_tree_span_uses_request_storage(pair, monkeypatch, path, strided, direct):
    base, target = pair
    state, _ = target.prefill([3, 5, 7, 9, 11, 13, 15])
    ref, _ = base.prefill([3, 5, 7, 9, 11, 13, 15])
    forward(base, target, ref, state, 'logits', 17)
    keep = 1
    storage = state.cache._buffer.tensor
    if strided:
        padded = storage.new_empty((*storage.shape[:-1], 2 * storage.size(-1)))
        padded[..., ::2].copy_(storage)
        state.cache.storage = storage = padded[..., ::2]
    before = storage[..., :state.seq_len, :].clone()
    expected = before.clone()
    expected[..., keep:keep + len(path), :] = before[..., [keep + p for p in path], :]
    sources = []
    copy_ = torch.Tensor.copy_

    def record(dest, src, *args, **kwargs):
        if dest.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr():
            sources.append(src.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr())
        return copy_(dest, src, *args, **kwargs)

    with monkeypatch.context() as spy:
        spy.setattr(torch.Tensor, 'copy_', record)
        target.gather_tree_tail(state, keep, path)
    assert sources == [direct]
    assert state.seq_len == keep + len(path)
    assert torch.equal(storage[..., :before.size(-2), :], expected)
    base.gather_tree_tail(ref, keep, path)
    forward(base, target, ref, state, 'predictions', 19)
    check_cache(state, ref)
