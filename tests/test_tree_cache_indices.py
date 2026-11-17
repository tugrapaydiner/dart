from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from dart.models import target as target_module
from dart.models.target import TargetModel, TargetState
from tests.test_sampling_devices import DEVICES


def cache_state(prefix, device, legacy):
    tensors = []
    cache = DynamicCache()
    for layer in range(2):
        keys = torch.arange(2 * (prefix + 16) * 4, device=device, dtype=torch.float32)
        keys = keys.reshape(1, 2, prefix + 16, 4) + layer
        values = keys + 0.5
        tensors.extend((keys, values))
        cache.update(keys, values, layer)
    if legacy:
        cache = SimpleNamespace(key_cache=tensors[::2], value_cache=tensors[1::2])
        cache.get_seq_length = lambda: cache.key_cache[0].shape[2]
    return TargetState(cache), tensors


def cache_tensors(state, legacy):
    if legacy:
        return [tensor for pair in zip(state.cache.key_cache, state.cache.value_cache) for tensor in pair]
    return [tensor for layer in state.cache.layers for tensor in (layer.keys, layer.values)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefix", [0, 16, 128, 1008])
@pytest.mark.parametrize("path", [[3, 4, 5], [0, 1, 8, 9], [15, 2, 15, 0], [15] * 20])
@pytest.mark.parametrize("legacy", [False, True])
def test_tree_indices_preserve_prefix_order_duplicates_and_input_storage(device, prefix, path, legacy):
    target = TargetModel.__new__(TargetModel)
    target.device = torch.device(device)
    target.last_hidden_block = None
    state, originals = cache_state(prefix, device, legacy)
    saved = [tensor.clone() for tensor in originals]
    target.gather_tree_tail(state, prefix, path)
    expected_positions = list(range(prefix)) + [prefix + offset for offset in path]
    assert state.seq_len == prefix + len(path)
    for actual, original, before in zip(cache_tensors(state, legacy), originals, saved):
        assert torch.equal(actual, before[:, :, expected_positions, :])
        assert torch.equal(original, before)
        assert actual.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()


def test_tree_indices_keep_their_buffer_alive_and_release_it_after_gather(monkeypatch):
    target = TargetModel.__new__(TargetModel)
    target.device = torch.device("cpu")
    state, originals = cache_state(16, "cpu", False)
    refs, indices = [], []
    array, select = target_module.array, torch.Tensor.index_select

    def track_buffer(*args):
        buffer = array(*args)
        refs.append(weakref.ref(buffer))
        return buffer

    def check_indices(tensor, dim, index):
        gc.collect()
        assert refs[-1]() is not None
        assert index.dtype == torch.long and index.device == tensor.device
        indices.append(index.tolist())
        return select(tensor, dim, index)

    with monkeypatch.context() as patch:
        patch.setattr(target_module, "array", track_buffer)
        patch.setattr(torch.Tensor, "index_select", check_indices)
        target.gather_tree_tail(state, 16, [3, 7])
    gc.collect()
    assert len(indices) == len(originals)
    assert all(index == list(range(16)) + [19, 23] for index in indices)
    assert len(refs) == 1 and refs[0]() is None


def test_tree_indices_fall_back_when_native_integer_width_differs(monkeypatch):
    class WideArray(list):
        itemsize = 16

    target = TargetModel.__new__(TargetModel)
    target.device = torch.device("cpu")
    state, originals = cache_state(16, "cpu", False)
    with monkeypatch.context() as patch:
        patch.setattr(target_module, "array", lambda code, values: WideArray(values))
        patch.setattr(torch, "frombuffer", lambda *a, **k: pytest.fail("used an incompatible integer buffer"))
        with torch.device("meta"):
            target.gather_tree_tail(state, 16, [7, 3])
    for actual, original in zip(cache_tensors(state, False), originals):
        assert actual.device.type == "cpu"
        assert torch.equal(actual, original[:, :, list(range(16)) + [23, 19], :])


def test_tree_indices_ignore_default_device_and_dtype():
    target = TargetModel.__new__(TargetModel)
    target.device = torch.device("cpu")
    state, originals = cache_state(16, "cpu", False)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            target.gather_tree_tail(state, 16, [7, 3])
    finally:
        torch.set_default_dtype(dtype)
    for actual, original in zip(cache_tensors(state, False), originals):
        assert actual.device.type == "cpu" and actual.dtype == torch.float32
        assert torch.equal(actual, original[:, :, list(range(16)) + [23, 19], :])
