from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dart.kv.residency import move_cache


class DeviceTensor(torch.Tensor):
    """CPU storage that reports a CUDA device and records requested moves."""

    @property
    def device(self):
        return self._reported

    def to(self, device, *args, **kwargs):
        self._moves.append(str(device))
        return self

    def pin_memory(self):
        self._moves.append("pin")
        return self


def fake_cache(index, moves):
    tensors = []
    for _ in range(4):
        tensor = torch.Tensor._make_subclass(DeviceTensor, torch.zeros(1, 2, 3, 4))
        tensor._reported, tensor._moves = torch.device("cuda", index), moves
        tensors.append(tensor)
    return SimpleNamespace(layers=[SimpleNamespace(keys=tensors[0], values=tensors[1]),
                                   SimpleNamespace(keys=tensors[2], values=tensors[3])])


@pytest.fixture
def cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", torch.device("cuda"), torch.device("cuda", 0)])
def test_resident_cuda_cache_is_not_counted_as_moved(device, cuda_device):
    moves = []
    stats = move_cache(fake_cache(0, moves), device)
    assert (stats.moved_tensors, stats.moved_bytes, moves) == (0, 0, [])


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_host_offload_pins_for_any_cpu_device_spelling(device, cuda_device):
    moves = []
    stats = move_cache(fake_cache(0, moves), device, pin=True)
    assert stats.moved_tensors == 4
    assert moves == ["cpu", "pin"] * 4


def test_cache_on_another_cuda_device_is_moved(cuda_device):
    moves = []
    stats = move_cache(fake_cache(1, moves), "cuda")
    assert stats.moved_tensors == 4 and stats.moved_bytes == 4 * 24 * 4
    assert moves == ["cuda"] * 4
