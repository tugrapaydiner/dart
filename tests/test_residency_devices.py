from __future__ import annotations

from threading import get_ident
from types import SimpleNamespace

import pytest
import torch

from dart.kv.residency import ResidencyStats, move_cache
from dart.runtime import engine as runtime
from dart.runtime.types import DecodeConfig


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
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None))


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
    assert moves.count("cpu") == moves.count("pin") == 4


def test_cache_on_another_cuda_device_is_moved(cuda_device):
    moves = []
    stats = move_cache(fake_cache(1, moves), "cuda")
    assert stats.moved_tensors == 4 and stats.moved_bytes == 4 * 24 * 4
    assert moves == ["cuda"] * 4


@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_finished_host_copy_precedes_cache_reuse(cuda_device, monkeypatch, pin, legacy):
    pending = {0: [], 1: []}
    events = []

    class HostCopy(torch.Tensor):
        def pin_memory(self):
            events.append("pin")
            return self.as_subclass(torch.Tensor).clone()

    def transfer(src, device, *, non_blocking):
        assert torch.device(device).type == "cpu"
        dst = torch.full_like(src.as_subclass(torch.Tensor), -1).as_subclass(HostCopy)
        if non_blocking:
            pending[src.device.index].append((dst, src))
        else:
            dst.copy_(src.as_subclass(torch.Tensor))
        events.append("copy")
        return dst

    def stream(device):
        def finish():
            events.append(("wait", device.index))
            for dst, src in pending[device.index]:
                dst.copy_(src.as_subclass(torch.Tensor))
            pending[device.index].clear()
        return SimpleNamespace(synchronize=finish)

    monkeypatch.setattr(DeviceTensor, "to", transfer)
    monkeypatch.setattr(torch.cuda, "current_stream", stream)
    cache = fake_cache(0, [])
    for index, layer in enumerate(cache.layers):
        for slot, t in enumerate((layer.keys, layer.values)):
            t._reported = torch.device("cuda", slot)
            t.fill_(2 * index + slot)
    if legacy:
        cache = SimpleNamespace(key_cache=[layer.keys for layer in cache.layers],
                                value_cache=[layer.values for layer in cache.layers])

    stats = move_cache(cache, "cpu", pin=pin)
    pairs = zip(cache.key_cache, cache.value_cache) if legacy else [(layer.keys, layer.values) for layer in cache.layers]
    assert [int(t.flatten()[0]) for pair in pairs for t in pair] == [0, 1, 2, 3]
    assert events[:4] == ["copy"] * 4
    assert set(events[4:6]) == {("wait", 0), ("wait", 1)}
    assert events[6:] == ["pin"] * (4 if pin else 0)
    assert stats == ResidencyStats(moved_tensors=4, moved_bytes=4 * 24 * 4)


def test_cpu_cache_avoids_cuda_probe(monkeypatch):
    def unexpected():
        pytest.fail("CPU cache requested CUDA availability")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=torch.ones(1), values=torch.zeros(1))])
    assert move_cache(cache, "cpu", pin=True) == ResidencyStats()


def test_park_waits_for_verifier_stream_in_worker(tiny_target, monkeypatch):
    class Target:
        device = "cuda:1"

        def __getattr__(self, name):
            return getattr(tiny_target, name)

    owner = get_ident()
    recorded, ready, consumed = [], [], []

    def current_stream(device):
        assert get_ident() == owner and torch.device(device) == torch.device("cuda:1")

        def record():
            event = len(recorded)
            recorded.append(event)

            def wait():
                assert get_ident() != owner
                ready.append(event)
            return SimpleNamespace(synchronize=wait)
        return SimpleNamespace(record_event=record)

    def transfer(cache, device, pin=False):
        if pin:
            assert get_ident() != owner and ready == recorded and len(ready) == len(consumed) + 1
            consumed.append(ready[-1])
        return ResidencyStats()

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(runtime, "reset_gpu_memory_stats", lambda *_: None)
    monkeypatch.setattr(runtime, "gpu_memory_gb", lambda *_: 0.0)
    monkeypatch.setattr(runtime, "move_cache", transfer)
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    prompt = [4, 7, 11]
    expected = tiny_target.greedy_generate(prompt, 4)
    with runtime.DartEngine(Target()) as engine:
        out = engine.generate(prompt, DecodeConfig(max_new_tokens=4, full_kv_residency="cpu_pinned"))
    assert out.token_ids == expected
    assert consumed == list(range(len(out.rounds) - 1)) == [0, 1]
