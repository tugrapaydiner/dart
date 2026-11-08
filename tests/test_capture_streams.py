import gc
import threading
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from dart.models import replay
from dart.models.replay import CudaGraphTarget, _ReplayBlock
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def cuda(monkeypatch):
    events, created = [], []
    origin = SimpleNamespace(fail_sync=False, wait_stream=lambda stream: None)

    def sync_origin():
        events.append(("owner_sync", 0))
        if origin.fail_sync:
            raise RuntimeError("owner sync failed")

    origin.synchronize = sync_origin

    class Stream:
        def __init__(self, device=None):
            self.device = device
            self.cuda_stream = len(created) + 1
            self.fail_sync = False
            created.append(self)

        def wait_stream(self, other):
            pass

        def synchronize(self):
            events.append(("sync", self.cuda_stream))
            if self.fail_sync:
                raise RuntimeError("stream sync failed")

    class Graph:
        def __init__(self):
            self.stream = None
            self.fail_reset = False

        def reset(self):
            events.append(("reset", self.stream.cuda_stream))
            if self.fail_reset:
                raise RuntimeError("graph reset failed")

    def graph(value, stream):
        value.stream = stream
        return nullcontext()

    monkeypatch.setattr(replay, "_capture_stream_pools", threading.local(), raising=False)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: origin)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    return SimpleNamespace(events=events, created=created, origin=origin)


def block(length=1):
    model = SimpleNamespace(forward=lambda **kwargs: SimpleNamespace(logits=kwargs["input_ids"].float().unsqueeze(-1)))
    workspace = SimpleNamespace(storage=torch.zeros(1, 2, 1, 1, 16, 4), positions=None)
    return _ReplayBlock(model, workspace, length, True, capture_predictions=True)


def test_closed_preparation_reuses_streams(cuda):
    for _ in range(3):
        prepared = [block(length) for length in (1, 2, 3)]
        for value in prepared:
            value.close()
        del prepared, value
        gc.collect()
    assert len(cuda.created) == 3


def test_live_graphs_keep_distinct_streams_and_reset_before_reuse(cuda):
    first, second = block(), block()
    first_stream = first._capture_stream
    second_stream = second._capture_stream
    held_graph = first.graph
    assert first_stream is not second_stream
    first.close()
    assert cuda.events[-2:] == [("sync", first_stream.cuda_stream), ("reset", first_stream.cuda_stream)]
    third = block()
    assert third._capture_stream is first_stream
    assert third.graph is not held_graph
    assert second._capture_stream is second_stream
    second.close()
    third.close()


@pytest.mark.parametrize("failure", ["sync", "reset"])
def test_failed_close_keeps_stream_out_of_reuse_until_retry(cuda, failure):
    first = block()
    stream = first._capture_stream
    graph = first.graph
    resource = stream if failure == "sync" else graph
    setattr(resource, "fail_" + failure, True)
    with pytest.raises(RuntimeError, match="failed"):
        first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.run()
    second = block()
    assert second._capture_stream is not stream
    setattr(resource, "fail_" + failure, False)
    first.close()
    third = block()
    assert third._capture_stream is stream
    first.close()
    second.close()
    third.close()


@pytest.mark.parametrize("stage", ["warmup", "capture"])
def test_failed_construction_returns_only_cleaned_streams(cuda, monkeypatch, stage):
    original = _ReplayBlock.forward
    calls = 0

    def fail(self):
        nonlocal calls
        calls += 1
        if calls == (1 if stage == "warmup" else 4):
            raise RuntimeError("prepare failed")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(_ReplayBlock, "forward", fail)
        with pytest.raises(RuntimeError, match="prepare failed"):
            block()
    later = block()
    assert len(cuda.created) == 1
    later.close()


def test_abandoned_block_closes_graph_before_reuse(cuda):
    first = block()
    old_stream = first._capture_stream
    reference = weakref.ref(first)
    graph = first.graph
    del first
    gc.collect()
    assert reference() is None
    assert ("reset", old_stream.cuda_stream) in cuda.events
    assert cuda.events.index(("owner_sync", 0)) < cuda.events.index(("reset", old_stream.cuda_stream))
    second = block()
    assert second._capture_stream is old_stream and second.graph is not graph
    second.close()


def test_pending_replay_failure_prevents_stream_reuse(cuda):
    first = block()
    old_stream = first._capture_stream
    cuda.origin.fail_sync = True
    with pytest.raises(RuntimeError, match="owner sync failed"):
        first.close()
    assert not any(event[0] == "reset" for event in cuda.events)
    cuda.origin.fail_sync = False
    second = block()
    assert second._capture_stream is not old_stream
    first.close()
    second.close()


def test_pool_bounds_idle_streams_and_rejects_live_handle_alias(cuda, monkeypatch):
    pool = replay._capture_pool(torch.device("cuda:0"))
    streams = [pool.acquire() for _ in range(17)]
    assert len({stream.cuda_stream for stream in streams}) == 17
    for stream in streams:
        pool.release(stream)
    assert len(pool.available) == 16 and not pool.active
    live = [pool.acquire() for _ in range(16)]
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: live[0])
    with pytest.raises(RuntimeError, match="already in use"):
        pool.acquire()
    assert len(pool.active) == 16
    for stream in live:
        pool.release(stream)


def test_device_and_thread_pools_stay_separate(cuda):
    first = replay._capture_pool(torch.device("cuda:0"))
    second = replay._capture_pool(torch.device("cuda:1"))
    assert first is not second
    stream = first.acquire()
    other_pools = []

    def other_thread():
        other_pools.append(replay._capture_pool(torch.device("cuda:0")))
        first.release(stream)

    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert other_pools[0] is not first
    assert first.acquire() is stream
    first.release(stream)


def test_target_close_drains_failures_and_can_retry(monkeypatch):
    base, _ = _synthetic_models(17)
    target = CudaGraphTarget(base, capacity=16, max_block_tokens=3, replay=False)
    called = []
    fail = True
    originals = [value.close for value in target._blocks.values()]

    def close(index):
        called.append(index)
        if index == 1 and fail:
            raise RuntimeError("cleanup failed")
        originals[index]()

    for index, value in enumerate(target._blocks.values()):
        monkeypatch.setattr(value, "close", lambda i=index: close(i))
    with pytest.raises(RuntimeError, match="cleanup failed"):
        target.close()
    assert sorted(called) == [0, 1, 2]
    with pytest.raises(RuntimeError, match="closed"):
        target.prefill([3, 5])
    fail = False
    target.close()
    assert target._workspace is None and not target._blocks


def test_partial_target_preparation_closes_retained_blocks(monkeypatch):
    base, _ = _synthetic_models(17)
    prepared = []
    original = replay._ReplayBlock

    def create(*args, **kwargs):
        if len(prepared) == 2:
            raise RuntimeError("third block failed")
        value = original(*args, **kwargs)
        prepared.append(value)
        return value

    monkeypatch.setattr(replay, "_ReplayBlock", create)
    with pytest.raises(RuntimeError, match="third block failed"):
        CudaGraphTarget(base, capacity=16, max_block_tokens=3, replay=False)
    assert len(prepared) == 2
    assert all(value._closed and value.workspace is None for value in prepared)
