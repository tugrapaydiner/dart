import threading

import pytest

from dart.kv.prefetch import ThreadPrefetcher
from dart.kv.residency import ResidencyStats
from dart.runtime import engine as runtime
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig


@pytest.mark.parametrize("stage", ["draft", "verify", "observe", "trace", "offload"])
def test_failed_request_closes_offload_worker_and_next_request_succeeds(tiny_target, monkeypatch, stage):
    workers = []

    class Worker(ThreadPrefetcher):
        closed = False

        def __init__(self):
            super().__init__()
            workers.append(self)

        def close(self):
            super().close()
            self.closed = True

    def fail(*args, **kwargs):
        raise RuntimeError("request failure")

    monkeypatch.setattr(runtime, "ThreadPrefetcher", Worker)
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    config = DecodeConfig(max_new_tokens=4, full_kv_residency="cpu_pinned")
    with DartEngine(tiny_target) as engine:
        try:
            with monkeypatch.context() as patch:
                if stage == "offload":
                    def move(cache, device, pin=False):
                        if pin:
                            fail()
                        return ResidencyStats()

                    patch.setattr(runtime, "move_cache", move)
                else:
                    obj, attribute = {"draft": (engine.drafter, "propose"), "verify": (engine, "verify"),
                                      "observe": (engine.drafter, "observe"), "trace": (engine.trace, "log_round")}[stage]
                    patch.setattr(obj, attribute, fail)
                with pytest.raises(RuntimeError, match="request failure"):
                    engine.generate([5, 6, 7], config)
            assert workers[0].closed
            with pytest.raises(RuntimeError, match="cannot schedule new futures"):
                workers[0].prefetch(lambda: None)
            expected = tiny_target.greedy_generate([5, 6, 7], 4)
            assert engine.generate([5, 6, 7], config).token_ids == expected
            assert workers[-1].closed
        finally:
            for worker in workers:
                worker.close()


def test_interrupted_request_waits_for_inflight_offload(tiny_target, monkeypatch):
    started, release, finished = (threading.Event() for _ in range(3))
    workers = []

    class Interrupted(BaseException):
        pass

    class Worker(ThreadPrefetcher):
        def __init__(self):
            super().__init__()
            workers.append(self)

        def close(self):
            release.set()
            super().close()

    def move(cache, device, pin=False):
        if pin:
            started.set()
            assert release.wait(5)
            finished.set()
        return ResidencyStats()

    failure = Interrupted("cancelled")
    calls = 0

    def propose(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert started.wait(5)
            raise failure
        return []

    monkeypatch.setattr(runtime, "ThreadPrefetcher", Worker)
    monkeypatch.setattr(runtime, "move_cache", move)
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    with DartEngine(tiny_target) as engine:
        monkeypatch.setattr(engine.drafter, "propose", propose)
        try:
            with pytest.raises(Interrupted) as caught:
                engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4, full_kv_residency="cpu_pinned"))
            assert caught.value is failure
            assert finished.is_set()
            with pytest.raises(RuntimeError, match="cannot schedule new futures"):
                workers[0].prefetch(lambda: None)
        finally:
            release.set()
            for worker in workers:
                worker.close()
