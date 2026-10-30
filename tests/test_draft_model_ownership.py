from __future__ import annotations

import gc
import weakref

import pytest

from dart.drafters.ar_draft import ARDrafter
from dart.models.replay import CudaGraphTarget
from dart.runtime import factory
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig

DRAFT = {"model": "tiny:1", "execution": "static", "cache_capacity": 64, "max_block_tokens": 5}
CONFIG = {"target": {"model": "tiny:0"}, "lanes": {"ar_draft": DRAFT}}


def static_draft(engine):
    draft = engine.drafter.dm
    assert isinstance(draft, CudaGraphTarget)
    return draft


def test_factory_engine_closes_its_draft_model():
    with factory.build_engine(CONFIG) as engine:
        draft = static_draft(engine)
        workspace = weakref.ref(draft._workspace)
        expected = engine.target.greedy_generate([5, 6, 7], 6)
        assert engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=6, draft_len=4)).token_ids == expected
    assert draft._closed and draft._workspace is None and not draft._blocks
    gc.collect()
    assert workspace() is None
    engine.close()


def test_borrowed_draft_model_stays_open():
    draft = factory.build_target(DRAFT)
    target = factory.build_target({"model": "tiny:0"})
    with DartEngine(target, ARDrafter(draft)) as engine:
        engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4))
    assert not draft._closed
    assert draft.greedy_generate([5, 6], 3)
    draft.close()


def test_failed_lane_construction_closes_earlier_draft_models(monkeypatch):
    built = []
    build_target = factory.build_target
    monkeypatch.setattr(factory, "build_target", lambda spec: built.append(build_target(spec)) or built[-1])
    with pytest.raises(ValueError, match="unknown lane"):
        factory.build_engine({**CONFIG, "lanes": {"ar_draft": DRAFT, "missing": {}}})
    assert len(built) == 2 and built[1]._closed


def test_draft_model_close_failure_still_closes_target_and_can_be_retried(monkeypatch):
    engine = factory.build_engine(CONFIG)
    draft = static_draft(engine)
    close, calls = draft.close, []

    def fail_once():
        calls.append("draft")
        if calls.count("draft") == 1:
            raise RuntimeError("draft close failed")
        close()

    monkeypatch.setattr(draft, "close", fail_once)
    monkeypatch.setattr(engine.target, "close", lambda: calls.append("target"), raising=False)
    with pytest.raises(RuntimeError, match="draft close failed"):
        engine.close()
    assert calls == ["target", "draft"]
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([5, 6], DecodeConfig(max_new_tokens=1))
    engine.close()
    engine.close()
    assert calls == ["target", "draft", "draft"] and draft._closed


@pytest.mark.parametrize("lane", ["compressed_kv", "sparse_kv"])
def test_lanes_sharing_the_target_close_it_once(lane, monkeypatch):
    target = factory.build_target({"model": "tiny:0"})
    calls = []
    monkeypatch.setattr(target, "close", lambda: calls.append("closed"), raising=False)
    monkeypatch.setattr(factory, "build_target", lambda spec: target)
    with factory.build_engine({"target": {}, "lanes": {lane: {}}}):
        pass
    assert calls == ["closed"]
