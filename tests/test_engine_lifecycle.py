from __future__ import annotations

import gc
import weakref

import pytest

from dart.runtime import factory
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.test_replay_predictions import execution as execution


@pytest.fixture
def owned_target(tiny_target, monkeypatch):
    calls = []
    monkeypatch.setattr(tiny_target, "close", lambda: calls.append("closed"), raising=False)
    monkeypatch.setattr(factory, "build_target", lambda spec: tiny_target)
    return tiny_target, calls


@pytest.mark.parametrize("options", [{}, {"lanes": {"ngram": {}}},
                                     {"lanes": {"ngram": {}}, "scheduler": {"type": "heuristic"}}])
def test_factory_engine_closes_its_target_once(owned_target, options):
    target, calls = owned_target
    with factory.build_engine({"target": {}, **options}) as engine:
        assert engine.target is target and calls == []
    assert calls == ["closed"]
    engine.close()
    assert calls == ["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([5, 6], DecodeConfig(max_new_tokens=0))


@pytest.mark.parametrize("execution", ["cpu_static"], indirect=True)
def test_factory_close_releases_actual_static_replay_workspace(execution, monkeypatch):
    base, target = execution
    workspace = weakref.ref(target._workspace)
    block = weakref.ref(target._blocks[1])
    monkeypatch.setattr(factory, "build_target", lambda spec: target)
    with factory.build_engine({"target": {}}) as engine:
        expected = base.greedy_generate([5, 6, 7], 4)
        assert engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4)).token_ids == expected
    assert target._closed and target._workspace is None and not target._blocks
    gc.collect()
    assert workspace() is None and block() is None
    assert engine.target is target  # Cleanup must work while the engine is still referenced.


def test_borrowed_target_remains_usable_by_another_engine(owned_target):
    target, calls = owned_target
    first, second = DartEngine(target), DartEngine(target)
    first.close()
    assert not calls
    expected = target.greedy_generate([5, 6, 7], 3)
    assert second.generate([5, 6, 7], DecodeConfig(max_new_tokens=3)).token_ids == expected
    second.close()
    assert not calls


@pytest.mark.parametrize("stage", ["defaults", "lane", "scheduler", "verifier"])
def test_factory_construction_failure_closes_created_target(owned_target, monkeypatch, stage):
    _, calls = owned_target
    config = {"target": {}}

    def fail(*args, **kwargs):
        raise ValueError("construction failed")

    if stage == "defaults":
        monkeypatch.setattr(factory, "decode_defaults", fail)
    elif stage == "lane":
        config["lanes"] = {"ngram": {}}
        monkeypatch.setattr(factory, "build_lane", fail)
    elif stage == "scheduler":
        config["lanes"] = {"ngram": {}}
        config["scheduler"] = {"type": "heuristic"}
        monkeypatch.setattr(factory, "HeuristicScheduler", fail)
    else:
        config["verifier"] = "unknown"
    with pytest.raises(ValueError):
        factory.build_engine(config)
    assert calls == ["closed"]


def test_owned_target_closes_on_request_failure(owned_target, monkeypatch):
    _, calls = owned_target
    engine = factory.build_engine({"target": {}})

    def fail(*args, **kwargs):
        raise RuntimeError("verification failed")

    monkeypatch.setattr(engine, "verify", fail)
    with pytest.raises(RuntimeError, match="verification failed"), engine:
        engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=3))
    assert calls == ["closed"]


def test_trace_failure_still_closes_owned_target(owned_target, monkeypatch):
    _, calls = owned_target
    engine = factory.build_engine({"target": {}})

    def fail():
        raise OSError("trace close failed")

    monkeypatch.setattr(engine.trace, "close", fail)
    with pytest.raises(OSError, match="trace close failed"):
        engine.close()
    assert calls == ["closed"]


def test_target_close_failure_can_be_retried(owned_target, monkeypatch):
    target, calls = owned_target
    engine = factory.build_engine({"target": {}})

    def fail_once():
        calls.append("attempt")
        if len(calls) == 1:
            raise RuntimeError("close failed")

    monkeypatch.setattr(target, "close", fail_once)
    with pytest.raises(RuntimeError, match="close failed"):
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([5], DecodeConfig(max_new_tokens=0))
    engine.close()
    engine.close()
    assert calls == ["attempt", "attempt"]


def test_explicit_target_ownership_and_trace_cleanup(owned_target, tmp_path):
    target, calls = owned_target
    engine = DartEngine(target, owns_target=True, trace_path=str(tmp_path / "trace.jsonl"))
    handle = engine.trace._fh
    engine.close()
    assert calls == ["closed"] and handle.closed


def test_factory_eager_target_without_close_is_supported(tiny_target, monkeypatch):
    assert not hasattr(tiny_target, "close")
    monkeypatch.setattr(factory, "build_target", lambda spec: tiny_target)
    engine = factory.build_engine({"target": {}})
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([5], DecodeConfig(max_new_tokens=0))


@pytest.mark.parametrize("value", [1, None, "true"])
def test_ownership_option_requires_boolean(tiny_target, value):
    with pytest.raises(ValueError, match="owns_target"):
        DartEngine(tiny_target, owns_target=value)
