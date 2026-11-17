from __future__ import annotations

import gc
import weakref
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from dart.runtime import factory
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _ReferenceDrafter, _synthetic_models

pytestmark = pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                                reason="static execution requires transformers >= 5.16.1")

REQUESTS = [(4, 8, 16), (8, 17, 32), (30, 20, 64), (6, 10, 16), (16, 0, 16)]  # prompt, budget, bucket


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    yield target, reference
    torch.set_num_threads(threads)


def buckets(target, capacities=(16, 32, 64), **options):
    from dart.models.replay import CapacityBuckets

    return CapacityBuckets(target, capacities, max_block_tokens=5, replay=False, **options)


def prompts():
    generator = torch.Generator().manual_seed(3)
    return [tuple(torch.randint(3, 67, (length,), generator=generator).tolist()) for length, _, _ in REQUESTS]


def test_requests_use_the_smallest_bucket_that_fits(models):
    target, reference = models
    decode = hf_decoder(reference)
    cases = [GenerationCase(f"case-{i}", prompt, budget) for i, (prompt, (_, budget, _)) in
             enumerate(zip(prompts(), REQUESTS))]
    expected = [decode(case) for case in cases]
    sequences = [case.prompt + tuple(output) for case, output in zip(cases, expected)]
    prepared = buckets(target)
    try:
        assert prepared.capacities == (16, 32, 64) and prepared.context_limit == 64
        with DartEngine(prepared, _ReferenceDrafter(sequences)) as engine:
            for case, output, (_, _, capacity) in zip(cases, expected, REQUESTS):
                result = engine.generate(list(case.prompt), DecodeConfig(max_new_tokens=case.max_new_tokens))
                assert result.token_ids == output
                assert result.metadata["cache_capacity"] == capacity
                assert prepared.for_request(len(case.prompt), case.max_new_tokens).capacity == capacity
    finally:
        prepared.close()


def test_requests_beyond_the_largest_bucket_are_rejected_before_execution(models, monkeypatch):
    target, _ = models
    prepared = buckets(target, (16, 32))
    calls = []
    forward = target.model.forward
    monkeypatch.setattr(target.model, "forward", lambda *a, **k: calls.append(1) or forward(*a, **k))
    try:
        with DartEngine(prepared) as engine:
            with pytest.raises(ValueError, match="exceeds replay capacity 32"):
                engine.generate([5] * 20, DecodeConfig(max_new_tokens=13))
            assert calls == []
            assert engine.generate([5] * 20, DecodeConfig(max_new_tokens=12)).metadata["cache_capacity"] == 32
    finally:
        prepared.close()


def test_greedy_generate_uses_a_fitting_bucket(models):
    target, reference = models
    prepared = buckets(target, (16, 64))
    try:
        prompt = prompts()[2]
        assert prepared.greedy_generate(list(prompt), 20) == hf_decoder(reference)(GenerationCase("g", prompt, 20))
        assert prepared.greedy_generate([5, 6], 0) == []
    finally:
        prepared.close()


@pytest.mark.parametrize("capacities, match", [
    ((), "one to four"), ((8, 16, 24, 32, 40), "one to four"), ((16, 16), "distinct"), ([16, 32], "tuple"),
    ((16.0,), "integers"), ((8, 16, 24, 32), "16 prepared blocks"),
])
def test_rejects_invalid_bucket_menus(models, capacities, match):
    with pytest.raises(ValueError, match=match):
        buckets(models[0], capacities)


def test_failed_bucket_preparation_closes_earlier_buckets(models, monkeypatch):
    from dart.models.replay import CudaGraphTarget

    closed, close = [], CudaGraphTarget.close
    monkeypatch.setattr(CudaGraphTarget, "close", lambda self: closed.append(self.capacity) or close(self))
    with pytest.raises(ValueError, match="replay capacity"):
        buckets(models[0], (16, 32, 65))
    assert closed == [32, 16]


def test_close_releases_every_bucket_workspace(models):
    prepared = buckets(models[0], (16, 32))
    workspaces = [weakref.ref(bucket._workspace) for bucket in prepared.buckets]
    prepared.close()
    prepared.close()
    gc.collect()
    assert all(bucket._closed for bucket in prepared.buckets) and all(ref() is None for ref in workspaces)
    with pytest.raises(RuntimeError, match="closed"):
        prepared.for_request(4, 4)


def test_factory_builds_buckets_from_a_capacity_list():
    from dart.models.replay import CapacityBuckets

    spec = {"model": "tiny:0", "execution": "static", "cache_capacity": [16, 64], "max_block_tokens": 4}
    engine = factory.build_engine({"target": spec, "lanes": {"ngram": {}}})
    prepared = engine.target
    assert isinstance(prepared, CapacityBuckets) and prepared.capacities == (16, 64)
    with engine:
        result = engine.generate([5, 6, 7, 5, 6, 7], DecodeConfig(max_new_tokens=8))
        assert result.metadata["cache_capacity"] == 16
    assert all(bucket._closed for bucket in prepared.buckets)


def test_draft_lanes_need_a_single_capacity():
    lane = {"model": "tiny:1", "execution": "static", "cache_capacity": [16, 64]}
    with pytest.raises(ValueError, match="single cache capacity"):
        factory.build_engine({"target": {"model": "tiny:0"}, "lanes": {"ar_draft": lane}})
