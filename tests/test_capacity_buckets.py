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


def test_replay_size_menu_runs_four_buckets(models):
    from dart.models.replay import CapacityBuckets

    target, reference = models
    sizes = [9, 1]
    prepared = CapacityBuckets(target, (16, 24, 32, 64), max_block_tokens=9, replay=False,
                               replay_block_sizes=sizes, capture_predictions=True)
    sizes.clear()
    assert prepared.replay_block_sizes == (1, 9)
    assert [tuple(bucket._blocks) for bucket in prepared.buckets] == [(1, 9)] * 4
    cases = [GenerationCase(str(i), prompt, budget) for i, (prompt, (_, budget, _)) in
             enumerate(zip(prompts(), REQUESTS))]
    expected = [hf_decoder(reference)(case) for case in cases]
    sequences = [case.prompt + tuple(output) for case, output in zip(cases, expected)]
    with DartEngine(prepared, _ReferenceDrafter(sequences)) as engine:
        for case, output, capacity in zip(cases, expected, (16, 32, 64, 16, 16)):
            result = engine.generate(list(case.prompt), DecodeConfig(max_new_tokens=case.max_new_tokens, draft_len=8))
            assert result.token_ids == output
            assert result.metadata["cache_capacity"] == capacity
            assert result.metadata["replay_block_sizes"] == [1, 9]


@pytest.mark.parametrize("sizes", [(), (1, 4), (2, 3)])
@pytest.mark.parametrize("predictions", [False, True])
def test_replay_size_menu_keeps_cache_and_outputs(models, monkeypatch, sizes, predictions):
    from dart.models import replay
    from dart.verification.state_sequences import StateOperation as Op
    from dart.verification.state_sequences import check_state_sequence

    base, reference = models
    target = replay.CudaGraphTarget(base, max_block_tokens=4, capacity=16, replay=False,
                                    capture_predictions=predictions, replay_block_sizes=sizes)
    try:
        assert target.replay_block_sizes == tuple(target._blocks) == sizes
        with pytest.raises(AttributeError):
            target.replay_block_sizes = (1,)
        monkeypatch.setattr(replay, "_ReplayBlock", lambda *a, **kw: pytest.fail("prepared another block"))
        operations = [Op("a", "prefill", (5, 7)), Op("b", "prefill", (11,)),
                      Op("a", "forward", (2,)), Op("b", "forward", (7, 8)),
                      Op("a", "forward", (3, 4, 5)), Op("a", "crop", length=3),
                      Op("b", "forward", (7, 8, 9, 10)), Op("a", "block", (13, 14, 15)),
                      Op("a", "forward", (4,) * 5, expect_error="ValueError"), Op("b", "forward", (17,))]
        checked = check_state_sequence(target, reference, operations)
        assert checked.passed, checked

        state, _ = target.prefill([5, 7])
        control, _ = base.prefill([5, 7])
        saved = []
        for width in (1, 2, 3, 4, 1):
            ids = [7] * width
            expected = base.forward_block(control, ids)
            if predictions:
                output = target.forward_predictions(state, ids)
                assert output == expected.argmax(-1).tolist()
            else:
                output = target.forward_block(state, ids)
                torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
            saved.append((output, expected))
        for output, expected in saved:
            if predictions:
                assert output == expected.argmax(-1).tolist()
            else:
                torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    finally:
        target.close()


def test_replay_size_menu_failed_fallback_can_retry(models, monkeypatch):
    from dart.models.replay import CudaGraphTarget

    base, _ = models
    target = CudaGraphTarget(base, capacity=16, max_block_tokens=3, replay=False, replay_block_sizes=(1,))
    try:
        state, _ = target.prefill([5, 7])
        control, _ = base.prefill([5, 7])
        target.forward_block(state, [3])
        base.forward_block(control, [3])
        workspace = target._workspace

        def fail(*args, **kwargs):
            workspace.storage.fill_(float("nan"))
            raise RuntimeError("interrupted fallback")

        with monkeypatch.context() as patch:
            patch.setattr(workspace, "update", fail)
            with pytest.raises(RuntimeError, match="interrupted fallback"):
                target.forward_block(state, [9, 11])
        assert state.seq_len == 3
        torch.testing.assert_close(target.forward_block(state, [9, 11]), base.forward_block(control, [9, 11]),
                                   atol=2e-6, rtol=2e-5)
    finally:
        target.close()


def test_replay_size_menu_counts_greedy_chunks(models):
    from dart.models.replay import CapacityBuckets

    kwargs = dict(max_block_tokens=9, capture_predictions=True, greedy_chunk_sizes=(2, 4), replay=False)
    with pytest.raises(ValueError, match="16 prepared blocks"):
        CapacityBuckets(models[0], (16, 24, 32, 64), replay_block_sizes=(1, 2, 9), **kwargs)
    prepared = CapacityBuckets(models[0], (16, 24, 32, 64), replay_block_sizes=(1, 9), **kwargs)
    try:
        assert sum(len(b._blocks) + len(b._greedy_blocks) for b in prepared.buckets) == 16
    finally:
        prepared.close()


@pytest.mark.parametrize("sizes", [[True], [1, 1], [0], [5], [1.0], [2, []], 1, "12", {1, 2}])
def test_replay_size_menu_rejects_bad_widths_before_loading(monkeypatch, sizes):
    monkeypatch.setattr(factory, "_build_target", lambda spec: pytest.fail("loaded model"))
    with pytest.raises(ValueError, match="replay_block_sizes"):
        factory.build_target({"execution": "static", "max_block_tokens": 4, "replay_block_sizes": sizes})


def test_replay_size_menu_factory():
    spec = {"model": "tiny:0", "execution": "static", "max_block_tokens": 4, "replay_block_sizes": [4, 1]}
    for capacity in (16, [16, 32]):
        target = factory.build_target(dict(spec, cache_capacity=capacity))
        try:
            assert target.replay_block_sizes == (1, 4)
        finally:
            target.close()
    with pytest.raises(ValueError, match="static or cuda_graph"):
        factory.build_target(dict(spec, execution="eager"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires a GPU")
def test_replay_size_menu_mixes_capture_and_fallback(models):
    from dart.models.replay import CudaGraphTarget
    from dart.models.target import TargetModel
    from dart.verification.state_sequences import StateOperation as Op
    from dart.verification.state_sequences import check_state_sequence

    base, reference = models
    target = CudaGraphTarget(TargetModel(base.model, device="cuda"), capacity=16, max_block_tokens=3,
                             replay_block_sizes=(1, 3), capture_predictions=True)
    try:
        assert tuple(target._blocks) == (1, 3)
        assert all(block.graph is not None for block in target._blocks.values())
        result = check_state_sequence(target, reference.to("cuda"), [
            Op("a", "prefill", (3, 5)), Op("b", "prefill", (7,)), Op("a", "forward", (11,)),
            Op("b", "forward", (4, 5)), Op("a", "block", (8, 9)), Op("b", "forward", (5, 6, 7)),
            Op("b", "crop", length=2), Op("b", "forward", (4, 8))])
        assert result.passed, result
    finally:
        target.close()
