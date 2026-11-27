from __future__ import annotations

import copy
import gc
import weakref
from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from dart.models import replay
from dart.models.replay import CudaGraphTarget
from dart.runtime.factory import build_engine
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_sequences import StateOperation as Op
from dart.verification.state_sequences import check_state_sequence
from dart.verification.synthetic import _synthetic_models

pytestmark = pytest.mark.skipif(Version(version('transformers')) < Version('5.16.1'),
                                reason='static execution requires transformers >= 5.16.1')


@pytest.fixture(autouse=True)
def no_cuda_calls(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('CPU static execution called a CUDA API')

    for name in ('current_stream', 'Stream', 'CUDAGraph', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, unexpected)


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    base, reference = _synthetic_models(17)
    base.eos_token_ids = frozenset()
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    yield base, reference
    torch.set_num_threads(threads)


@pytest.mark.parametrize('capacity', [4, 8, 16])
@pytest.mark.parametrize('predictions', [False, True])
def test_static_cpu_capacity_shapes_and_interleaved_requests(models, capacity, predictions):
    base, reference = models
    target = CudaGraphTarget(base, capacity=capacity, max_block_tokens=3, replay=False,
                             capture_predictions=predictions)
    try:
        assert target.execution_backend == 'static' and target._stream is None
        assert target._workspace.storage.device.type == 'cpu'
        assert set(target._blocks) == {1, 2, 3}
        assert all(block.graph is None for block in target._blocks.values())
        case = GenerationCase('capacity', (5, 6), capacity - 2)
        assert target.greedy_generate(list(case.prompt), case.max_new_tokens) == hf_decoder(reference)(case)
        operations = [Op('a', 'prefill', (5,) * (capacity - 2)), Op('b', 'prefill', (13,)),
                      Op('a', 'forward', (7, 8)), Op('a', 'forward', (9,), expect_error='ValueError'),
                      Op('b', 'forward', (17,)), Op('a', 'crop', length=0),
                      Op('a', 'forward', (19, 23, 29)),
                      Op('b', 'forward', (7, 8, 9, 10), expect_error='ValueError'),
                      Op('b', 'forward', (11,)), Op('b', 'block', (31,)),
                      Op('a', 'release'), Op('a', 'prefill', (32,)), Op('a', 'forward', (37, 41))]
        report = check_state_sequence(target, reference, operations)
        assert report.passed, report
        assert len(report.steps) == len(operations)
    finally:
        target.close()


@pytest.mark.parametrize('predictions', [False, True])
def test_factory_static_cpu_owns_and_releases_the_real_workspace(predictions):
    config = {'target': {'model': 'tiny:23', 'device': 'cpu', 'precision': 'fp32',
                        'execution': 'static', 'cache_capacity': 8, 'max_block_tokens': 3,
                        'capture_predictions': predictions, 'hidden_size': 32, 'intermediate_size': 64,
                        'num_layers': 1, 'vocab_size': 67}}
    with build_engine(config) as engine:
        target = engine.target
        workspace, block = weakref.ref(target._workspace), weakref.ref(target._blocks[1])
        reference = copy.deepcopy(target.model)
        case = GenerationCase('factory', (5, 6, 7), 5)
        assert engine.generate(list(case.prompt), DecodeConfig(max_new_tokens=5, draft_len=2)).token_ids == hf_decoder(reference)(case)
    gc.collect()
    assert workspace() is None and block() is None
    assert target._closed and target._workspace is None and not target._blocks
    target.close()
    with pytest.raises(RuntimeError, match='closed'):
        target.prefill([5, 6])
    with pytest.raises(RuntimeError, match='closed'):
        target.greedy_generate([5, 6], 0)


def test_cpu_static_snapshots_cannot_cross_targets(models):
    base, reference = models
    targets = [CudaGraphTarget(base, capacity=n, max_block_tokens=3, replay=False) for n in (8, 16)]
    first, second = targets
    try:
        state, _ = first.prefill([5, 6])
        first.forward_block(state, [7])
        with pytest.raises(ValueError, match='different replay target'):
            second.forward_block(state, [8])
        with pytest.raises(ValueError, match='different replay target'):
            second.crop(state, 1)
        assert state.seq_len == 3
        expected = hf_decoder(reference)(GenerationCase('owner', (5, 6, 7), 2))
        assert first.forward_block(state, [expected[0]]).argmax(-1).tolist() == expected[1:]
        first.close()
        assert second.greedy_generate([5, 6, 7], 2) == expected
    finally:
        for target in targets:
            target.close()


@pytest.mark.parametrize('options,match', [
    ({'capacity': 0}, 'capacity'), ({'capacity': True}, 'capacity'), ({'capacity': 65}, 'capacity'),
    ({'max_block_tokens': 0}, 'max_block_tokens'), ({'max_block_tokens': 17}, 'max_block_tokens'),
    ({'capacity': 2, 'max_block_tokens': 3}, 'max_block_tokens'),
    ({'replay': True}, 'float32 CUDA'),
])
def test_cpu_static_rejects_invalid_preparation_before_workspace_allocation(models, monkeypatch, options, match):
    monkeypatch.setattr(replay, '_Workspace', lambda *args: pytest.fail('invalid configuration allocated a workspace'))
    with pytest.raises(ValueError, match=match):
        CudaGraphTarget(models[0], **{'capacity': 16, 'max_block_tokens': 3, 'replay': False, **options})


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cpu_static_keeps_fp32_requirement(models, monkeypatch, dtype):
    monkeypatch.setattr(models[0], 'dtype', dtype)
    with pytest.raises(ValueError, match='float32'):
        CudaGraphTarget(models[0], replay=False)


def test_cpu_static_keeps_model_storage_guard(models, monkeypatch):
    target = CudaGraphTarget(models[0], capacity=8, replay=False)
    try:
        layer = target.model.model.layers[0].self_attn.q_proj
        monkeypatch.setattr(layer, 'weight', torch.nn.Parameter(layer.weight.detach().clone()))
        with pytest.raises(ValueError, match='tensor storage changed'):
            target.prefill([5, 6])
    finally:
        target.close()


@pytest.mark.parametrize("mode,width", [("prepared", 3), ("fallback", 2), ("predictions", 1), ("chunk", 4)])
@pytest.mark.parametrize("at_limit", [False, True])
@torch.inference_mode()
def test_active_columns_match_live_keys_and_mask(models, monkeypatch, mode, width, at_limit):
    base, reference = models
    capacity = 16
    prompt = [5] * (capacity - width if at_limit else 3)
    target = CudaGraphTarget(base, capacity=capacity, max_block_tokens=3, replay=False,
                             replay_block_sizes=(1, 3), capture_predictions=True, greedy_chunk_sizes=(4,))
    try:
        target.eos_token_ids = frozenset()
        state, _ = target.prefill(prompt)
        owner = (target._workspace if mode == "fallback" else
                 target._greedy_blocks[width] if mode == "chunk" else target._blocks[width])
        forward = owner.forward_model
        address = target._workspace.storage.untyped_storage().data_ptr()
        seen = []

        def observe(**kwargs):
            cache = kwargs["past_key_values"]
            update = cache.update
            q, columns = kwargs["attention_mask"].shape[-2:]

            def rows(keys, values, layer, *args, **options):
                k, v = update(keys, values, layer, *args, **options)
                assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() == address
                seen.append((layer, q, columns, k.size(-2), v.size(-2)))
                return k, v

            with monkeypatch.context() as patch:
                patch.setattr(cache, "update", rows)
                return forward(**kwargs)

        monkeypatch.setattr(owner, "forward_model", observe)
        ids = list(range(9, 9 + width))
        if mode == "chunk":
            got = target.forward_greedy(state, ids[0], width)
            wanted = hf_decoder(reference)(GenerationCase("chunk", tuple(prompt + ids[:1]), width))
            assert got == wanted
            sizes = [(1, len(prompt) + j + 1) for j in range(width)]
        else:
            if mode == "predictions":
                assert target.forward_predictions(state, ids) == hf_decoder(reference)(
                    GenerationCase("single", tuple(prompt + ids), 1))
            else:
                logits = target.forward_block(state, ids)
                native = reference(input_ids=torch.tensor([prompt + ids])).logits[0, -width:]
                torch.testing.assert_close(logits, native, rtol=2e-5, atol=2e-6)
            sizes = [(width, len(prompt) + width)]
        assert seen == [(i, q, end, end, end) for q, end in sizes for i in range(reference.config.num_hidden_layers)]
        assert state.seq_len == len(prompt) + width
        assert target._workspace.storage.size(-2) == capacity
    finally:
        target.close()


@pytest.mark.parametrize("width,start,active", [(1, 0, True), (1, 15, True), (3, 0, True),
                                                (3, 7, True), (3, 13, True), (3, 7, False), (1, 5, False)])
def test_causal_mask_skips_visible_prefix(monkeypatch, width, start, active):
    capacity = 16
    workspace = SimpleNamespace(columns=torch.arange(capacity), storage=torch.zeros(1, 1, capacity, 2))
    end = start + width if active else None
    columns = end if active else capacity
    low = torch.finfo(workspace.storage.dtype).min
    expected = torch.tensor([[0.0 if k <= start + i else low for k in range(columns)] for i in range(width)])
    comparisons = []
    greater = torch.Tensor.__gt__

    def compare(left, right):
        comparisons.append(torch.broadcast_shapes(left.shape, right.shape).numel())
        return greater(left, right)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__gt__", compare)
        mask = replay._causal_logits(lambda **args: SimpleNamespace(logits=args["attention_mask"]), workspace,
                                     torch.zeros(1, width, dtype=torch.long), torch.arange(start, start + width), end=end)
    assert torch.equal(mask[0, 0], expected)
    if active:
        assert sum(comparisons) <= (width * width if width > 1 else 0)


@pytest.mark.parametrize("mode", ["prepared", "fallback", "chunk"])
@pytest.mark.parametrize("component", [0, 1], ids=["keys", "values"])
@torch.inference_mode()
def test_active_columns_isolate_a_shorter_request(models, mode, component):
    base, reference = models
    target = CudaGraphTarget(base, capacity=16, max_block_tokens=3, replay=False,
                             replay_block_sizes=(1, 3), capture_predictions=True, greedy_chunk_sizes=(4,))
    try:
        target.eos_token_ids = frozenset()
        longer, _ = target.prefill([5] * 10)
        target.forward_block(longer, [7])
        longer.cache.storage[:, component, :, :, 8, :].fill_(float("nan"))  # A mutable caller-owned KV row.
        assert not target.forward_block(longer, [9]).isfinite().all()
        prompt, ids = [5, 7, 8], [11, 13] if mode == "fallback" else [11]
        short, _ = target.prefill(prompt)
        if mode == "chunk":
            got = target.forward_greedy(short, ids[0], 4)
            wanted = hf_decoder(reference)(GenerationCase("short", tuple(prompt + ids), 4))
            assert got == wanted
        else:
            expected = reference(input_ids=torch.tensor([prompt + ids])).logits[0, -len(ids):]
            actual = target.forward_block(short, ids)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert longer.seq_len == 12 and longer.cache.storage[:, component, :, :, 8, :].isnan().all()
    finally:
        target.close()


@pytest.mark.parametrize("chunk", [False, True])
@torch.inference_mode()
def test_fixed_replay_clears_stale_columns_after_an_uncaptured_round(models, monkeypatch, chunk):
    base, reference = models
    target = CudaGraphTarget(base, capacity=16, max_block_tokens=3, replay=False,
                             replay_block_sizes=(1, 3), capture_predictions=True, greedy_chunk_sizes=(4,))
    try:
        target.eos_token_ids = frozenset()
        long, _ = target.prefill([5] * 10)
        target.forward_block(long, [7])
        slot = 5 if chunk else 8  # The first chunk step cannot see a later step.
        long.cache.storage[..., slot, :].fill_(float("nan"))
        target.forward_block(long, [9])
        prompt = [5, 6, 7]
        middle, _ = target.prefill(prompt)
        target.forward_block(middle, [11, 12])  # Short fallback leaves the old tail in storage.
        short, _ = target.prefill(prompt)
        block = target._greedy_blocks[4] if chunk else target._blocks[3]
        forward, columns = block.forward_model, []

        def observe(**kwargs):
            columns.append(kwargs["attention_mask"].size(-1))
            return forward(**kwargs)

        class FixedReplay:
            def replay(self):
                block.output = block.forward()

            def reset(self):
                pass

        monkeypatch.setattr(block, "forward_model", observe)
        monkeypatch.setattr(block, "graph", FixedReplay())
        if chunk:
            actual = target.forward_greedy(short, 11, 4)
            assert actual == hf_decoder(reference)(GenerationCase("fixed", tuple(prompt + [11]), 4))
        else:
            ids = [11, 12, 13]
            actual = target.forward_block(short, ids)
            expected = reference(input_ids=torch.tensor([prompt + ids])).logits[0, -len(ids):]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert columns == [16] * (4 if chunk else 1)
        assert long.cache.storage[..., slot, :].isnan().all()
    finally:
        target.close()
