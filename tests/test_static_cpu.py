from __future__ import annotations

import copy
import gc
import weakref
from importlib.metadata import version

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
