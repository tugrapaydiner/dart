from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from dart.drafters.base import NullDrafter
from dart.models.replay import CudaGraphTarget
from dart.runtime import metrics
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def cuda_counters(monkeypatch):
    calls = []

    def available():
        calls.append(('available', None))
        return True

    def peak(device=None):
        calls.append(('peak', None if device is None else str(device)))
        return 3_000_000_000

    def reset(device=None):
        calls.append(('reset', None if device is None else str(device)))

    def free(device=None):
        calls.append(('free', None if device is None else str(device)))
        return 4_000_000_000, 8_000_000_000

    monkeypatch.setattr(torch.cuda, 'is_available', available)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', peak)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', free)
    return calls


@pytest.mark.parametrize('device', ['cpu', torch.device('cpu'), 'mps'])
def test_non_cuda_memory_queries_do_not_touch_cuda(device, cuda_counters):
    assert metrics.gpu_memory_gb(device) == 0.0
    metrics.reset_gpu_memory_stats(device)
    assert DartEngine._gpu_free_gb(device) == float('inf')
    assert cuda_counters == []


@pytest.mark.parametrize('device', [None, 'cuda:1', torch.device('cuda:2')])
def test_memory_queries_use_the_selected_cuda_device(device, cuda_counters):
    assert metrics.gpu_memory_gb(device) == 3.0
    metrics.reset_gpu_memory_stats(device)
    assert DartEngine._gpu_free_gb(device) == 4.0
    selected = None if device is None else str(device)
    assert [call for call in cuda_counters if call[0] != 'available'] == [
        ('peak', selected), ('reset', selected), ('free', selected),
    ]


def test_unavailable_cuda_does_not_read_memory(cuda_counters, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert metrics.gpu_memory_gb('cuda:1') == 0.0
    metrics.reset_gpu_memory_stats('cuda:1')
    assert DartEngine._gpu_free_gb('cuda:1') == float('inf')
    assert cuda_counters == []


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield _synthetic_models(17)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('execution', ['eager', pytest.param('static', marks=pytest.mark.skipif(
    Version(version('transformers')) < Version('5.16.1'), reason='static execution requires transformers >= 5.16.1',
))])
@pytest.mark.parametrize('scheduled', [False, True])
@pytest.mark.parametrize('budget', [0, 5])
def test_cpu_generation_does_not_reset_or_read_unrelated_gpu_stats(models, execution, scheduled, budget, monkeypatch):
    base, reference = models
    target = CudaGraphTarget(base, capacity=16, replay=False) if execution == 'static' else base
    expected = hf_decoder(reference)(GenerationCase('memory', (5, 6, 7), budget))
    calls, states = [], []

    def unexpected(*args, **kwargs):
        calls.append(True)
        pytest.fail('CPU request touched CUDA memory accounting')

    for name in ('is_available', 'max_memory_allocated', 'reset_peak_memory_stats', 'mem_get_info'):
        monkeypatch.setattr(torch.cuda, name, unexpected)

    class Scheduler(HeuristicScheduler):
        def choose_action(self, state, **kwargs):
            states.append(state.gpu_free_gb)
            return super().choose_action(state, **kwargs)

    options = {}
    if scheduled:
        options = {'lanes': {'target_only': NullDrafter()},
                   'scheduler': Scheduler(SchedulerConfig(available_lanes=['target_only']))}
    try:
        with DartEngine(target, owns_target=True, **options) as engine:
            out = engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=budget))
        assert out.token_ids == expected and not calls
        assert out.metadata['gpu_peak_gb'] == 0.0 and out.metadata['gpu_memory_device'] is None
        assert out.metadata['gpu_memory_scope'] == 'process allocations on target device'
        if scheduled and budget:
            assert states and all(value == float('inf') for value in states)
    finally:
        if hasattr(target, 'close'):
            target.close()


def test_empty_cuda_request_passes_its_device_to_memory_accounting(cuda_counters):
    target = SimpleNamespace(device='cuda:1', model=SimpleNamespace(config=SimpleNamespace(name_or_path='synthetic')),
                             tokenizer=None, validate_generation=lambda *args: None, validate_prompt=lambda *args: None)
    with DartEngine(target) as engine:
        output = engine.generate([5, 6], DecodeConfig(max_new_tokens=0))
    assert output.token_ids == []
    assert output.metadata['gpu_peak_gb'] == 3.0 and output.metadata['gpu_memory_device'] == 'cuda:1'
    assert [call for call in cuda_counters if call[0] != 'available'] == [('reset', 'cuda:1'), ('peak', 'cuda:1')]
