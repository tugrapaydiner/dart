from __future__ import annotations

import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.ar_draft import ARDrafter
from dart.drafters.base import NullDrafter
from dart.drafters.block_sparse_kv import BlockSparseKVDrafter
from dart.drafters.compact_diffusion import (
    CompactDiffusionDrafter,
    CompactDiffusionModel,
    CompactDrafterConfig,
)
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.drafters.mtp import MTPDrafter, MTPHeads
from dart.drafters.sparse_attention import SparseAttentionDrafter
from dart.kv.cache import QuantSpec
from dart.kv.compressors.token_drop import TokenDropSpec
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    target.record_hidden = True
    target.eos_token_ids = frozenset()
    reference.config.eos_token_id = None
    reference.generation_config.eos_token_id = None
    yield target, reference
    torch.set_num_threads(previous)


def build_drafter(name, target):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        if name == 'ar':
            return ARDrafter(target, seed=23)
        if name == 'compressed':
            return CompressedKVDrafter(target, QuantSpec(bits=4, sink_len=1, window_len=2))
        if name == 'sparse':
            return SparseAttentionDrafter(target, TokenDropSpec(sink_len=1, window_len=3))
        if name == 'block_sparse':
            return BlockSparseKVDrafter(target, block_size=2, top_k=1, sink_blocks=1, recent_blocks=1)
        if name == 'mtp':
            return MTPDrafter(target, MTPHeads(32, 67, num_offsets=3))
        if name == 'compact':
            model = CompactDiffusionModel(target, CompactDrafterConfig(hidden_size=32, num_layers=1, num_heads=4))
            return CompactDiffusionDrafter(model, canvas_len=4, denoise_steps=1)
        model = DiffusionDrafterModel(DiffusionDrafterConfig(67, hidden_size=32, num_layers=1,
                                                          num_heads=4, ctx_window=8, max_canvas=4))
        return DiffusionDrafter(model, canvas_len=4, denoise_steps=1,
                               reconcile=ReconcilePolicy(mode='reconcile', repair_steps=1))


def assert_cleared(drafter, target):
    for name in ('_state', '_last_canvas', '_warm', '_hidden', '_entropy'):
        assert getattr(drafter, name, None) is None, name
    for name in ('_processed', '_context', '_committed_ids'):
        assert getattr(drafter, name, []) == [], name
    assert target.last_hidden_block is None


@pytest.mark.parametrize('lane', ['ar', 'compressed', 'sparse', 'block_sparse', 'diffusion', 'compact', 'mtp'])
def test_completed_requests_release_drafter_data_and_keep_statistics(models, monkeypatch, lane):
    target, reference = models
    drafter = build_drafter(lane, target)
    observed, retained = [], []
    kv_stats, rng_states = [], []
    observe = drafter.observe

    def capture(result):
        observe(result)
        observed.append(dict(getattr(drafter, 'stats', {})))
        if hasattr(drafter, 'kv_stats'):
            kv_stats.append(drafter.kv_stats())
        if hasattr(drafter, 'gen'):
            rng_states.append(drafter.gen.get_state().clone())
        for item in (getattr(drafter, '_state', None), target.last_hidden_block):
            if item is not None:
                retained.append(weakref.ref(item))

    monkeypatch.setattr(drafter, 'observe', capture)
    engine = DartEngine(target, drafter)
    prompt = [5, 6, 7, 8, 9, 10, 11, 12]
    expected = hf_decoder(reference)(GenerationCase('cleanup', tuple(prompt), 9))
    result = engine.generate(prompt, DecodeConfig(max_new_tokens=9, draft_len=3, canvas_len=3))
    assert result.token_ids == expected and observed
    assert_cleared(drafter, target)
    assert getattr(drafter, 'stats', {}) == observed[-1]
    if kv_stats:
        assert drafter.kv_stats() == {}
        assert drafter.last_kv_stats == kv_stats[-1]
        assert drafter.last_kv_stats['logical_compressed_bytes'] > 0
        json.dumps(drafter.last_kv_stats)
    if rng_states:
        assert torch.equal(drafter.gen.get_state(), rng_states[-1])
    gc.collect()
    assert retained and all(item() is None for item in retained)
    assert engine.generate(prompt, DecodeConfig(max_new_tokens=9, draft_len=3, canvas_len=3)).token_ids == expected
    assert_cleared(drafter, target)


class Cancelled(BaseException):
    pass


@pytest.mark.parametrize('error_type', [RuntimeError, Cancelled])
def test_failed_incremental_draft_does_not_poison_the_next_request(models, monkeypatch, error_type):
    target, reference = models
    drafter = ARDrafter(target)
    engine = DartEngine(target, drafter)
    original = drafter.propose
    failed_states = []
    error = error_type('draft layer failed')
    count = 0

    def fail(module, args):
        failed_states.append(drafter._state)
        raise error

    def propose(committed, budget):
        nonlocal count
        count += 1
        if count != 2:
            return original(committed, budget)
        hook = target.model.model.layers[1].register_forward_pre_hook(fail)
        try:
            return original(committed, budget)
        finally:
            hook.remove()

    prompt, config = [5, 6, 7], DecodeConfig(max_new_tokens=12, draft_len=3)
    with monkeypatch.context() as patch:
        patch.setattr(drafter, 'propose', propose)
        with pytest.raises(error_type) as caught:
            engine.generate(prompt, config)
        assert caught.value is error
    assert count == 2 and failed_states[0]._failed
    cache = failed_states[0].cache
    assert cache.layers[0].keys.shape[-2] > cache.layers[1].keys.shape[-2]
    expected = hf_decoder(reference)(GenerationCase('recovered', tuple(prompt), 12))
    assert engine.generate(prompt, config).token_ids == expected
    assert_cleared(drafter, target)


@pytest.mark.parametrize('stage', ['prefill', 'draft', 'verify', 'observe', 'trace', 'decode'])
@pytest.mark.parametrize('error_type', [RuntimeError, Cancelled])
def test_request_errors_clear_owned_draft_state(models, monkeypatch, stage, error_type):
    target, _ = models
    drafter = ARDrafter(target)
    drafter.propose([5, 6, 7], 3)
    assert drafter._state is not None
    engine = DartEngine(target, drafter)
    error = error_type('request failed')

    def fail(*args, **kwargs):
        raise error

    if stage == 'decode':
        target.tokenizer = SimpleNamespace(decode=fail)
    else:
        obj, attr = {'prefill': (target, 'prefill'), 'draft': (drafter, 'propose'),
                     'verify': (engine, 'verify'), 'observe': (drafter, 'observe'),
                     'trace': (engine.trace, 'log_round')}[stage]
        monkeypatch.setattr(obj, attr, fail)
    with pytest.raises(error_type) as caught:
        engine.generate([13, 17, 19], DecodeConfig(max_new_tokens=8, draft_len=3))
    assert caught.value is error
    assert_cleared(drafter, target)


def test_cleanup_runs_once_per_drafter_and_drains_after_a_hook_fails(models):
    target, _ = models
    calls = []
    error = RuntimeError('cleanup failed')

    class Lane(NullDrafter):
        def __init__(self, name):
            self.name = name

        def finish_request(self):
            calls.append(self.name)
            if self.name == 'second':
                raise error

    first, second = Lane('first'), Lane('second')
    scheduler = SimpleNamespace(requires_entropy=False)
    engine = DartEngine(target, lanes={'a': first, 'b': second, 'alias': first}, scheduler=scheduler)
    with pytest.raises(RuntimeError) as caught:
        engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=1))
    assert caught.value is error and sorted(calls) == ['first', 'second']
    assert target.last_hidden_block is None


def test_sampling_cleanup_preserves_the_advanced_rng(models, monkeypatch):
    target, _ = models
    drafter = ARDrafter(target, seed=23)
    initial = drafter.gen.get_state().clone()
    final = []
    finish = drafter.finish_request

    def capture():
        final.append(drafter.gen.get_state().clone())
        finish()

    monkeypatch.setattr(drafter, 'finish_request', capture)
    result = DartEngine(target, drafter).generate([5, 6, 7], DecodeConfig(
        mode='exact_sampling', max_new_tokens=9, draft_len=3, seed=29,
    ))
    assert len(result.token_ids) == 9 and len(final) == 1
    assert not torch.equal(initial, final[0])
    assert torch.equal(drafter.gen.get_state(), final[0])
    assert_cleared(drafter, target)


@pytest.mark.parametrize('budget', [0, -1])
def test_requests_without_execution_do_not_clear_existing_external_state(models, monkeypatch, budget):
    target, _ = models
    target.prefill([5, 6, 7])
    hidden = target.last_hidden_block
    drafter = ARDrafter(target)

    def unexpected():
        pytest.fail('cleanup ran without executing a request')

    monkeypatch.setattr(drafter, 'finish_request', unexpected, raising=False)
    engine = DartEngine(target, drafter)
    if budget < 0:
        with pytest.raises(ValueError):
            engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=budget))
    else:
        assert engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=0)).token_ids == []
    assert target.last_hidden_block is hidden
