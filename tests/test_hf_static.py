from __future__ import annotations

import copy
import gc
import weakref

import pytest
import torch
from transformers import StaticCache

from benchmarks.run_bench import build_hf_method
from dart.models.replay import CudaGraphTarget
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def models():
    target, reference = _synthetic_models(17)
    target.eos_token_ids = frozenset()
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    return target, reference


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("budget", [1, 7, 12])
def test_native_static_outputs_capacity_and_request_lifetime(models, monkeypatch, attention, budget):
    target, reference = models
    target.model.set_attn_implementation(attention)
    reference.set_attn_implementation(attention)
    prior_cache = object()
    target.model._cache = prior_cache
    before = copy.deepcopy(target.model.generation_config)
    caches, tensors = [], []

    def observe(module, args, kwargs, output):
        cache = kwargs["past_key_values"]
        assert type(cache) is StaticCache and output.past_key_values is cache
        if not caches or caches[-1]() is not cache:
            caches.append(weakref.ref(cache))
            for layer in cache.layers:
                assert layer.keys.shape[-2] == layer.values.shape[-2] == 16
                tensors.extend([weakref.ref(layer.keys), weakref.ref(layer.values)])

    generate = target.model.generate

    def inspect(*args, **kwargs):
        assert kwargs["attention_mask"].all()
        assert kwargs["disable_compile"] is True
        return generate(*args, **kwargs)

    monkeypatch.setattr(target.model, "generate", inspect)
    monkeypatch.setattr(target.model, "get_compiled_call", lambda *a, **k: pytest.fail("unexpected compilation"))
    handle = target.model.register_forward_hook(observe, with_kwargs=True)
    method = build_hf_method(target, budget, record_hidden=False, static_capacity=16)
    try:
        for prompt in ([3, 5, 7], [0, 0, 5, 0], [3, 5, 7]):
            expected = hf_decoder(reference)(GenerationCase("static", tuple(prompt), budget))
            assert method(prompt).token_ids == expected
            gc.collect()
            assert all(ref() is None for ref in [*caches, *tensors])
            assert target.model.generation_config == before and target.model._cache is prior_cache
        assert len(caches) == 3
    finally:
        handle.remove()


@pytest.mark.parametrize("eos_offset", [0, 2, 5])
def test_native_static_keeps_eos_and_ignores_inherited_generation_settings(models, monkeypatch, eos_offset):
    target, reference = models
    prompt, budget = [3, 5, 7], 6
    expected = hf_decoder(reference)(GenerationCase("eos", tuple(prompt), budget))
    eos = expected[eos_offset]
    target.eos_token_ids = frozenset([eos])
    expected = expected[:expected.index(eos) + 1]
    config = target.model.generation_config
    config.do_sample, config.num_beams, config.num_return_sequences = True, 3, 2
    config.repetition_penalty, config.min_new_tokens = 2.0, 20
    config.forced_bos_token_id, config.forced_eos_token_id = 1, 2
    config.suppress_tokens, config.cache_implementation = expected[:1], "static"
    config.return_dict_in_generate, config.output_scores = True, True
    config.eos_token_id, config.disable_compile = None, False
    before, generate = copy.deepcopy(config), target.model.generate

    def inspect(*args, **kwargs):
        assert kwargs["disable_compile"] is True and kwargs["cache_implementation"] is None
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        return generate(*args, **kwargs)

    monkeypatch.setattr(target.model, "generate", inspect)
    assert build_hf_method(target, budget, static_capacity=16)(prompt).token_ids == expected
    assert target.model.generation_config == before and not hasattr(target.model, "_cache")


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, 65])
def test_bad_capacity_is_rejected_before_model_work(models, monkeypatch, capacity):
    monkeypatch.setattr(models[0].model, "generate", lambda *a, **k: pytest.fail("unexpected generation"))
    with pytest.raises(ValueError, match="static_capacity"):
        build_hf_method(models[0], 4, static_capacity=capacity)


@pytest.mark.parametrize("prompt,budget", [([], 1), ([3, 5], 7), ([3], -1), ([3], True), ([-1], 1), ([67], 1)])
def test_static_request_admission_precedes_cache_allocation(models, monkeypatch, prompt, budget):
    monkeypatch.setattr("benchmarks.run_bench.StaticCache", lambda *a, **k: pytest.fail("unexpected cache allocation"))
    with pytest.raises(ValueError):
        build_hf_method(models[0], budget, static_capacity=8)(prompt)


def test_static_zero_budget_does_no_model_work(models, monkeypatch):
    monkeypatch.setattr("benchmarks.run_bench.StaticCache", lambda *a, **k: pytest.fail("unexpected cache allocation"))
    monkeypatch.setattr(models[0].model, "generate", lambda *a, **k: pytest.fail("unexpected generation"))
    assert build_hf_method(models[0], 0, static_capacity=8)([0, 5, 0]).token_ids == []


@pytest.mark.parametrize("case", ["lookup", "dtype", "model", "attention"])
def test_static_rejects_unsupported_controls(models, monkeypatch, case):
    target = models[0]
    options = {"prompt_lookup_num_tokens": 2} if case == "lookup" else {}
    if case == "dtype":
        target.dtype = torch.float16
    elif case == "model":
        target.model = torch.nn.Linear(2, 2)
    elif case == "attention":
        target.model.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="static"):
        build_hf_method(target, 4, static_capacity=8, **options)


def test_failed_static_forward_does_not_poison_next_request(models, monkeypatch):
    target, reference = models
    layer = target.model.model.layers[-1]
    forward, caches = layer.forward, []

    def fail(*args, **kwargs):
        caches.append(weakref.ref(kwargs["past_key_values"]))
        raise RuntimeError("later layer failed")

    method = build_hf_method(target, 5, static_capacity=16)
    with monkeypatch.context() as patch:
        patch.setattr(layer, "forward", fail)
        with pytest.raises(RuntimeError, match="later layer failed"):
            method([3, 5, 7])
    gc.collect()
    assert caches and all(ref() is None for ref in caches)
    assert layer.forward == forward and not hasattr(target.model, "_cache")
    expected = hf_decoder(reference)(GenerationCase("recovery", (3, 5, 7), 5))
    assert method([3, 5, 7]).token_ids == expected


def test_native_static_does_not_reuse_or_change_dart_replay_cache(models, monkeypatch):
    base, reference = models
    target = CudaGraphTarget(base, capacity=16, max_block_tokens=3, replay=False)
    try:
        state, _ = target.prefill([11, 13, 17])
        target.forward_block(state, [19])
        original = state.cache
        saved = original.storage[..., :state.seq_len, :].clone()
        monkeypatch.setattr("benchmarks.hf_replay.build_replay_generate", lambda *a: pytest.fail("unexpected DART adapter"))
        expected = hf_decoder(reference)(GenerationCase("native", (3, 5, 7), 5))
        assert build_hf_method(target, 5, static_capacity=16)([3, 5, 7]).token_ids == expected
        assert state.cache is original and state.seq_len == 4
        torch.testing.assert_close(original.storage[..., :state.seq_len, :], saved, atol=0, rtol=0)
        continuation = hf_decoder(reference)(GenerationCase("live", (11, 13, 17, 19), 2))
        assert target.forward_block(state, [continuation[0]]).argmax(-1).tolist() == continuation[1:]
    finally:
        target.close()
    for budget in (0, 5):
        with pytest.raises(RuntimeError, match="closed"):
            build_hf_method(target, budget, static_capacity=16)([3])
