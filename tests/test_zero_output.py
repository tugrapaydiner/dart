from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dart.drafters.ar_draft import ARDrafter
from dart.runtime import engine as runtime
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models
from tests.test_replay_predictions import execution as execution


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield _synthetic_models(17)
    torch.set_num_threads(previous)


def unexpected(*args, **kwargs):
    pytest.fail("zero output request started model or request execution")


@pytest.mark.parametrize("mode,verifier", [
    ("exact", "block"), ("exact", "reference"), ("exact", "tree"),
    ("exact_sampling", "block"), ("approx", "block"),
])
@pytest.mark.parametrize("residency", ["gpu", "cpu_pinned"])
def test_engine_zero_output_skips_execution_and_preserves_metadata(models, monkeypatch, mode, verifier, residency):
    target, _ = models
    monkeypatch.setattr(target, "prefill", unexpected)
    monkeypatch.setattr(target.model, "forward", unexpected)
    monkeypatch.setattr(runtime, "SyncPrefetcher", unexpected)
    monkeypatch.setattr(runtime, "ThreadPrefetcher", unexpected)
    monkeypatch.setattr(runtime, "reset_gpu_memory_stats", lambda: None)
    monkeypatch.setattr(runtime, "gpu_memory_gb", lambda: 0.125)
    drafter = SimpleNamespace(name="unused", requires_entropy=False, after_prefill=unexpected,
                              propose=unexpected, observe=unexpected)
    with DartEngine(target, drafter, verifier=verifier) as engine:
        monkeypatch.setattr(engine.trace, "log_round", unexpected)
        out = engine.generate([5, 6, 7], DecodeConfig(mode=mode, max_new_tokens=0,
                                                   full_kv_residency=residency, seed=29))
    assert out.token_ids == out.rounds == [] and out.text is None
    meta = out.metadata
    for key in ("tokens_generated", "tokens_verified", "draft_tokens_proposed", "draft_tokens_accepted",
                "rounds", "rejections", "draft_ms", "verify_ms", "gpu_kv_gb", "acceptance_rate", "avg_accept_run"):
        assert meta[key] == 0
    assert meta["lanes_used"] == [] and meta["output_hash"] == output_hash([])
    assert meta["finished_eos"] is False and meta["exact_greedy_equivalent"] is None
    assert meta["mode"] == mode and meta["full_kv_residency"] == residency
    assert meta["greedy_verified"] is (mode != "exact_sampling")
    assert meta["verifier"] == ("sampling" if mode == "exact_sampling" else verifier)
    assert meta["wall_time_ms"] >= 0 and meta["gpu_peak_gb"] == 0.125
    if residency == "cpu_pinned":
        assert meta["kv_residency_moves"] == meta["kv_residency_moved_mb"] == 0
    if mode == "exact_sampling":
        assert meta["distribution_exact"] and meta["seed"] == 29
        assert meta["temperature"] == 1.0
    if mode == "approx":
        assert meta["verify_skip_policy"] == {"verify_every": 1}
        assert meta["unverified_tokens"] == meta["trailing_unverified"] == meta["pending_divergence"] == 0
        assert meta["quality_risk"] is False


@pytest.mark.parametrize("prompt", [[5, 6], torch.tensor([5, 6]), torch.tensor([[5, 6]])])
def test_eager_baseline_zero_output_does_not_forward(models, monkeypatch, prompt):
    target, _ = models
    monkeypatch.setattr(target.model, "forward", unexpected)
    assert target.greedy_generate(prompt, 0) == []


@pytest.mark.parametrize("entry", ["engine", "baseline"])
@pytest.mark.parametrize("prompt", [[], [-1], [67], [5] * 65, [[5], [6]]])
def test_zero_output_still_rejects_invalid_prompts(models, monkeypatch, entry, prompt):
    target, _ = models
    monkeypatch.setattr(target.model, "forward", unexpected)
    with pytest.raises(ValueError):
        if entry == "engine":
            DartEngine(target).generate(prompt, DecodeConfig(max_new_tokens=0))
        else:
            target.greedy_generate(prompt, 0)


def test_zero_output_decodes_empty_text_without_prefill(models, monkeypatch):
    target, _ = models
    seen = []

    def decode(ids, skip_special_tokens):
        seen.append((ids, skip_special_tokens))
        return ""

    target.tokenizer = SimpleNamespace(encode=lambda text: [5, 6], decode=decode)
    monkeypatch.setattr(target, "prefill", unexpected)
    out = DartEngine(target).generate("hello", DecodeConfig(max_new_tokens=0))
    assert out.text == "" and seen == [([], False)]


def test_empty_request_preserves_next_request_outputs_and_other_live_state(models):
    target, reference = models
    state, _ = target.prefill([13, 17, 19])
    cache_before = [(layer.keys.clone(), layer.values.clone()) for layer in state.cache.layers]
    expected = hf_decoder(reference)(GenerationCase("next", (5, 6, 7), 8))
    engine = DartEngine(target, ARDrafter(target, seed=23))
    engine.generate([23, 29, 31], DecodeConfig(max_new_tokens=6, draft_len=2))
    assert engine.generate([5] * 64, DecodeConfig(max_new_tokens=0)).token_ids == []
    for layer, (keys, values) in zip(state.cache.layers, cache_before):
        torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
    assert engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=8)).token_ids == expected


@pytest.mark.parametrize("execution", ["cpu_static"], indirect=True)
@pytest.mark.parametrize("entry", ["engine", "baseline"])
def test_replay_zero_output_skips_prefill(execution, monkeypatch, entry):
    _, target = execution
    monkeypatch.setattr(target.model, "forward", unexpected)
    if entry == "engine":
        assert DartEngine(target).generate([5, 6, 7], DecodeConfig(max_new_tokens=0)).token_ids == []
    else:
        assert target.greedy_generate([5, 6, 7], 0) == []


@pytest.mark.parametrize("execution", ["cpu_static"], indirect=True)
@pytest.mark.parametrize("entry", ["engine", "baseline"])
def test_zero_output_does_not_bypass_closed_replay_target(execution, entry):
    _, target = execution
    target.close()
    with pytest.raises(RuntimeError, match="closed"):
        if entry == "engine":
            DartEngine(target).generate([5, 6, 7], DecodeConfig(max_new_tokens=0))
        else:
            target.greedy_generate([5, 6, 7], 0)
