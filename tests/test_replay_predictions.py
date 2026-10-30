from dataclasses import asdict
from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version
from transformers import LlamaConfig, LlamaForCausalLM

from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CudaGraphTarget
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_target
from dart.runtime.types import DecodeConfig
from dart.verification.block_verify import verify_block_greedy


@pytest.fixture(params=["cpu_static", "cuda_graph"])
def execution(request):
    if Version(version("transformers")) < Version("5.16.1"):
        pytest.skip("static replay requires transformers >= 5.16.1")
    device = "cuda" if request.param == "cuda_graph" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA capture needs a GPU")
    torch.set_num_threads(1)
    torch.manual_seed(481)
    config = LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                         eos_token_id=None, pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    base = TargetModel(LlamaForCausalLM(config), device=device)
    target = CudaGraphTarget(base, capacity=64, max_block_tokens=5, capture_predictions=True,
                             replay=device == "cuda")
    yield base, target
    target.close()


def hf_ids(base, prompt, budget):
    ids = torch.tensor([prompt], device=base.device)
    with torch.inference_mode():
        return base.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=budget, use_cache=True)[0, len(prompt):].tolist()


@pytest.mark.parametrize("budget", [1, 9, 19])
def test_complete_outputs(execution, budget):
    base, target = execution
    prompt = [5, 6, 7] * 3
    expected = hf_ids(base, prompt, budget)
    with DartEngine(target, NGramDrafter(min_ngram=1, max_ngram=4)) as engine:
        actual = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, draft_len=4)).token_ids
    assert actual == expected
    assert target.greedy_generate(prompt, budget) == expected


@pytest.mark.parametrize("rejection", [None, 0, 1, 2, 3, "empty"])
def test_rejection_and_followup(execution, rejection):
    base, target = execution
    prompt = [5, 7, 9]
    expected = hf_ids(base, prompt, 5)
    pending, drafts = expected[0], expected[1:]
    if rejection == "empty":
        drafts = []
    elif rejection is not None:
        drafts[rejection] = (drafts[rejection] + 1) % base.model.config.vocab_size
    a, _ = base.prefill(prompt)
    b, _ = target.prefill(prompt)
    wanted = verify_block_greedy(base, a, pending, drafts, compute_entropy=False)
    actual = verify_block_greedy(target, b, pending, drafts, compute_entropy=False)
    assert asdict(actual) == asdict(wanted)
    assert b.seq_len == a.seq_len == len(prompt) + len(actual.committed)
    for index, layer in enumerate(a.cache.layers):
        torch.testing.assert_close(b.cache.storage[index, 0, ..., :b.seq_len, :], layer.keys, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(b.cache.storage[index, 1, ..., :b.seq_len, :], layer.values, atol=2e-6, rtol=2e-5)
    next_ids = [actual.committed[-1]]
    assert target.forward_predictions(b, next_ids) == base.forward_block(a, next_ids).argmax(-1).tolist()


def test_accepted_eos(execution):
    base, target = execution
    prompt = [5, 7, 9]
    expected = hf_ids(base, prompt, 5)
    base.eos_token_ids = target.eos_token_ids = frozenset([expected[2]])
    a, _ = base.prefill(prompt)
    b, _ = target.prefill(prompt)
    wanted = verify_block_greedy(base, a, expected[0], expected[1:], compute_entropy=False)
    actual = verify_block_greedy(target, b, expected[0], expected[1:], compute_entropy=False)
    assert asdict(actual) == asdict(wanted)
    assert actual.finished and actual.committed[-1] in target.eos_token_ids
    assert a.seq_len == b.seq_len
    target.eos_token_ids = frozenset([expected[0]])
    assert target.greedy_generate(prompt, 5) == expected[:1]
    with DartEngine(target, NGramDrafter()) as engine:
        assert engine.generate(prompt, DecodeConfig(max_new_tokens=5)).token_ids == expected[:1]


def test_interleaving_and_output_ownership(execution):
    base, target = execution
    a, _ = target.prefill([5, 7, 9])
    b, _ = target.prefill([3, 2, 8, 6])
    ar, _ = base.prefill([5, 7, 9])
    br, _ = base.prefill([3, 2, 8, 6])
    first = target.forward_block(a, [4, 5])
    first_saved = first.clone()
    torch.testing.assert_close(first, base.forward_block(ar, [4, 5]), atol=2e-6, rtol=2e-5)
    predictions = target.forward_predictions(b, [6, 7])
    saved = list(predictions)
    assert predictions == base.forward_block(br, [6, 7]).argmax(-1).tolist()
    assert target.forward_predictions(a, [3]) == base.forward_block(ar, [3]).argmax(-1).tolist()
    assert predictions == saved
    torch.testing.assert_close(first, first_saved, atol=0, rtol=0)


def test_guards_and_context(execution):
    _, target = execution
    with pytest.raises(AttributeError):
        target.capture_predictions = False
    state, _ = target.prefill([4, 5, 6])
    target.forward_predictions(state, [7])
    length = state.seq_len
    with pytest.raises(ValueError, match="prepared replay size"):
        target.forward_predictions(state, [2] * 6)
    assert state.seq_len == length
    owner = state.cache.owner
    state.cache.owner = object()
    with pytest.raises(ValueError, match="different replay target"):
        target.forward_predictions(state, [2])
    state.cache.owner = owner
    with pytest.raises(ValueError, match="context limit"):
        target.forward_predictions(state, [2] * 64)
    assert state.seq_len == length
    target.close()
    with pytest.raises(RuntimeError, match="closed"):
        target.forward_predictions(state, [2])


def test_prediction_route_avoids_unused_logits(tiny_target, monkeypatch):
    state, logits = tiny_target.prefill([5, 7, 9])
    forward = tiny_target.forward_block
    calls = []

    def predictions(state, tokens):
        calls.append(list(tokens))
        return forward(state, tokens).argmax(-1).tolist()

    def unused(*args, **kwargs):
        raise AssertionError("unused full logits requested")

    monkeypatch.setattr(tiny_target, "capture_predictions", True, raising=False)
    monkeypatch.setattr(tiny_target, "forward_predictions", predictions, raising=False)
    monkeypatch.setattr(tiny_target, "forward_block", unused)
    pending = int(logits.argmax())
    result = verify_block_greedy(tiny_target, state, pending, [], compute_entropy=False)
    assert calls == [[pending]] and result.entropy == 0.0


def test_entropy_keeps_full_logits(tiny_target, monkeypatch):
    def unused(*args, **kwargs):
        raise AssertionError("entropy needs the full logits")

    monkeypatch.setattr(tiny_target, "capture_predictions", True, raising=False)
    monkeypatch.setattr(tiny_target, "forward_predictions", unused, raising=False)
    state, logits = tiny_target.prefill([5, 7, 9])
    result = verify_block_greedy(tiny_target, state, int(logits.argmax()), [], compute_entropy=True)
    assert result.entropy > 0


@pytest.mark.parametrize("value", [1, None, "true"])
def test_prediction_option_requires_boolean(value):
    with pytest.raises(ValueError, match="capture_predictions must be a boolean"):
        CudaGraphTarget(SimpleNamespace(), capture_predictions=value)
    with pytest.raises(ValueError, match="capture_predictions must be a boolean"):
        build_target({"model": "invalid", "capture_predictions": value})


def test_eager_rejects_prediction_option_before_loading_model():
    with pytest.raises(ValueError, match="require static or cuda_graph execution"):
        build_target({"model": "invalid", "capture_predictions": True})
