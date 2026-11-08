from contextlib import contextmanager
from functools import wraps

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.replay import CudaGraphTarget
from dart.models.target import TargetModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA capture required")


def apply_scripted_predictions(model):
    original = model.forward

    @wraps(original)
    def forward(input_ids=None, **kwargs):
        result = original(input_ids=input_ids, **kwargs)
        ids = input_ids[:, -result.logits.shape[1]:]
        logits = torch.zeros_like(result.logits)
        logits.scatter_(-1, ((ids + 1) % model.config.vocab_size).unsqueeze(-1), 1)
        result.logits = logits
        return result

    model.forward = forward


@contextmanager
def prepared(*, scripted=False):
    torch.set_num_threads(1)
    torch.manual_seed(617)
    config = LlamaConfig(vocab_size=97, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         eos_token_id=None, pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    base = TargetModel(LlamaForCausalLM(config), device="cuda")
    if scripted:
        apply_scripted_predictions(base.model)
    target = CudaGraphTarget(base, capacity=64, max_block_tokens=3, capture_predictions=True,
                             greedy_chunk_sizes=(2, 4))
    try:
        assert all(block.graph is not None for block in target._greedy_blocks.values())
        yield base, target
    finally:
        target.close()


def hf_ids(base, prompt, budget, eos=None):
    ids = torch.tensor([prompt], device=base.device)
    with torch.inference_mode():
        return base.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=budget, use_cache=True, eos_token_id=eos)[0, len(prompt):].tolist()


@pytest.mark.parametrize("budget", [1, 5, 17])
def test_captured_complete_outputs(budget):
    with prepared() as (base, target):
        prompt = [3, 5, 7] * 4
        expected = hf_ids(base, prompt, budget)
        assert target.greedy_generate(prompt, budget) == expected
        assert target.greedy_generate(prompt, budget) == expected
        assert target.greedy_generate(prompt, 0) == []


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_captured_eos_at_each_offset(offset):
    with prepared(scripted=True) as (base, target):
        prompt = [3, 5, 9]
        eos = 11 + offset
        target.eos_token_ids = frozenset([eos])
        state, logits = target.prefill(prompt)
        pending = int(logits.argmax())
        assert pending == 10
        actual = target.forward_greedy(state, pending, 4)
        assert actual == list(range(11, eos + 1))
        assert state.seq_len == len(prompt) + offset + 1
        assert [pending, *actual] == hf_ids(base, prompt, 9, eos=eos)
        assert target.greedy_generate(prompt, 9) == [pending, *actual]


def test_captured_state_interleaving_and_followup():
    with prepared() as (base, target):
        prompt_a, prompt_b = [3, 5, 7], [8, 6, 4, 2]
        a, logits_a = target.prefill(prompt_a)
        b, logits_b = target.prefill(prompt_b)
        ar, _ = base.prefill(prompt_a)
        first = target.forward_greedy(a, int(logits_a.argmax()), 4)
        saved = list(first)
        assert [int(logits_a.argmax()), *first] == hf_ids(base, prompt_a, 5)
        assert target.forward_greedy(b, int(logits_b.argmax()), 2) == hf_ids(base, prompt_b, 3)[1:]
        base.forward_block(ar, [int(logits_a.argmax()), *first[:-1]])
        assert a.seq_len == ar.seq_len
        for index, layer in enumerate(ar.cache.layers):
            torch.testing.assert_close(a.cache.storage[index, 0, ..., :a.seq_len, :], layer.keys,
                                       atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a.cache.storage[index, 1, ..., :a.seq_len, :], layer.values,
                                       atol=2e-6, rtol=2e-5)
        assert target.forward_predictions(a, [first[-1], 11]) == base.forward_block(ar, [first[-1], 11]).argmax(-1).tolist()
        assert first == saved
        expected = []
        pending = 12
        for _ in range(2):
            pending = int(base.forward_block(ar, [pending]).argmax())
            expected.append(pending)
        assert target.forward_greedy(a, 12, 2) == expected


def test_captured_exception_preserves_request(monkeypatch):
    with prepared() as (base, target):
        prompt = [3, 5, 7]
        state, logits = target.prefill(prompt)
        pending = int(logits.argmax())
        first = target.forward_greedy(state, pending, 2)
        length = state.seq_len
        prefix = state.cache.storage[..., :length, :].clone()
        block = target._greedy_blocks[4]
        run = block.run

        def fail_after_replay():
            run()
            torch.cuda.synchronize()
            raise RuntimeError("synthetic post-replay error")

        monkeypatch.setattr(block, "run", fail_after_replay)
        with pytest.raises(RuntimeError, match="post-replay"):
            target.forward_greedy(state, first[-1], 4)
        assert state.seq_len == length
        torch.testing.assert_close(state.cache.storage[..., :length, :], prefix, atol=0, rtol=0)
        monkeypatch.setattr(block, "run", run)
        assert [pending, *first, *target.forward_greedy(state, first[-1], 4)] == hf_ids(base, prompt, 7)


def test_captured_context_stream_and_close():
    with prepared() as (_, target):
        state, _ = target.prefill([3, 5, 7] * 20 + [3, 5])
        with pytest.raises(ValueError, match="context limit"):
            target.forward_greedy(state, 3, 4)
        assert state.seq_len == 62
        target.forward_greedy(state, 3, 2)
        assert state.seq_len == 64
        state, _ = target.prefill([3, 5, 7])
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(ValueError, match="stream"):
                target.forward_greedy(state, 3, 2)
        target.close()
        for count in (1, 2, 4):
            with pytest.raises(RuntimeError, match="closed"):
                target.forward_greedy(state, 3, count)
