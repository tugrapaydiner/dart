import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.replay import CapacityBuckets, CudaGraphTarget
from dart.models.target import TargetModel
from dart.runtime.factory import build_target


@pytest.fixture
def pair():
    torch.set_num_threads(1)
    torch.manual_seed(617)
    config = LlamaConfig(vocab_size=97, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         eos_token_id=None, pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    base = TargetModel(LlamaForCausalLM(config), device="cpu")
    target = CudaGraphTarget(base, capacity=64, max_block_tokens=5, replay=False, capture_predictions=True, greedy_chunk_sizes=(2, 4))
    yield base, target
    target.close()


def reference(base, prompt, budget, eos=None):
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        return base.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=budget, use_cache=True, eos_token_id=eos)[0, len(prompt):].tolist()


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 8, 17])
@pytest.mark.parametrize("prompt", [[4, 7, 9], [2, 4, 6, 8, 5, 3, 2] * 3])
def test_full_outputs(pair, prompt, budget):
    base, target = pair
    assert target.greedy_generate(prompt, budget) == reference(base, prompt, budget)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_cache_and_followup(pair, count):
    base, target = pair
    prompt = [3, 4, 8, 6]
    a, logits = base.prefill(prompt)
    b, _ = target.prefill(prompt)
    pending = int(logits.argmax())
    expected = []
    for _ in range(count):
        pending = int(base.forward_block(a, [pending])[-1].argmax())
        expected.append(pending)
    actual = target.forward_greedy(b, int(logits.argmax()), count)
    assert actual == expected
    assert b.seq_len == a.seq_len == len(prompt) + count
    for index, layer in enumerate(a.cache.layers):
        torch.testing.assert_close(b.cache.storage[index, 0, ..., :b.seq_len, :], layer.keys, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(b.cache.storage[index, 1, ..., :b.seq_len, :], layer.values, atol=2e-6, rtol=2e-5)
    assert target.forward_predictions(b, [actual[-1]]) == base.forward_block(a, [expected[-1]]).argmax(-1).tolist()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_real_model_eos(pair, index):
    base, target = pair
    prompt = [3, 4, 8, 6]
    expected = reference(base, prompt, 9)
    eos = expected[index]
    target.eos_token_ids = frozenset([eos])
    actual = target.greedy_generate(prompt, 9)
    assert actual == reference(base, prompt, 9, eos=eos)
    assert actual[-1] == eos and eos not in actual[:-1]


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_eos_crops_every_chunk_position(pair, offset, monkeypatch):
    _, target = pair
    state, _ = target.prefill([3, 4, 8, 6])
    block = target._greedy_blocks[4]
    original = block.forward_model

    def scripted(**kwargs):
        result = original(**kwargs)
        logits = torch.zeros_like(result.logits)
        logits.scatter_(-1, (kwargs["input_ids"] + 1).unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(block, "forward_model", scripted)
    target.eos_token_ids = frozenset([11 + offset])
    assert target.forward_greedy(state, 10, 4) == list(range(11, 12 + offset))
    assert state.seq_len == 4 + offset + 1


def test_interleaving_and_linear_transition(pair):
    base, target = pair
    a, logits_a = target.prefill([3, 5, 7])
    b, logits_b = target.prefill([8, 6, 4, 2])
    ar, _ = base.prefill([3, 5, 7])
    first = target.forward_greedy(a, int(logits_a.argmax()), 4)
    saved = list(first)
    base.forward_block(ar, [int(logits_a.argmax()), *first[:-1]])
    assert target.forward_greedy(b, int(logits_b.argmax()), 2) == reference(base, [8, 6, 4, 2], 3)[1:]
    assert target.forward_predictions(a, [first[-1], 11]) == base.forward_block(ar, [first[-1], 11]).argmax(-1).tolist()
    assert first == saved
    next_a = target.forward_greedy(a, 12, 2)
    expected = []
    pending = 12
    for _ in range(2):
        pending = int(base.forward_block(ar, [pending]).argmax())
        expected.append(pending)
    assert next_a == expected


def test_context_and_failure_leave_state_unchanged(pair, monkeypatch):
    _, target = pair
    state, _ = target.prefill([3, 5, 7] * 20 + [3, 5])
    original_cache = state.cache
    with pytest.raises(ValueError, match="context limit"):
        target.forward_greedy(state, 3, 4)
    assert state.cache is original_cache and state.seq_len == 62
    target.forward_greedy(state, 3, 2)
    assert state.seq_len == 64
    with pytest.raises(ValueError, match="context limit"):
        target.forward_greedy(state, 3, 1)
    with pytest.raises(ValueError):
        target.greedy_generate([4] * 60, 5)

    state, _ = target.prefill([3, 5, 7])
    original_cache = state.cache

    def fail():
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(target._greedy_blocks[4], "run", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        target.forward_greedy(state, 3, 4)
    assert state.cache is original_cache and state.seq_len == 3


def test_guards_and_close(pair):
    _, target = pair
    state, _ = target.prefill([3, 5, 7])
    for count in [0, 3, True, 2.0]:
        with pytest.raises(ValueError, match="not prepared"):
            target.forward_greedy(state, 3, count)
    for token in [-1, 97, True]:
        with pytest.raises(ValueError):
            target.forward_greedy(state, token, 2)
    target.forward_greedy(state, 3, 2)
    owner = state.cache.owner
    state.cache.owner = object()
    with pytest.raises(ValueError, match="different replay target"):
        target.forward_greedy(state, 3, 4)
    state.cache.owner = owner
    target.close()
    with pytest.raises(RuntimeError, match="closed"):
        target.forward_greedy(state, 3, 4)
    with pytest.raises(RuntimeError, match="closed"):
        target.greedy_generate([3, 5], 3)


def test_partial_execution_error_preserves_request_and_recovers(pair, monkeypatch):
    base, target = pair
    state, logits = target.prefill([3, 5, 7])
    pending = int(logits.argmax())
    first = target.forward_greedy(state, pending, 2)
    cache = state.cache
    prefix = cache.storage[..., :cache.length, :].clone()
    length = cache.length
    block = target._greedy_blocks[4]
    original = block.forward_model
    calls = 0

    def fail_after_one(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic partial execution")
        return original(**kwargs)

    monkeypatch.setattr(block, "forward_model", fail_after_one)
    with pytest.raises(RuntimeError, match="partial execution"):
        target.forward_greedy(state, first[-1], 4)
    assert calls == 2 and state.cache is cache and state.seq_len == length
    torch.testing.assert_close(cache.storage[..., :length, :], prefix, atol=0, rtol=0)
    monkeypatch.setattr(block, "forward_model", original)
    actual = target.forward_greedy(state, first[-1], 4)
    assert [pending, *first, *actual] == reference(base, [3, 5, 7], 7)


def test_target_close_rejects_all_chunk_sizes(pair):
    _, target = pair
    state, _ = target.prefill([3, 5, 7])
    target.close()
    for count in (1, 2, 4):
        with pytest.raises(RuntimeError, match="closed"):
            target.forward_greedy(state, 3, count)


def test_zero_output_budget(pair):
    _, target = pair
    assert target.greedy_generate([3, 5, 7], 0) == []


@pytest.mark.parametrize("count", [2, 4])
def test_failed_request_is_rejected_before_chunk_work(pair, count, monkeypatch):
    _, target = pair
    state, _ = target.prefill([3, 5, 7])
    state._failed = True
    calls = []
    run = target._greedy_blocks[count].run

    def observed():
        calls.append(True)
        return run()

    monkeypatch.setattr(target._greedy_blocks[count], "run", observed)
    with pytest.raises(RuntimeError, match="invalid after"):
        target.forward_greedy(state, 3, count)
    assert not calls


def test_zero_budget_still_checks_prompt_ids(pair):
    _, target = pair
    with pytest.raises(ValueError):
        target.greedy_generate([97], 0)


@pytest.mark.parametrize("sizes", [[2, 4], None, 2, (), (1,), (9,), (True,), (2.0,), (2, 2), ([],)])
def test_chunk_admission(pair, sizes):
    base, _ = pair
    if sizes == ():
        target = CudaGraphTarget(base, capacity=16, max_block_tokens=1, replay=False)
        assert target.greedy_chunk_sizes == () and target._greedy_blocks == {}
        target.close()
    else:
        with pytest.raises(ValueError, match="greedy chunk sizes"):
            CudaGraphTarget(base, capacity=16, replay=False, capture_predictions=True, greedy_chunk_sizes=sizes)


def test_chunk_graph_limits_and_prediction_requirement(pair):
    base, _ = pair
    with pytest.raises(ValueError, match="capture_predictions"):
        CudaGraphTarget(base, replay=False, greedy_chunk_sizes=(2, 4))
    with pytest.raises(ValueError, match="capacity"):
        CudaGraphTarget(base, capacity=3, max_block_tokens=1, replay=False,
                        capture_predictions=True, greedy_chunk_sizes=(4,))
    with pytest.raises(ValueError, match="16 prepared"):
        CudaGraphTarget(base, capacity=32, max_block_tokens=15, replay=False,
                        capture_predictions=True, greedy_chunk_sizes=(2, 4))
    with pytest.raises(ValueError, match="16 prepared"):
        CapacityBuckets(base, (16, 32), max_block_tokens=7, replay=False,
                        capture_predictions=True, greedy_chunk_sizes=(2, 4))


@pytest.mark.parametrize("capacities", [32, [16, 32]])
def test_factory_chunks_and_shared_weights(capacities):
    target = build_target({"model": "tiny:41", "hidden_size": 16, "intermediate_size": 32, "num_layers": 1,
                           "execution": "static", "cache_capacity": capacities, "max_block_tokens": 3,
                           "capture_predictions": True, "greedy_chunk_sizes": [2, 4]})
    try:
        buckets = target.buckets if isinstance(target, CapacityBuckets) else (target,)
        assert all(bucket.greedy_chunk_sizes == (2, 4) for bucket in buckets)
        assert sum(len(bucket._blocks) + len(bucket._greedy_blocks) for bucket in buckets) <= 16
        assert len({id(bucket.model) for bucket in buckets}) == 1
        for prompt in ([5, 7, 9], [5, 7, 9] * 6, [5, 7, 9]):
            ids = torch.tensor([prompt])
            with torch.inference_mode():
                expected = target.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                  max_new_tokens=9, do_sample=False)[0, len(prompt):].tolist()
            assert target.greedy_generate(prompt, 9) == expected
    finally:
        target.close()


@pytest.mark.parametrize("spec", [
    {"execution": "eager", "greedy_chunk_sizes": [2]},
    {"execution": "static", "greedy_chunk_sizes": [2]},
    {"execution": "static", "capture_predictions": True, "greedy_chunk_sizes": [1]},
    {"execution": "static", "capture_predictions": True, "greedy_chunk_sizes": None},
])
def test_factory_invalid_chunks_fail_before_loading(spec):
    with pytest.raises(ValueError):
        build_target(spec)


def test_chunks_release_storage_on_close(pair):
    _, target = pair
    workspace = weakref.ref(target._workspace)
    blocks = [weakref.ref(block) for block in target._greedy_blocks.values()]
    target.greedy_generate([3, 5, 7], 9)
    target.close()
    gc.collect()
    assert workspace() is None and all(block() is None for block in blocks)
    assert target.greedy_chunk_sizes == (2, 4)
    with pytest.raises(AttributeError):
        target.greedy_chunk_sizes = ()


def test_partial_preparation_releases_blocks(pair, monkeypatch):
    import dart.models.replay as replay

    base, _ = pair
    old = replay._GreedyBlock
    refs = []

    def fail_second(*args):
        if refs:
            raise RuntimeError("second chunk failed")
        result = old(*args)
        refs.extend([weakref.ref(result), weakref.ref(result.workspace)])
        return result

    monkeypatch.setattr(replay, "_GreedyBlock", fail_second)
    with pytest.raises(RuntimeError, match="second chunk failed"):
        CudaGraphTarget(base, capacity=32, replay=False, capture_predictions=True, greedy_chunk_sizes=(2, 4))
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_copy_failure_invalidates_state(pair, monkeypatch):
    _, target = pair
    state, logits = target.prefill([3, 5, 7])
    pending = int(logits.argmax())
    first = target.forward_greedy(state, pending, 2)
    address = state.cache.storage.untyped_storage().data_ptr()
    copy = torch.Tensor.copy_

    def fail_request_copy(tensor, source, *args, **kwargs):
        result = copy(tensor, source, *args, **kwargs)
        if tensor.untyped_storage().data_ptr() == address:
            raise RuntimeError("request copy failed")
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", fail_request_copy)
    with pytest.raises(RuntimeError, match="request copy failed"):
        target.forward_greedy(state, first[-1], 4)
    with pytest.raises(RuntimeError, match="invalid after"):
        target.forward_greedy(state, first[-1], 2)
    with pytest.raises(RuntimeError, match="invalid after"):
        target.forward_predictions(state, [first[-1]])


def test_chunk_generation_does_not_add_target_steps(pair, monkeypatch):
    base, target = pair
    calls = []
    original = base.model.forward

    def measured(*args, **kwargs):
        calls.append(kwargs["input_ids"].shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(base.model, "forward", measured)
    for block in (*target._blocks.values(), *target._greedy_blocks.values()):
        monkeypatch.setattr(block, "forward_model", measured)
    target.greedy_generate([3, 5, 7], 9)
    assert calls == [3, *([1] * 8)]


