import copy
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from benchmarks.run_bench import build_hf_method
from dart.models.replay import CudaGraphTarget, ReplayCache
from dart.runtime.factory import build_tiny_target


@pytest.mark.parametrize("amount", [torch.tensor(-2), torch.tensor(-2, dtype=torch.int32), torch.tensor(0)])
def test_replay_crop_accepts_hf_integer_scalars(amount):
    cache = ReplayCache(torch.empty(2, 2, 1, 2, 8, 4), 3, object())
    cache.crop(amount)
    assert cache.length == 3 + amount.item()
    assert type(cache.length) is int


@pytest.mark.parametrize("amount", [torch.tensor([-1]), torch.tensor(-1.0), torch.tensor(True), torch.tensor(-4), torch.tensor(1)])
def test_replay_crop_rejects_invalid_tensor_counts(amount):
    cache = ReplayCache(torch.empty(2, 2, 1, 2, 8, 4), 3, object())
    with pytest.raises(ValueError):
        cache.crop(amount)
    assert cache.length == 3


@pytest.fixture(scope="module")
def targets():
    if not torch.cuda.is_available() or Version(version("transformers")) < Version("5.16.1"):
        pytest.skip("HF replay requires CUDA and transformers >= 5.16.1")
    original = build_tiny_target(789, {"device": "cuda", "precision": "fp32"})
    graph = CudaGraphTarget(original, max_block_tokens=5, capacity=64)
    static = CudaGraphTarget(original, max_block_tokens=5, capacity=64, replay=False)
    yield original, graph, static
    graph.close()
    static.close()


@pytest.fixture(params=[1, 2], ids=["cuda_graph", "static"])
def target_pair(targets, request):
    return targets[0], targets[request.param]


@pytest.mark.parametrize("lookup", [None, 1, 4])
@pytest.mark.parametrize("budget", [1, 3, 11])
@pytest.mark.parametrize("prompt", [[5, 7, 9], [5, 6, 7] * 8])
def test_hf_replay_matches_complete_ordinary_generation(target_pair, lookup, budget, prompt):
    original, target = target_pair
    expected = build_hf_method(original, budget)(prompt).token_ids
    method = build_hf_method(target, budget, prompt_lookup_num_tokens=lookup)
    assert method(prompt).token_ids == expected


def test_hf_replay_shares_weights_without_replacing_original_forward(targets, monkeypatch):
    original, graph, _ = targets
    saved_forward = original.model.forward
    monkeypatch.setattr(original.model.generation_config, "repetition_penalty", 3.0)
    settings = copy.deepcopy(original.model.generation_config)
    pointers = [parameter.data_ptr() for parameter in original.model.parameters()]
    calls = []

    def record(module, args, kwargs, output):
        calls.append((module is original.model, kwargs["input_ids"].shape[1], output.logits.shape[1]))
        assert [parameter.data_ptr() for parameter in module.parameters()] == pointers
        assert output.hidden_states is None

    handle = original.model.register_forward_hook(record, with_kwargs=True)
    try:
        actual = build_hf_method(graph, 7)([5, 6, 7] * 4).token_ids
    finally:
        handle.remove()
    assert calls and all(not original_wrapper for original_wrapper, _, _ in calls)
    assert calls[0][1:] == (12, 1)  # Keep HF's last-logit-only prefill.
    assert len(calls) == len(actual)
    assert original.model.forward == saved_forward
    assert original.model.generation_config == settings
    assert build_hf_method(original, 7)([5, 6, 7] * 4).token_ids == actual


@pytest.mark.parametrize("prompt", [[0] * 12, [0, 0, 0, 0, 5, 0, 0, 0, 0, 6, 0]], ids=["accept", "reject"])
def test_prompt_lookup_replays_blocks_and_crops_rejected_tail(target_pair, monkeypatch, prompt):
    original, target = target_pair
    weight = original.model.lm_head.weight
    saved = weight.detach().clone()
    lengths, removals = [], []
    crop = ReplayCache.crop

    def record_crop(cache, count):
        before = cache.length
        crop(cache, count)
        removals.append(before - cache.length)
        assert type(cache.length) is int

    monkeypatch.setattr(ReplayCache, "crop", record_crop)
    for length, block in target._blocks.items():
        run = block.run

        def record_run(run=run, length=length):
            lengths.append(length)
            return run()

        monkeypatch.setattr(block, "run", record_run)
    try:
        weight.zero_()
        method = build_hf_method(target, 12, prompt_lookup_num_tokens=4)
        assert method(prompt).token_ids == [0] * 12
        assert method(prompt).token_ids == build_hf_method(original, 12)(prompt).token_ids
        assert 5 in lengths and removals
        if 5 in prompt:
            assert max(removals) > 0
        else:
            assert len(lengths) < 2 * 11
    finally:
        weight.copy_(saved)


@pytest.mark.parametrize("lookup", [None, 4])
def test_hf_replay_stops_at_eos_after_prefill(target_pair, monkeypatch, lookup):
    original, target = target_pair
    prompt = [5, 9, 11]
    monkeypatch.setattr(original, "eos_token_ids", frozenset())
    oracle = build_hf_method(original, 12)(prompt).token_ids
    index = next(i for i in range(1, len(oracle)) if oracle[i] not in oracle[:i])
    monkeypatch.setattr(original, "eos_token_ids", frozenset({oracle[index]}))
    monkeypatch.setattr(target, "eos_token_ids", original.eos_token_ids)
    expected = oracle[:index + 1]
    method = build_hf_method(target, 12, prompt_lookup_num_tokens=lookup)
    assert method(prompt).token_ids == expected
    assert method(prompt).token_ids == build_hf_method(original, 12)(prompt).token_ids


@pytest.mark.parametrize("lookup", [None, 4])
def test_hf_replay_stops_at_first_eos(target_pair, monkeypatch, lookup):
    original, target = target_pair
    prompt = [5, 9, 11]
    first = build_hf_method(original, 1)(prompt).token_ids[0]
    monkeypatch.setattr(target, "eos_token_ids", frozenset({first}))
    assert build_hf_method(target, 12, prompt_lookup_num_tokens=lookup)(prompt).token_ids == [first]


def test_hf_replay_checks_context_and_prepared_block_size(target_pair):
    _, target = target_pair
    with pytest.raises(ValueError, match="prepared replay block size"):
        build_hf_method(target, 8, prompt_lookup_num_tokens=8)
    method = build_hf_method(target, 8, prompt_lookup_num_tokens=4)
    with pytest.raises(ValueError, match="capacity"):
        method([5] * 54)
    assert method([5] * 53).token_ids
    assert build_hf_method(target, 1, prompt_lookup_num_tokens=8)([5] * 63).token_ids


def test_hf_replay_rejects_hidden_states_and_invalid_ids(target_pair):
    _, target = target_pair
    with pytest.raises(ValueError, match="no hidden-state"):
        build_hf_method(target, 8, record_hidden=True)
    method = build_hf_method(target, 8, prompt_lookup_num_tokens=4)
    for prompt in [[-1, 5], [5, 211]]:
        with pytest.raises(ValueError, match="vocabulary"):
            method(prompt)
    assert method([5, 6, 7]).token_ids


def test_hf_replay_failure_does_not_change_target_or_poison_next_request(targets, monkeypatch):
    original, graph, _ = targets
    saved_forward = original.model.forward
    method = build_hf_method(graph, 8, prompt_lookup_num_tokens=4)

    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    with monkeypatch.context() as patch:
        patch.setattr(ReplayCache, "crop", fail)
        with pytest.raises(RuntimeError, match="test failure"):
            method([5, 6, 7] * 4)
    assert original.model.forward == saved_forward
    expected = build_hf_method(original, 8)([5, 6, 7] * 4).token_ids
    assert method([5, 6, 7] * 4).token_ids == expected
    assert graph.greedy_generate([5, 6, 7] * 4, 8) == expected
