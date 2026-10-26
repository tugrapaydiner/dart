from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version
from transformers.cache_utils import DynamicCache

from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CudaGraphTarget, ReplayCache
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_target, build_tiny_target
from dart.runtime.types import DecodeConfig
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference

modern_cache = pytest.mark.skipif(Version(version("transformers")) < Version("5.0"), reason="replay requires transformers 5")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay needs a GPU")


def dynamic_cache(length=3):
    cache = DynamicCache()
    for layer in range(2):
        keys = torch.arange(2 * length * 4).reshape(1, 2, length, 4).float() + layer
        cache.update(keys, keys + 100, layer)
    return cache


@modern_cache
def test_snapshot_owns_prefix_and_crop_only_changes_valid_length():
    source = dynamic_cache()
    first = ReplayCache.from_dynamic(source, 8, object())
    second = ReplayCache.from_dynamic(source, 8, object())
    for index, layer in enumerate(source.layers):
        torch.testing.assert_close(first.storage[index, 0, ..., :3, :], layer.keys, rtol=0, atol=0)
        torch.testing.assert_close(first.storage[index, 1, ..., :3, :], layer.values, rtol=0, atol=0)
    saved = second.storage[..., :3, :].clone()
    first.storage[..., :3, :].zero_()
    torch.testing.assert_close(second.storage[..., :3, :], saved, rtol=0, atol=0)
    assert source.layers[0].keys.sum() > 0
    second.crop(-2)
    assert second.get_seq_length() == 1
    assert second.get_mask_sizes(2) == (3, 0)
    second.crop(0)
    second.crop(-1)
    assert second.get_seq_length() == 0
    torch.testing.assert_close(second.storage[..., :3, :], saved, rtol=0, atol=0)
    with pytest.raises(ValueError, match="through CudaGraphTarget"):
        second.update(None)


@modern_cache
@pytest.mark.parametrize("amount", [1, -4, True, -1.5])
def test_snapshot_rejects_invalid_crop(amount):
    cache = ReplayCache.from_dynamic(dynamic_cache(), 8, object())
    with pytest.raises(ValueError, match="negative removal"):
        cache.crop(amount)
    assert cache.get_seq_length() == 3


@modern_cache
def test_snapshot_rejects_incompatible_cache():
    for cache, capacity in [(DynamicCache(), 8), (object(), 8), (dynamic_cache(), 2), (dynamic_cache(), True)]:
        with pytest.raises(ValueError):
            ReplayCache.from_dynamic(cache, capacity, object())
    cache = dynamic_cache()
    cache.layers[1].values = cache.layers[1].values.double()
    with pytest.raises(ValueError, match="matching shapes, devices and dtypes"):
        ReplayCache.from_dynamic(cache, 8, object())
    cache = DynamicCache()
    cache.update(torch.zeros(2, 2, 3, 4), torch.zeros(2, 2, 3, 4), 0)
    with pytest.raises(ValueError, match="batch-one"):
        ReplayCache.from_dynamic(cache, 8, object())


def test_default_target_stays_eager_and_cpu_replay_fails(tiny_target):
    default = build_target({"model": "tiny:42"})
    assert type(default) is TargetModel
    assert default.greedy_generate([5, 6, 7], 2)
    with pytest.raises(ValueError, match="float32 CUDA"):
        CudaGraphTarget(tiny_target)


def test_unknown_execution_fails_before_loading_model():
    with pytest.raises(ValueError, match="target execution"):
        build_target({"execution": "unknown"})


@pytest.mark.parametrize("verifier,options", [
    ("block", {"mode": "exact_sampling"}), ("block", {"mode": "approx"}),
    ("block", {"full_kv_residency": "cpu_pinned"}), ("reference", {}), ("tree", {}),
])
def test_engine_rejects_unsupported_replay_modes_before_prefill(verifier, options):
    target = SimpleNamespace(uses_replay_cache=True)
    with DartEngine(target, verifier=verifier) as engine:
        with pytest.raises(ValueError, match="exact mode, block verification and resident GPU KV"):
            engine.generate([5, 6], DecodeConfig(max_new_tokens=2, **options))


@pytest.fixture(scope="module")
def replay_targets():
    if not torch.cuda.is_available() or Version(version("transformers")) < Version("5.0"):
        pytest.skip("CUDA replay requires a GPU and transformers 5")
    original = build_tiny_target(1234, {"device": "cuda", "precision": "fp32"})
    graph = CudaGraphTarget(original, max_block_tokens=5, capacity=64)
    static = CudaGraphTarget(original, max_block_tokens=5, capacity=64, replay=False)
    yield original, graph, static
    graph.close()
    static.close()


@pytest.fixture(params=[1, 2], ids=["cuda_graph", "static"])
def replay_target(replay_targets, request):
    return replay_targets[0], replay_targets[request.param]


def assert_logits(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert actual.argmax(-1).tolist() == expected.argmax(-1).tolist()


@pytest.mark.parametrize("budget", [0, 1, 2, 9, 17])
@pytest.mark.parametrize("prompt", [[5, 7, 9], [5, 6, 7] * 9])
def test_complete_outputs_match_hf(replay_target, prompt, budget):
    original, replay = replay_target
    if budget:
        with torch.inference_mode():
            ids = torch.tensor([prompt], device=original.device)
            expected = original.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                               do_sample=False, max_new_tokens=budget, use_cache=True)[0, len(prompt):].tolist()
    else:
        expected = []
    assert original.greedy_generate(prompt, budget) == expected
    assert replay.greedy_generate(prompt, budget) == expected
    with DartEngine(replay, drafter=NGramDrafter(max_ngram=4, min_ngram=1)) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, draft_len=4))
    assert output.token_ids == expected


@pytest.mark.parametrize("rejection", [None, 0, 2, "empty"])
def test_verification_and_followup_after_rollback(replay_target, monkeypatch, rejection):
    original, replay = replay_target
    monkeypatch.setattr(original, "eos_token_ids", frozenset())
    monkeypatch.setattr(replay, "eos_token_ids", frozenset())
    prompt = [5, 9, 11, 14]
    oracle = original.greedy_generate(prompt, 6)
    drafts = oracle[1:5]
    if rejection == "empty":
        drafts = []
    elif rejection is not None:
        drafts[rejection] = (drafts[rejection] + 1) % original.model.config.vocab_size
    state, logits = replay.prefill(prompt)
    control, _ = original.prefill(prompt)
    pending = int(logits.argmax().item())
    result = verify_block_greedy(replay, state, pending, drafts)
    expected = verify_greedy_reference(original, control, pending, drafts)
    assert result == expected
    assert result.n_draft_accepted == (4 if rejection is None else 0 if rejection == "empty" else rejection)
    assert state.seq_len == control.seq_len
    assert_logits(replay.forward_block(state, [result.committed[-1]]),
                  original.forward_block(control, [expected.committed[-1]]))


@pytest.mark.parametrize("accepted", [True, False])
def test_eos_in_verified_block(replay_target, monkeypatch, accepted):
    original, replay = replay_target
    prompt = [5, 9, 11, 14]
    monkeypatch.setattr(original, "eos_token_ids", frozenset())
    oracle = original.greedy_generate(prompt, 6)
    eos = oracle[1]
    monkeypatch.setattr(original, "eos_token_ids", frozenset({eos}))
    monkeypatch.setattr(replay, "eos_token_ids", frozenset({eos}))
    drafts = oracle[1:5] if accepted else [(eos + 1) % original.model.config.vocab_size] * 4
    state, _ = replay.prefill(prompt)
    control, _ = original.prefill(prompt)
    result = verify_block_greedy(replay, state, oracle[0], drafts)
    assert result == verify_greedy_reference(original, control, oracle[0], drafts)
    assert result.finished and result.committed == [eos]
    assert result.n_draft_accepted == int(accepted)
    assert state.seq_len == control.seq_len == len(prompt) + 1


def test_first_token_eos_and_request_reset(replay_target, monkeypatch):
    original, replay = replay_target
    prompt = [5, 8, 17]
    first = original.greedy_generate(prompt, 1)[0]
    monkeypatch.setattr(replay, "eos_token_ids", frozenset({first}))
    with DartEngine(replay, drafter=NGramDrafter()) as engine:
        for _ in range(2):
            assert engine.generate(prompt, DecodeConfig(max_new_tokens=8)).token_ids == [first]
    assert replay.greedy_generate(prompt, 8) == [first]


def test_interleaved_requests_crop_to_empty_and_output_lifetime(replay_target):
    original, replay = replay_target
    a, _ = replay.prefill([5, 6, 7])
    b, _ = replay.prefill([17, 19])
    saved = replay.forward_block(a, [8, 9, 10, 11, 12])
    snapshot = saved.clone()
    replay.forward_block(b, [20, 21])
    replay.crop(a, 5)
    control, _ = original.prefill([5, 6, 7, 8, 9])
    assert_logits(replay.forward_block(a, [30, 31]), original.forward_block(control, [30, 31]))
    control, _ = original.prefill([17, 19, 20, 21])
    assert_logits(replay.forward_block(b, [22]), original.forward_block(control, [22]))
    torch.testing.assert_close(saved, snapshot, rtol=0, atol=0)
    assert a.cache.storage.data_ptr() != b.cache.storage.data_ptr()
    replay.crop(a, 0)
    assert_logits(replay.forward_block(a, [40]), original.all_logits([40]))
    assert a.seq_len == 1 and b.seq_len == 5


def test_capture_inputs_survive_allocator_reuse(replay_targets):
    original, graph, _ = replay_targets
    state, _ = graph.prefill([5, 6, 7])
    control, _ = original.prefill([5, 6, 7])
    for length in [1, 5, 2, 4, 3, 1]:
        pressure = [torch.full((n,), 100000, dtype=torch.long, device="cuda")
                    for n in [1, 2, 3, 4, 5, 64] for _ in range(8)]
        tokens = list(range(30, 30 + length))
        assert_logits(graph.forward_block(state, tokens), original.forward_block(control, tokens))
        assert pressure[-1][-1].item() == 100000
    assert all(block.graph is not None for block in graph._blocks.values())


def test_context_and_block_boundaries_leave_state_usable(replay_target):
    original, replay = replay_target
    state, _ = replay.prefill([5] * 62)
    replay.forward_block(state, [6, 7])
    assert state.seq_len == 64
    with pytest.raises(ValueError, match="capacity"):
        replay.forward_block(state, [8])
    assert state.seq_len == 64
    with pytest.raises(ValueError, match="capacity"):
        replay.greedy_generate([5] * 63, 2)
    state, _ = replay.prefill([5, 6])
    before = state.cache
    with pytest.raises(ValueError, match="prepared replay size"):
        replay.forward_block(state, [7] * 6)
    assert state.seq_len == 2 and state.cache is before
    assert_logits(replay.forward_block(state, [7]), original.all_logits([5, 6, 7])[-1:])


@pytest.mark.parametrize("tokens", [[], [[5], [6]], [-1], [211]])
def test_invalid_ids_fail_before_gpu_indexing(replay_target, tokens):
    _, replay = replay_target
    state, _ = replay.prefill([5, 6])
    for value in (tokens, torch.tensor(tokens, device="cuda", dtype=torch.long)):
        with pytest.raises(ValueError, match="token id"):
            replay.forward_block(state, value)
        assert state.seq_len == 2
    assert torch.isfinite(replay.forward_block(state, [7])).all()


def test_replay_cache_cannot_cross_targets(replay_targets):
    _, graph, static = replay_targets
    state, _ = graph.prefill([5, 6])
    graph.forward_block(state, [7])
    with pytest.raises(ValueError, match="different replay target"):
        static.forward_block(state, [8])
    with pytest.raises(ValueError, match="different replay target"):
        static.crop(state, 1)
    assert state.seq_len == 3
    graph.forward_block(state, [8])


def test_replay_rejects_settings_changes_and_other_streams(replay_targets, monkeypatch):
    original, graph, _ = replay_targets
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(ValueError, match="stream that prepared"):
            graph.prefill([5, 6])
    for obj, attribute, value in [(graph, "record_hidden", True), (original.model, "training", True),
                                  (original.model.config, "_attn_implementation", "sdpa")]:
        with monkeypatch.context() as patch:
            patch.setattr(obj, attribute, value)
            with pytest.raises(ValueError, match="unchanged inference"):
                graph.prefill([5, 6])
    parameter = original.model.model.embed_tokens.weight
    with monkeypatch.context() as patch:
        patch.setattr(original.model.model.embed_tokens, "weight", torch.nn.Parameter(parameter.detach().clone()))
        with pytest.raises(ValueError, match="tensor storage changed"):
            graph.prefill([5, 6])
    projection = original.model.model.layers[0].self_attn.q_proj
    with monkeypatch.context() as patch:
        transposed = torch.nn.Parameter(projection.weight.detach().t())
        assert transposed.data_ptr() == projection.weight.data_ptr() and transposed.shape == projection.weight.shape
        patch.setattr(projection, "weight", transposed)
        with pytest.raises(ValueError, match="tensor storage changed"):
            graph.prefill([5, 6])
    assert graph.greedy_generate([5, 6], 2) == original.greedy_generate([5, 6], 2)


def test_unsupported_cache_and_tree_are_explicit(replay_target):
    _, replay = replay_target
    with pytest.raises(ValueError, match="own request cache"):
        replay.prefill([5, 6], cache=DynamicCache())
    with pytest.raises(ValueError, match="block verification"):
        replay.forward_tree(None, [], [], [])
    state, _ = replay.prefill([5, 6])
    replay.forward_block(state, [7])
    for length in [-1, 4, True, 1.5]:
        with pytest.raises(ValueError):
            replay.crop(state, length)
        assert state.seq_len == 3


@cuda
@modern_cache
@pytest.mark.parametrize("execution", ["static", "cuda_graph"])
def test_factory_replay_and_close(execution):
    target = build_target({"model": "tiny:72", "device": "cuda", "precision": "fp32", "execution": execution,
                           "cache_capacity": 16, "max_block_tokens": 1})
    assert isinstance(target, CudaGraphTarget) and target.execution_backend == execution
    state, _ = target.prefill([5, 6])
    assert target.forward_block(state, [7]).shape == (1, 211)
    assert target.setup_ms > 0
    target.close()
    target.close()
    with pytest.raises(RuntimeError, match="closed"):
        target.prefill([5, 6])
    with pytest.raises(RuntimeError, match="closed"):
        target.crop(state, 1)


@pytest.mark.parametrize("options", [
    {"capacity": 0}, {"capacity": 2049}, {"capacity": True}, {"max_block_tokens": 0},
    {"max_block_tokens": 17}, {"max_block_tokens": 2, "capacity": 1}, {"replay": 1},
])
def test_invalid_preparation_bounds(replay_targets, options):
    with pytest.raises(ValueError):
        CudaGraphTarget(replay_targets[0], **options)


def test_replay_rejects_non_fp32_and_hidden_recording(replay_targets, monkeypatch):
    original = replay_targets[0]
    with monkeypatch.context() as patch:
        patch.setattr(original, "dtype", torch.bfloat16)
        with pytest.raises(ValueError, match="float32 CUDA"):
            CudaGraphTarget(original)
    with monkeypatch.context() as patch:
        patch.setattr(original, "record_hidden", True)
        with pytest.raises(ValueError, match="no hidden-state recording"):
            CudaGraphTarget(original)


def test_wrapping_preserves_target_eos_override(replay_targets, monkeypatch):
    original = replay_targets[0]
    prompt = [5, 8, 17]
    eos = original.greedy_generate(prompt, 1)[0]
    monkeypatch.setattr(original, "eos_token_ids", frozenset({eos}))
    wrapped = CudaGraphTarget(original, capacity=16, max_block_tokens=1, replay=False)
    try:
        assert wrapped.eos_token_ids == original.eos_token_ids
        assert wrapped.greedy_generate(prompt, 8) == [eos]
    finally:
        wrapped.close()
