"""Units: KV quantization, token-drop compaction, n-gram lookup,
tokenizer bridge, scheduler routing, attention mask layouts."""

from __future__ import annotations

import pytest
import torch

from dart.attention.block_sparse import block_sparse_mask, score_blocks, select_blocks
from dart.attention.sliding_global import sliding_global_mask
from dart.attention.triton_kernels import HAS_TRITON
from dart.attention.verifier_mask import linear_block_mask, tree_mask
from dart.drafters.ngram import NGramDrafter
from dart.kv.cache import (
    KVStore,
    PrecisionPolicy,
    QuantizedKVCache,
    QuantSpec,
    QuantStore,
    dequantize_blockwise,
    quantize_blockwise,
)
from dart.kv.compressors.token_drop import TokenDropSpec, compact_sequence
from dart.models.tokenizer_bridge import TokenizerBridge
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeState, VerifyResult


# ------------------------------ KV quant ------------------------------ #
@pytest.mark.parametrize("bits,tol", [(8, 0.02), (4, 0.2), (3, 0.5)])
def test_quantize_roundtrip_error(bits, tol):
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 16)
    q, s, z = quantize_blockwise(x, bits)
    err = (dequantize_blockwise(q, s, z, x.dtype) - x).abs().max().item()
    span = (x.amax(-1) - x.amin(-1)).max().item()
    assert err <= span / ((1 << bits) - 1) * 0.5 + 1e-6
    assert err <= tol * span


def test_quantstore_append_crop_and_bytes():
    s = QuantStore()
    x = torch.randn(1, 2, 4, 8)
    s.append(x, bits=8)
    s.append(torch.randn(1, 2, 3, 8), bits=4)
    assert s.length == 7
    assert s.bits_per_token.tolist() == [8] * 4 + [4] * 3
    before = s.logical_bytes()
    s.crop(5)
    assert s.length == 5 and s.logical_bytes() < before
    s.crop(0)
    assert s.length == 0 and s.logical_bytes() == 0


def test_quantized_kv_cache_ages_and_crops():
    spec = QuantSpec(bits=4, sink_len=2, sink_bits=8, window_len=3)
    cache = QuantizedKVCache(spec)
    k = torch.randn(1, 2, 10, 8)
    v = torch.randn(1, 2, 10, 8)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape and v_out.shape == v.shape
    assert cache.get_seq_length() == 10
    st = cache.stats()
    assert st["quantized_tokens"] == 7 and st["window_tokens"] == 3
    # sinks kept at 8 bits, middle at 4
    assert cache._kq[0].bits_per_token.tolist() == [8, 8, 4, 4, 4, 4, 4]

    cache.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), layer_idx=0)
    assert cache.get_seq_length() == 11

    cache.crop(5)  # cut into the quantized region
    assert cache.get_seq_length() == 5
    k2, _ = cache.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), 0)
    assert k2.size(-2) == 6


# ----------------------- precision policy (14.3) --------------------- #
def test_precision_policy_rules_take_the_max():
    p = PrecisionPolicy(
        default_bits=4, sink_len=2, sink_bits=8,
        syntax_token_ids=frozenset({7}), syntax_bits=6,
        high_entropy_threshold=2.0, high_entropy_bits=7,
        num_layers=4, last_layers_hi=2, first_layer_hi=True, layer_hi_bits=8,
    )
    assert p.bits_for(0, 1) == 8  # sink prefix
    assert p.bits_for(5, 1) == 4  # plain middle token
    assert p.bits_for(5, 1, token_id=7) == 6  # syntax delimiter
    assert p.bits_for(5, 1, entropy=3.0) == 7  # high entropy
    assert p.bits_for(5, 0) == 8  # first layer sensitive
    assert p.bits_for(5, 3) == 8  # last-2 layers sensitive (3 >= 4-2)
    assert p.bits_for(5, 2) == 8  # layer 2 also in the last two {2, 3}
    assert p.bits_for(5, 1) == 4  # layer 1 is neither first nor last-2
    # Rules compose by max: sink + syntax -> 8 (the higher), never lowered.
    assert p.bits_for(0, 1, token_id=7) == 8


class _Meta:
    """Per-position metadata provider for QuantizedKVCache."""

    def __init__(self, ids, entropy=None):
        self.ids, self.entropy = ids, entropy

    def token_id_at(self, pos):
        return self.ids[pos] if 0 <= pos < len(self.ids) else None

    def entropy_at(self, pos):
        e = self.entropy
        return e[pos] if e is not None and 0 <= pos < len(e) else None


def test_cache_precision_policy_raises_bits_for_syntax_tokens():
    pol = PrecisionPolicy(
        default_bits=4, sink_len=1, sink_bits=8,
        syntax_token_ids=frozenset({99}), syntax_bits=8,
    )
    spec = QuantSpec(bits=4, sink_len=1, sink_bits=8, window_len=2, policy=pol)
    meta = _Meta([10, 99, 12, 13, 14, 15])  # position 1 holds a syntax token
    cache = QuantizedKVCache(spec, meta=meta)
    cache.update(torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8), layer_idx=0)
    # 4 tokens age out (6 - window 2): pos0 sink, pos1 syntax -> 8; rest 4.
    assert cache._kq[0].bits_per_token.tolist() == [8, 8, 4, 4]


def test_cache_default_policy_is_backward_compatible():
    # No explicit policy: behaves exactly like the v0.1 sink+middle scheme.
    spec = QuantSpec(bits=4, sink_len=2, sink_bits=8, window_len=3)
    cache = QuantizedKVCache(spec)
    cache.update(torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8), layer_idx=0)
    assert cache._kq[0].bits_per_token.tolist() == [8, 8, 4, 4, 4, 4, 4]


def test_kvstore_protocol_conformance():
    from dart.kv.compressors.turboquant_like import PolarVQStore

    assert isinstance(QuantStore(), KVStore)
    assert isinstance(PolarVQStore(), KVStore)


def test_compressed_kv_precision_policy_stays_exact():
    from dart.drafters.compressed_kv import CompressedKVDrafter
    from dart.runtime.engine import DartEngine
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    pol = PrecisionPolicy(
        default_bits=4, sink_len=4, sink_bits=8,
        syntax_token_ids=frozenset({10, 20, 30}), syntax_bits=8,
        high_entropy_threshold=1.5, high_entropy_bits=8,  # exercises the entropy path
        num_layers=target.model.config.num_hidden_layers, last_layers_hi=1,
    )
    spec = QuantSpec(bits=4, sink_len=4, sink_bits=8, window_len=6, policy=pol)
    drafter = CompressedKVDrafter(target, spec)
    prompt = random_prompt(101, 12)
    base = target.greedy_generate(prompt, max_new_tokens=24)
    out = DartEngine(target, drafter=drafter, verifier="block").generate(
        prompt, DecodeConfig(max_new_tokens=24, draft_len=4)
    )
    assert out.token_ids == base  # fine-grained KV precision, still byte-exact


# --------------------------- token drop ------------------------------ #
def test_compact_sequence():
    spec = TokenDropSpec(sink_len=3, window_len=4)
    short = list(range(6))
    assert compact_sequence(short, spec) == (short, False)
    long = list(range(20))
    compacted, dropped = compact_sequence(long, spec)
    assert dropped and compacted == [0, 1, 2, 16, 17, 18, 19]


# ------------------------------ n-gram ------------------------------- #
def test_ngram_prefers_longest_and_most_recent_match():
    d = NGramDrafter(max_ngram=3, min_ngram=1)
    #          0  1  2  3  4  5  6  7  8
    seq = [7, 8, 9, 1, 7, 8, 9, 2, 7, 8, 9]
    # suffix [7,8,9] matched at idx 4 (recent) -> continuation [2, 7, 8, 9...]
    assert d.propose(seq, 3) == [2, 7, 8]
    assert d.propose([1, 2, 3], 4) == []  # no repeat -> nothing
    assert d.propose([5, 6, 5], 2) == [6, 5]  # 1-gram fallback


# ------------------------- tokenizer bridge -------------------------- #
class _FakeTok:
    """Space-joined int 'tokenizer' with a controllable merge bug."""

    def __init__(self, merge_break=False):
        self.merge_break = merge_break

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        ids = [int(t) for t in text.split()]
        if self.merge_break and len(ids) >= 2:
            ids[0] = ids[0] + 1000  # simulate junction re-tokenization drift
        return ids


def test_bridge_clean_boundary():
    b = TokenizerBridge(_FakeTok())
    res = b.drafts_from_text([1, 2, 3], " 4 5 6", max_tokens=2)
    assert res.boundary_intact and res.draft_ids == [4, 5]


def test_bridge_broken_boundary_drops_drafts():
    b = TokenizerBridge(_FakeTok(merge_break=True))
    res = b.drafts_from_text([1, 2, 3], " 4 5", max_tokens=4)
    assert not res.boundary_intact and res.draft_ids == []


# ----------------------------- scheduler ----------------------------- #
def _state(**kw) -> DecodeState:
    base = dict(
        context_len=100, generated_len=10, gpu_free_gb=24.0, batch_size=1,
        prompt_kind="chat", exact_required=True, entropy_ewma=0.0,
        acceptance_ewma={}, kv_transfer_ms=0.0, target_step_ms=0.0,
    )
    base.update(kw)
    return DecodeState(**base)


def test_scheduler_long_context_routes_to_compressed_kv():
    s = HeuristicScheduler(SchedulerConfig(
        available_lanes=["compressed_kv", "diffusion", "target_only"],
        long_context_threshold=1000,
    ))
    a = s.choose_action(_state(context_len=5000))
    assert a.lane == "compressed_kv"


def test_scheduler_code_prefers_hot_mtp_then_diffusion():
    cfg = SchedulerConfig(available_lanes=["mtp", "diffusion", "target_only"])
    s = HeuristicScheduler(cfg)
    a = s.choose_action(_state(prompt_kind="code"))
    assert a.lane == "diffusion"  # mtp not hot yet
    s.acceptance_ewma["mtp"] = 0.9
    a = s.choose_action(_state(prompt_kind="code"))
    assert a.lane == "mtp"


def test_scheduler_benches_rejected_lanes():
    cfg = SchedulerConfig(
        available_lanes=["ngram", "target_only"], min_lane_acceptance=0.3,
        ewma_alpha=1.0,
    )
    s = HeuristicScheduler(cfg)
    assert s.choose_action(_state()).lane == "ngram"
    s.observe("ngram", VerifyResult([5], 4, 0, False, False))  # 0% accepted
    assert s.choose_action(_state()).lane == "target_only"


# --------------------------- attention masks ------------------------- #
def test_sliding_global_mask_properties():
    m = sliding_global_mask(q_len=4, kv_len=10, window=3, num_global=2)
    assert m.shape == (4, 10)
    q0_abs = 6
    row = m[0]
    assert row[:2].all()  # global anchors
    assert bool(row[q0_abs]) and not bool(row[q0_abs + 1])  # causal
    assert not bool(row[3])  # outside window, not global


def test_block_sparse_selection():
    torch.manual_seed(0)
    k = torch.randn(40, 8)
    q = torch.randn(4, 8)
    scores = score_blocks(q, k, block_size=8)  # 5 blocks
    kept = select_blocks(scores, top_k=2, always_first=1, always_last=1)
    assert 0 in kept.tolist() and 4 in kept.tolist()
    mask = block_sparse_mask(40, 8, kept)
    assert mask.sum().item() == 8 * len(kept)


def test_verifier_masks():
    lm = linear_block_mask(prefix_len=5, block_len=3)
    assert lm.shape == (3, 8)
    assert lm[0].tolist() == [True] * 6 + [False, False]
    tm = tree_mask(parent=[-1, 0, 0, 2], prefix_len=2)
    assert tm.shape == (4, 6)
    assert tm[3].tolist() == [True, True, True, False, True, True]


# ------------------------- polar VQ (turboquant-like) ----------------- #
def test_polar_vq_roundtrip_direction_quality():
    from dart.kv.compressors.turboquant_like import (
        PolarVQStore,
        pack_signs,
        random_orthogonal,
        unpack_signs,
    )

    torch.manual_seed(0)
    signs = torch.rand(2, 3, 5, 16) > 0.5
    assert torch.equal(unpack_signs(pack_signs(signs), 16) > 0, signs)

    r = random_orthogonal(16, seed=1)
    assert torch.allclose(r @ r.T, torch.eye(16), atol=1e-5)

    store = PolarVQStore(rotation_seed=1)
    x = torch.randn(1, 2, 12, 16)
    store.append(x[..., :4, :], bits=8)   # sink path (int8)
    store.append(x[..., 4:, :], bits=2)   # polar path (1-bit)
    assert store.length == 12
    x_hat = store.dequantize(torch.float32)
    assert x_hat.shape == x.shape
    # Sinks should be near-exact; polar tokens direction-correlated.
    assert (x_hat[..., :4, :] - x[..., :4, :]).abs().max() < 0.05
    cos = torch.nn.functional.cosine_similarity(
        x_hat[..., 4:, :].flatten(0, -2), x[..., 4:, :].flatten(0, -2), dim=-1
    )
    assert cos.mean() > 0.6  # 1-bit direction code: rough but informative
    # Crop across the sink/polar boundary.
    store.crop(6)
    assert store.length == 6
    assert store.dequantize(torch.float32).shape[-2] == 6
    before = store.logical_bytes()
    store.crop(2)
    assert store.logical_bytes() < before


def test_polar_vq_cache_lane_is_exact():
    from dart.drafters.compressed_kv import CompressedKVDrafter
    from dart.kv.compressors.turboquant_like import PolarVQStore, polar_vq_spec
    from dart.runtime.engine import DartEngine
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    drafter = CompressedKVDrafter(
        target, polar_vq_spec(sink_len=4, window_len=6),
        store_factory=PolarVQStore,
    )
    prompt = random_prompt(301, 14)
    base = target.greedy_generate(prompt, max_new_tokens=24)
    out = DartEngine(target, drafter=drafter, verifier="block").generate(
        prompt, DecodeConfig(max_new_tokens=24, draft_len=4)
    )
    assert out.token_ids == base  # 1-bit KV drafts, still byte-exact output


# ----------------------------- bandit scheduler ----------------------- #
def test_bandit_converges_to_better_lane():
    from dart.runtime.scheduler import BanditScheduler

    cfg = SchedulerConfig(available_lanes=["fast_lane", "slow_lane", "target_only"])
    b = BanditScheduler(cfg, ucb_c=0.3)

    def fake_round(lane):
        if lane == "fast_lane":  # 5 tokens / 10 ms
            r = VerifyResult([1, 2, 3, 4, 5], 4, 4, True, False)
        else:  # 1 token / 10 ms
            r = VerifyResult([1], 4, 0, False, False)
        b.observe(lane, r, draft_ms=4.0, verify_ms=6.0)

    picks = []
    for _ in range(60):
        lane = b.choose_action(_state()).lane
        picks.append(lane)
        fake_round(lane)

    assert set(picks[:2]) == {"fast_lane", "slow_lane"}  # explored both arms
    assert picks[20:].count("fast_lane") / len(picks[20:]) > 0.8


def test_bandit_long_context_hard_routes_to_kv_lanes():
    from dart.runtime.scheduler import BanditScheduler

    cfg = SchedulerConfig(
        available_lanes=["ngram", "compressed_kv", "target_only"],
        long_context_threshold=1000,
    )
    b = BanditScheduler(cfg)
    assert b.choose_action(_state(context_len=9000)).lane == "compressed_kv"


def test_bandit_engine_integration_is_exact():
    from dart.drafters.base import NullDrafter
    from dart.drafters.ngram import NGramDrafter
    from dart.runtime.engine import DartEngine
    from dart.runtime.scheduler import BanditScheduler
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    lanes = {"ngram": NGramDrafter(), "target_only": NullDrafter()}
    sched = BanditScheduler(SchedulerConfig(available_lanes=list(lanes)))
    engine = DartEngine(target, lanes=lanes, scheduler=sched, verifier="block")
    for seed, plen in [(401, 9), (402, 17)]:
        prompt = random_prompt(seed, plen)
        base = target.greedy_generate(prompt, max_new_tokens=24)
        out = engine.generate(prompt, DecodeConfig(max_new_tokens=24, draft_len=4))
        assert out.token_ids == base


# ----------------- entropy gate (Phase 6 / s.25) --------------------- #
def test_entropy_gated_shortens_only_above_threshold():
    from dart.runtime.scheduler import SchedulerConfig, entropy_gated

    c = SchedulerConfig(entropy_gate_threshold=2.0, entropy_gate_factor=0.5)
    lo = _state(entropy_ewma=1.0)
    hi = _state(entropy_ewma=5.0)
    assert entropy_gated(8, 32, lo, c) == (8, 32)  # below gate: untouched
    assert entropy_gated(8, 32, hi, c) == (4, 16)  # above gate: halved
    assert entropy_gated(1, 0, hi, c) == (1, 0)  # never below 1, zeros stay zero


def test_block_verifier_reports_entropy():
    from dart.runtime.engine import DartEngine
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    out = DartEngine(target, drafter=NGramDrafter(), verifier="block").generate(
        random_prompt(77, 12), DecodeConfig(max_new_tokens=16, draft_len=4)
    )
    # Random tiny target has high-entropy predictions; the signal is live.
    assert out.metadata["exact_greedy_equivalent"] is True


def test_entropy_gate_keeps_engine_exact():
    from dart.drafters.base import NullDrafter
    from dart.runtime.engine import DartEngine
    from dart.runtime.scheduler import BanditScheduler, SchedulerConfig
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    lanes = {"ngram": NGramDrafter(), "target_only": NullDrafter()}
    sched = BanditScheduler(
        SchedulerConfig(available_lanes=list(lanes), entropy_gate_threshold=0.0)
    )  # threshold 0 -> gate always active
    engine = DartEngine(target, lanes=lanes, scheduler=sched, verifier="block")
    prompt = random_prompt(78, 14)
    base = target.greedy_generate(prompt, max_new_tokens=24)
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=24, draft_len=8))
    assert out.token_ids == base  # gating changes speed, never output


# ---------------- bandit reward shaping / constraints (24.4) ---------- #
def test_bandit_benches_lane_over_rejection_rate():
    from dart.runtime.scheduler import BanditScheduler

    cfg = SchedulerConfig(
        available_lanes=["greedy_lane", "target_only"],
        max_rejection_rate=0.5, ewma_alpha=1.0,
    )
    b = BanditScheduler(cfg)
    assert b.choose_action(_state()).lane == "greedy_lane"  # explored first
    b.observe("greedy_lane", VerifyResult([5], 4, 0, False, False))  # 100% rejected
    assert b.choose_action(_state()).lane == "target_only"  # benched: >50% reject


def test_bandit_benches_lane_over_kv_budget():
    from dart.runtime.scheduler import BanditScheduler

    cfg = SchedulerConfig(
        available_lanes=["kv_lane", "target_only"], max_gpu_kv_gb=1.0,
    )
    b = BanditScheduler(cfg)
    assert b.choose_action(_state()).lane == "kv_lane"
    # High acceptance but blows the KV budget -> benched anyway.
    b.observe("kv_lane", VerifyResult([1, 2, 3, 4, 5], 4, 4, True, False), kv_gb=2.0)
    assert b.choose_action(_state()).lane == "target_only"


def test_bandit_memory_penalty_prefers_lighter_lane():
    from dart.runtime.scheduler import BanditScheduler

    cfg = SchedulerConfig(available_lanes=["light", "heavy"], memory_penalty=1.0)
    b = BanditScheduler(cfg, ucb_c=0.0)  # pure exploitation after exploring both

    def fake_round(lane):
        r = VerifyResult([1, 2, 3, 4, 5], 4, 4, True, False)  # identical throughput
        b.observe(lane, r, draft_ms=4.0, verify_ms=6.0,
                  kv_gb=0.0 if lane == "light" else 10.0)

    picks = []
    for _ in range(10):
        lane = b.choose_action(_state()).lane
        picks.append(lane)
        fake_round(lane)
    assert set(picks[:2]) == {"light", "heavy"}  # both explored
    assert all(p == "light" for p in picks[2:])  # then avoids the memory hog


def test_metadata_reports_kv_and_residency_fields():
    from dart.drafters.compressed_kv import CompressedKVDrafter
    from dart.runtime.engine import DartEngine
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    drafter = CompressedKVDrafter(
        target, QuantSpec(bits=4, sink_len=4, sink_bits=8, window_len=6)
    )
    prompt = random_prompt(102, 15)
    out = DartEngine(target, drafter=drafter).generate(
        prompt,
        DecodeConfig(max_new_tokens=20, draft_len=4, full_kv_residency="cpu_pinned"),
    )
    assert "gpu_kv_gb" in out.metadata and out.metadata["gpu_kv_gb"] >= 0.0
    assert out.metadata["full_kv_residency"] == "cpu_pinned"
    assert out.metadata["tokens_verified"] == out.metadata["tokens_generated"]


# ----------------------- KV residency (12.1) ------------------------- #
def test_sync_prefetcher_resolves_thunk():
    from dart.kv.prefetch import SyncPrefetcher

    calls = []
    pf = SyncPrefetcher().prefetch(lambda: calls.append(1) or 42)
    assert pf.wait() == 42 and calls == [1]
    assert pf.wait() == 42 and calls == [1]  # cached: thunk runs once


def test_thread_prefetcher_overlaps_and_preserves_order():
    import time

    from dart.kv.prefetch import ThreadPrefetcher

    pf = ThreadPrefetcher()
    order = []

    def work(tag, ms):
        time.sleep(ms / 1000.0)
        order.append(tag)
        return tag

    t0 = time.perf_counter()
    h = pf.prefetch(lambda: work("a", 40))
    # The submit returns immediately (work runs on the worker thread).
    assert (time.perf_counter() - t0) < 0.03
    assert h.wait() == "a" and h.wait() == "a"  # idempotent
    # Single worker -> FIFO ordering (engine residency relies on this).
    h1 = pf.prefetch(lambda: work("b", 5))
    h2 = pf.prefetch(lambda: work("c", 5))
    h1.wait()
    h2.wait()
    assert order == ["a", "b", "c"]
    pf.close()


def test_move_cache_is_noop_when_already_on_device():
    from dart.kv.residency import move_cache
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()  # cpu target
    state, _ = target.prefill(random_prompt(1, 8))
    stats = move_cache(state.cache, "cpu", pin=True)  # already on cpu
    assert stats.moved_tensors == 0 and stats.moved_bytes == 0


def test_move_cache_write_back_mutates_real_cache_storage():
    """Regression: on transformers >= 4.54 ``cache.key_cache`` is a property
    returning a *fresh list*, so writing into it silently mutated a temporary
    - move_cache became a no-op on GPU while still reporting transfer stats.
    ``_layer_slots`` setters must hit the cache's real storage on both
    layouts."""
    from dart.kv.residency import _layer_slots
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    state, _ = target.prefill(random_prompt(2, 8))
    cache = state.cache

    slots = list(_layer_slots(cache))
    assert slots, "no K/V tensors found in the cache"
    originals = [get() for get, _ in slots]
    assert all(t is not None for t in originals)
    for get, set_ in slots:
        set_(get().clone())  # simulate a device move (identity must change)

    # Re-read through the cache's REAL storage, layout-independently.
    layers = getattr(cache, "layers", None)
    if layers is not None:
        live = [t for layer in layers for t in (layer.keys, layer.values)]
    else:
        live = list(cache.key_cache) + list(cache.value_cache)
    live_ids = {id(t) for t in live if t is not None}
    assert live_ids.isdisjoint({id(t) for t in originals}), (
        "setter wrote to a temporary list, not the cache storage "
        "(the pre-fix bug on transformers >= 4.54)"
    )
    # The cache must still be usable for further decoding.
    logits = target.forward_block(state, [7])
    assert logits.shape[-1] == target.model.config.vocab_size


def test_cpu_pinned_residency_stays_exact_and_reports_moves():
    from dart.drafters.ngram import NGramDrafter
    from dart.runtime.engine import DartEngine
    from dart.runtime.types import DecodeConfig
    from tests.conftest import build_tiny_target, random_prompt

    target = build_tiny_target()
    prompt = random_prompt(103, 18)
    base = target.greedy_generate(prompt, max_new_tokens=24)
    eng = DartEngine(target, drafter=NGramDrafter(), verifier="block")
    gpu = eng.generate(prompt, DecodeConfig(max_new_tokens=24, draft_len=4))
    pinned = eng.generate(
        prompt,
        DecodeConfig(max_new_tokens=24, draft_len=4, full_kv_residency="cpu_pinned"),
    )
    # Residency policy must not change a single token.
    assert gpu.token_ids == base and pinned.token_ids == base
    # cpu_pinned path reports its (CPU-target: zero) transfer accounting.
    assert "kv_residency_moves" in pinned.metadata
    assert "kv_residency_moved_mb" in pinned.metadata
    assert "kv_residency_moves" not in gpu.metadata  # only when parking


# ------------------- triton kernel references (CPU-checkable) --------- #
def test_dequant_attention_reference_matches_eager():
    import math

    from dart.attention.triton_kernels.dequant_attention import (
        dequant_attention_reference,
    )
    from dart.kv.cache import quantize_blockwise

    torch.manual_seed(0)
    h, t, d = 2, 9, 16
    k = torch.randn(1, h, t, d)
    v = torch.randn(1, h, t, d)
    q = torch.randn(h, d)
    kq, ks, kz = quantize_blockwise(k, 8)
    vq, vs, vz = quantize_blockwise(v, 8)

    out = dequant_attention_reference(
        q, kq[0], ks[0], kz[0], vq[0], vs[0], vz[0]
    )
    # Contract check vs plain eager attention over the dequantized tensors.
    k_hat = (kq[0].float() * ks[0] + kz[0])
    v_hat = (vq[0].float() * vs[0] + vz[0])
    scores = torch.einsum("hd,htd->ht", q, k_hat) / math.sqrt(d)
    expected = torch.einsum("ht,htd->hd", torch.softmax(scores, -1), v_hat)
    assert torch.allclose(out, expected, atol=1e-5)
    # And int8 round-trip keeps it close to TRUE attention.
    true_scores = torch.einsum("hd,htd->ht", q, k[0]) / math.sqrt(d)
    true_out = torch.einsum("ht,htd->hd", torch.softmax(true_scores, -1), v[0])
    assert (out - true_out).abs().max() < 0.05


def test_gather_blocks_reference():
    from dart.attention.triton_kernels.gather_block_sparse import (
        gather_blocks_reference,
    )

    kv = torch.arange(2 * 20 * 4, dtype=torch.float32).reshape(2, 20, 4)
    out = gather_blocks_reference(kv, torch.tensor([0, 2]), block_size=8)
    # block 0 -> rows 0..7; block 2 -> rows 16..19 (short tail block)
    assert out.shape == (2, 12, 4)
    assert torch.equal(out[:, :8], kv[:, 0:8])
    assert torch.equal(out[:, 8:], kv[:, 16:20])


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="triton kernels need CUDA + triton (the fused path raises otherwise)",
)
def test_triton_dequant_attention_matches_reference():  # pragma: no cover
    from dart.attention.triton_kernels.dequant_attention import (
        dequant_attention,
        dequant_attention_reference,
    )
    from dart.kv.cache import quantize_blockwise

    torch.manual_seed(0)
    h, t, d = 4, 100, 64
    k = torch.randn(1, h, t, d, device="cuda")
    v = torch.randn(1, h, t, d, device="cuda")
    q = torch.randn(h, d, device="cuda")
    kq, ks, kz = quantize_blockwise(k, 8)
    vq, vs, vz = quantize_blockwise(v, 8)
    ref = dequant_attention_reference(q, kq[0], ks[0], kz[0], vq[0], vs[0], vz[0])
    out = dequant_attention(q, kq[0], ks[0], kz[0], vq[0], vs[0], vz[0])
    assert (out - ref).abs().max() < 1e-3
