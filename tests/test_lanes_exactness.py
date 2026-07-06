"""Every draft lane - however lossy - must yield byte-identical greedy
output.  This is the whole point of DART; if one of these fails, the lane
is broken or the verifier is, and either way nothing else matters."""

from __future__ import annotations

import pytest
import torch

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.ar_draft import ARDrafter
from dart.drafters.base import NullDrafter
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.drafters.mtp import MTPDrafter, MTPHeads
from dart.drafters.ngram import NGramDrafter
from dart.drafters.sparse_attention import SparseAttentionDrafter
from dart.kv.cache import QuantSpec
from dart.kv.compressors.token_drop import TokenDropSpec
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeConfig
from tests.conftest import VOCAB, build_tiny_target, random_prompt

N_NEW = 24
PROMPTS = [(101, 7), (102, 15), (103, 28)]


def _check_lane(target, drafter, seed, plen, draft_len=4, canvas_len=8):
    prompt = random_prompt(seed, plen)
    base = target.greedy_generate(prompt, max_new_tokens=N_NEW)
    engine = DartEngine(target, drafter=drafter, verifier="block")
    out = engine.generate(
        prompt,
        DecodeConfig(max_new_tokens=N_NEW, draft_len=draft_len, canvas_len=canvas_len),
    )
    assert out.token_ids == base, f"lane {drafter.name} diverged"
    return out


@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_ngram_lane_exact(tiny_target, seed, plen):
    _check_lane(tiny_target, NGramDrafter(max_ngram=4), seed, plen)


def test_ngram_lane_exact_on_repetitive_prompt(tiny_target):
    span = random_prompt(111, 9)
    prompt = span + random_prompt(112, 3) + span  # repetition: ngram will fire
    base = tiny_target.greedy_generate(prompt, max_new_tokens=N_NEW)
    engine = DartEngine(tiny_target, drafter=NGramDrafter(), verifier="block")
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=N_NEW, draft_len=4))
    assert out.token_ids == base
    assert out.metadata["draft_tokens_proposed"] > 0  # it actually drafted


@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_ar_draft_lane_exact(tiny_target, seed, plen):
    draft_model = build_tiny_target(seed=4321)  # different weights on purpose
    _check_lane(tiny_target, ARDrafter(draft_model), seed, plen)


def test_ar_draft_self_oracle_full_acceptance(tiny_target):
    """Drafting with the target itself == oracle: full acceptance, exact."""
    prompt = random_prompt(121, 10)
    base = tiny_target.greedy_generate(prompt, max_new_tokens=N_NEW)
    if any(t in tiny_target.eos_token_ids for t in base):
        pytest.skip("EOS in stream; acceptance accounting differs")
    out = _check_lane(tiny_target, ARDrafter(tiny_target), 121, 10)
    assert out.metadata["acceptance_rate"] == 1.0


@pytest.mark.parametrize("bits", [8, 4, 3])
@pytest.mark.parametrize("seed,plen", PROMPTS[:2])
def test_compressed_kv_lane_exact(tiny_target, bits, seed, plen):
    spec = QuantSpec(bits=bits, sink_len=4, sink_bits=8, window_len=6)
    _check_lane(tiny_target, CompressedKVDrafter(tiny_target, spec), seed, plen)


@pytest.mark.parametrize("seed,plen", PROMPTS[:2])
def test_sparse_kv_lane_exact(tiny_target, seed, plen):
    spec = TokenDropSpec(sink_len=4, window_len=12)  # forces real dropping
    _check_lane(tiny_target, SparseAttentionDrafter(tiny_target, spec), seed, plen)


@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_block_sparse_kv_lane_exact(tiny_target, seed, plen):
    from dart.drafters.block_sparse_kv import BlockSparseKVDrafter

    # Small blocks + budget so generation actually crosses into compaction.
    drafter = BlockSparseKVDrafter(
        tiny_target, block_size=4, top_k=2, sink_blocks=1, recent_blocks=2
    )
    _check_lane(tiny_target, drafter, seed, plen)
    if plen >= 28:  # long enough that block selection actually fired
        assert drafter.compacted_last_round


def _diffusion_drafter(mode: str, seed: int = 5) -> DiffusionDrafter:
    torch.manual_seed(seed)
    model = DiffusionDrafterModel(
        DiffusionDrafterConfig(
            vocab_size=VOCAB, hidden_size=64, num_layers=2, num_heads=4,
            ctx_window=24, max_canvas=16,
        )
    )
    return DiffusionDrafter(
        model, canvas_len=8, denoise_steps=4,
        reconcile=ReconcilePolicy(mode=mode, repair_steps=2),
    )


@pytest.mark.parametrize("mode", ["discard", "reconcile"])
@pytest.mark.parametrize("seed,plen", PROMPTS[:2])
def test_diffusion_lane_exact(tiny_target, mode, seed, plen):
    out = _check_lane(tiny_target, _diffusion_drafter(mode), seed, plen)
    assert out.metadata["draft_tokens_proposed"] > 0


def test_diffusion_reconcile_uses_warm_canvases(tiny_target):
    drafter = _diffusion_drafter("reconcile")
    _check_lane(tiny_target, drafter, 131, 12)
    # Untrained drafter -> rejections every round -> warm canvases existed.
    assert drafter.stats["warm_rounds"] > 0


def test_mtp_lane_exact():
    target = build_tiny_target()
    target.record_hidden = True
    torch.manual_seed(9)
    heads = MTPHeads(
        hidden_size=target.model.config.hidden_size, vocab_size=VOCAB, num_offsets=3
    )
    _check_lane(target, MTPDrafter(target, heads), 141, 11, draft_len=3)


def test_scheduled_multilane_engine_exact(tiny_target):
    lanes = {
        "target_only": NullDrafter(),
        "ngram": NGramDrafter(),
        "compressed_kv": CompressedKVDrafter(
            tiny_target, QuantSpec(bits=4, sink_len=4, window_len=6)
        ),
        "diffusion": _diffusion_drafter("reconcile"),
    }
    sched = HeuristicScheduler(
        SchedulerConfig(available_lanes=list(lanes), draft_len=4, canvas_len=8)
    )
    engine = DartEngine(tiny_target, lanes=lanes, scheduler=sched, verifier="block")
    for seed, plen in PROMPTS:
        prompt = random_prompt(seed, plen)
        base = tiny_target.greedy_generate(prompt, max_new_tokens=N_NEW)
        out = engine.generate(prompt, DecodeConfig(max_new_tokens=N_NEW, draft_len=4))
        assert out.token_ids == base
