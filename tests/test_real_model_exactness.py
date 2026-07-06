"""Exactness on a REAL pretrained model (docs/design.md §13.4).

The rest of the suite proves the contract on random tiny targets, which is
correct (exactness is weight-independent) but leaves one question open: does
the pipeline still hold with a real tokenizer, real vocabulary, and real
attention/RoPE?  This test answers it on ``sshleifer/tiny-gpt2`` (a small
real GPT-2 checkpoint), and is skipped automatically when the model cannot
be fetched (offline CI).
"""

from __future__ import annotations

import pytest
import torch

from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.drafters.ngram import NGramDrafter
from dart.drafters.sparse_attention import SparseAttentionDrafter
from dart.kv.cache import QuantSpec
from dart.kv.compressors.token_drop import TokenDropSpec
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig

MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def real_target():
    try:
        return TargetModel.from_pretrained(MODEL, device="cpu", dtype=torch.float32)
    except Exception as e:  # offline / hub error -> skip, do not fail
        pytest.skip(f"could not load real model {MODEL}: {type(e).__name__}")


PROMPTS = [
    "def fibonacci(n):\n    ",
    "The capital of France is",
    '{"name": "ada", "age":',
]


def test_dart_baseline_matches_hf_generate_on_real_model(real_target):
    tok = real_target.tokenizer
    for text in PROMPTS:
        ids = tok.encode(text)
        base = real_target.greedy_generate(ids, max_new_tokens=24)
        hf = real_target.model.generate(
            torch.tensor([ids]), max_new_tokens=24, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=real_target.model.config.eos_token_id,
        )[0, len(ids):].tolist()
        assert hf[: len(base)] == base[: len(hf)]  # DART baseline == HF greedy


@pytest.mark.parametrize("lane_name", ["ngram", "compressed_kv_int4", "sparse_kv"])
def test_lanes_byte_exact_on_real_model(real_target, lane_name):
    lanes = {
        "ngram": lambda: NGramDrafter(max_ngram=4),
        "compressed_kv_int4": lambda: CompressedKVDrafter(
            real_target, QuantSpec(bits=4, sink_len=16, window_len=64)
        ),
        "sparse_kv": lambda: SparseAttentionDrafter(
            real_target, TokenDropSpec(sink_len=16, window_len=128)
        ),
    }
    tok = real_target.tokenizer
    for text in PROMPTS:
        ids = tok.encode(text)
        base = real_target.greedy_generate(ids, max_new_tokens=24)
        out = DartEngine(
            real_target, drafter=lanes[lane_name](), verifier="block"
        ).generate(ids, DecodeConfig(max_new_tokens=24, draft_len=4))
        assert out.token_ids == base, f"{lane_name} diverged on {text!r}"
        assert out.metadata["exact_greedy_equivalent"] is True
