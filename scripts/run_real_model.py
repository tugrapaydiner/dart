"""Real-model exactness validation (docs/design.md §13.4).

The exactness suite proves DART against *random* tiny targets - which is
the right test for the contract, since exactness is weight-independent.
This script closes the obvious credibility question that random weights
leave open: does the whole pipeline - real tokenizer, real vocabulary,
real attention/RoPE - still produce byte-identical greedy output on an
actual pretrained model?

It runs three independent references and asserts they all agree:

  1. HuggingFace ``model.generate`` greedy (the ecosystem ground truth);
  2. ``TargetModel.greedy_generate`` (DART's trusted K=0 baseline);
  3. ``DartEngine.generate`` exact mode, per draft lane.

    python scripts/run_real_model.py --model sshleifer/tiny-gpt2
    python scripts/run_real_model.py --model gpt2 --max-new 64
    python scripts/run_real_model.py --model Qwen/Qwen2.5-0.5B-Instruct

Default model is tiny so the script is runnable anywhere; point --model at
a real checkpoint on a capable box.  Wall-clock here is not a claim
(docs/design.md §5.3 / README): CPU timings only show the contract
holds, not that DART is faster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.drafters.ar_draft import ARDrafter
from dart.drafters.block_sparse_kv import BlockSparseKVDrafter
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.drafters.ngram import NGramDrafter
from dart.drafters.sparse_attention import SparseAttentionDrafter
from dart.kv.cache import QuantSpec
from dart.kv.compressors.token_drop import TokenDropSpec
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig

DEFAULT_PROMPTS = [
    "def fibonacci(n):\n    ",
    "The capital of France is",
    '{"name": "ada", "age":',
    "import numpy as np\n",
]


def load_prompts(path: str | None) -> list[str]:
    """Curated workload prompts from a JSONL file (docs/design.md §13.1),
    e.g. benchmarks/prompts/code.jsonl; falls back to DEFAULT_PROMPTS."""
    if not path:
        return DEFAULT_PROMPTS
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(ln)["prompt"] for ln in lines if ln.strip()]


def hf_generate_greedy(target: TargetModel, prompt_ids: list[int], n: int) -> list[int]:
    """Pure HuggingFace greedy continuation (no DART involved)."""
    ids = torch.tensor([prompt_ids], device=target.device)
    out = target.model.generate(
        ids,
        max_new_tokens=n,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=target.model.config.eos_token_id,
    )
    return out[0, len(prompt_ids):].tolist()


def build_lanes(target: TargetModel) -> dict:
    return {
        "ngram": NGramDrafter(max_ngram=4),
        "compressed_kv_int4": CompressedKVDrafter(
            target, QuantSpec(bits=4, sink_len=16, window_len=64)
        ),
        "sparse_kv": SparseAttentionDrafter(
            target, TokenDropSpec(sink_len=16, window_len=128)
        ),
        "block_sparse_kv": BlockSparseKVDrafter(
            target, block_size=16, top_k=3, sink_blocks=1, recent_blocks=2
        ),
        "ar_draft_self": ARDrafter(target),  # self-draft: a perfect oracle drafter
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--draft-len", type=int, default=4)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--prompts-file",
        help="JSONL of {'prompt': ...} workloads, e.g. benchmarks/prompts/code.jsonl",
    )
    args = ap.parse_args()
    prompt_texts = load_prompts(args.prompts_file)

    print(f"loading {args.model} ...")
    target = TargetModel.from_pretrained(
        args.model, device="cpu", dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    tok = target.tokenizer
    prompts = [tok.encode(p) for p in prompt_texts]

    print(f"\nmodel={args.model}  vocab={target.model.config.vocab_size}  "
          f"max_new={args.max_new}  prompts={len(prompts)}\n")
    all_exact = True
    for pi, (text, ids) in enumerate(zip(prompt_texts, prompts)):
        hf = hf_generate_greedy(target, ids, args.max_new)
        base = target.greedy_generate(ids, max_new_tokens=args.max_new)
        ref_hash = output_hash(base)
        # DART baseline must already equal HF greedy.
        hf_ok = hf == base
        all_exact &= hf_ok
        print(f"[prompt {pi}] {text!r}")
        print(f"   HF==DART-baseline: {hf_ok}   tokens={len(hf)}/{len(base)}   hash={ref_hash[:12]}")
        for name, drafter in build_lanes(target).items():
            out = DartEngine(target, drafter=drafter, verifier="block").generate(
                ids, DecodeConfig(max_new_tokens=args.max_new, draft_len=args.draft_len)
            )
            ok = out.token_ids == base
            all_exact &= ok
            tag = "OK " if ok else "MISMATCH"
            print(f"   {tag} {name:20s} accept={out.metadata['acceptance_rate']:.3f} "
                  f"hash={out.metadata['output_hash'][:12]}")
        print()

    if all_exact:
        print(f"EXACTNESS HOLDS on real model {args.model}: "
              f"every lane byte-identical to HF greedy.")
    else:
        print("EXACTNESS FAILURE on real model", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
