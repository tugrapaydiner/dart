"""Measure exactness vs KV bit-width: unverified quantized decoding vs DART.

The claim behind Figure 2 (README §2): decoding *directly* on a quantized KV
cache drifts from the full-precision target as bits fall, while DART - which
uses the same quantized cache only to draft and verifies on full KV - stays
byte-exact at every bit-width.  This script measures both arms and writes the
artifact the figure is generated from, so the figure can never drift from
data (docs/benchmark_methodology.md).

Arms, per bit-width:
  * ``unverified``  - greedy decoding whose *only* KV cache is the quantized
                      cache (the trusted baseline path, just with lossy KV);
  * ``dart``        - exact mode with a CompressedKVDrafter at the same spec
                      (drafts on quantized KV, verifies on full KV).

Both are compared token-for-token against the full-precision greedy baseline.
The quantization spec uses a deliberately small sink/window so aged-out
(quantized) tokens dominate and the bit-width actually binds.

    python scripts/measure_exactness_vs_bits.py --prompts 12 --max-new 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_bench import synthetic_prompts
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.kv.cache import QuantizedKVCache, QuantSpec
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_tiny_target
from dart.runtime.types import DecodeConfig


def greedy_on_cache(target: TargetModel, prompt: list[int], max_new: int, cache):
    """Greedy decode trusting ``cache`` as the model's only KV store."""
    state, logits = target.prefill(prompt, cache=cache)
    out: list[int] = []
    tok = int(torch.argmax(logits).item())
    for _ in range(max_new):
        out.append(tok)
        if tok in target.eos_token_ids or len(out) == max_new:
            break
        step = target.forward_block(state, [tok])
        tok = int(torch.argmax(step[-1]).item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--draft-len", type=int, default=4)
    ap.add_argument("--bits", default="8,6,4,3,2")
    ap.add_argument("--sink-len", type=int, default=4)
    ap.add_argument("--window-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="benchmarks/results/exactness_vs_bits.md")
    args = ap.parse_args()

    target = build_tiny_target(seed=args.seed)
    vocab = target.model.config.vocab_size
    prompts = synthetic_prompts(args.prompts, vocab)
    baselines = [target.greedy_generate(p, max_new_tokens=args.max_new) for p in prompts]

    rows = []
    for bits in (int(b) for b in args.bits.split(",")):
        spec = QuantSpec(
            bits=bits, sink_len=args.sink_len, sink_bits=8,
            window_len=args.window_len,
        )
        unverified = sum(
            greedy_on_cache(target, p, args.max_new, QuantizedKVCache(spec)) == base
            for p, base in zip(prompts, baselines)
        )
        engine = DartEngine(
            target, drafter=CompressedKVDrafter(target, spec), verifier="block"
        )
        accepts, dart_exact = [], 0
        for p, base in zip(prompts, baselines):
            out = engine.generate(
                p, DecodeConfig(max_new_tokens=args.max_new, draft_len=args.draft_len)
            )
            dart_exact += out.token_ids == base
            accepts.append(out.metadata["acceptance_rate"])
        rows.append({
            "bits": bits,
            "unverified_exact_rate": round(unverified / len(prompts), 4),
            "dart_exact_rate": round(dart_exact / len(prompts), 4),
            "dart_acceptance_rate": round(sum(accepts) / len(accepts), 4),
        })
        print(f"bits={bits}: unverified exact {rows[-1]['unverified_exact_rate']:.3f}  "
              f"dart exact {rows[-1]['dart_exact_rate']:.3f}  "
              f"dart acceptance {rows[-1]['dart_acceptance_rate']:.3f}")

    if any(r["dart_exact_rate"] != 1.0 for r in rows):
        print("EXACTNESS FAILURE in verified arm", file=sys.stderr)
        sys.exit(1)

    run = {
        "target": "tiny-seeded", "device": "cpu",
        "prompts": len(prompts), "max_new": args.max_new,
        "draft_len": args.draft_len,
        "sink_len": args.sink_len, "window_len": args.window_len,
        "note": "exact-match rate is hardware-independent (argmax comparisons)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exactness vs KV bit-width (README Figure 2)",
        "",
        f"Target: tiny seeded Llama · {len(prompts)} prompts × {args.max_new} tokens · "
        f"sink {args.sink_len} @8-bit · fp window {args.window_len} · CPU "
        "(exact-match rate is hardware-independent)",
        "",
        "| KV bits (aged tokens) | unverified quantized decode: exact rate | DART (same cache, drafts only): exact rate | DART acceptance |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bits']} | {r['unverified_exact_rate']:.3f} "
            f"| {r['dart_exact_rate']:.3f} | {r['dart_acceptance_rate']:.3f} |"
        )
    lines += [
        "",
        "Unverified decoding on a quantized cache loses byte-exactness as bits "
        "fall; the verified path stays at 1.000 at every bit-width because the "
        "quantized cache is only ever used to *draft*. Quantization error shows "
        "up in the acceptance column (a throughput cost) instead of the output.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"run": run, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
