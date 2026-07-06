"""Baseline runner: trusted greedy decode + hashing + timing.

    python scripts/run_baseline.py --config configs/dart_exact.yaml --max-new 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_bench import synthetic_prompts
from dart.runtime.factory import engine_from_yaml
from dart.runtime.metrics import output_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dart_exact.yaml")
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    engine = engine_from_yaml(args.config)
    target = engine.target
    vocab = target.model.config.vocab_size
    prompts = synthetic_prompts(args.prompts, vocab)

    records = []
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        ids = target.greedy_generate(p, max_new_tokens=args.max_new)
        ms = (time.perf_counter() - t0) * 1000
        rec = {
            "prompt_idx": i,
            "prompt_len": len(p),
            "generated": len(ids),
            "hash": output_hash(ids),
            "wall_ms": round(ms, 1),
            "tok_per_s": round(len(ids) / (ms / 1000), 1) if ms > 0 else None,
        }
        records.append(rec)
        print(json.dumps(rec))

    out = Path("benchmarks/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {out / 'baseline_report.json'}")


if __name__ == "__main__":
    main()
