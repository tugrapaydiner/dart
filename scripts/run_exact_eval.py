"""Exact-match evaluation of a configured engine vs the trusted baseline.

    python scripts/run_exact_eval.py --config configs/dart_diffusion.yaml \
        --prompts 20 --max-new 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.exact_match import run_exact_match
from benchmarks.run_bench import synthetic_prompts
from dart.runtime.factory import engine_from_yaml
from dart.runtime.types import DecodeConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dart_exact.yaml")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--draft-len", type=int, default=4)
    args = ap.parse_args()

    engine = engine_from_yaml(args.config)
    vocab = engine.target.model.config.vocab_size
    prompts = synthetic_prompts(args.prompts, vocab)

    report = run_exact_match(
        engine.target,
        engine,
        prompts,
        args.max_new,
        DecodeConfig(max_new_tokens=args.max_new, draft_len=args.draft_len),
    )
    print(json.dumps(report.summary(), indent=2))
    if report.mismatches:
        print(json.dumps(report.mismatches, indent=2), file=sys.stderr)
        sys.exit(1)
    print("exactness contract holds: 100% greedy match")


if __name__ == "__main__":
    main()
