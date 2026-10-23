"""Compare fixed reconciliation policies on a local target and corpus split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.reconciliation import measure_reconciliation
from dart.data.corpus import SPLITS
from dart.data.storage import read_split
from dart.diffusion.compact_training import load_compact_checkpoint
from dart.models.target import TargetModel


def _select_records(records, per_workload, max_new):
    counts = Counter()
    selected = []
    for record in sorted(records, key=lambda row: (row.workload, row.id)):
        if per_workload and counts[record.workload] >= per_workload:
            continue
        counts[record.workload] += 1
        selected.append(replace(record, max_new_tokens=max_new) if max_new is not None else record)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="existing local model directory")
    parser.add_argument("--checkpoint", required=True, help="compact drafter checkpoint")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--draft-len", "--canvas-len", type=int, default=4)
    parser.add_argument("--denoise-steps", "--full-steps", type=int, default=2)
    parser.add_argument("--repair-steps", type=int, default=1)
    parser.add_argument("--keep-confidence-above", type=float, default=0.5)
    parser.add_argument("--max-keep-fraction", type=float, default=0.75)
    parser.add_argument("--per-workload", type=int, default=1, help="first N prompt IDs per workload; 0 selects all")
    parser.add_argument("--max-new", type=int, help="override each prompt's recorded token budget")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile", action="store_true", help="run separate synchronized phase diagnostics")
    parser.add_argument("--out")
    args = parser.parse_args()
    if not Path(args.model).is_dir() or not Path(args.corpus).is_dir() or not Path(args.checkpoint).is_file():
        parser.error("use existing local model, corpus, and checkpoint paths")
    if args.threads < 1 or args.draft_len < 1 or not 1 <= args.repair_steps < args.denoise_steps:
        parser.error("use threads >= 1, draft-len >= 1, and 1 <= repair-steps < denoise-steps")
    if args.per_workload < 0 or (args.max_new is not None and not 1 <= args.max_new <= 1024):
        parser.error("use per-workload >= 0 and max-new from 1 to 1024")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("use warmup >= 1 and repeats >= 2")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (args.keep_confidence_above, args.max_keep_fraction)):
        parser.error("keep-confidence-above and max-keep-fraction must be finite and in [0, 1]")
    output = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "runs/reconciliation" / f"{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    if output.exists():
        parser.error("output already exists; choose a new path")
    torch.set_num_threads(args.threads)
    target = TargetModel.from_pretrained(
        str(Path(args.model).resolve()), device=args.device, dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention, record_hidden=True,
    )
    records, manifest = read_split(args.corpus, args.split, target.tokenizer)
    selected = _select_records(records, args.per_workload, args.max_new)
    model, checkpoint = load_compact_checkpoint(args.checkpoint, target)
    checkpoint_info = {
        "sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "training_step": checkpoint["step"], "training_config": checkpoint["training_config"],
        "training_metadata": checkpoint["metadata"],
    }
    del checkpoint
    print(f"Checking and timing {len(selected)} {args.split} prompts, {args.repeats} repeats per method", flush=True)
    report = measure_reconciliation(
        target, model, selected, canvas_len=args.draft_len,
        full_steps=args.denoise_steps, repair_steps=args.repair_steps,
        keep_confidence_above=args.keep_confidence_above, max_keep_fraction=args.max_keep_fraction,
        warmup=args.warmup, repeats=args.repeats, seed=args.seed, profile=args.profile,
    )
    report["checkpoint"] = checkpoint_info
    report["corpus"] = {key: manifest[key] for key in ("version", "corpus_hash", "tokenizer_hash", "tokenization")}
    report["selection"] = {"rule": "first prompt IDs within each workload", "per_workload": args.per_workload,
                           "max_new_override": args.max_new, "available_prompts": len(records)}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(payload)
    summary = report["summary"]
    reference = summary["fastest_baseline"]
    print(f"Fastest target-only baseline: {reference}")
    for name, row in summary["timing"].items():
        speed = summary["speedups"][reference][name]
        low, high = speed["ci95_speedup"]
        print(f"{name}: {row['mean_ms']:.2f} ms, {speed['speedup']:.3f}x [{low:.3f}, {high:.3f}]")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
