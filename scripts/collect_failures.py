from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data.failures import MAX_RECORDS, write_failures
from dart.data.teacher import TraceDataset
from dart.diffusion.compact_training import load_compact_checkpoint
from dart.models.target import TargetModel


def select_indices(traces, per_workload):
    counts = Counter()
    indices = []
    for index, row in sorted(enumerate(traces.manifest["shards"]), key=lambda item: (item[1]["workload"], item[1]["prompt_id"])):
        if not per_workload or counts[row["workload"]] < per_workload:
            indices.append(index)
            counts[row["workload"]] += 1
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verified draft errors into a bounded local manifest.")
    parser.add_argument("--model", required=True, help="existing local target model directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True, help="new JSON output file")
    parser.add_argument("--canvas-len", type=int, default=4)
    parser.add_argument("--per-prompt", type=int, default=8)
    parser.add_argument("--per-workload", type=int, default=0, help="first N prompt IDs per workload; 0 selects all")
    parser.add_argument("--max-storage-mib", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not Path(args.model).is_dir() or not Path(args.checkpoint).is_file() or not Path(args.traces).is_dir():
        parser.error("use existing local model, checkpoint, and trace paths")
    if Path(args.out).exists():
        parser.error("output already exists; choose a new file")
    if args.threads < 1 or not 1 <= args.canvas_len <= 64 or not 1 <= args.per_prompt <= 64 or args.per_workload < 0:
        parser.error("use positive threads, canvas/per-prompt counts from 1 to 64, and per-workload >= 0")
    traces = TraceDataset(args.traces)
    indices = select_indices(traces, args.per_workload)
    if traces.manifest["split"] != "train" or len(indices) * args.per_prompt > MAX_RECORDS:
        parser.error("use training traces and at most 4096 retained windows")
    torch.set_num_threads(args.threads)
    target = TargetModel.from_pretrained(
        str(Path(args.model).resolve()), device=args.device, dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention, record_hidden=True,
    )
    model, checkpoint = load_compact_checkpoint(args.checkpoint, target)
    del checkpoint

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Verified {done}/{total} prompts", flush=True)

    manifest = write_failures(
        target, model, traces, args.out, indices=indices, canvas_len=args.canvas_len,
        per_prompt=args.per_prompt, seed=args.seed, max_bytes=args.max_storage_mib * 1024 ** 2, progress=progress,
    )
    print(json.dumps({"prompts": len(manifest["prompts"]), "windows": len(manifest["records"]),
                      "bytes": Path(args.out).stat().st_size, "out": args.out}))


if __name__ == "__main__":
    main()
