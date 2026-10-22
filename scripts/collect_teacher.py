from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data.teacher import write_traces
from dart.models.target import TargetModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bounded token-label and hidden-feature shards.")
    parser.add_argument("--model", required=True, help="existing local model directory")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-storage-mib", type=int, default=256)
    args = parser.parse_args()
    if not Path(args.model).is_dir() or args.threads < 1 or args.max_storage_mib < 1:
        parser.error("use a local model, positive threads, and a positive storage cap")
    torch.set_num_threads(args.threads)
    target = TargetModel.from_pretrained(
        args.model, device=args.device, dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention, record_hidden=True,
    )

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Collected {done}/{total} prompts", flush=True)

    manifest = write_traces(
        target, args.corpus, args.split, args.out,
        max_bytes=args.max_storage_mib * 1024 ** 2, progress=progress,
    )
    print(json.dumps({"records": manifest["records"], "shard_bytes": manifest["shard_bytes"], "out": args.out}))


if __name__ == "__main__":
    main()
