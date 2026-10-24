from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.corpus_eval import evaluate_corpus
from dart.data.corpus import SPLITS
from dart.data.storage import read_split
from dart.diffusion.compact_training import load_compact_checkpoint
from dart.drafters.compact_diffusion import CompactDiffusionDrafter
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a versioned prompt split against plain HF greedy.")
    parser.add_argument("--model", required=True, help="existing local model directory")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--lane", choices=("target_only", "ngram", "compact"), default="ngram")
    parser.add_argument("--checkpoint", help="compact drafter checkpoint")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not Path(args.model).is_dir() or args.threads < 1 or args.draft_len < 0:
        parser.error("use a local model directory, threads >= 1, and draft-len >= 0")
    if (args.lane == "compact") != bool(args.checkpoint):
        parser.error("--checkpoint is required for compact drafting and only applies to that lane")
    if args.lane == "compact" and args.draft_len < 1:
        parser.error("compact drafting requires draft-len >= 1")
    torch.set_num_threads(args.threads)
    target = TargetModel.from_pretrained(args.model, device=args.device, dtype=getattr(torch, args.dtype), attn_implementation=args.attention, record_hidden=args.lane == "compact")
    records, manifest = read_split(args.corpus, args.split, target.tokenizer)
    drafter = NGramDrafter() if args.lane == "ngram" else None
    drafter_info = None
    if args.lane == "compact":
        model, checkpoint = load_compact_checkpoint(args.checkpoint, target)
        drafter = CompactDiffusionDrafter(model, canvas_len=args.draft_len)
        drafter_info = {
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "training_step": checkpoint["step"], "model_config": checkpoint["model_config"],
            "training_config": checkpoint["training_config"],
            "training_corpus_hash": checkpoint["metadata"]["corpus_hash"],
            "learned_parameters": sum(p.numel() for p in model.parameters()),
            "canvas_len": args.draft_len, "denoise_steps": drafter.denoise_steps,
            "reconcile": drafter.policy.mode,
        }
        del checkpoint

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Checked {done}/{total} prompts", flush=True)

    with DartEngine(target, drafter) as engine:
        report = evaluate_corpus(engine, records, DecodeConfig(draft_len=args.draft_len, canvas_len=args.draft_len), progress=progress)
    report["drafter"] = drafter_info
    report["corpus"] = {key: manifest[key] for key in ("version", "corpus_hash", "tokenizer_hash", "tokenization")}
    report["model"] = {
        "path": args.model, "device": target.device, "dtype": str(target.dtype),
        "attention": target.model.config._attn_implementation, "threads": torch.get_num_threads(),
        "lane": args.lane, "context_limit": target.model.config.max_position_embeddings,
    }
    output = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "runs/eval" / f"{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
