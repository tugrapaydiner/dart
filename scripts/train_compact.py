from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data.teacher import TraceDataset
from dart.diffusion.compact_training import CompactTrainer, CompactTrainingConfig
from dart.models.target import TargetModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact drafter from bounded teacher shards.")
    parser.add_argument("--model", required=True, help="existing local target model directory")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True, help="empty output directory for this run")
    parser.add_argument("--resume", help="checkpoint to resume with matching options")
    parser.add_argument("--steps", type=int, required=True, help="total optimizer steps, including resumed steps")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--canvas-len", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.0, help="fraction of loss from teacher distributions (0 to 1)")
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--drafter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not Path(args.model).is_dir() or min(args.steps, args.save_every, args.threads) < 1:
        parser.error("use a local model and positive step, save, and thread counts")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; resume into a new directory")
    torch.set_num_threads(args.threads)
    target = TargetModel.from_pretrained(args.model, device=args.device, dtype=getattr(torch, args.dtype), attn_implementation=args.attention)
    trainer = CompactTrainer(target, TraceDataset(args.traces), config=CompactTrainingConfig(
        seed=args.seed, batch_size=args.batch_size, canvas_len=args.canvas_len,
        learning_rate=args.learning_rate, dtype=args.drafter_dtype,
        distill_weight=args.distill_weight, distill_temperature=args.distill_temperature,
    ))
    if args.resume:
        trainer.resume(args.resume)
    if trainer.step >= args.steps:
        parser.error("steps must be greater than the checkpoint step")
    output.mkdir(parents=True, exist_ok=True)
    if trainer.step == 0:
        trainer.save(output / "initial.pt")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (output / "losses.jsonl").open("x", encoding="utf-8") as log:
        while trainer.step < args.steps:
            loss = trainer.train_step()
            row = {"step": trainer.step, "loss": loss, "elapsed_s": time.perf_counter() - started}
            log.write(json.dumps(row) + "\n")
            if trainer.step % 25 == 0 or trainer.step == args.steps:
                log.flush()
                print(json.dumps(row), flush=True)
            if trainer.step % args.save_every == 0 or trainer.step == args.steps:
                trainer.save(output / "latest.pt")
    report = {
        "steps": trainer.step, "last_loss": trainer.last_loss,
        "learned_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "metadata": trainer.metadata,
        "model_config": asdict(trainer.model.cfg), "training_config": asdict(trainer.config),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
