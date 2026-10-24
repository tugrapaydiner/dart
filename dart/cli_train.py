from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path


def add_arguments(parser) -> None:
    parser.add_argument("--model", required=True, help="existing local target model directory")
    parser.add_argument("--traces", required=True, help="bounded training trace directory")
    parser.add_argument("--failures", help="bounded proposal-failure manifest")
    parser.add_argument("--failure-masking", choices=("none", "observed", "random_matched"), default="none")
    parser.add_argument("--out", required=True, help="empty output directory; resume into a new directory")
    parser.add_argument("--resume", help="checkpoint to resume with matching options, traces and target settings")
    parser.add_argument("--steps", type=int, required=True, help="total optimizer steps, including resumed steps")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--canvas-len", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--full-mask-fraction", type=float, default=0.5)
    parser.add_argument("--distill-weight", type=float, default=0.0, help="teacher distribution loss fraction (default: %(default)s)")
    parser.add_argument("--distill-temperature", type=float, default=1.0, help="distillation temperature (default: %(default)s)")
    parser.add_argument("--embedding-init-std", type=float, default=1.0, help="position/type embedding std (default: %(default)s)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="same device used to collect the traces")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--drafter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--threads", type=int, default=1)


def add_command(commands) -> None:
    parser = commands.add_parser("train", help="train or resume a compact drafter from local traces",
                                  description="Train a compact drafter using the selected scaled-distillation pilot defaults.")
    add_arguments(parser)
    parser.set_defaults(distill_weight=0.5, distill_temperature=2.0, embedding_init_std=0.02)


def run(args) -> int:
    from dart.cli_data import empty_output
    from dart.presets import local_model_path

    model = local_model_path(args.model)
    output = empty_output(args.out)
    if min(args.steps, args.save_every, args.threads) < 1:
        raise ValueError("use positive step, save, and thread counts")
    if (args.failure_masking != "none") != bool(args.failures):
        raise ValueError("--failures and an explicit --failure-masking mode are required together")
    for name in ("failures", "resume"):
        if getattr(args, name) and not Path(getattr(args, name)).is_file():
            raise ValueError(f"{name} must be an existing local file")

    import torch

    from dart.data.failures import FailureDataset
    from dart.data.teacher import TraceDataset
    from dart.diffusion.compact_training import CompactTrainer, CompactTrainingConfig
    from dart.drafters.compact_diffusion import CompactDrafterConfig
    from dart.models.target import TargetModel

    model_config = CompactDrafterConfig(embedding_init_std=args.embedding_init_std)
    config = CompactTrainingConfig(
        seed=args.seed, batch_size=args.batch_size, canvas_len=args.canvas_len,
        learning_rate=args.learning_rate, dtype=args.drafter_dtype,
        full_mask_fraction=args.full_mask_fraction, failure_masking=args.failure_masking,
        distill_weight=args.distill_weight, distill_temperature=args.distill_temperature,
    )
    if config.canvas_len > model_config.max_canvas:
        raise ValueError("training canvas exceeds model capacity")
    traces = TraceDataset(args.traces)
    if traces.manifest["split"] != "train":
        raise ValueError("optimizer updates require training traces")
    failures = FailureDataset(args.failures, traces) if args.failures else None
    torch.set_num_threads(args.threads)
    target = TargetModel.from_pretrained(
        str(model), device=args.device, dtype=getattr(torch, args.dtype), attn_implementation=args.attention,
        local_files_only=True, trust_remote_code=False,
    )
    trainer = CompactTrainer(target, traces, model_config=model_config, config=config, failures=failures)
    if args.resume:
        trainer.resume(args.resume)
    if trainer.step >= args.steps:
        raise ValueError("steps must be greater than the checkpoint step")
    empty_output(output)
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
        "memory_scope": "training updates and checkpoint writes; excludes initialization",
        "metadata": trainer.metadata,
        "model_config": asdict(trainer.model.cfg), "training_config": asdict(trainer.config),
    }
    with (output / "report.json").open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(json.dumps(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a compact drafter from bounded teacher shards.")
    add_arguments(parser)  # Keep the standalone script's original defaults.
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
