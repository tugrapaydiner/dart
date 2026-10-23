from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dart import __version__


def _parser() -> argparse.ArgumentParser:
    from dart.cli_data import add_commands
    from dart.cli_train import add_command

    parser = argparse.ArgumentParser(description="Train, check and benchmark local Supra-50M drafting.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="compare complete outputs with HF greedy")
    bench = commands.add_parser("bench", help="repeat latency comparisons and separate phase profiling")
    demo = commands.add_parser("demo", help="serve the comparison page on localhost")
    for command in (check, bench, demo):
        command.add_argument("--model", required=True, help="existing local Supra-50M model directory")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for command in (check, bench):
        command.add_argument("--corpus", help="versioned corpus directory; otherwise generate the fixed pilot")
        command.add_argument("--split", choices=("train", "validation", "test"), default="validation")
        command.add_argument("--seed", type=int, default=17, help="pilot seed; also controls benchmark ordering")
        command.add_argument("--out", help="new report path; default: runs/<command>_<time>_<id>.json")
    bench.add_argument("--per-workload", type=int, default=1, help="first sorted IDs per workload, 1-32 (default: 1)")
    bench.add_argument("--warmup", type=int, default=2, help="warm-up passes, 1-10 (default: 2)")
    bench.add_argument("--repeats", type=int, default=10, help="measured repeats, 2-100 (default: 10)")
    demo.add_argument("--port", type=int, default=8400, help="localhost port; 0 picks a free port (default: 8400)")
    add_commands(commands)
    add_command(commands)
    return parser


def _records(args, target):
    from dart.data.corpus import TOKENIZATION, corpus_hash
    from dart.data.pilot import VERSION, build_pilot
    from dart.data.storage import read_split, tokenizer_hash

    if args.corpus:
        records, manifest = read_split(args.corpus, args.split, target.tokenizer)
    else:
        all_records = build_pilot(args.seed)
        records = [record for record in all_records if record.split == args.split]
        manifest = {
            "version": VERSION, "seed": args.seed, "corpus_hash": corpus_hash(all_records),
            "tokenizer_hash": tokenizer_hash(target.tokenizer), "tokenization": TOKENIZATION,
        }
    return sorted(records, key=lambda record: record.id), manifest


def _run(args) -> int:
    import torch

    from benchmarks.corpus_eval import evaluate_corpus
    from dart.data.teacher import target_fingerprint
    from dart.presets import supra_config
    from dart.runtime.factory import build_engine

    if args.command == "bench":
        if not 1 <= args.warmup <= 10 or not 2 <= args.repeats <= 100:
            raise ValueError("use warmup from 1 to 10 and repeats from 2 to 100")
        if not 1 <= args.per_workload <= 32 or args.seed < 0:
            raise ValueError("use per-workload from 1 to 32 and a nonnegative seed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.out) if args.out else Path("runs") / f"{args.command}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = supra_config(args.model, args.device)

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Checked {done}/{total} prompts", file=sys.stderr, flush=True)

    with build_engine(config) as engine:
        records, manifest = _records(args, engine.target)
        fingerprint = target_fingerprint(engine.target)
        if args.command == "check":
            report = evaluate_corpus(engine, records, engine.default_decode_config, progress=progress)
        else:
            from benchmarks.corpus_bench import benchmark_corpus, select_records

            selected = select_records(records, args.per_workload)
            report = benchmark_corpus(
                engine, selected, warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            report["selection"] = {
                "rule": "first sorted IDs per workload", "per_workload": args.per_workload,
                "available_prompts": len(records), "selected_prompts": len(selected),
            }
    report.update({
        "command": args.command, "preset": config, "target": fingerprint, "corpus": manifest,
        "environment": {
            "dart": __version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
            "platform": platform.platform(), "processor": platform.processor(), "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(engine.target.device) if args.device == "cuda" else platform.processor(),
            **{name: version(name) for name in ("torch", "transformers", "numpy", "PyYAML", "safetensors")},
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    if args.command == "check":
        print(json.dumps(report["summary"], indent=2))
    else:
        for name, row in report["summary"]["timing"].items():
            low, high = row["ci95_ms"]
            print(f"{name}: {row['mean_ms']:.2f} ms, 95% CI [{low:.2f}, {high:.2f}]")
        comparison = report["summary"]["vs_fastest"]["dart"]
        low, high = comparison["ci95_speedup"]
        print(f"DART vs {report['summary']['fastest_baseline']}: {comparison['speedup']:.3f}x, 95% CI [{low:.3f}, {high:.3f}]")
        for row in report["correctness"]["outputs"]:
            print(f"{row['id']}: matching hash {row['output_hash']}, {row['generated_tokens']} tokens, "
                  f"draft acceptance {row['draft_tokens_accepted']}/{row['draft_tokens_proposed']}")
        print("All measured outputs matched HF. Intervals describe repeats of these fixed prompts.")
    print(f"Wrote {output}")
    return 0


def _demo(args) -> int:
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be from 0 to 65535")
    from dart.presets import supra_config
    from dart.runtime.factory import build_engine
    from dart.runtime.server import serve

    with build_engine(supra_config(args.model, args.device)) as engine:
        try:
            serve(engine, port=args.port)
        except KeyboardInterrupt:
            return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from dart.cli_train import run

            return run(args)
        if args.command in ("corpus", "collect"):
            from dart.cli_data import run

            return run(args)
        if args.command == "demo":
            return _demo(args)
        return _run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
