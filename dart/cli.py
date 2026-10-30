from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from contextlib import ExitStack
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
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="existing local Supra-50M model directory")
    source.add_argument("--synthetic", action="store_true", help="check tiny random CPU models; no download or default report")
    check.add_argument("--state", action="store_true", help="check interleaved request caches; requires --synthetic")
    check.add_argument("--execution", choices=("eager", "static"),
                       help="synthetic CPU backend; defaults to eager, or the saved backend with --replay")
    check.add_argument("--reduce", type=int, nargs="?", const=32, metavar="N",
                       help="shrink a state failure with 2-32 checks (default: 32); requires --synthetic --state")
    for command in (check, bench, demo):
        if command is not check:
            command.add_argument("--model", required=True, help="existing local Supra-50M model directory")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--native-rms-norm", action="store_true", help="use PyTorch RMSNorm for float32 Llama layers")
    for command in (check, bench):
        command.add_argument("--norm-control", action="store_true", help="compare with original RMSNorm; requires --native-rms-norm")
        command.add_argument("--corpus", help="versioned corpus directory; otherwise generate the fixed pilot")
        command.add_argument("--split", choices=("train", "validation", "test"), default="validation")
        seed_options = command.add_mutually_exclusive_group() if command is check else command
        seed_options.add_argument("--seed", type=int, default=17, help="pilot seed; also controls benchmark ordering")
        if command is check:
            seed_options.add_argument("--replay", help="replay a saved local report; requires --synthetic --state")
        command.add_argument("--out", help="new report path; synthetic checks write only when set, model runs default to runs/")
    bench.add_argument("--per-workload", type=int, default=1, help="first sorted IDs per workload, 1-32 (default: 1)")
    bench.add_argument("--warmup", type=int, default=2, help="warm-up passes, 1-10 (default: 2)")
    bench.add_argument("--repeats", type=int, default=10, help="measured repeats, 2-100 (default: 10)")
    bench.add_argument("--skip-profile", action="store_true", help="save latency comparisons without separate phase diagnostics")
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

    if args.norm_control and not args.native_rms_norm:
        raise ValueError("norm-control requires native-rms-norm")
    if args.command == "bench" and args.norm_control and not args.skip_profile:
        raise ValueError("norm-control benchmarks require skip-profile; use separate decode traces")
    if args.command == "bench":
        if not 1 <= args.warmup <= 10 or not 2 <= args.repeats <= 100:
            raise ValueError("use warmup from 1 to 10 and repeats from 2 to 100")
        if not 1 <= args.per_workload <= 32 or args.seed < 0:
            raise ValueError("use per-workload from 1 to 32 and a nonnegative seed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.out) if args.out else Path("runs") / f"{args.command}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    timing_output = output.with_suffix(".timing.json")
    if args.command == "bench" and timing_output.exists():
        raise FileExistsError(f"timing output already exists: {timing_output}")
    config = supra_config(args.model, args.device, native_rms_norm=args.native_rms_norm)

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Checked {done}/{total} prompts", file=sys.stderr, flush=True)

    with ExitStack() as stack:
        engine = stack.enter_context(build_engine(config))
        control = stack.enter_context(build_engine(supra_config(args.model, args.device))) if args.norm_control else None
        records, manifest = _records(args, engine.target)
        fingerprint = target_fingerprint(engine.target)
        metadata = {
            "command": args.command, "preset": config, "target": fingerprint, "corpus": manifest,
            **({"norm_control_target": target_fingerprint(control.target)} if control is not None else {}),
            "environment": {
                "dart": __version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
                "platform": platform.platform(), "processor": platform.processor(), "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(engine.target.device) if args.device == "cuda" else platform.processor(),
                **{name: version(name) for name in ("torch", "transformers", "numpy", "PyYAML", "safetensors")},
            },
        }
        if args.command == "check":
            report = evaluate_corpus(engine, records, engine.default_decode_config, progress=progress,
                                     reference_target=control.target if control is not None else None)
        else:
            from benchmarks.corpus_bench import benchmark_corpus, select_records

            selected = select_records(records, args.per_workload)
            selection = {
                "rule": "first sorted IDs per workload", "per_workload": args.per_workload,
                "available_prompts": len(records), "selected_prompts": len(selected),
            }

            def save_timing(partial):
                timing_output.parent.mkdir(parents=True, exist_ok=True)
                with timing_output.open("x", encoding="utf-8") as destination:
                    json.dump({**partial, **metadata, "selection": selection}, destination, indent=2)
                    destination.write("\n")
                phase = "profiling was skipped" if args.skip_profile else "profiling is still pending"
                print(f"Saved timings to {timing_output}; {phase}", file=sys.stderr, flush=True)

            report = benchmark_corpus(
                engine, selected, warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                on_timing=save_timing,
                profile=not args.skip_profile,
                norm_control=control,
            )
            report["selection"] = selection
    report.update(metadata)
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
        if "norm_ablation" in report["summary"]:
            ablation = report["summary"]["norm_ablation"]
            low, high = ablation["ci95_speedup"]
            print(f"Native vs original RMSNorm: {ablation['speedup']:.3f}x, 95% CI [{low:.3f}, {high:.3f}]")
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

    with build_engine(supra_config(args.model, args.device, native_rms_norm=args.native_rms_norm)) as engine:
        try:
            serve(engine, port=args.port)
        except KeyboardInterrupt:
            return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check" and args.execution is not None and not args.synthetic:
            raise ValueError("--execution requires --synthetic")
        if args.command == "check" and args.state and not args.synthetic:
            raise ValueError("--state requires --synthetic")
        if args.command == "check" and (args.replay is not None or args.reduce is not None) and not (args.synthetic and args.state):
            raise ValueError("--replay and --reduce require --synthetic --state")
        if args.command == "check" and args.synthetic:
            from dart.cli_check import run_synthetic

            return run_synthetic(args)
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
