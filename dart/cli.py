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
    parser = argparse.ArgumentParser(description="Evaluate local Supra-50M decoding.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="compare complete outputs with HF greedy")
    check.add_argument("--model", required=True, help="existing local Supra-50M model directory")
    check.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    check.add_argument("--corpus", help="versioned corpus directory; otherwise generate the fixed pilot")
    check.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    check.add_argument("--seed", type=int, default=17, help="seed for the built-in pilot")
    check.add_argument("--out", help="new report path; default: runs/check_<time>_<id>.json")
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


def _check(args) -> int:
    import torch

    from benchmarks.corpus_eval import evaluate_corpus
    from dart.data.teacher import target_fingerprint
    from dart.presets import supra_config
    from dart.runtime.factory import build_engine

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.out) if args.out else Path("runs") / f"check_{stamp}_{uuid.uuid4().hex[:8]}.json"
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = supra_config(args.model, args.device)

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Checked {done}/{total} prompts", file=sys.stderr, flush=True)

    with build_engine(config) as engine:
        records, manifest = _records(args, engine.target)
        fingerprint = target_fingerprint(engine.target)
        report = evaluate_corpus(engine, records, engine.default_decode_config, progress=progress)
    report.update({
        "command": "check", "preset": config, "target": fingerprint, "corpus": manifest,
        "environment": {
            "dart": __version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
            **{name: version(name) for name in ("torch", "transformers", "numpy", "PyYAML", "safetensors")},
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _check(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
