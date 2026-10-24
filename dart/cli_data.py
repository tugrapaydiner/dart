from __future__ import annotations

import json
import sys
from pathlib import Path


def add_commands(commands) -> None:
    from dart.data.pilot import VERSION

    corpus = commands.add_parser("corpus", help="build versioned procedural splits with a local tokenizer")
    collect = commands.add_parser("collect", help="collect bounded train/validation traces in float32/eager",
                                  description="Collect bounded local teacher traces using float32 and eager attention.")
    for command in (corpus, collect):
        command.add_argument("--model", required=True, help="existing local Supra-50M model directory")
    corpus.add_argument("--out", default="runs/corpus/pilot-v1", help="empty output directory (default: runs/corpus/pilot-v1)")
    corpus.add_argument("--corpus-version", default=VERSION)
    corpus.add_argument("--seed", type=int, default=17)
    corpus.add_argument("--train-per-family", type=int, default=16)
    corpus.add_argument("--validation-per-family", type=int, default=8)
    corpus.add_argument("--test-per-family", type=int, default=16)
    collect.add_argument("--corpus", required=True, help="versioned corpus directory")
    collect.add_argument("--split", choices=("train", "validation"), default="train")
    collect.add_argument("--out", required=True, help="empty trace directory")
    collect.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    collect.add_argument("--max-storage-mib", type=int, default=256, help="total trace storage cap (default: 256 MiB)")


def empty_output(path: str | Path) -> Path:
    directory = Path(path)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("output must be an empty or new directory")
    return directory


def _corpus(args, model: Path, output: Path) -> dict:
    from dart.data.pilot import build_pilot

    if not args.corpus_version.strip():
        raise ValueError("provide a nonempty corpus version")
    records = build_pilot(args.seed, args.train_per_family, args.validation_per_family, args.test_per_family)

    from transformers import AutoConfig, AutoTokenizer

    from dart.data.storage import write_corpus

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    manifest = write_corpus(output, records, tokenizer, config.max_position_embeddings, args.corpus_version, args.seed)
    return {
        "version": manifest["version"], "corpus_hash": manifest["corpus_hash"],
        "tokenizer_hash": manifest["tokenizer_hash"],
        "splits": {split: {key: info[key] for key in ("records", "workloads", "prompt_tokens", "generation_budgets")}
                   for split, info in manifest["splits"].items()},
    }


def _collect(args, model: Path, output: Path) -> dict:
    if args.max_storage_mib < 1:
        raise ValueError("max-storage-mib must be positive")
    if not (Path(args.corpus) / "manifest.json").is_file():
        raise ValueError("provide a versioned corpus directory with manifest.json")

    import torch

    from dart.data.teacher import write_traces
    from dart.models.target import TargetModel

    torch.set_num_threads(1)
    target = TargetModel.from_pretrained(
        str(model), device=args.device, dtype=torch.float32, attn_implementation="eager",
        record_hidden=True, local_files_only=True, trust_remote_code=False,
    )

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"Collected {done}/{total} prompts", file=sys.stderr, flush=True)

    manifest = write_traces(target, args.corpus, args.split, output,
                            max_bytes=args.max_storage_mib * 1024 ** 2, progress=progress)
    return {"records": manifest["records"], "shard_bytes": manifest["shard_bytes"], "target": manifest["target"]}


def run(args) -> int:
    from dart.presets import local_model_path

    model = local_model_path(args.model)
    output = empty_output(args.out)
    report = _corpus(args, model, output) if args.command == "corpus" else _collect(args, model, output)
    print(json.dumps({"command": args.command, "out": str(output), **report}, indent=2))
    return 0
