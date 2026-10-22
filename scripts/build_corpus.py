from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data.pilot import VERSION, build_pilot
from dart.data.storage import write_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped procedural decoding workloads using a local tokenizer.")
    parser.add_argument("--model", required=True, help="existing local model directory")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/corpus/pilot-v1"))
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-per-family", type=int, default=16)
    parser.add_argument("--validation-per-family", type=int, default=8)
    parser.add_argument("--test-per-family", type=int, default=16)
    args = parser.parse_args()
    model = Path(args.model)
    if not model.is_dir():
        parser.error("--model must be an existing local directory")
    try:
        records = build_pilot(args.seed, args.train_per_family, args.validation_per_family, args.test_per_family)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        manifest = write_corpus(args.out, records, tokenizer, config.max_position_embeddings, args.version, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"version": manifest["version"], "corpus_hash": manifest["corpus_hash"], "splits": {
        split: {key: value for key, value in info.items() if key in ("records", "workloads", "prompt_tokens", "generation_budgets")}
        for split, info in manifest["splits"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
