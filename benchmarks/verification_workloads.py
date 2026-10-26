from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from benchmarks.workloads import (
    FIXTURES,
    PROMPT_LIMIT,
    _digest,
    _real_candidates,
    near_duplicate,
    read_source,
)
from dart.data.corpus import (
    SPLITS,
    WORKLOADS,
    Prompt,
    corpus_hash,
    encode_prompt,
    read_records,
    validate_records,
)
from dart.data.storage import tokenizer_hash, write_corpus

VERSION = "dart-verification-workloads-v1"
SEED = 307
PROMPTS = FIXTURES / "verification-prompts.jsonl"
SOURCES = FIXTURES / "verification-sources.json"
COUNTS = {"train": 2, "validation": 4, "test": 12}


def repetition_candidates() -> list[Prompt]:
    groups = {"train": ("alternating_dialogue",), "validation": ("shell_transcript",),
              "test": ("ini_stanzas", "html_badges")}
    rows = []
    for split, families in groups.items():
        for family in families:
            for index in range(32):
                rng = random.Random(_digest(f"{SEED}:{family}:{index}"))
                first, second = rng.sample(("amber", "birch", "cedar", "delta", "elm", "fjord", "grove", "harbor"), 2)
                number = rng.randrange(100, 999)
                pattern = {
                    "alternating_dialogue": f"Operator: route {first} to bay {number}.\nDispatcher: {second} confirmed.\n",
                    "shell_transcript": f"$ echo {first}-{number}\n{first}-{number}\n$ echo {second}\n{second}\n",
                    "ini_stanzas": f"[{first}]\nport={number}\nmode={second}\n\n",
                    "html_badges": f'<span class="{first}">{number}</span><small>{second}</small>\n',
                }[family]
                text = "Continue the transcript:\n" + pattern * (2, 8, 24)[index % 3]
                source = f"generated:verification:{family}:{index:02d}"
                rows.append(Prompt(source, source, "verification:" + family, "repetition", split,
                                   text, format="raw"))
    return rows


def validate_fresh(rows: list[Prompt], previous: list[Prompt]) -> None:
    validate_records(rows)
    sources = {row.source for row in previous}
    families = {row.family for row in previous}
    if any(row.source in sources or row.family in families for row in rows):
        raise ValueError("verification sources or families overlap the earlier corpus")
    for index, row in enumerate(rows):
        comparisons = previous + [other for other in rows[:index] if other.split != row.split]
        if any(near_duplicate(row, other) for other in comparisons):
            raise ValueError(f"{row.id}: lexical overlap with an earlier corpus or another split")


def select_workloads(mbpp, dolly, tokenizer, context_limit: int) -> list[Prompt]:
    if context_limit < PROMPT_LIMIT + 64 + 7:
        raise ValueError("verification workloads need at least 967 context positions")
    previous = read_records(FIXTURES / "v0.3-prompts.jsonl")
    pools = _real_candidates(mbpp, dolly)
    pools["repetition"] = repetition_candidates()
    pools = {key: sorted(rows, key=lambda row: (_digest(f"{SEED}:{row.source}"), row.id))
             for key, rows in pools.items()}
    selected: list[Prompt] = []
    used_sources = {row.source for row in previous}
    used_families = {row.family for row in previous}
    for split in SPLITS:
        for workload in WORKLOADS:
            for index in range(COUNTS[split]):
                kind = {"instruction": ("summarization", "open_qa"), "code": ("mbpp",),
                        "structured": ("information_extraction", "classification"),
                        "repetition": ("repetition",)}[workload]
                kind = kind[index % len(kind)]
                minimum, maximum = (384, PROMPT_LIMIT) if index % 6 == 0 and workload != "code" else (1, 383)
                for candidate in pools[kind]:
                    if candidate.split != split or candidate.source in used_sources or candidate.family in used_families:
                        continue
                    row = replace(candidate, id=f"v03-execution-{split}-{workload}-{index:02d}",
                                  max_new_tokens=(16, 32, 64)[index % 3])
                    try:
                        length = len(encode_prompt(row, tokenizer, context_limit - 7))
                    except ValueError as error:
                        if "exceeds context" not in str(error):
                            raise
                        continue
                    if not minimum <= length <= maximum:
                        continue
                    comparisons = previous + [other for other in selected if other.split != split]
                    if any(near_duplicate(row, other) for other in comparisons):
                        continue
                    selected.append(row)
                    used_sources.add(row.source)
                    if not row.source.startswith("generated:"):
                        used_families.add(row.family)
                    break
                else:
                    raise ValueError(f"not enough independent {workload} candidates in {split}: {index}")
    validate_fresh(selected, previous)
    return sorted(selected, key=lambda row: row.id)


def load_frozen(tokenizer, context_limit: int) -> list[Prompt]:
    metadata = json.loads(SOURCES.read_text(encoding="utf-8"))
    rows = read_records(PROMPTS)
    previous = read_records(FIXTURES / "v0.3-prompts.jsonl")
    if corpus_hash(rows) != metadata["corpus_hash"] or corpus_hash(previous) != metadata["previous_corpus_hash"]:
        raise ValueError("frozen verification or earlier workloads changed")
    if tokenizer_hash(tokenizer) != metadata["tokenizer_hash"]:
        raise ValueError("tokenizer fingerprint changed")
    validate_fresh(rows, previous)
    for row in rows:
        encode_prompt(row, tokenizer, context_limit - 7)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare the frozen verification workloads with local model files.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mbpp", type=Path)
    parser.add_argument("--dolly", type=Path)
    args = parser.parse_args()
    if bool(args.mbpp) != bool(args.dolly):
        parser.error("provide both source files or neither")
    from transformers import AutoConfig, AutoTokenizer

    from dart.presets import local_model_path

    model = local_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rows = load_frozen(tokenizer, config.max_position_embeddings)
    if args.mbpp:
        rebuilt = select_workloads(read_source(args.mbpp, "mbpp"), read_source(args.dolly, "dolly"),
                                   tokenizer, config.max_position_embeddings)
        if [asdict(row) for row in rebuilt] != [asdict(row) for row in rows]:
            raise ValueError("rebuilt verification selection differs from the frozen workloads")
    manifest = write_corpus(args.out, rows, tokenizer, config.max_position_embeddings, VERSION, SEED)
    print(json.dumps({"version": VERSION, "corpus_hash": manifest["corpus_hash"],
                      "splits": {split: info["records"] for split, info in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
