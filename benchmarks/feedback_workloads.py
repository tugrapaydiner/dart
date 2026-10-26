"""Fresh inputs for confirming entropy-feedback overhead and workload coverage."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from benchmarks.verification_workloads import validate_fresh
from benchmarks.workloads import (
    FIXTURES,
    PROMPT_LIMIT,
    _digest,
    _real_candidates,
    near_duplicate,
    read_source,
)
from dart.data.corpus import WORKLOADS, Prompt, corpus_hash, encode_prompt, read_records
from dart.data.storage import tokenizer_hash

VERSION = "dart-feedback-workloads-v1"
SEED = 311
PROMPTS = FIXTURES / "feedback-prompts.jsonl"
SOURCES = FIXTURES / "feedback-sources.json"
PREVIOUS = ("v0.3-prompts.jsonl", "verification-prompts.jsonl")


def prior_workloads():
    return [row for name in PREVIOUS for row in read_records(FIXTURES / name)]


def repetition_candidates() -> list[Prompt]:
    rows = []
    for family in ("csv_inventory", "markdown_inventory"):
        for index in range(32):
            rng = random.Random(_digest(f"{SEED}:{family}:{index}"))
            first, second = rng.sample(("cobalt", "linen", "marble", "ochre", "pearl", "silver", "umber", "wheat"), 2)
            number = rng.randrange(100, 999)
            pattern = {
                "csv_inventory": f"{first},{number},shelf-A\n{second},{number + 1},shelf-B\n",
                "markdown_inventory": f"| {first} | {number} | in stock |\n| {second} | {number + 1} | ordered |\n",
            }[family]
            text = "Continue the inventory:\n" + pattern * (2, 8, 24)[index % 3]
            source = f"generated:feedback:{family}:{index:02d}"
            rows.append(Prompt(source, source, "feedback:" + family, "repetition", "test", text, format="raw"))
    return rows


def select_workloads(mbpp, dolly, tokenizer, context_limit: int) -> list[Prompt]:
    if context_limit < PROMPT_LIMIT + 64 + 7:
        raise ValueError("feedback workloads need at least 967 context positions")
    previous = prior_workloads()
    pools = _real_candidates(mbpp, dolly)
    pools["repetition"] = repetition_candidates()
    pools = {key: sorted(rows, key=lambda row: (_digest(f"{SEED}:{row.source}"), row.id)) for key, rows in pools.items()}
    used_sources = {row.source for row in previous}
    used_families = {row.family for row in previous}
    selected = []
    for workload in WORKLOADS:
        for index in range(12):
            kinds = {"code": ("mbpp",), "instruction": ("summarization", "open_qa"),
                     "structured": ("information_extraction", "classification"), "repetition": ("repetition",)}[workload]
            minimum, maximum = (384, PROMPT_LIMIT) if index % 6 == 0 and workload != "code" else (1, 383)
            for candidate in pools[kinds[index % len(kinds)]]:
                if candidate.split != "test" or candidate.source in used_sources or candidate.family in used_families:
                    continue
                row = replace(candidate, id=f"v03-feedback-test-{workload}-{index:02d}", max_new_tokens=(16, 32, 64)[index % 3])
                try:
                    length = len(encode_prompt(row, tokenizer, context_limit - 7))
                except ValueError as error:
                    if "exceeds context" not in str(error):
                        raise
                    continue
                real = not row.source.startswith("generated:")
                comparisons = previous + (selected if real else [])
                if not minimum <= length <= maximum or any(near_duplicate(row, other) for other in comparisons):
                    continue
                selected.append(row)
                used_sources.add(row.source)
                if real:
                    used_families.add(row.family)
                break
            else:
                raise ValueError(f"not enough independent {workload} candidates: {index}")
    validate_fresh(selected, previous)
    return sorted(selected, key=lambda row: row.id)


def load_frozen(tokenizer, context_limit: int) -> list[Prompt]:
    metadata = json.loads(SOURCES.read_text(encoding="utf-8"))
    rows = read_records(PROMPTS)
    previous_hashes = {name: corpus_hash(read_records(FIXTURES / name)) for name in PREVIOUS}
    if corpus_hash(rows) != metadata["corpus_hash"] or previous_hashes != metadata["previous_corpus_hashes"]:
        raise ValueError("frozen feedback or earlier workloads changed")
    if tokenizer_hash(tokenizer) != metadata["tokenizer_hash"]:
        raise ValueError("tokenizer fingerprint changed")
    expected_ids = {f"v03-feedback-test-{workload}-{index:02d}" for workload in WORKLOADS for index in range(12)}
    if {row.id for row in rows} != expected_ids or any(row.split != "test" for row in rows):
        raise ValueError("feedback case coverage changed")
    for workload in WORKLOADS:
        if Counter(row.max_new_tokens for row in rows if row.workload == workload) != {16: 4, 32: 4, 64: 4}:
            raise ValueError("feedback output budgets changed")
    validate_fresh(rows, prior_workloads())
    for row in rows:
        encode_prompt(row, tokenizer, context_limit - 7)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True, help="new JSONL file for the final inputs")
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
        rebuilt = select_workloads(read_source(args.mbpp, "mbpp"), read_source(args.dolly, "dolly"), tokenizer, config.max_position_embeddings)
        if [asdict(row) for row in rebuilt] != [asdict(row) for row in rows]:
            raise ValueError("rebuilt feedback selection differs from the frozen workloads")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"version": VERSION, "corpus_hash": corpus_hash(rows), "cases": len(rows)}))


if __name__ == "__main__":
    main()
