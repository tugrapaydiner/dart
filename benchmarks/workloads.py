from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path

from dart.data.corpus import (
    SPLITS,
    WORKLOADS,
    Prompt,
    corpus_hash,
    encode_prompt,
    read_records,
    validate_records,
)
from dart.data.pilot import build_pilot
from dart.data.storage import tokenizer_hash, write_corpus

VERSION = "dart-workloads-v0.3"
SEED = 303
PROMPT_LIMIT = 896
FIXTURES = Path(__file__).with_name("fixtures")
SOURCE_HASHES = {
    "mbpp": "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
    "dolly": "e442d96b320010510c67a6cc0a86ea988a2a6f71c7570d1afbdf4fa631896e58",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _shingles(row: Prompt) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", _normalize(row.text + " " + row.input))
    return {tuple(words[i:i + 4]) for i in range(len(words) - 3)}


def near_duplicate(first: Prompt, second: Prompt) -> bool:
    if _normalize(first.text + " " + first.input) == _normalize(second.text + " " + second.input):
        return True
    a, b = _shingles(first), _shingles(second)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.5


def _real_candidates(mbpp: list[dict], dolly: list[dict]) -> dict[str, list[Prompt]]:
    pools = {key: [] for key in ("mbpp", "open_qa", "summarization", "classification", "information_extraction")}
    for row in mbpp:
        task = row["task_id"]
        split = "test" if 11 <= task <= 510 else "validation" if 511 <= task <= 600 else "train"
        source = f"mbpp:{task}"
        pools["mbpp"].append(Prompt(
            source, source, "mbpp:" + _digest(_normalize(row["prompt"])), "code", split,
            row["prompt"], input="Tests:\n" + "\n".join(row["test_list"]), max_new_tokens=64,
        ))
    for index, row in enumerate(dolly):
        category = row["category"]
        if category not in pools:
            continue
        if category == "classification" and not re.search(r"\b(classify|categorize|identify)\b", row["instruction"], re.IGNORECASE):
            continue
        family = "dolly:" + _digest(_normalize(row["context"] or row["instruction"]))
        bucket = int(_digest(f"{SEED}:{family}")[:8], 16) % 5
        split = "train" if bucket == 0 else "validation" if bucket == 1 else "test"
        source = f"dolly:{index}"
        pools[category].append(Prompt(
            source, source, family, "instruction" if category in ("open_qa", "summarization") else "structured",
            split, row["instruction"], input=row["context"], max_new_tokens=64,
        ))
    return pools


def select_workloads(mbpp: list[dict], dolly: list[dict], tokenizer, context_limit: int) -> list[Prompt]:
    if context_limit < PROMPT_LIMIT + 64 + 7:
        raise ValueError("workload protocol needs at least 967 context positions")
    pools = _real_candidates(mbpp, dolly)
    procedural = build_pilot(SEED)
    selected: list[Prompt] = []
    used_sources: set[str] = set()
    used_real_families: set[str] = set()

    def take(pool, split, workload, *, long=False):
        ranked = sorted(pool, key=lambda row: (_digest(f"{SEED}:{row.source}"), row.id))
        for row in ranked:
            if row.split != split or row.workload != workload or row.source in used_sources:
                continue
            real = row.source.startswith(("mbpp:", "dolly:"))
            if real and row.family in used_real_families:
                continue
            if any(other.split != split and (row.family == other.family or near_duplicate(row, other)) for other in selected):
                continue
            row = replace(row, max_new_tokens=64)
            try:
                length = len(encode_prompt(row, tokenizer, context_limit))
            except ValueError as error:
                if "exceeds context" not in str(error):
                    raise
                continue
            if length > PROMPT_LIMIT or (long and length < 384):
                continue
            used_sources.add(row.source)
            if real:
                used_real_families.add(row.family)
            return row
        raise ValueError(f"not enough independent {workload} candidates in {split}, long={long}")

    for split in SPLITS:
        for workload in WORKLOADS:
            count = 2 if split == "train" else 6
            kinds = {
                "instruction": ["summarization", "open_qa", "summarization", "open_qa"],
                "code": ["mbpp"] * 4,
                "structured": ["information_extraction", "classification", "information_extraction", "classification"],
                "repetition": [],
            }[workload]
            if split == "train":
                kinds = kinds[:1]
            for index in range(count):
                kind = kinds[index] if index < len(kinds) else "procedural"
                long = index == 0 and workload in ("instruction", "structured")
                row = take(procedural if kind == "procedural" else pools[kind], split, workload, long=long)
                selected.append(replace(row, id=f"v03-{split}-{workload}-{index:02d}", max_new_tokens=(16, 32, 64)[index % 3]))
    validate_records(selected)
    return sorted(selected, key=lambda row: row.id)


def read_source(path: Path, name: str):
    with path.open("rb") as source:
        raw = source.read(16 * 1024 ** 2 + 1)
    if len(raw) > 16 * 1024 ** 2:
        raise ValueError("source exceeds 16 MiB")
    raw = raw.replace(b"\r\n", b"\n").rstrip(b"\n")
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError(f"{name}: source hash differs from the pinned dataset")
    return json.loads(raw) if name == "mbpp" else [json.loads(line) for line in raw.splitlines()]


def load_frozen(tokenizer, context_limit: int) -> list[Prompt]:
    metadata = json.loads((FIXTURES / "v0.3-sources.json").read_text(encoding="utf-8"))
    rows = read_records(FIXTURES / "v0.3-prompts.jsonl")
    if corpus_hash(rows) != metadata["corpus_hash"] or tokenizer_hash(tokenizer) != metadata["tokenizer_hash"]:
        raise ValueError("frozen workloads or tokenizer fingerprint changed")
    for row in rows:
        encode_prompt(row, tokenizer, context_limit - 7)  # Allow drafts up to eight tokens.
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare the frozen v0.3 workloads with a local tokenizer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mbpp", type=Path, help="optional pinned MBPP source, used with --dolly to rebuild selection")
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
            raise ValueError("rebuilt selection differs from the frozen workloads")
    manifest = write_corpus(args.out, rows, tokenizer, config.max_position_embeddings, VERSION, SEED)
    print(json.dumps({"version": VERSION, "corpus_hash": manifest["corpus_hash"],
                      "splits": {split: info["records"] for split, info in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
