from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dart.data.corpus import (
    SCHEMA,
    SPLITS,
    TOKENIZATION,
    Prompt,
    corpus_hash,
    encode_prompt,
    read_records,
    validate_records,
)


def tokenizer_hash(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("versioned corpora require a fast tokenizer")
    settings = {key: getattr(tokenizer, key, None) for key in (
        "bos_token_id", "eos_token_id", "pad_token_id", "add_bos_token", "add_eos_token",
    )}
    definition = json.loads(backend.to_str())
    for key in ("padding", "truncation"):
        definition.pop(key, None)  # Per-call options do not change the tokenizer.
    payload = {"backend": definition, "special_tokens": tokenizer.special_tokens_map, **settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_corpus(
    directory: str | Path, records: list[Prompt], tokenizer, context_limit: int,
    version: str, seed: int,
) -> dict:
    if not isinstance(version, str) or not version.strip() or type(seed) is not int:
        raise ValueError("provide a version name and integer seed")
    if len(records) > 10_000 or set(row.split for row in records) != set(SPLITS):
        raise ValueError("corpus needs all three splits and at most 10000 records")
    validate_records(records)
    manifest = {
        "schema": SCHEMA, "version": version, "seed": seed,
        "corpus_hash": corpus_hash(records), "tokenizer_hash": tokenizer_hash(tokenizer),
        "tokenization": dict(TOKENIZATION), "context_limit": context_limit, "splits": {},
    }
    payloads = {}
    for split in SPLITS:
        rows = sorted((row for row in records if row.split == split), key=lambda row: row.id)
        lengths = [len(encode_prompt(row, tokenizer, context_limit)) for row in rows]
        payload = "".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        if len(payload) > 16 * 1024 ** 2:
            raise ValueError("split exceeds the 16 MiB input limit")
        payloads[split] = payload
        manifest["splits"][split] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "corpus_hash": corpus_hash(rows),
            "records": len(rows), "sources": sorted({row.source for row in rows}),
            "families": sorted({row.family for row in rows}),
            "workloads": dict(sorted(Counter(row.workload for row in rows).items())),
            "prompt_tokens": {"min": min(lengths), "max": max(lengths)},
            "generation_budgets": sorted({row.max_new_tokens for row in rows}),
        }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(manifest_bytes) > 1024 ** 2:
        raise ValueError("manifest exceeds the 1 MiB input limit")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("corpus directory is not empty; choose a new version")
    directory.mkdir(parents=True, exist_ok=True)
    for split, payload in payloads.items():
        with (directory / f"{split}.jsonl").open("xb") as output:
            output.write(payload)
    with (directory / "manifest.json").open("xb") as output:  # Incomplete corpora have no manifest.
        output.write(manifest_bytes)
    return manifest


def _read_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("corpus file exceeds the input limit")
    return data


def read_split(directory: str | Path, split: str, tokenizer=None) -> tuple[list[Prompt], dict]:
    if split not in SPLITS:
        raise ValueError("choose train, validation, or test explicitly")
    directory = Path(directory)
    manifest = json.loads(_read_bytes(directory / "manifest.json", 1024 ** 2))
    if manifest.get("schema") != SCHEMA or manifest.get("tokenization") != TOKENIZATION:
        raise ValueError("unsupported corpus schema or tokenization")
    if tokenizer is not None and tokenizer_hash(tokenizer) != manifest.get("tokenizer_hash"):
        raise ValueError("tokenizer does not match the corpus fingerprint")
    metadata = manifest.get("splits", {})
    if set(metadata) != set(SPLITS):
        raise ValueError("manifest must describe all three splits")
    for key in ("sources", "families"):
        seen = set()
        for info in metadata.values():
            groups = set(info[key])
            if groups & seen:
                raise ValueError(f"manifest {key} cross splits")
            seen.update(groups)
    path = directory / f"{split}.jsonl"
    expected = metadata[split]
    if hashlib.sha256(_read_bytes(path, 16 * 1024 ** 2)).hexdigest() != expected["sha256"]:
        raise ValueError(f"{split}: content hash changed")
    records = read_records(path)
    if len(records) != expected["records"] or corpus_hash(records) != expected["corpus_hash"]:
        raise ValueError(f"{split}: record metadata changed")
    if any(row.split != split for row in records):
        raise ValueError("requested split does not match its records")
    if sorted({row.source for row in records}) != expected["sources"] or sorted({row.family for row in records}) != expected["families"]:
        raise ValueError("source or template groups do not match the manifest")
    return records, manifest
