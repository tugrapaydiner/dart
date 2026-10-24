from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "dart-corpus-v1"
SPLITS = ("train", "validation", "test")
WORKLOADS = ("instruction", "code", "structured", "repetition")
FORMAT_SOURCE = "https://huggingface.co/SupraLabs/Supra-50M-Instruct#inference"
TOKENIZATION = {"format_version": "supra-alpaca-v1", "add_special_tokens": True, "truncation": False}


@dataclass(frozen=True)
class Prompt:
    id: str
    source: str
    family: str
    workload: str
    split: str
    text: str
    input: str = ""
    format: str = "supra"
    max_new_tokens: int = 64

    def __post_init__(self):
        for name in ("id", "source", "family", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.split not in SPLITS or self.workload not in WORKLOADS or self.format not in ("supra", "raw"):
            raise ValueError("unknown split, workload, or prompt format")
        if not isinstance(self.input, str) or (self.format == "raw" and self.input):
            raise ValueError("input must be text and is only supported with the supra format")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be an integer from 1 to 1024")


def render_prompt(prompt: Prompt) -> str:
    if prompt.format == "raw":
        return prompt.text
    if prompt.input.strip():
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{prompt.text}\n\n"
            f"### Input:\n{prompt.input}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt.text}\n\n### Response:\n"
    )


def encode_prompt(prompt: Prompt, tokenizer, context_limit: int) -> list[int]:
    if type(context_limit) is not int or context_limit < 1:
        raise ValueError("context_limit must be a positive integer")
    ids = tokenizer.encode(render_prompt(prompt), add_special_tokens=True, truncation=False)
    if not ids:
        raise ValueError(f"{prompt.id}: tokenizer returned no prompt tokens")
    if len(ids) + prompt.max_new_tokens > context_limit:
        raise ValueError(f"{prompt.id}: {len(ids)} prompt + {prompt.max_new_tokens} generated tokens exceeds context {context_limit}")
    return ids


def validate_records(records: list[Prompt]) -> None:
    if not records:
        raise ValueError("corpus must not be empty")
    ids, texts = set(), set()
    owners = {"source": {}, "family": {}}
    for row in records:
        if row.id in ids:
            raise ValueError(f"duplicate prompt id: {row.id}")
        ids.add(row.id)
        text = " ".join(render_prompt(row).split())
        if text in texts:
            raise ValueError(f"duplicate prompt text: {row.id}")
        texts.add(text)
        for key, groups in owners.items():
            group = getattr(row, key)
            if groups.setdefault(group, row.split) != row.split:
                raise ValueError(f"{key} group crosses splits: {group}")


def corpus_hash(records: list[Prompt]) -> str:
    payload = json.dumps([asdict(row) for row in sorted(records, key=lambda row: row.id)], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_records(path: str | Path, max_records: int = 10_000, max_bytes: int = 16 * 1024 ** 2) -> list[Prompt]:
    path = Path(path)
    if type(max_records) is not int or type(max_bytes) is not int or max_records < 1 or max_bytes < 1 or path.stat().st_size > max_bytes:
        raise ValueError("corpus exceeds the configured input limit")
    records = []
    consumed = 0
    line_number = 0
    with path.open("rb") as source:
        while line := source.readline(max_bytes - consumed + 1):
            line_number += 1
            consumed += len(line)
            if consumed > max_bytes:
                raise ValueError("corpus exceeds the configured input limit")
            if not line.strip():
                continue
            if len(records) >= max_records:
                raise ValueError("corpus exceeds the configured input limit")
            try:
                records.append(Prompt(**json.loads(line)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path.name}:{line_number}: invalid prompt record") from error
    validate_records(records)
    return records
