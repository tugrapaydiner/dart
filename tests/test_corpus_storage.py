from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dart.data.corpus import SPLITS, Prompt
from dart.data.storage import read_split, tokenizer_hash, write_corpus


class Tokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 0
    special_tokens_map = {"bos_token": "<s>"}

    @property
    def backend_tokenizer(self):
        return self

    def to_str(self):
        return json.dumps({"vocab": ["a", "b", "c"]})

    def encode(self, text, **kwargs):
        return [0, 1, 2]


def records():
    return [Prompt(f"p-{split}", f"source-{split}", f"family-{split}", "instruction", split, f"Explain {split}.") for split in SPLITS]


def test_corpus_versions_are_deterministic_and_splits_load_explicitly(tmp_path):
    tokenizer = Tokenizer()
    a, b = tmp_path / "a", tmp_path / "b"
    manifest = write_corpus(a, records(), tokenizer, 1024, "pilot-1", 17)
    assert manifest == write_corpus(b, records()[::-1], tokenizer, 1024, "pilot-1", 17)
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()
    for split in SPLITS:
        rows, loaded = read_split(a, split, tokenizer)
        assert len(rows) == 1 and rows[0].split == split
        assert loaded == manifest
        assert manifest["splits"][split]["prompt_tokens"] == {"min": 3, "max": 3}
        assert (a / f"{split}.jsonl").read_bytes() == (b / f"{split}.jsonl").read_bytes()


def test_existing_corpus_cannot_be_overwritten(tmp_path):
    write_corpus(tmp_path, records(), Tokenizer(), 1024, "pilot-1", 17)
    original = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="new version"):
        write_corpus(tmp_path, records(), Tokenizer(), 1024, "changed", 99)
    assert (tmp_path / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("change", ["backend", "special_token"])
def test_tokenizer_changes_are_detected(tmp_path, change):
    original, changed = Tokenizer(), Tokenizer()
    write_corpus(tmp_path, records(), original, 1024, "pilot-1", 17)
    if change == "backend":
        changed.to_str = lambda: json.dumps({"vocab": ["c", "b", "a"]})
    else:
        changed.eos_token_id = 1
    assert tokenizer_hash(original) != tokenizer_hash(changed)
    with pytest.raises(ValueError, match="fingerprint"):
        read_split(tmp_path, "train", changed)


def test_tampered_split_and_manifest_groups_are_rejected(tmp_path):
    write_corpus(tmp_path, records(), Tokenizer(), 1024, "pilot-1", 17)
    with (tmp_path / "train.jsonl").open("a", encoding="utf-8") as target:
        target.write("\n")
    with pytest.raises(ValueError, match="content hash"):
        read_split(tmp_path, "train")
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["validation"]["families"] = manifest["splits"]["train"]["families"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cross splits"):
        read_split(tmp_path, "validation")


def test_invalid_or_overflowing_corpus_leaves_no_output(tmp_path):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="exceeds context"):
        write_corpus(destination, records(), Tokenizer(), 64, "pilot-1", 17)
    assert not destination.exists()
    rows = records()
    rows[1] = replace(rows[1], source=rows[0].source)
    with pytest.raises(ValueError, match="crosses splits"):
        write_corpus(destination, rows, Tokenizer(), 1024, "pilot-1", 17)
    assert not destination.exists()


def test_loading_one_split_does_not_read_another_split(tmp_path):
    write_corpus(tmp_path, records(), Tokenizer(), 1024, "pilot-1", 17)
    (tmp_path / "test.jsonl").unlink()
    rows, _ = read_split(tmp_path, "train", Tokenizer())
    assert rows[0].split == "train"


def test_tokenizer_fingerprint_ignores_previous_call_padding():
    original, padded = Tokenizer(), Tokenizer()
    padded.to_str = lambda: json.dumps({"vocab": ["a", "b", "c"], "padding": {"length": 32}, "truncation": {"max_length": 16}})
    assert tokenizer_hash(original) == tokenizer_hash(padded)


def test_manifest_size_is_checked_before_writing(tmp_path):
    rows = [replace(row, source=row.source + "x" * 400_000) for row in records()]
    destination = tmp_path / "large"
    with pytest.raises(ValueError, match="manifest exceeds"):
        write_corpus(destination, rows, Tokenizer(), 1024, "large", 17)
    assert not destination.exists()
