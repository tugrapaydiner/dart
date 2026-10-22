from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from dart.data.corpus import (
    Prompt,
    corpus_hash,
    encode_prompt,
    read_records,
    render_prompt,
    validate_records,
)


def row(**changes):
    return replace(Prompt("p1", "source-1", "family-1", "instruction", "train", "Say hello."), **changes)


def test_supra_format_matches_instruction_and_input_boundaries():
    text = render_prompt(row(input="Alice"))
    assert text.startswith("Below is an instruction that describes a task, paired with an input")
    assert text.endswith("### Instruction:\nSay hello.\n\n### Input:\nAlice\n\n### Response:\n")
    without_input = render_prompt(row())
    assert "### Input:" not in without_input
    assert without_input.endswith("### Instruction:\nSay hello.\n\n### Response:\n")
    assert render_prompt(row(text="def f(x):\n    ", format="raw")) == "def f(x):\n    "


def test_tokenization_keeps_special_tokens_and_rejects_overflow():
    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": True, "truncation": False}
            assert text == "abc"
            return [1, 2, 3, 4]

    prompt = row(text="abc", format="raw", max_new_tokens=6)
    assert encode_prompt(prompt, Tokenizer(), 10) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="4 prompt \\+ 6 generated"):
        encode_prompt(prompt, Tokenizer(), 9)


@pytest.mark.parametrize("field", ["source", "family"])
def test_shared_source_or_template_family_cannot_cross_splits(field):
    other = row(id="p2", source="source-2", family="family-2", split="validation", text="Say goodbye.")
    other = replace(other, **{field: getattr(row(), field)})
    with pytest.raises(ValueError, match="crosses splits"):
        validate_records([row(), other])


@pytest.mark.parametrize("duplicate", [row(text="different"), row(id="p2", text=" Say   hello. ")])
def test_duplicate_ids_and_whitespace_variants_are_rejected(duplicate):
    with pytest.raises(ValueError, match="duplicate prompt"):
        validate_records([row(), duplicate])


def test_corpus_hash_is_stable_and_covers_budgets_and_grouping():
    first, second = row(), row(id="p2", text="Another request.")
    assert corpus_hash([first, second]) == corpus_hash([second, first])
    for change in ({"max_new_tokens": 32}, {"family": "changed"}, {"source": "changed"}, {"format": "raw"}):
        assert corpus_hash([first]) != corpus_hash([replace(first, **change)])


def test_bounded_jsonl_round_trip_and_invalid_line(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(asdict(row())) + "\n\n", encoding="utf-8")
    assert read_records(path, max_records=1) == [row()]
    with pytest.raises(ValueError, match="input limit"):
        read_records(path, max_bytes=3)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(asdict(row(id="p2", text="Another request."))) + "\n")
    with pytest.raises(ValueError, match="input limit"):
        read_records(path, max_records=1)
    path.write_text('{"text": "missing metadata"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="train.jsonl:1: invalid"):
        read_records(path)


@pytest.mark.parametrize("changes", [
    {"id": ""}, {"split": "unknown"}, {"workload": "unknown"}, {"format": "unknown"},
    {"max_new_tokens": True}, {"max_new_tokens": 0}, {"max_new_tokens": 1025},
    {"input": None}, {"format": "raw", "input": "ignored"},
])
def test_invalid_prompt_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        row(**changes)
