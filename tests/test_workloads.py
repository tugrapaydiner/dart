from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest

from benchmarks import workloads
from dart.data.corpus import corpus_hash, read_records
from tests.test_corpus_eval import records


def test_frozen_inputs_match_the_protocol_and_keep_splits_separate():
    rows = read_records(workloads.FIXTURES / "v0.3-prompts.jsonl")
    metadata = json.loads((workloads.FIXTURES / "v0.3-sources.json").read_text())
    assert corpus_hash(rows) == metadata["corpus_hash"]
    assert Counter(row.split for row in rows) == {"train": 8, "validation": 24, "test": 24}
    real = [row for row in rows if not row.source.startswith("generated:")]
    assert len({row.family for row in real}) == len(real)
    for split in ("validation", "test"):
        group = [row for row in rows if row.split == split]
        assert Counter(row.workload for row in group) == dict.fromkeys(workloads.WORKLOADS, 6)
        assert Counter(row.max_new_tokens for row in group) == {16: 8, 32: 8, 64: 8}
        assert Counter(row.source.split(":")[0] for row in group) == {"mbpp": 4, "dolly": 8, "generated": 12}
    for index, row in enumerate(rows):
        for other in rows[index + 1:]:
            if row.split != other.split:
                assert not workloads.near_duplicate(row, other), (row.id, other.id)


def test_duplicate_detection_ignores_case_and_spacing_and_catches_shared_passages():
    first, other = records()
    assert workloads.near_duplicate(first, replace(first, text="  AAA  "))
    passage = " ".join(f"word{i}" for i in range(30))
    first = replace(first, text="Summarize this passage", input=passage, format="supra")
    other = replace(other, text="Name the main subject", input=passage, format="supra")
    assert workloads.near_duplicate(first, other)
    assert not workloads.near_duplicate(first, replace(other, input="unrelated text about music"))


def test_changed_source_is_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="source hash differs"):
        workloads.read_source(path, "mbpp")


def test_classification_pool_excludes_mislabelled_opinion_questions():
    dolly = [{"category": "classification", "instruction": instruction, "context": ""}
             for instruction in ("Who is the best player?", "Classify these animals: cat, salmon.")]
    pool = workloads._real_candidates([], dolly)["classification"]
    assert [row.source for row in pool] == ["dolly:1"]


def test_selection_fails_when_candidates_or_context_are_insufficient():
    with pytest.raises(ValueError, match="967 context"):
        workloads.select_workloads([], [], None, 966)
    with pytest.raises(ValueError, match="not enough independent"):
        workloads.select_workloads([], [], None, 1024)


def test_frozen_inputs_reject_a_different_tokenizer(monkeypatch):
    monkeypatch.setattr(workloads, "tokenizer_hash", lambda tokenizer: "different")
    with pytest.raises(ValueError, match="fingerprint changed"):
        workloads.load_frozen(None, 1024)


def test_frozen_inputs_reserve_lookup_context(monkeypatch):
    metadata = json.loads((workloads.FIXTURES / "v0.3-sources.json").read_text())
    monkeypatch.setattr(workloads, "tokenizer_hash", lambda tokenizer: metadata["tokenizer_hash"])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [7] * 1008  # The first 16-token budget fits only without lookup lookahead.

    with pytest.raises(ValueError, match="exceeds context"):
        workloads.load_frozen(Tokenizer(), 1024)
