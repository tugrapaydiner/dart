from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest

from benchmarks import verification_workloads as workloads
from dart.data.corpus import corpus_hash, read_records


def test_frozen_workloads_have_fresh_sources_and_balanced_final_cases():
    rows = read_records(workloads.PROMPTS)
    previous = read_records(workloads.FIXTURES / "v0.3-prompts.jsonl")
    metadata = json.loads(workloads.SOURCES.read_text(encoding="utf-8"))
    assert corpus_hash(rows) == metadata["corpus_hash"]
    assert corpus_hash(previous) == metadata["previous_corpus_hash"]
    workloads.validate_fresh(rows, previous)
    assert Counter(row.split for row in rows) == {"train": 8, "validation": 16, "test": 48}
    final = [row for row in rows if row.split == "test"]
    assert Counter(row.workload for row in final) == dict.fromkeys(workloads.WORKLOADS, 12)
    assert Counter(row.max_new_tokens for row in final) == {16: 16, 32: 16, 64: 16}
    timed = [row for row in final if row.id in metadata["protocol"]["timed_cases"]]
    assert len(timed) == len(set(metadata["protocol"]["timed_cases"])) == 12
    assert Counter(row.workload for row in timed) == dict.fromkeys(workloads.WORKLOADS, 3)
    for workload in workloads.WORKLOADS:
        assert {row.max_new_tokens for row in timed if row.workload == workload} == {16, 32, 64}


@pytest.mark.parametrize("field", ["source", "family"])
def test_prior_sources_and_families_cannot_be_relabelled(field):
    previous = read_records(workloads.FIXTURES / "v0.3-prompts.jsonl")[:1]
    new = read_records(workloads.PROMPTS)[0]
    with pytest.raises(ValueError, match="overlap the earlier corpus"):
        workloads.validate_fresh([replace(new, **{field: getattr(previous[0], field)})], previous)


def test_prior_text_cannot_be_reused_under_new_identifiers():
    previous = read_records(workloads.FIXTURES / "v0.3-prompts.jsonl")[:1]
    copy = replace(previous[0], id="new", source="new-source", family="new-family", split="test")
    with pytest.raises(ValueError, match="lexical overlap"):
        workloads.validate_fresh([copy], previous)


def test_lexical_duplicates_cannot_cross_new_splits():
    row = read_records(workloads.PROMPTS)[0]
    copy = replace(row, id="new", source="new-source", family="new-family", split="validation",
                   text=row.text + " Please.")
    with pytest.raises(ValueError, match="lexical overlap"):
        workloads.validate_fresh([row, copy], [])


def test_generated_families_are_disjoint_and_selection_fails_closed():
    rows = workloads.repetition_candidates()
    assert rows == workloads.repetition_candidates()
    workloads.validate_fresh(rows, [])
    with pytest.raises(ValueError, match="967 context"):
        workloads.select_workloads([], [], None, 966)
    with pytest.raises(ValueError, match="not enough independent"):
        workloads.select_workloads([], [], None, 1024)


def test_load_rejects_changed_inputs_and_tokenizer(monkeypatch, tmp_path):
    metadata = json.loads(workloads.SOURCES.read_text(encoding="utf-8"))
    monkeypatch.setattr(workloads, "tokenizer_hash", lambda tokenizer: "different")
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        workloads.load_frozen(None, 1024)
    metadata["corpus_hash"] = "changed"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(workloads, "SOURCES", path)
    with pytest.raises(ValueError, match="workloads changed"):
        workloads.load_frozen(None, 1024)


def test_load_keeps_the_verification_lookahead_inside_context(monkeypatch):
    metadata = json.loads(workloads.SOURCES.read_text(encoding="utf-8"))
    monkeypatch.setattr(workloads, "tokenizer_hash", lambda tokenizer: metadata["tokenizer_hash"])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [7] * 1008

    with pytest.raises(ValueError, match="exceeds context"):
        workloads.load_frozen(Tokenizer(), 1024)
