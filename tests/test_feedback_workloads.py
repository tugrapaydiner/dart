import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from benchmarks import feedback_workloads as workloads
from dart.data.corpus import corpus_hash, read_records


def test_confirmation_cases_are_fresh_balanced_and_all_timed():
    rows = read_records(workloads.PROMPTS)
    metadata = json.loads(workloads.SOURCES.read_text(encoding="utf-8"))
    assert len(rows) == 48
    assert {row.split for row in rows} == {"test"}
    assert corpus_hash(rows) == metadata["corpus_hash"]
    assert metadata["previous_corpus_hashes"] == {
        name: corpus_hash(read_records(workloads.FIXTURES / name)) for name in workloads.PREVIOUS
    }
    workloads.validate_fresh(rows, workloads.prior_workloads())
    assert Counter(row.workload for row in rows) == dict.fromkeys(workloads.WORKLOADS, 12)
    assert metadata["protocol"]["timed_cases"] == [row.id for row in rows]
    for workload in workloads.WORKLOADS:
        assert Counter(row.max_new_tokens for row in rows if row.workload == workload) == {16: 4, 32: 4, 64: 4}


def test_inventory_templates_are_deterministic_and_different_from_prior_inputs():
    rows = workloads.repetition_candidates()
    assert rows == workloads.repetition_candidates()
    workloads.validate_fresh(rows, workloads.prior_workloads())
    assert len({row.family for row in rows}) == 2


@pytest.mark.parametrize("context", [966, 1024])
def test_selection_fails_instead_of_reusing_exposed_inputs(context):
    with pytest.raises(ValueError, match="967 context|not enough independent"):
        workloads.select_workloads([], [], None, context)


@pytest.mark.parametrize("changed", ["corpus", "v0.3-prompts.jsonl", "verification-prompts.jsonl", "tokenizer"])
def test_frozen_fingerprints_are_checked(changed, monkeypatch, tmp_path):
    metadata = json.loads(workloads.SOURCES.read_text(encoding="utf-8"))
    actual_tokenizer = metadata["tokenizer_hash"]
    if changed == "corpus":
        metadata["corpus_hash"] = "changed"
    elif changed == "tokenizer":
        metadata["tokenizer_hash"] = "changed"
    else:
        metadata["previous_corpus_hashes"][changed] = "changed"
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(workloads, "SOURCES", path)
    monkeypatch.setattr(workloads, "tokenizer_hash", lambda _: actual_tokenizer)
    with pytest.raises(ValueError, match="changed"):
        workloads.load_frozen(None, 1024)


def test_context_guard_includes_verification_lookahead(monkeypatch):
    metadata = json.loads(workloads.SOURCES.read_text(encoding="utf-8"))
    monkeypatch.setattr(workloads, "tokenizer_hash", lambda _: metadata["tokenizer_hash"])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [7] * 1002

    with pytest.raises(ValueError, match="exceeds context"):
        workloads.load_frozen(Tokenizer(), 1024)


def test_cli_exports_final_inputs_without_training_splits(monkeypatch, tmp_path, capsys):
    from transformers import AutoConfig, AutoTokenizer

    from dart import presets

    rows = read_records(workloads.PROMPTS)
    monkeypatch.setattr(presets, "local_model_path", lambda _: tmp_path)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(max_position_embeddings=1024))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(workloads, "load_frozen", lambda *args: rows)
    output = tmp_path / "final.jsonl"
    monkeypatch.setattr(sys, "argv", ["feedback_workloads", "--model", str(tmp_path), "--out", str(output)])
    workloads.main()
    assert read_records(output) == rows
    assert json.loads(capsys.readouterr().out)["corpus_hash"] == corpus_hash(rows)
    with pytest.raises(FileExistsError):
        workloads.main()
