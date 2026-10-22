from __future__ import annotations

import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
from scripts import build_corpus

from dart.data.corpus import (
    SPLITS,
    WORKLOADS,
    corpus_hash,
    read_records,
    validate_records,
)
from dart.data.pilot import FAMILIES, build_pilot
from tests.test_corpus_storage import Tokenizer


def test_default_pilot_has_grouped_balanced_workloads_and_varied_budgets():
    rows = build_pilot()
    validate_records(rows)
    assert Counter(row.split for row in rows) == {"train": 192, "validation": 64, "test": 128}
    for split in SPLITS:
        subset = [row for row in rows if row.split == split]
        counts = Counter(row.workload for row in subset)
        assert set(counts) == set(WORKLOADS) and len(set(counts.values())) == 1
        assert {row.max_new_tokens for row in subset} == {32, 64, 128}
    families = [name for groups in FAMILIES.values() for names in groups.values() for name in names]
    assert len(families) == len(set(families)) == 28
    assert len({row.source for row in rows}) == len(rows)


def test_more_training_data_does_not_change_validation_or_test():
    small, large = build_pilot(), build_pilot(train_per_family=24)
    for split in ("validation", "test"):
        assert [row for row in small if row.split == split] == [row for row in large if row.split == split]
    assert corpus_hash(small) == corpus_hash(build_pilot())
    assert corpus_hash(small) != corpus_hash(build_pilot(seed=19))


def test_code_completion_prefixes_are_valid_python_when_completed():
    for row in build_pilot(train_per_family=3, validation_per_family=3, test_per_family=3):
        if row.workload == "code":
            assert row.format == "raw" and not row.input
            compile(row.text + "pass\n", "<prompt>", "exec")


def test_schema_specifications_do_not_repeat_field_names():
    for row in build_pilot():
        if row.family == "json_schema":
            fields = row.input.split(": ", 1)[1].split(", ")
            assert len(fields) == len(set(fields)), row.id


@pytest.mark.parametrize("kwargs", [{"train_per_family": 0}, {"seed": True}, {"test_per_family": 1.5}, {"train_per_family": 1000}])
def test_invalid_or_excessive_pilot_sizes_fail(kwargs):
    with pytest.raises(ValueError):
        build_pilot(**kwargs)


def test_corpus_command_loads_only_local_tokenizer_and_config(monkeypatch, tmp_path):
    model, output = tmp_path / "model", tmp_path / "corpus"
    model.mkdir()

    def load(path, **kwargs):
        assert path == model and kwargs == {"local_files_only": True}
        return Tokenizer()

    monkeypatch.setattr(build_corpus.AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(build_corpus.AutoConfig, "from_pretrained", lambda path, **kw: SimpleNamespace(max_position_embeddings=1024))
    monkeypatch.setattr(sys, "argv", [
        "build_corpus", "--model", str(model), "--out", str(output),
        "--train-per-family", "1", "--validation-per-family", "1", "--test-per-family", "1",
    ])
    build_corpus.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["splits"]["train"]["records"] == 12
    assert len(read_records(output / "test.jsonl")) == 8
