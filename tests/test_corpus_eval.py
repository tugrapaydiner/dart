from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.corpus_eval import evaluate_corpus, output_stats
from dart.data.corpus import Prompt
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig, GenerationOutput
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        return [10 + ord(char) % 100 for char in text]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


def records():
    return [
        Prompt("a", "source-a", "family-a", "instruction", "validation", "aaa", format="raw", max_new_tokens=3),
        Prompt("b", "source-b", "family-b", "code", "validation", "bbb", format="raw", max_new_tokens=7),
    ]


def test_output_statistics_distinguish_eos_budget_and_repetition():
    stats = output_stats([5] * 6 + [2], frozenset([2]), 8)
    assert stats["early_eos"] and stats["finished_eos"] and not stats["hit_token_limit"]
    assert stats["repeated_4gram_fraction"] == 0.5
    assert stats["max_identical_token_run"] == 6
    assert not output_stats([2], frozenset([2]), 1)["early_eos"]
    assert output_stats([], frozenset([2]), 1)["max_identical_token_run"] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("early_eos", [False, True])
def test_corpus_evaluation_honors_each_budget_and_groups_complete_matches(device, early_eos):
    target = TargetModel(build_tiny_target().model, tokenizer=Tokenizer(), device=device)
    rows = records()
    if early_eos:
        target.eos_token_ids = frozenset(target.greedy_generate(Tokenizer().encode("aaa", add_special_tokens=True, truncation=False), 1))
    config = DecodeConfig(max_new_tokens=99, draft_len=2)
    progress = []
    with DartEngine(target, NGramDrafter()) as engine:
        report = evaluate_corpus(engine, rows, config, progress=lambda done, total: progress.append((done, total)))
    assert config.max_new_tokens == 99
    assert report["summary"]["exact_match_rate"] == 1
    assert set(report["by_budget"]) == {"3", "7"}
    assert set(report["by_workload"]) == {"instruction", "code"}
    assert progress == [(1, 2), (2, 2)]
    for row in report["outputs"]:
        assert len(row["token_ids"]) <= row["max_new_tokens"]
    if early_eos:
        assert report["outputs"][0]["early_eos"]
        assert report["outputs"][0]["generated_tokens"] == 1


def test_short_or_mismatched_output_cannot_count_as_exact(monkeypatch):
    target = TargetModel(build_tiny_target().model, tokenizer=Tokenizer())
    engine = DartEngine(target)
    monkeypatch.setattr(engine, "generate", lambda *args: GenerationOutput([], "", {}))
    with pytest.raises(ValueError, match="complete token mismatch"):
        evaluate_corpus(engine, records())


def test_mixed_splits_and_duplicate_tokenized_inputs_fail_before_generation(monkeypatch):
    target = TargetModel(build_tiny_target().model, tokenizer=Tokenizer())
    engine = DartEngine(target)

    def fail(*args):
        pytest.fail("Invalid corpus reached generation")

    monkeypatch.setattr(engine, "generate", fail)
    rows = records()
    with pytest.raises(ValueError, match="one explicit split"):
        evaluate_corpus(engine, [rows[0], replace(rows[1], split="test")])
    monkeypatch.setattr(target.tokenizer, "encode", lambda *args, **kwargs: [7, 8, 9])
    with pytest.raises(ValueError, match="duplicate tokenized"):
        evaluate_corpus(engine, rows)
