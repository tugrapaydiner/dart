from __future__ import annotations

import hashlib
import json
import sys

import pytest

from dart.diffusion.compact_training import CompactTrainer
from scripts import run_corpus_eval
from tests.test_compact_training import MODEL, TRAIN, training_inputs


def test_eval_command_loads_checkpoint_and_records_training_provenance(tmp_path, monkeypatch):
    target, dataset = training_inputs(tmp_path)
    target.tokenizer.decode = lambda ids, **kwargs: " ".join(map(str, ids))
    trainer = CompactTrainer(target, dataset, MODEL, TRAIN)
    trainer.train_step()
    checkpoint = tmp_path / "drafter.pt"
    trainer.save(checkpoint)

    def load_target(*args, **kwargs):
        assert kwargs["record_hidden"] is True
        return target

    monkeypatch.setattr(run_corpus_eval.TargetModel, "from_pretrained", load_target)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_corpus_eval", "--model", str(tmp_path), "--corpus", str(tmp_path / "corpus"),
        "--lane", "compact", "--checkpoint", str(checkpoint), "--out", str(report),
    ])
    run_corpus_eval.main()
    result = json.loads(report.read_text())
    assert result["summary"]["exact_match_rate"] == 1
    assert result["model"]["lane"] == "compact"
    assert result["drafter"]["training_step"] == 1
    assert result["drafter"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["drafter"]["training_corpus_hash"] == dataset.manifest["corpus_hash"]
    assert result["drafter"]["reconcile"] == "discard" and result["drafter"]["denoise_steps"] == 1


@pytest.mark.parametrize("options", [["--lane", "compact"], ["--lane", "ngram", "--checkpoint", "unused.pt"], ["--lane", "compact", "--checkpoint", "unused.pt", "--draft-len", "0"]])
def test_invalid_checkpoint_options_fail_before_model_loading(tmp_path, monkeypatch, options):
    monkeypatch.setattr(run_corpus_eval.TargetModel, "from_pretrained", lambda *a, **k: pytest.fail("loaded before validation"))
    monkeypatch.setattr(sys, "argv", ["run_corpus_eval", "--model", str(tmp_path), "--corpus", str(tmp_path)] + options)
    with pytest.raises(SystemExit) as error:
        run_corpus_eval.main()
    assert error.value.code == 2
