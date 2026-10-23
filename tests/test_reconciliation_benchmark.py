from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from statistics import mean

import pytest
from scripts import run_reconciliation_ablation as command

from benchmarks import reconciliation
from benchmarks.headroom import BASELINES
from dart.diffusion.compact_training import CompactTrainer
from dart.runtime.types import GenerationOutput
from tests.test_compact_training import MODEL, TRAIN, training_inputs
from tests.test_corpus_eval import Tokenizer, records
from tests.test_reconciliation_arms import compact_model
from tests.test_sampling_devices import DEVICES

ARMS = {"fresh_one_pass", "fresh_multi_pass", "reuse_equal_budget", "reuse_reduced_budget"}
METHODS = set(BASELINES) | {"ngram"} | ARMS


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("early_eos", [False, True])
def test_measurement_keeps_each_budget_and_all_cases(device, early_eos, monkeypatch):
    target, model = compact_model(device)
    target.tokenizer = Tokenizer()
    rows = records()
    if early_eos:
        target.eos_token_ids = frozenset(target.greedy_generate(target.tokenizer.encode("aaa", add_special_tokens=True, truncation=False), 1))
    original = reconciliation.measure_methods
    hidden_calls = {name: [] for name in METHODS}

    def measure(methods, *args, **kwargs):
        checked = {}
        for name, method in methods.items():
            def run(prompt, name=name, method=method):
                def before(module, args, kwargs):
                    hidden_calls[name].append(bool(kwargs.get("output_hidden_states", module.config.output_hidden_states)))
                hook = target.model.register_forward_pre_hook(before, with_kwargs=True)
                try:
                    return method(prompt)
                finally:
                    hook.remove()
            checked[name] = run
        return original(checked, *args, **kwargs)

    monkeypatch.setattr(reconciliation, "measure_methods", measure)
    report = reconciliation.measure_reconciliation(target, model, rows, warmup=1, repeats=2, profile=True)
    assert target.record_hidden is True
    assert set(report["summary"]["timing"]) == METHODS
    assert set(report["policies"]) == ARMS
    assert set(report["by_workload"]) == {"instruction", "code"}
    assert [row["id"] for row in report["by_prompt"]] == [row.id for row in rows]
    assert len(report["timing"]["samples"]) == len(report["profile"]["samples"]) == 32
    assert report["record_hidden_by_method"] == {name: name in ARMS for name in METHODS}
    for name in METHODS:
        assert hidden_calls[name] and all(value == (name in ARMS) for value in hidden_calls[name])
    for sample in report["timing"]["samples"] + report["profile"]["samples"]:
        prompt = report["prompts"][sample["prompt_index"]]
        assert sample["output_hash"] == prompt["output_hash"]
        assert sample["tokens"] == prompt["generated_tokens"] <= rows[sample["prompt_index"]].max_new_tokens
        if not early_eos:
            assert sample["tokens"] == rows[sample["prompt_index"]].max_new_tokens
    if early_eos:
        assert report["prompts"][0]["early_eos"] and report["prompts"][0]["generated_tokens"] == 1
    for workload, group in report["by_workload"].items():
        index = next(i for i, row in enumerate(rows) if row.workload == workload)
        assert group["prompts"] == 1
        assert group["fastest_baseline"] == min(BASELINES, key=lambda name: group["timing"][name]["mean_ms"])
        for name in METHODS:
            samples = [row for row in report["timing"]["samples"] if row["method"] == name and row["prompt_index"] == index]
            assert group["timing"][name]["mean_ms"] == pytest.approx(mean(row["latency_ms"] for row in samples))
            assert name in group["speedups"][group["fastest_baseline"]]
            phases = group["profile"][name]
            assert phases["request_ms"] == pytest.approx(phases["setup_ms"] + phases["prefill_ms"] + phases["decode_ms"])
        for name, samples in report["acceptance_samples"].items():
            counts = group["acceptance"][name]
            assert counts["draft_tokens_proposed"] == samples[index]["draft_tokens_proposed"]
            assert counts["diffusion_forwards"] == samples[index]["diffusion_forwards"]
    for name, counts in report["summary"]["acceptance"].items():
        assert counts["diffusion_forwards"] == sum(group["acceptance"][name]["diffusion_forwards"] for group in report["by_workload"].values())
        if counts["diffusion_forwards"]:
            assert counts["accepted_tokens_per_diffusion_forward"] == counts["draft_tokens_accepted"] / counts["diffusion_forwards"]
    json.dumps(report, allow_nan=False)


def test_mismatched_tokens_fail_before_timing(monkeypatch):
    target, model = compact_model()
    target.tokenizer = Tokenizer()
    monkeypatch.setattr(reconciliation.DartEngine, "generate", lambda *a: GenerationOutput([], "", {}))
    monkeypatch.setattr(reconciliation, "measure_methods", lambda *a, **k: pytest.fail("timed an invalid method"))
    with pytest.raises(ValueError, match="complete token mismatch"):
        reconciliation.measure_reconciliation(target, model, records(), warmup=1, repeats=2)


def test_invalid_corpus_fails_before_model_forwards(monkeypatch):
    target, model = compact_model()
    target.tokenizer = Tokenizer()
    monkeypatch.setattr(target.model, "forward", lambda *a, **k: pytest.fail("invalid input reached the model"))
    rows = records()
    with pytest.raises(ValueError, match="one corpus split"):
        reconciliation.measure_reconciliation(target, model, [rows[0], replace(rows[1], split="test")])
    with pytest.raises(ValueError, match="context"):
        reconciliation.measure_reconciliation(target, model, [replace(rows[0], max_new_tokens=1024)])
    monkeypatch.setattr(target.tokenizer, "encode", lambda *a, **k: [7, 8, 9])
    with pytest.raises(ValueError, match="duplicate tokenized"):
        reconciliation.measure_reconciliation(target, model, rows)


def test_selection_is_balanced_stable_and_does_not_change_source_budgets():
    rows = records()
    extra = [replace(row, id=row.id + "2", text=row.text + "x", max_new_tokens=11) for row in rows]
    selected = command._select_records(extra + rows[::-1], 1, None)
    assert [row.id for row in selected] == ["b", "a"]
    assert [row.max_new_tokens for row in selected] == [7, 3]
    overridden = command._select_records(extra + rows, 0, 5)
    assert len(overridden) == 4 and all(row.max_new_tokens == 5 for row in overridden)
    assert [row.max_new_tokens for row in rows] == [3, 7]


def test_command_loads_checkpoint_and_writes_complete_report(tmp_path, monkeypatch):
    target, dataset = training_inputs(tmp_path)
    target.tokenizer.decode = lambda ids, **kwargs: " ".join(map(str, ids))
    trainer = CompactTrainer(target, dataset, MODEL, TRAIN)
    trainer.train_step()
    checkpoint = tmp_path / "drafter.pt"
    trainer.save(checkpoint)
    manifest = (tmp_path / "corpus/manifest.json").read_bytes()

    def load_target(*args, **kwargs):
        assert kwargs["record_hidden"] is True
        return target

    monkeypatch.setattr(command.TargetModel, "from_pretrained", load_target)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_reconciliation_ablation", "--model", str(tmp_path), "--corpus", str(tmp_path / "corpus"),
        "--checkpoint", str(checkpoint), "--out", str(output), "--warmup", "1", "--repeats", "2", "--max-new", "3",
    ])
    command.main()
    report = json.loads(output.read_text())
    assert report["checkpoint"]["training_step"] == 1
    assert report["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["checkpoint"]["training_metadata"]["corpus_hash"] == dataset.manifest["corpus_hash"]
    assert len(report["timing"]["samples"]) == 16 and report["profile"] is None
    assert report["prompts"][0]["max_new_tokens"] == report["selection"]["max_new_override"] == 3
    assert (tmp_path / "corpus/manifest.json").read_bytes() == manifest
    original = output.read_bytes()
    monkeypatch.setattr(command.TargetModel, "from_pretrained", lambda *a, **k: pytest.fail("loaded before rejecting overwrite"))
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2 and output.read_bytes() == original


@pytest.mark.parametrize("options", [
    ["--repeats", "1"], ["--warmup", "0"], ["--per-workload", "-1"], ["--max-new", "0"],
    ["--max-new", "1025"], ["--repair-steps", "2"], ["--draft-len", "0"], ["--threads", "0"],
    ["--keep-confidence-above", "nan"], ["--max-keep-fraction", "1.5"], ["--model", "missing-model"],
])
def test_invalid_options_fail_before_model_loading(tmp_path, monkeypatch, options):
    checkpoint = tmp_path / "drafter.pt"
    checkpoint.write_bytes(b"unused")
    monkeypatch.setattr(command.TargetModel, "from_pretrained", lambda *a, **k: pytest.fail("loaded before validation"))
    monkeypatch.setattr(sys, "argv", [
        "run_reconciliation_ablation", "--model", str(tmp_path), "--corpus", str(tmp_path),
        "--checkpoint", str(checkpoint), "--out", str(tmp_path / "report.json"),
    ] + options)
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
