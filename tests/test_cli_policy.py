from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dart import cli
from dart.models.target import TargetModel
from dart.presets import supra_config
from dart.runtime import factory
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
from tests.conftest import build_tiny_target
from tests.test_cli_check import Tokenizer
from tests.test_corpus_eval import records


@pytest.mark.parametrize("policy", ["fixed", "time", "cost_table"])
def test_preset_policy_is_explicit_and_does_not_change_the_default(tmp_path, policy):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    config = supra_config(tmp_path, policy=policy)
    if policy == "fixed":
        assert "scheduler" not in config
    else:
        assert config["scheduler"]["type"] == policy
    if policy == "cost_table":
        assert config["scheduler"]["draft_lengths"] == [2, 4, 8]
        assert config["scheduler"]["verifiers"] == ["block", "tree"]
        config["scheduler"]["draft_lengths"].append(16)
        assert supra_config(tmp_path, policy=policy)["scheduler"]["draft_lengths"] == [2, 4, 8]
    assert "scheduler" not in supra_config(tmp_path)
    assert config["target"]["local_files_only"] is True
    assert config["target"]["attn_implementation"] == "sdpa"


@pytest.mark.parametrize("policy", [None, False, "adaptive", []])
def test_invalid_policy_is_rejected_before_model_path_lookup(tmp_path, policy):
    with pytest.raises(ValueError, match="policy"):
        supra_config(tmp_path, policy=policy)


@pytest.fixture
def policy_model(tmp_path, monkeypatch):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    built, outputs = [], []
    original_build = factory.build_engine

    def load(spec):
        assert spec["local_files_only"] and not spec["trust_remote_code"]
        model = build_tiny_target().model
        model.config.eos_token_id = model.generation_config.eos_token_id = None
        model.set_attn_implementation(spec["attn_implementation"])
        return TargetModel(model, tokenizer=Tokenizer(), device="cpu")

    def build(config):
        engine = original_build(config)
        generate = engine.generate

        def record(*args, **kwargs):
            output = generate(*args, **kwargs)
            outputs.append(output)
            return output

        monkeypatch.setattr(engine, "generate", record)
        built.append(engine)
        return engine

    monkeypatch.setattr(factory, "_build_target", load)
    monkeypatch.setattr(factory, "build_engine", build)
    rows = [replace(row, text=row.text * 4, max_new_tokens=24) for row in records()]
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: rows)
    return directory, built, outputs


@pytest.mark.parametrize("policy, scheduler", [("time", TimeScheduler), ("cost_table", CostTableScheduler)])
def test_check_runs_the_selected_policy_against_hf(policy_model, tmp_path, policy, scheduler):
    directory, built, outputs = policy_model
    path = tmp_path / "check.json"
    assert cli.main(["check", "--model", str(directory), "--policy", policy, "--out", str(path)]) == 0
    report = json.loads(path.read_text())
    assert report["preset"]["scheduler"]["type"] == policy
    assert report["summary"]["exact_match_rate"] == 1 and len(built) == 1
    assert type(built[0].scheduler) is scheduler and built[0]._closed
    assert len(outputs) == 2 and all(len(out.token_ids) == 24 for out in outputs)
    if policy == "cost_table":
        assert all({step.verifier for step in out.rounds} == {"block", "tree"} for out in outputs)
        assert {"ngram:8:block", "ngram:8:tree"} <= set(built[0].scheduler.counts)


def test_benchmark_records_adaptive_policy_and_matching_controls(policy_model, tmp_path):
    directory, built, outputs = policy_model
    path = tmp_path / "bench.json"
    assert cli.main(["bench", "--model", str(directory), "--policy", "cost_table", "--skip-profile",
                     "--warmup", "1", "--repeats", "2", "--out", str(path)]) == 0
    report = json.loads(path.read_text())
    partial = json.loads(path.with_suffix(".timing.json").read_text())
    assert report["preset"] == partial["preset"]
    assert report["preset"]["scheduler"]["verifiers"] == ["block", "tree"]
    assert report["profile"] is None and not report["profiling_requested"]
    assert report["correctness"]["summary"]["exact_match_rate"] == 1
    samples = report["timing"]["samples"]
    assert len(samples) == 20
    assert {row["method"] for row in samples} == {
        "hf_greedy", "hf_prompt_lookup", "target_greedy", "target_only", "dart",
    }
    for sample in samples:
        assert sample["output_hash"] == report["correctness"]["outputs"][sample["prompt_index"]]["output_hash"]
    assert len(built) == 1 and built[0]._closed
    assert all({step.verifier for step in out.rounds} == {"block", "tree"} for out in outputs)


@pytest.mark.parametrize("args, message", [
    (["check", "--synthetic", "--policy", "time"], "--policy requires --model"),
    (["bench", "--model", "unused", "--policy", "cost_table"], "cost_table benchmarks require --skip-profile"),
    (["bench", "--model", "unused", "--policy", "time", "--native-rms-norm", "--norm-control", "--skip-profile"],
     "norm-control benchmarks require the fixed policy"),
])
def test_unsupported_policy_options_fail_before_loading(monkeypatch, capsys, args, message):
    monkeypatch.setattr(factory, "build_engine", lambda *a, **k: pytest.fail("unexpected model load"))
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2 and message in capsys.readouterr().err
