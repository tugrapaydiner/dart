from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from benchmarks import corpus_bench, corpus_eval
from dart import cli
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_cli_check import local_model as local_model
from tests.test_corpus_eval import Tokenizer, records
from tests.test_sampling_devices import DEVICES


def engines(device="cpu"):
    model, tokenizer = build_tiny_target().model, Tokenizer()
    original = TargetModel(deepcopy(model), tokenizer=tokenizer, device=device)
    native = TargetModel(model, tokenizer=tokenizer, device=device, native_rms_norm=True)
    config = DecodeConfig(max_new_tokens=9, draft_len=2)
    return (DartEngine(native, NGramDrafter(), default_decode_config=config),
            DartEngine(original, NGramDrafter(), default_decode_config=config))


@pytest.mark.parametrize("device", DEVICES)
def test_norm_control_uses_original_reference_and_preserves_all_timing_checks(device):
    native, original = engines(device)
    with native, original:
        report = corpus_bench.benchmark_corpus(native, records(), norm_control=original, profile=False, warmup=1, repeats=2)
    assert "original RMSNorm" in report["correctness"]["reference"]
    assert "native_rms_norms" not in report["correctness"]["reference_target"]
    assert report["normalization_control"] == "dart_model_norm"
    assert "dart_model_norm" in report["baseline_methods"]
    assert "norm_ablation" in report["summary"]
    assert "two target models" in report["memory_scope"]
    assert len(report["timing"]["samples"]) == 24
    assert set(report["record_hidden_by_method"].values()) == {False}
    for sample in report["timing"]["samples"]:
        assert sample["output_hash"] == report["correctness"]["outputs"][sample["prompt_index"]]["output_hash"]


def test_original_norm_can_remain_the_fastest_baseline():
    costs = {"hf_greedy": 5, "hf_prompt_lookup": 2, "target_greedy": 4, "target_only": 4,
             "dart": 1, "dart_model_norm": 0.5}
    samples = [{"method": name, "repeat": repeat, "latency_ms": cost, "tokens": 4, "peak_gpu_bytes": 0}
               for name, cost in costs.items() for repeat in range(2)]
    result = corpus_bench._summarize(samples, 303)
    assert result["fastest_baseline"] == "dart_model_norm"
    assert result["norm_ablation"] == {"speedup": 0.5, "ci95_speedup": [0.5, 0.5]}
    assert result["vs_fastest"]["dart"]["speedup"] == 0.5


@pytest.mark.parametrize("change", ["weights", "eos", "native_reference", "tokenizer"])
def test_changed_reference_settings_fail_before_generation(monkeypatch, change):
    native, original = engines()
    with native, original:
        if change == "weights":
            with torch.no_grad():
                original.target.model.lm_head.weight[0, 0] += 1
        elif change == "eos":
            original.target.eos_token_ids = frozenset([3])
        elif change == "native_reference":
            original.target = native.target
        else:
            original.target.tokenizer = Tokenizer()
        monkeypatch.setattr(native, "generate", lambda *a: pytest.fail("invalid reference reached generation"))
        with pytest.raises(ValueError):
            corpus_eval.evaluate_corpus(native, records(), reference_target=original.target)


def test_original_reference_mismatch_is_not_hidden_by_native_agreement(monkeypatch):
    native, original = engines()
    build = corpus_eval.build_hf_method

    def shortened(target, *args, **kwargs):
        method = build(target, *args, **kwargs)

        def run(prompt):
            output = method(prompt)
            return replace(output, token_ids=output.token_ids[:-1]) if target is original.target else output

        return run

    monkeypatch.setattr(corpus_eval, "build_hf_method", shortened)
    with native, original, pytest.raises(ValueError, match="complete token mismatch"):
        corpus_eval.evaluate_corpus(native, records(), reference_target=original.target)


@pytest.mark.parametrize("change", ["profile", "policy", "hidden", "verifier"])
def test_mismatched_control_configuration_fails_before_generation(monkeypatch, change):
    native, original = engines()
    with native, original:
        if change == "policy":
            original.drafter.max_ngram = 3
        elif change == "hidden":
            original.target.record_hidden = True
        elif change == "verifier":
            original.verifier_name = "reference"
        monkeypatch.setattr(corpus_bench, "evaluate_corpus", lambda *a, **k: pytest.fail("invalid control reached generation"))
        with pytest.raises(ValueError, match="norm controls"):
            corpus_bench.benchmark_corpus(native, records(), norm_control=original, profile=change == "profile")


@pytest.mark.parametrize("command", ["check", "bench"])
def test_public_commands_record_a_matched_original_norm_control(local_model, tmp_path, monkeypatch, command):
    directory, built = local_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    path = tmp_path / f"{command}.json"
    args = [command, "--model", str(directory), "--native-rms-norm", "--norm-control", "--out", str(path)]
    if command == "bench":
        args.extend(["--skip-profile", "--warmup", "1", "--repeats", "2"])
    assert cli.main(args) == 0
    report = json.loads(path.read_text())
    assert len(built) == 2
    assert report["target"]["native_rms_norms"] == 5
    assert "native_rms_norms" not in report["norm_control_target"]
    assert report["target"]["weights_sha256"] == report["norm_control_target"]["weights_sha256"]
    checks = report["correctness"] if command == "bench" else report
    assert checks["reference_target"] == report["norm_control_target"]
    assert "original RMSNorm" in checks["reference"]
    if command == "bench":
        assert len(report["timing"]["samples"]) == 24
        assert report["normalization_control"] == "dart_model_norm" and report["profile"] is None


@pytest.mark.parametrize("args", [["check", "--norm-control"], ["bench", "--norm-control"],
                                  ["bench", "--native-rms-norm", "--norm-control"]])
def test_invalid_control_options_fail_before_loading(local_model, args):
    directory, built = local_model
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--model", str(directory)])
    assert error.value.code == 2 and not built
