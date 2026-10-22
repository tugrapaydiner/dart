from __future__ import annotations

import json
import sys

import pytest

from benchmarks import run_bench
from dart.diffusion.compact_training import CompactTrainer
from tests.conftest import build_tiny_target
from tests.test_benchmark_rows import unused_model
from tests.test_compact_training import MODEL, TRAIN, training_inputs


def test_named_checkpoints_use_features_without_charging_baselines(tmp_path, monkeypatch):
    target, dataset = training_inputs(tmp_path)
    target.tokenizer.decode = lambda ids, **kwargs: " ".join(map(str, ids))
    trainer = CompactTrainer(target, dataset, MODEL, TRAIN)
    initial, trained = tmp_path / "initial.pt", tmp_path / "trained.pt"
    trainer.save(initial)
    trainer.train_step()
    trainer.save(trained)
    monkeypatch.setattr(run_bench, "build_tiny_target", lambda **kwargs: target)
    monkeypatch.setattr(run_bench, "DiffusionDrafterModel", unused_model)
    monkeypatch.setattr(run_bench, "MTPHeads", unused_model)
    measure = run_bench.measure_methods
    feature_methods = {"initial", "trained"}

    def check_methods(methods, prompts, *args, **kwargs):
        for name, method in methods.items():
            flags = []
            hook = target.model.register_forward_pre_hook(
                lambda module, args, kwargs, observed=flags: observed.append(bool(kwargs.get("output_hidden_states", module.config.output_hidden_states))),
                with_kwargs=True,
            )
            try:
                method(prompts[0])
            finally:
                hook.remove()
            assert flags and all(flag is (name in feature_methods) for flag in flags)
            assert target.record_hidden is True
        return measure(methods, prompts, *args, **kwargs)

    monkeypatch.setattr(run_bench, "measure_methods", check_methods)
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--prompts", "2", "--max-new", "6", "--draft-len", "3",
        "--warmup", "1", "--repeats", "2", "--profile", "--headroom",
        "--compact-checkpoint", f"initial={initial}", "--compact-checkpoint", f"trained={trained}",
        "--out", str(tmp_path / "bench"), "--tag", "compact",
    ])
    run_bench.main()
    report = json.loads((tmp_path / "bench/compact.json").read_text())
    methods = {"hf_greedy", "target_greedy", "target_only", "ngram", "initial", "trained"}
    assert set(report["timing"]["summary"]) == methods
    assert len(report["timing"]["samples"]) == 24
    assert report["run"]["record_hidden_by_method"] == {name: name in feature_methods for name in methods}
    assert report["lanes"]["initial"]["checkpoint"]["training_step"] == 0
    assert report["lanes"]["trained"]["checkpoint"]["training_step"] == 1
    assert report["lanes"]["initial"]["checkpoint"]["sha256"] != report["lanes"]["trained"]["checkpoint"]["sha256"]
    assert report["lanes"]["trained"]["checkpoint"]["canvas_len"] == 3
    assert set(report["profile"]["summary"]) == methods
    assert report["headroom"]["scope"].startswith("offline")
    for prompt_index in range(2):
        assert len({row["output_hash"] for row in report["timing"]["samples"] if row["prompt_index"] == prompt_index}) == 1
    assert target.record_hidden is True


def test_unselected_checkpoint_is_not_loaded(monkeypatch):
    monkeypatch.setattr(run_bench, "load_compact_checkpoint", unused_model)
    rows = run_bench.build_rows(build_tiny_target(), selected={"ngram"}, compact_checkpoints={"compact": "missing.pt"})
    assert set(rows) == {"ngram", "target_only"}


@pytest.mark.parametrize("name", ["hf_greedy", "target_only", "ngram", "mtp_untrained", "ar_draft"])
def test_checkpoint_names_cannot_replace_references_or_existing_lanes(name, monkeypatch):
    monkeypatch.setattr(run_bench, "load_compact_checkpoint", unused_model)
    with pytest.raises(ValueError, match="reserved"):
        run_bench.build_rows(build_tiny_target(), tiny_drafters=False, selected={name}, compact_checkpoints={name: "missing.pt"})


@pytest.mark.parametrize("pairs", [["missing-separator"], ["name="], ["name=a", "name=b"], ["Upper=a"]])
def test_invalid_checkpoint_arguments_fail_before_model_loading(pairs, monkeypatch):
    monkeypatch.setattr(run_bench, "build_tiny_target", unused_model)
    args = ["run_bench"]
    for pair in pairs:
        args += ["--compact-checkpoint", pair]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        run_bench.main()
    assert error.value.code == 2


def test_hidden_recording_is_restored_when_a_method_fails():
    target = build_tiny_target()
    target.record_hidden = True

    def fail(prompt):
        assert target.record_hidden is False
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        run_bench.with_hidden_recording(target, False, fail)([7, 8, 9])
    assert target.record_hidden is True
