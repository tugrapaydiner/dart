from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from benchmarks import corpus_bench
from dart import cli
from dart.drafters.base import NullDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.test_cli_policy import policy_model as policy_model
from tests.test_corpus_bench import build_engine
from tests.test_corpus_eval import records

CONTROLS = {"fixed2", "fixed4", "fixed8", "tree2", "tree4", "tree8", "time4"}


@pytest.mark.parametrize("case", ["ordinary", "accepted", "eos"])
def test_controls_share_the_target_and_match_full_outputs(monkeypatch, case):
    with build_engine() as engine:
        target = engine.target
        engine.drafter.min_ngram, engine.drafter.max_ngram = 2, 3
        engine.drafter.tree_width, engine.drafter.max_scan = 2, 11
        ngram = dict(vars(engine.drafter))
        if case == "accepted":
            with torch.no_grad():
                target.model.lm_head.weight.zero_()
        elif case == "eos":
            prompt = target.tokenizer.encode("aaa", add_special_tokens=True, truncation=False)
            target.eos_token_ids = frozenset(target.greedy_generate(prompt, 1))
        created, closed, saved, requests = [], [], [], {}
        initialize, close = DartEngine.__init__, DartEngine.close
        generate = DartEngine.generate

        def initialize_control(self, *args, **kwargs):
            initialize(self, *args, **kwargs)
            created.append(self)

        def close_control(self):
            closed.append(self)
            close(self)

        def record_request(self, prompt, config):
            requests.setdefault(id(self), []).append(config.draft_len)
            return generate(self, prompt, config)

        monkeypatch.setattr(DartEngine, "__init__", initialize_control)
        monkeypatch.setattr(DartEngine, "close", close_control)
        monkeypatch.setattr(DartEngine, "generate", record_request)
        report = corpus_bench.benchmark_corpus(engine, records(), compare_policies=True,
                                               profile=False, warmup=1, repeats=2, on_timing=saved.append)
        controls = report["policy_controls"]
        assert set(controls) == CONTROLS and saved[0]["policy_controls"] == controls
        assert CONTROLS <= set(report["baseline_methods"])
        assert len(created) == 8 and created == list(reversed(closed))
        assert all(control.target is target for control in created)
        assert all(control._closed for control in created[1:])
        drafts = [control.lanes["ngram"] if control.lanes is not None else control.drafter for control in created[1:]]
        assert len({id(draft) for draft in drafts}) == 7
        assert all(draft is not engine.drafter and vars(draft) == ngram for draft in drafts)
        for control, settings in zip(created[1:], controls.values()):
            assert requests[id(control)] == [settings["draft_len"]] * 8
            assert control.verifier_name == settings["verifier"]
        assert target.record_hidden and engine.default_decode_config.draft_len == 2
        assert engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=3)).token_ids == target.greedy_generate([5, 6, 7], 3)
    assert len(report["timing"]["samples"]) == 48
    for sample in report["timing"]["samples"]:
        expected = report["correctness"]["outputs"][sample["prompt_index"]]
        assert sample["output_hash"] == expected["output_hash"]
        assert sample["tokens"] == expected["generated_tokens"]
    for name, settings in controls.items():
        assert settings["draft_len"] == int(name[-1])
        assert settings["verifier"] == ("tree" if name.startswith("tree") else "block")
        assert settings["ngram"] == ngram
        assert report["record_hidden_by_method"][name] is True
    assert controls["time4"]["scheduler"]["warmup_rounds"] == 2
    assert controls["time4"]["scheduler"]["retry_interval"] == 16
    if case == "eos":
        assert report["correctness"]["outputs"][0]["early_eos"]


@pytest.mark.parametrize("fastest", sorted(CONTROLS))
def test_every_policy_can_be_the_strongest_control(fastest):
    names = [*corpus_bench.COMPARISON_BASELINES, *sorted(CONTROLS), "dart"]
    samples = [{"method": name, "prompt_index": 0, "repeat": repeat, "tokens": 4,
                "latency_ms": 1.0 if name == fastest else 2.0, "peak_gpu_bytes": 0}
               for name in names for repeat in range(2)]
    summary = corpus_bench._summarize(samples, 17)
    assert summary["fastest_baseline"] == fastest
    assert summary["vs_fastest"]["dart"]["speedup"] == 0.5


@pytest.mark.parametrize("failure_call", [1, 3, 5], ids=["preflight", "warmup", "timing"])
def test_policy_mismatch_stops_before_saving_and_closes_controls(monkeypatch, failure_call):
    measure = corpus_bench.measure_methods
    closed, close = [], DartEngine.close

    def corrupt(methods, *args, **kwargs):
        original, calls = methods["tree8"], 0

        def changed(prompt):
            nonlocal calls
            calls += 1
            output = original(prompt)
            return replace(output, token_ids=output.token_ids[:-1]) if calls == failure_call else output

        methods["tree8"] = changed
        return measure(methods, *args, **kwargs)

    monkeypatch.setattr(corpus_bench, "measure_methods", corrupt)
    monkeypatch.setattr(DartEngine, "close", lambda self: closed.append(self) or close(self))
    saved = []
    with build_engine() as engine:
        with pytest.raises(ValueError, match="tree8: complete token mismatch"):
            corpus_bench.benchmark_corpus(engine, records(), compare_policies=True, profile=False,
                                          warmup=1, repeats=2, on_timing=saved.append)
        assert not saved and len(closed) == 8
        assert all(control._closed for control in closed[:-1])
        assert engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=3)).token_ids == engine.target.greedy_generate([5, 6, 7], 3)


@pytest.mark.parametrize("case", ["no_ngram", "disabled_ngram", "residency", "replay", "invalid_bool", "norm"])
def test_unsupported_policy_comparisons_fail_before_generation(monkeypatch, case):
    monkeypatch.setattr(corpus_bench, "evaluate_corpus", lambda *a, **k: pytest.fail("unexpected generation"))
    with build_engine() as engine:
        config = engine.default_decode_config
        if case == "no_ngram":
            engine.drafter = NullDrafter()
        elif case == "disabled_ngram":
            config = replace(config, allowed_lanes=["target_only"])
        elif case == "residency":
            config = replace(config, full_kv_residency="cpu_pinned")
        elif case == "replay":
            engine.target.uses_replay_cache = True
        with pytest.raises(ValueError, match="polic"):
            corpus_bench.benchmark_corpus(engine, records(), config, compare_policies=1 if case == "invalid_bool" else True,
                                          profile=False, norm_control=engine if case == "norm" else None)


def test_cli_runs_and_records_the_optional_comparison(policy_model, monkeypatch, tmp_path):
    directory, built, _ = policy_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    output = tmp_path / "policies.json"
    assert cli.main(["bench", "--model", str(directory), "--policy", "cost_table", "--compare-policies",
                     "--skip-profile", "--warmup", "1", "--repeats", "2", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    partial = json.loads(output.with_suffix(".timing.json").read_text())
    assert len(built) == 1 and len(report["timing"]["samples"]) == 48
    assert set(report["policy_controls"]) == CONTROLS
    assert partial["policy_controls"] == report["policy_controls"]


def test_invalid_cli_comparison_fails_before_loading(policy_model, capsys):
    directory, built, _ = policy_model
    with pytest.raises(SystemExit) as error:
        cli.main(["bench", "--model", str(directory), "--compare-policies", "--skip-profile", "--native-rms-norm", "--norm-control"])
    assert error.value.code == 2 and not built
    assert "policy comparisons cannot use" in capsys.readouterr().err
