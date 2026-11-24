from __future__ import annotations

import json
from contextlib import closing
from functools import wraps

import pytest
import torch

from benchmarks import profiling, run_bench
from dart import cli
from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CapacityBuckets
from dart.runtime import engine as runtime
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.tree_verify import DraftTree
from tests.conftest import build_tiny_target
from tests.test_cli_policy import policy_model as policy_model
from tests.test_corpus_eval import records
from tests.test_round_actions import ScriptedScheduler


def target_and_prompt():
    target = build_tiny_target()
    with torch.no_grad():
        target.model.lm_head.weight.zero_()
    return target, [5, 6, 7, 0, 3, 5, 6, 7, 0, 4, 5, 6, 7]


@pytest.mark.parametrize("verifier,widths", [("block", [4, 2]), ("reference", [1] * 6), ("tree", [5, 3])])
def test_verifier_work_matches_model_inputs(monkeypatch, verifier, widths):
    target, prompt = target_and_prompt()
    drafter = NGramDrafter()
    monkeypatch.setattr(drafter, "propose", lambda history, count: [0] * count)
    monkeypatch.setattr(drafter, "propose_tree", lambda history, count: DraftTree.from_branches(
        [[0] * count, [1], [2, 8][:count]]))
    target.eos_token_ids = frozenset({2})
    config = DecodeConfig(max_new_tokens=7, draft_len=3)
    observed = []

    def record(module, args, kwargs):
        observed.append(kwargs["input_ids"].numel())

    with DartEngine(target, drafter, verifier=verifier) as engine:
        with target.model.register_forward_pre_hook(record, with_kwargs=True):
            assert engine.generate(prompt, config).token_ids == [0] * 7
        assert observed == [len(prompt), *widths]
        draft = engine._round_drafts

        def draft_with_forward(*args, **kwargs):
            target.prefill([7, 8])
            return draft(*args, **kwargs)

        monkeypatch.setattr(engine, "_round_drafts", draft_with_forward)
        report = profiling.profile_methods({"hf_greedy": run_bench.build_hf_method(target, 7),
                                            "dart": lambda ids: engine.generate(ids, config)},
                                           [prompt], target, {"dart": engine}, repeats=2)
    for name, expected in (("dart", (len(widths), sum(widths))), ("hf_greedy", (None, None))):
        rows = [row for row in report["samples"] if row["method"] == name]
        for row in [*rows, report["summary"][name]]:
            assert (row["verify_target_calls"], row["verify_input_positions"]) == expected
    assert "_run_model" not in vars(target)


def test_verifier_work_counts_selected_replay_bucket(monkeypatch):
    pytest.importorskip("transformers", minversion="5.16.1")
    target, _ = target_and_prompt()
    prompts = [[0] * 3, [0] * 17]
    drafter = NGramDrafter()
    monkeypatch.setattr(drafter, "propose", lambda history, count: [0] * count)
    with closing(CapacityBuckets(target, (16, 32), max_block_tokens=3, capture_predictions=True,
                                 replay=False)) as prepared:
        with DartEngine(prepared, drafter) as engine:
            methods = {"fixed": lambda ids: engine.generate(ids, DecodeConfig(max_new_tokens=6, draft_len=2)),
                       "one": lambda ids: engine.generate(ids, DecodeConfig(max_new_tokens=1))}
            for name, expected in (("fixed", (2, 5)), ("one", (0, 0))):
                report = profiling.profile_methods({name: methods[name]}, prompts, prepared, {name: engine},
                                                   reference=name, repeats=1)
                for row in report["samples"]:
                    assert (row["verify_target_calls"], row["verify_input_positions"]) == expected
        assert all("_forward" not in vars(bucket) for bucket in prepared.buckets)


@pytest.mark.parametrize("default", ["block", "tree", "reference"])
def test_every_rounds_drafting_and_verification_time_is_counted(monkeypatch, default):
    target, prompt = target_and_prompt()
    drafter = NGramDrafter()
    clock, draft_costs, verify_costs, entropy = [0.0], [], [], []
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: clock[0])

    def charge(call, cost, costs, verifier=None):
        @wraps(call)
        def measured(*args, **kwargs):
            output = call(*args, **kwargs)
            clock[0] += cost
            costs.append(cost)
            if verifier in ("block", "tree"):
                entropy.append(kwargs.get("compute_entropy"))
            return output
        return measured

    monkeypatch.setattr(drafter, "propose", charge(drafter.propose, 0.03, draft_costs))
    monkeypatch.setattr(drafter, "propose_tree", charge(drafter.propose_tree, 0.07, draft_costs))
    for name, cost in (("block", 0.02), ("tree", 0.04), ("reference", 0.06)):
        monkeypatch.setitem(runtime._VERIFIERS, name, charge(runtime._VERIFIERS[name], cost, verify_costs, name))
    verifiers = dict(runtime._VERIFIERS)
    scheduler = ScriptedScheduler([(2, "block"), (2, "tree"), (2, "reference")], lane="ngram")
    outputs = []
    with DartEngine(target, lanes={"ngram": drafter}, scheduler=scheduler, verifier=default) as engine:
        verify = engine.verify

        def generate(ids):
            output = engine.generate(ids, DecodeConfig(max_new_tokens=12))
            outputs.append(output)
            return output

        report = profiling.profile_methods({"hf_greedy": run_bench.build_hf_method(target, 12), "dart": generate},
                                            [prompt], target, {"dart": engine}, repeats=1)
        assert engine.verify is verify and runtime._VERIFIERS == verifiers
        assert "_round_drafts" not in vars(engine) and "_resolve_verifier" not in vars(engine)
    row = report["summary"]["dart"]
    assert {round.verifier for round in outputs[0].rounds} == {"block", "tree", "reference"}
    assert 0.03 in draft_costs and 0.07 in draft_costs
    assert set(verify_costs) == {0.02, 0.04, 0.06}
    assert (row["draft_ms"], row["verify_ms"]) == pytest.approx((sum(draft_costs) * 1000, sum(verify_costs) * 1000))
    assert row["draft_ms"] + row["verify_ms"] == pytest.approx(row["decode_ms"])
    assert entropy and all(value is False for value in entropy)
    assert len({sample["output_hash"] for sample in report["samples"]}) == 1
    assert not target.model._forward_hooks and not target.model._forward_pre_hooks


@pytest.mark.parametrize("stage", ["draft", "verify"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_tree_profile_failure_restores_hooks_and_allows_reuse(monkeypatch, stage, error):
    target, prompt = target_and_prompt()
    drafter = NGramDrafter()
    with DartEngine(target, drafter, verifier="tree") as engine:
        verify = engine.verify
        methods = {"hf_greedy": run_bench.build_hf_method(target, 5),
                   "dart": lambda ids: engine.generate(ids, DecodeConfig(max_new_tokens=5))}

        def fail(*args, **kwargs):
            raise error("interrupted phase")

        with monkeypatch.context() as patch:
            if stage == "draft":
                patch.setattr(drafter, "propose_tree", fail)
            else:
                patch.setattr(engine, "verify", fail)
            with pytest.raises(error, match="interrupted phase"):
                profiling.profile_methods(methods, [prompt], target, {"dart": engine}, repeats=1)
        assert engine.verify is verify
        assert "_round_drafts" not in vars(engine) and "_resolve_verifier" not in vars(engine)
        assert not target.model._forward_hooks and not target.model._forward_pre_hooks
        assert methods["dart"](prompt).token_ids == methods["hf_greedy"](prompt).token_ids


@pytest.mark.parametrize("compare", [False, True])
def test_cli_profiles_adaptive_and_fixed_tree_methods(policy_model, monkeypatch, tmp_path, compare):
    directory, _, _ = policy_model
    monkeypatch.setattr("dart.data.pilot.build_pilot", lambda seed: records())
    output = tmp_path / "profiled.json"
    assert cli.main(["bench", "--model", str(directory), "--policy", "cost_table",
                     *(["--compare-policies"] if compare else []),
                     "--warmup", "1", "--repeats", "2", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    partial = json.loads(output.with_suffix(".timing.json").read_text())
    assert report["profiling_requested"] and partial["profiling_requested"]
    count = 48 if compare else 20
    assert len(report["timing"]["samples"]) == len(report["profile"]["samples"]) == count
    for sample in report["profile"]["samples"]:
        expected = report["correctness"]["outputs"][sample["prompt_index"]]
        assert sample["output_hash"] == expected["output_hash"]
        if sample["method"] in {"dart", "target_only", *report["policy_controls"]}:
            assert sample["draft_ms"] >= 0 and sample["verify_ms"] > 0
            assert 0 < sample["verify_target_calls"] <= sample["verify_input_positions"]
        else:
            assert sample["draft_ms"] is None and sample["verify_ms"] is None
            assert sample["verify_target_calls"] is None and sample["verify_input_positions"] is None
    assert partial["timing"] == report["timing"] and "profile" not in partial
    assert "adaptive" in report["profile"]["policy_scope"]
