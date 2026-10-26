import copy
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

from benchmarks import overhead_bench as bench
from benchmarks.execution_protocol import digest, schedule


@pytest.fixture
def cached_study(study, monkeypatch, tmp_path):
    plan, _, reports = study("validation")
    plan["settings"].update(device="cuda", dtype="float32", attention="eager", native_rms_norm=True,
                            reference_native_rms_norm=False, threads=1, matmul_precision="highest")
    plan["protocol_hash"] = digest({k: v for k, v in plan.items() if k != "protocol_hash"})
    reports[0]["protocol_hash"] = plan["protocol_hash"]
    monkeypatch.setattr(bench, "model_hashes", lambda _: plan["model_hashes"])
    monkeypatch.setattr(bench, "version", lambda _: "1.0")
    monkeypatch.setattr(bench, "source_hashes", lambda: {"benchmarks/overhead_bench.py": "f" * 64})
    return plan, reports[0], tmp_path


def test_cached_protocol_freezes_outputs_and_stays_development_only(cached_study):
    plan, report, model = cached_study
    current = bench.make_protocol(plan, report, model)
    assert current["sessions"] == 3 and current["timeout_seconds"] == 120
    assert current["methods"] == list(bench.METHODS)
    assert current["cached_outputs"] == {key: row["token_ids"] for key, row in report["references"].items()}
    assert current["reference_protocol_hash"] == plan["protocol_hash"]
    assert "not fresh held-out" in current["scope"]


@pytest.mark.parametrize("change", ["hash", "tokens", "model", "packages", "settings"])
def test_cached_reference_mismatch_is_rejected_before_execution(cached_study, monkeypatch, change):
    plan, report, model = cached_study
    if change == "hash":
        plan["protocol_hash"] = "0" * 64
    elif change == "tokens":
        next(iter(report["references"].values()))["token_ids"] = [7]
    elif change == "model":
        monkeypatch.setattr(bench, "model_hashes", lambda _: {})
    elif change == "packages":
        monkeypatch.setattr(bench, "version", lambda _: "changed")
    else:
        plan["settings"]["reference_native_rms_norm"] = True
        plan["protocol_hash"] = digest({k: v for k, v in plan.items() if k != "protocol_hash"})
        report["protocol_hash"] = plan["protocol_hash"]
    with pytest.raises(ValueError):
        bench.make_protocol(plan, report, model)


def test_live_source_drift_fails_but_archived_analysis_can_read(cached_study, monkeypatch):
    plan, report, model = cached_study
    current = bench.make_protocol(plan, report, model)
    path = model / "plan.json"
    bench.write_json(path, current)
    assert bench.read_protocol(path) == current
    monkeypatch.setattr(bench, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="source changed"):
        bench.read_protocol(path)
    assert bench.read_protocol(path, check_source=False) == current


def test_methods_switch_the_guard_and_share_the_same_target(monkeypatch):
    from benchmarks import run_bench
    from dart.runtime.engine import DartEngine

    graph = SimpleNamespace(model=torch.nn.Linear(2, 2), _signature=lambda: ("current",))

    def generate(engine, prompt, config):
        assert engine.target is graph and graph._signature() == graph._model_signature
        return type(engine.drafter), config.max_new_tokens, graph._model_signature

    def baseline(target, budget, **kwargs):
        assert target is graph
        return lambda _: (budget, graph._signature(), kwargs)

    monkeypatch.setattr(DartEngine, "generate", generate)
    monkeypatch.setattr(run_bench, "build_hf_method", baseline)
    monkeypatch.setattr(run_bench, "build_target_method", baseline)
    with ExitStack() as stack:
        methods = bench.build_methods(graph, [16, 64], stack)
        for budget, calls in methods.items():
            assert calls["prior_dart"]([7]) == (bench.PriorNGram, budget, bench.prior_signature(graph))
            assert calls["guard_only"]([7]) == (bench.PriorNGram, budget, ("current",))
            assert calls["current_dart"]([7]) == (bench.NGramDrafter, budget, ("current",))
            assert calls["hf_lookup"]([7]) == (budget, ("current",), {"prompt_lookup_num_tokens": 4})
            assert calls["direct_greedy"]([7]) == calls["hf_greedy"]([7]) == (budget, ("current",), {})


@pytest.mark.parametrize("damage", [None, "missing", "reference", "process", "one_loss"])
def test_analysis_retains_losses_and_rejects_incomplete_evidence(cached_study, damage):
    old, template, output = cached_study
    plan = bench.make_protocol(old, template, output)
    bench.write_json(output / "protocol.json", plan)
    processes = []
    for session in range(3):
        report = copy.deepcopy(template)
        report.update(session=session, pid=100 + session, protocol_hash=plan["protocol_hash"],
                      resident_model_copies=1, memory_scope=plan["memory_scope"], warmups=[], samples=[])
        for call in schedule(plan, session):
            latency = 100 if call["method"] == "prior_dart" else 80
            if damage == "one_loss" and session == 2 and call["method"] == "current_dart":
                latency = 110
            row = {**call, **copy.deepcopy(report["references"][call["case"]]), "latency_ms": latency, "match": True}
            report["samples" if call["phase"] == "timed" else "warmups"].append(row)
        if damage == "missing" and session == 2:
            report["samples"].pop()
        if damage == "reference" and session == 2:
            next(iter(report["references"].values()))["token_ids"] = [7]
        bench.write_json(output / f"session-{session}.json", report)
        processes.append({"session": session, "worker_pid": 100 + session, "returncode": 0, "timed_out": False})
    if damage == "process":
        processes[0]["returncode"] = 1
    bench.write_json(output / "status.json", {"status": "complete", "protocol_hash": plan["protocol_hash"],
                                            "elapsed_seconds": 90, "processes": processes})
    if damage in {"missing", "reference", "process"}:
        with pytest.raises(ValueError):
            bench.analyze(output)
    else:
        result = bench.analyze(output)
        assert not result["confirmation_passed"]
        assert result["development_passed"] is (damage is None)
        assert result["full_output_matches"] == len(plan["cases"]) * 3 * (1 + 3 * len(bench.METHODS))
