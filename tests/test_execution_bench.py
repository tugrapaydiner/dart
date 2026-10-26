from __future__ import annotations

import copy
import json
import os
import sys
import time
import venv
from types import SimpleNamespace

import pytest

from benchmarks import execution_bench as bench
from benchmarks import execution_protocol as protocol


@pytest.fixture
def clock(monkeypatch):
    value = [0.0]
    monkeypatch.setattr(bench.time, "perf_counter", lambda: value[0])
    return value


def small_plan():
    return {"protocol_hash": "test", "sessions": 3, "repeats": 2, "warmups": 1, "timeout_seconds": 300,
            "cases": [{"id": "case", "prompt_ids": [11, 12], "max_new_tokens": 4}],
            "timed_ids": ["case"], "methods": list(protocol.METHODS), "candidate": "replay_dart",
            "gates": {"maximum_peak_allocated_bytes": 1000}}


def measure(plan, clock, session=0, *, ids=None, methods=None, progress=None):
    pending = [0.0]

    def generate(prompt):
        clock[0] += 0.004
        pending[0] += 0.006
        return SimpleNamespace(token_ids=list([7, 8, 2] if ids is None else ids))

    def synchronize():
        clock[0] += pending[0]
        pending[0] = 0.0

    report = {"schema": "dart-execution-session-v1", "status": "running", "session": session,
              "protocol_hash": plan["protocol_hash"], "peak_allocated_bytes": 0}
    calls = dict.fromkeys(protocol.METHODS, generate)
    calls.update(methods or {})
    bench.measure_session(plan, session, {4: generate}, {4: calls}, report,
                          synchronize=synchronize, memory=lambda: 123, eos_ids={2}, vocab_size=32,
                          progress=(progress or (lambda message: None)))
    report["status"] = "complete"
    return report


def test_timing_includes_pending_work_and_excludes_checkpoint_writes(clock):
    def progress(message):
        clock[0] += 5

    report = measure(small_plan(), clock, progress=progress)
    assert len(report["checks"]) == 1 and len(report["warmups"]) == 6 and len(report["samples"]) == 12
    assert all(row["latency_ms"] == pytest.approx(10) for row in report["samples"])
    assert all(row["early_eos"] and row["token_ids"] == [7, 8, 2] for row in report["samples"])
    bench.validate_session(report, small_plan(), 0)


def test_a_timed_truncation_stops_the_run_and_keeps_the_failed_output(clock):
    count, snapshots = [0], []

    def unstable(prompt):
        count[0] += 1
        clock[0] += 0.01
        return SimpleNamespace(token_ids=[7] if count[0] == 4 else [7, 8, 2])

    plan = small_plan()
    report = {"peak_allocated_bytes": 0}

    def reference(prompt):
        clock[0] += 0.01
        return SimpleNamespace(token_ids=[7, 8, 2])

    methods = dict.fromkeys(protocol.METHODS, reference)
    methods["replay_dart"] = unstable
    with pytest.raises(ValueError, match="complete token mismatch"):
        bench.measure_session(plan, 0, {4: reference}, {4: methods}, report,
                              synchronize=lambda: None, memory=lambda: 123, eos_ids={2}, vocab_size=32,
                              progress=lambda message: snapshots.append(copy.deepcopy(report)))
    assert report["samples"][-1]["match"] is False
    assert report["samples"][-1]["token_ids"] == [7]
    assert snapshots[-1]["samples"] == report["samples"]


@pytest.mark.parametrize("ids", [[], [7] * 5, [7, 2, 8], [True], [-1], [32]])
def test_invalid_reference_outputs_fail_before_comparisons(clock, ids):
    with pytest.raises(ValueError, match="invalid token|after EOS"):
        measure(small_plan(), clock, ids=ids)


def test_prompt_mutation_is_rejected(clock):
    def mutate(prompt):
        prompt.append(9)
        clock[0] += 0.01
        return SimpleNamespace(token_ids=[7, 8, 2])

    plan = small_plan()
    with pytest.raises(ValueError, match="changed its input"):
        measure(plan, clock, methods={"replay_dart": mutate})
    assert plan["cases"][0]["prompt_ids"] == [11, 12]


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "tokens", "reference_hash", "latency", "boolean_latency", "eos", "memory", "session_type"])
def test_incomplete_or_corrupt_sessions_cannot_be_accepted(clock, change):
    plan = small_plan()
    report = measure(plan, clock)
    if change == "missing":
        report["samples"].pop()
    elif change == "duplicate":
        report["samples"][1] = copy.deepcopy(report["samples"][0])
    elif change == "order":
        report["samples"].reverse()
    elif change == "tokens":
        report["samples"][0]["token_ids"].pop()
    elif change == "reference_hash":
        report["references"]["case"]["output_hash"] = "changed"
    elif change == "latency":
        report["samples"][0]["latency_ms"] = float("nan")
    elif change == "boolean_latency":
        report["samples"][0]["latency_ms"] = True
    elif change == "eos":
        report["samples"][0]["early_eos"] = False
    elif change == "memory":
        report["peak_allocated_bytes"] = 1001
    else:
        report["session"] = False
    with pytest.raises(ValueError):
        bench.validate_session(report, plan, 0)


@pytest.mark.parametrize("custom_protocol", [False, True])
def test_supervisor_launches_each_session_separately(clock, monkeypatch, tmp_path, custom_protocol):
    plan, launched = small_plan(), []
    monkeypatch.setattr(bench, "read_protocol", lambda path: copy.deepcopy(plan))

    def process(command, log, timeout, env):
        session = int(command[-1])
        report = measure(plan, clock, session)
        report["pid"] = 400 + session
        bench.write_json(log.with_suffix(".json"), report)
        launched.append((session, timeout, command[:3], env["HF_HUB_OFFLINE"]))
        return {"pid": report["pid"], "returncode": 0, "timed_out": False, "elapsed_seconds": 1}

    monkeypatch.setattr(bench, "run_process", process)
    options = {"protocol_reader": lambda path: copy.deepcopy(plan), "worker_module": "benchmarks.feedback_bench"} if custom_protocol else {}
    result = bench.run_study(tmp_path / "plan.json", tmp_path / "model", tmp_path / "out", **options)
    assert result["status"] == "complete"
    assert [row[0] for row in launched] == [0, 1, 2]
    assert launched[0][1] > launched[1][1] > launched[2][1]
    module = "benchmarks.feedback_bench" if custom_protocol else "benchmarks.execution_bench"
    assert all(row[2] == [sys.executable, "-m", module] and row[3] == "1" for row in launched)
    assert len({item["pid"] for item in result["processes"]}) == 3
    with pytest.raises(FileExistsError):
        bench.run_study(tmp_path / "plan.json", tmp_path / "model", tmp_path / "out")


@pytest.mark.parametrize("failure", ["timeout", "exit", "changed_reference", "wrong_pid"])
def test_supervisor_stops_on_failure_and_preserves_partial_evidence(clock, monkeypatch, tmp_path, failure):
    plan, launched = small_plan(), []
    monkeypatch.setattr(bench, "read_protocol", lambda path: copy.deepcopy(plan))

    def process(command, log, timeout, env):
        session = int(command[-1])
        launched.append(session)
        report = measure(plan, clock, session, ids=[7, 9, 2] if session and failure == "changed_reference" else None)
        report["pid"] = 500 + session
        bench.write_json(log.with_suffix(".json"), report)
        return {"pid": 999 if failure == "wrong_pid" else report["pid"],
                "returncode": 1 if failure == "exit" else 0, "timed_out": failure == "timeout", "elapsed_seconds": 1}

    monkeypatch.setattr(bench, "run_process", process)
    folder = tmp_path / "out"
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        bench.run_study(tmp_path / "plan.json", tmp_path / "model", folder)
    assert launched == ([0, 1] if failure == "changed_reference" else [0])
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == ("timeout" if failure == "timeout" else "failed")
    assert (folder / "session-0.json").is_file() and (folder / "protocol.json").is_file()


def test_real_subprocess_records_identity_and_keeps_timeout_log(tmp_path):
    success = bench.run_process([sys.executable, "-c", "import os; print(os.getpid(), flush=True)"],
                                tmp_path / "success.log", 10, os.environ.copy())
    assert success["returncode"] == 0 and not success["timed_out"]
    assert int((tmp_path / "success.log").read_text()) == success["pid"]
    timeout = bench.run_process([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
                                tmp_path / "timeout.log", 1, os.environ.copy())
    assert timeout["timed_out"] and timeout["returncode"] != 0
    assert "started" in (tmp_path / "timeout.log").read_text()


def test_supervisor_accepts_a_worker_started_by_a_venv_redirector(clock, monkeypatch, tmp_path):
    plan = small_plan()
    monkeypatch.setattr(bench, "read_protocol", lambda path: copy.deepcopy(plan))

    def process(command, log, timeout, env):
        session = int(command[-1])
        report = measure(plan, clock, session)
        report.update(pid=800 + session, parent_pid=700 + session)
        bench.write_json(log.with_suffix(".json"), report)
        return {"pid": report["parent_pid"], "returncode": 0, "timed_out": False, "elapsed_seconds": 1}

    monkeypatch.setattr(bench, "run_process", process)
    result = bench.run_study(tmp_path / "plan.json", tmp_path / "model", tmp_path / "out")
    assert result["status"] == "complete"
    assert [(row["pid"], row["worker_pid"]) for row in result["processes"]] == [(700, 800), (701, 801), (702, 802)]


def test_real_venv_timeout_stops_the_worker_behind_the_launcher(tmp_path):
    directory = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(directory)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = tmp_path / "still-running.txt"
    program = ("import json, os, time; from pathlib import Path; "
               "print(json.dumps({'pid': os.getpid(), 'parent_pid': os.getppid()}), flush=True); "
               f"time.sleep(3); Path({str(marker)!r}).write_text('still running')")
    result = bench.run_process([str(python), "-c", program], tmp_path / "venv.log", 2, os.environ.copy())
    identity = json.loads((tmp_path / "venv.log").read_text().splitlines()[0])
    assert result["pid"] in (identity["pid"], identity["parent_pid"])
    assert result["timed_out"] and result["returncode"] != 0
    time.sleep(3.2)
    assert not marker.exists()


def test_protocol_reader_rejects_changed_limits_and_source(tmp_path, monkeypatch):
    checks, timed = protocol.selected_ids("validation")
    plan = {"schema": protocol.SCHEMA, "split": "validation", "sessions": 1, "timeout_seconds": 300,
            "repeats": 2, "warmups": 1, "methods": list(protocol.METHODS), "timed_ids": timed,
            "cases": [{"id": name} for name in checks], "source_hashes": {"source": "hash"}}
    plan["protocol_hash"] = protocol.digest(plan)
    path = tmp_path / "protocol.json"
    monkeypatch.setattr(bench, "source_hashes", lambda: {"source": "hash"})
    bench.write_json(path, plan)
    assert bench.read_protocol(path) == plan
    plan["timeout_seconds"] = 9999
    plan["protocol_hash"] = protocol.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    bench.write_json(path, plan)
    with pytest.raises(ValueError, match="settings changed"):
        bench.read_protocol(path)
    plan["timeout_seconds"] = 300
    plan["protocol_hash"] = protocol.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    bench.write_json(path, plan)
    monkeypatch.setattr(bench, "source_hashes", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="source changed"):
        bench.read_protocol(path)
