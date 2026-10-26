"""Validate archived execution runs and report paired session-level evidence."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path, PurePosixPath
from statistics import fmean, stdev

from benchmarks.execution_bench import validate_session
from benchmarks.execution_protocol import (
    BASELINES,
    METHODS,
    MODEL_FILES,
    PACKAGES,
    WORKLOADS,
    digest,
    file_hash,
    protocol_settings,
    selected_ids,
    write_protocol,
)
from benchmarks.verification_workloads import PROMPTS
from dart.data.corpus import corpus_hash, read_records

T95_DF2 = 4.302652729911275


def number(value, name: str, *, minimum=0, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"invalid {name}")
    return value


def hash_string(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(letter in "0123456789abcdef" for letter in value)


def validate_archive_protocol(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise ValueError("execution protocol must be an object")
    body = {key: value for key, value in plan.items() if key != "protocol_hash"}
    if plan.get("protocol_hash") != digest(body):
        raise ValueError("protocol fingerprint changed")
    context = number(plan["context_limit"], "context limit", minimum=1, integer=True)
    fixed = protocol_settings(plan["split"], context)
    fields = set(fixed) | {"cases", "corpus_hash", "tokenizer_hash", "model_hashes", "source_hashes", "packages", "protocol_hash"}
    if set(plan) != fields or digest({key: plan.get(key) for key in fixed}) != digest(fixed):
        raise ValueError("protocol settings or acceptance gates changed")
    rows = read_records(PROMPTS)
    records = {row.id: row for row in rows}
    checks, _ = selected_ids(plan["split"])
    if plan["corpus_hash"] != corpus_hash(rows) or [case["id"] for case in plan["cases"]] != checks:
        raise ValueError("protocol corpus or case selection changed")
    for case in plan["cases"]:
        row = records[case["id"]]
        metadata = {key: getattr(row, key) for key in ("id", "source", "family", "workload", "max_new_tokens")}
        if digest({key: case.get(key) for key in metadata}) != digest(metadata):
            raise ValueError("protocol case metadata changed")
        tokens = case["prompt_ids"]
        if (not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens)
                or len(tokens) + case["max_new_tokens"] + 3 > context or case["prompt_hash"] != digest(tokens)):
            raise ValueError("invalid recorded prompt tokens or context budget")
    hashes = plan["model_hashes"]
    if (not isinstance(hashes, dict) or not set(MODEL_FILES[:4]) <= set(hashes) <= set(MODEL_FILES)
            or not all(hash_string(value) for value in hashes.values()) or not hash_string(plan["tokenizer_hash"])):
        raise ValueError("model or tokenizer fingerprints are incomplete")
    sources = plan["source_hashes"]
    required = {"benchmarks/execution_protocol.py", "benchmarks/execution_bench.py", "benchmarks/hf_replay.py",
                "dart/models/replay.py", "dart/runtime/engine.py"}
    if not isinstance(sources, dict) or not required <= set(sources):
        raise ValueError("execution source fingerprints are incomplete")
    for name, value in sources.items():
        path = PurePosixPath(name)
        if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
                or path.parts[0] not in ("dart", "benchmarks") or not hash_string(value)):
            raise ValueError("invalid execution source fingerprint")
    if set(plan["packages"]) != set(PACKAGES) or any(not isinstance(value, str) or not value for value in plan["packages"].values()):
        raise ValueError("package versions are incomplete")


def validate_study(plan: dict, status: dict, reports: list[dict]) -> None:
    validate_archive_protocol(plan)
    if (status.get("schema") != "dart-execution-run-v1" or status.get("status") != "complete"
            or status.get("protocol_hash") != plan["protocol_hash"]
            or digest(status.get("timeout_seconds")) != digest(plan["timeout_seconds"])):
        raise ValueError("study is incomplete or belongs to another protocol")
    elapsed = number(status["elapsed_seconds"], "study duration", minimum=1e-12)
    if elapsed > plan["timeout_seconds"] or len(reports) != plan["sessions"] or len(status["processes"]) != len(reports):
        raise ValueError("study has missing sessions or exceeded its time limit")
    identities = set()
    frozen = reports[0]
    process_seconds = 0.0
    for index, (report, process) in enumerate(zip(reports, status["processes"])):
        validate_session(report, plan, index)
        if (type(process["session"]) is not int or process["session"] != index
                or type(process["returncode"]) is not int or process["returncode"] != 0 or process["timed_out"] is not False):
            raise ValueError("study contains a failed or reordered process")
        worker = number(report["pid"], "worker PID", minimum=1, integer=True)
        launcher = number(process["pid"], "launcher PID", minimum=1, integer=True)
        parent = number(report["parent_pid"], "parent PID", minimum=1, integer=True)
        if type(process["worker_pid"]) is not int or process["worker_pid"] != worker or (worker != launcher and parent != launcher):
            raise ValueError("session process identity changed")
        started = datetime.fromisoformat(report["started_at"])
        identity = (worker, started)
        if started.tzinfo is None or identity in identities:
            raise ValueError("sessions must identify separately launched processes")
        identities.add(identity)  # Sequential processes may reuse a PID.
        worker_seconds = number(report["elapsed_seconds"], "worker duration", minimum=1e-12)
        process_seconds += number(process["elapsed_seconds"], "process duration", minimum=worker_seconds)
        environment = report["environment"]
        if (digest(environment) != digest(frozen["environment"]) or environment["threads"] != 1
                or type(environment["threads"]) is not int or environment["matmul_precision"] != "highest"
                or environment["torch"] != plan["packages"]["torch"]):
            raise ValueError("session execution environment changed")
        for key in ("python", "system", "torch", "cuda", "device"):
            if not isinstance(environment[key], str) or not environment[key]:
                raise ValueError("session execution environment is incomplete")
        number(environment["device_memory_bytes"], "device memory", minimum=1, integer=True)
        vocab = number(report["vocab_size"], "vocabulary size", minimum=1, integer=True)
        eos = report["eos_token_ids"]
        if (not isinstance(eos, list) or not eos or len(eos) != len(set(eos))
                or any(type(token) is not int or not 0 <= token < vocab for token in eos)):
            raise ValueError("invalid EOS vocabulary")
        if any(any(token >= vocab for token in case["prompt_ids"]) or case["prompt_ids"][-1] in eos for case in plan["cases"]):
            raise ValueError("recorded prompts disagree with the model vocabulary")
        for key in ("vocab_size", "eos_token_ids", "target_parameters", "extra_learned_parameters", "resident_model_copies"):
            if digest(report[key]) != digest(frozen[key]):
                raise ValueError("session model metadata changed")
        number(report["target_parameters"], "target parameters", minimum=1, integer=True)
        number(report["extra_learned_parameters"], "extra learned parameters", integer=True)
        if (type(report["resident_model_copies"]) is not int or report["resident_model_copies"] != 2
                or report["memory_scope"] != plan["memory_scope"]
                or report["output_hash_schema"] != "canonical-json-sha256"):
            raise ValueError("session memory or output scope changed")
        setup = report["setup_ms"]
        if set(setup) != {"reference_model", "native_model", "replay", "static"}:
            raise ValueError("backend preparation measurements are incomplete")
        for value in setup.values():
            number(value, "backend preparation time")
        capture = number(report["capture_and_warmup_ms"], "capture time")
        if capture > setup["replay"] or sum(setup.values()) > worker_seconds * 1000:
            raise ValueError("backend preparation exceeds its enclosing duration")
        cases = {case["id"]: case for case in plan["cases"]}
        for name, row in report["references"].items():
            if row["token_ids"] != frozen["references"][name]["token_ids"]:
                raise ValueError("reference outputs changed between processes")
            if len(row["token_ids"]) < cases[name]["max_new_tokens"] and not row["eos"]:
                raise ValueError("reference output stopped before its budget without EOS")
    if process_seconds > elapsed:
        raise ValueError("process durations exceed the total study duration")


def paired_ratio(control: list[float], candidate: list[float]) -> dict:
    if len(control) != len(candidate) or len(control) not in (1, 3):
        raise ValueError("paired analysis needs one validation session or three final sessions")
    for value in [*control, *candidate]:
        number(value, "session mean latency", minimum=1e-12)
    logs = [math.log(left) - math.log(right) for left, right in zip(control, candidate)]
    center = fmean(logs)
    interval = None
    if len(logs) == 3:
        margin = T95_DF2 * stdev(logs) / math.sqrt(3)
        interval = [math.exp(center - margin), math.exp(center + margin)]
    return {"arithmetic_mean_speed_ratio": fmean(control) / fmean(candidate),
            "mean_latency_reduction": 1 - fmean(candidate) / fmean(control),
            "geometric_session_ratio": math.exp(center), "session_ratio_ci95": interval,
            "session_speed_ratios": [left / right for left, right in zip(control, candidate)]}


def summarize_group(plan: dict, reports: list[dict], ids: list[str]) -> dict:
    sessions = []
    for report in reports:
        samples = [row for row in report["samples"] if row["case"] in ids]
        sessions.append({method: fmean(row["latency_ms"] for row in samples if row["method"] == method) for method in METHODS})
    means = {method: fmean(session[method] for session in sessions) for method in METHODS}
    candidate = plan["candidate"]
    comparisons = {method: paired_ratio([row[method] for row in sessions], [row[candidate] for row in sessions])
                   for method in METHODS if method != candidate}
    fastest = min(BASELINES, key=means.__getitem__)
    return {"cases": ids, "mean_latency_ms": means, "session_mean_latency_ms": sessions,
            "candidate_comparisons": comparisons, "fastest_matched_baseline": fastest,
            "candidate_beats_fastest_baseline": means[candidate] < means[fastest]}


def analyze(plan: dict, status: dict, reports: list[dict]) -> dict:
    validate_study(plan, status, reports)
    ids = plan["timed_ids"]
    cases = {case["id"]: case for case in plan["cases"]}
    aggregate = summarize_group(plan, reports, ids)
    workloads = {name: summarize_group(plan, reports, [key for key in ids if cases[key]["workload"] == name]) for name in WORKLOADS}
    budgets = {str(budget): summarize_group(plan, reports, [key for key in ids if cases[key]["max_new_tokens"] == budget])
               for budget in sorted({cases[key]["max_new_tokens"] for key in ids})}
    per_case = {key: {**summarize_group(plan, reports, [key]), "workload": cases[key]["workload"],
                      "prompt_tokens": len(cases[key]["prompt_ids"]), "max_new_tokens": cases[key]["max_new_tokens"],
                      "output_tokens": len(reports[0]["references"][key]["token_ids"])} for key in ids}
    control = aggregate["candidate_comparisons"][plan["control"]]
    limits = plan["gates"]
    gates = {
        "mean_latency_reduction": aggregate["mean_latency_ms"][plan["candidate"]] <= aggregate["mean_latency_ms"][plan["control"]] * (1 - limits["minimum_mean_latency_reduction"]),
        "session_ratio_ci95": control["session_ratio_ci95"] is not None and control["session_ratio_ci95"][0] > limits["ratio_ci95_lower_exceeds"],
        "each_session_improves": all(value > 1 for value in control["session_speed_ratios"]),
        "beat_fastest_matched_baseline": aggregate["candidate_beats_fastest_baseline"],
        "workload_regression": all(group["mean_latency_ms"][plan["candidate"]] <= group["mean_latency_ms"][plan["control"]] * (1 + limits["maximum_workload_regression"])
                                   for group in workloads.values()),
        "memory": all(report["peak_allocated_bytes"] <= limits["maximum_peak_allocated_bytes"] for report in reports),
        "extra_learned_parameters": all(report["extra_learned_parameters"] <= limits["maximum_extra_learned_parameters"] for report in reports),
        "complete_outputs": True, "complete_protocol": True,
    }
    final = plan["split"] == "test"
    return {
        "schema": "dart-execution-analysis-v1", "protocol_hash": plan["protocol_hash"], "split": plan["split"],
        "provenance": {key: plan[key] for key in ("control_commit", "corpus_hash", "tokenizer_hash", "model_hashes", "source_hashes", "packages", "settings")},
        "uncertainty": plan["uncertainty"],
        "uncertainty_limits": "Approximate log-normal session ratios; only three sessions, fixed cases on one host. Case/workload intervals are descriptive, without multiplicity correction. No population or independent reproduction claim.",
        "archive_validation": "Recorded fingerprints, fixed corpus metadata, complete outputs and matrices checked; archived model files and tokenization are not re-executed.",
        "timing_scope": plan["timing_scope"], "memory_scope": plan["memory_scope"],
        "aggregate": aggregate, "workloads": workloads, "output_budgets": budgets, "cases": per_case,
        "losses": {method: [key for key, group in per_case.items() if group["mean_latency_ms"][plan["candidate"]] > group["mean_latency_ms"][method]]
                   for method in METHODS if method != plan["candidate"]},
        "correctness": {"unique_cases": len(plan["cases"]), "candidate_checks": sum(len(report["checks"]) for report in reports),
                        "warmup_outputs": sum(len(report["warmups"]) for report in reports), "timed_outputs": sum(len(report["samples"]) for report in reports),
                        "natural_eos_cases": [key for key, row in reports[0]["references"].items() if row["eos"]],
                        "early_eos_cases": [key for key, row in reports[0]["references"].items() if row["early_eos"]]},
        "sessions": [{key: report[key] for key in ("session", "pid", "parent_pid", "started_at", "elapsed_seconds", "environment", "setup_ms",
                                                   "capture_and_warmup_ms", "peak_allocated_bytes", "target_parameters", "extra_learned_parameters", "resident_model_copies", "setup_scope")}
                     for report in reports],
        "elapsed_seconds": status["elapsed_seconds"], "gate_limits": limits,
        "evaluation_gates": gates if final else None,
        "evaluation_passed": all(gates.values()) if final else None,
        "release_complete": False,
        "verdict_scope": "Final quantitative evaluation only; design rationale, reproduction, public evidence and release review remain separate gates." if final else "Development pilot only; no promotion verdict or session uncertainty interval.",
    }


def analyze_directory(directory: Path) -> dict:
    def read(name, limit):
        path = directory / name
        if path.stat().st_size > limit:
            raise ValueError(f"{name} exceeds its size limit")
        return json.loads(path.read_text(encoding="utf-8"))

    plan = read("protocol.json", 2 * 1024**2)
    validate_archive_protocol(plan)
    names = [f"session-{index}.json" for index in range(plan["sessions"])]
    if {path.name for path in directory.glob("session-*.json")} != set(names):
        raise ValueError("study directory has missing or extra session files")
    status = read("status.json", 64 * 1024)
    result = analyze(plan, status, [read(name, 16 * 1024**2) for name in names])
    result["evidence_sha256"] = {name: file_hash(directory / name) for name in ["protocol.json", "status.json", *names]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed execution run directory")
    parser.add_argument("--out", type=Path, required=True, help="new analysis JSON file")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("analysis file already exists")
    try:
        result = analyze_directory(args.run)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    write_protocol(args.out, result)
    print(json.dumps({key: result[key] for key in ("split", "evaluation_passed", "correctness", "verdict_scope")}))


if __name__ == "__main__":
    main()
