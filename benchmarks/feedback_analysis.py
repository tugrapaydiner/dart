"""Analyze the frozen feedback study, including its non-repetitive target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.execution_analysis import summarize_group, validate_study
from benchmarks.execution_protocol import WORKLOADS, file_hash, write_protocol
from benchmarks.feedback_protocol import validate_archive


def analyze(plan, status, reports):
    validate_study(plan, status, reports, protocol_validator=validate_archive,
                   setup_names=("reference_model", "native_model", "replay"))
    cases = {case["id"]: case for case in plan["cases"]}
    ids = plan["timed_ids"]
    aggregate = summarize_group(plan, reports, ids)
    nonrepetitive = summarize_group(plan, reports, [name for name in ids if cases[name]["workload"] != "repetition"])
    workloads = {workload: summarize_group(plan, reports, [name for name in ids if cases[name]["workload"] == workload])
                 for workload in WORKLOADS}
    budgets = {str(budget): summarize_group(plan, reports, [name for name in ids if cases[name]["max_new_tokens"] == budget])
               for budget in sorted({cases[name]["max_new_tokens"] for name in ids})}
    per_case = {name: {**summarize_group(plan, reports, [name]), "workload": cases[name]["workload"],
                       "prompt_tokens": len(cases[name]["prompt_ids"]), "max_new_tokens": cases[name]["max_new_tokens"],
                       "output_tokens": len(reports[0]["references"][name]["token_ids"])} for name in ids}
    candidate = plan["candidate"]
    control = aggregate["candidate_comparisons"][plan["control"]]
    greedy = nonrepetitive["candidate_comparisons"]["replay_greedy"]
    limits = plan["gates"]

    def interval_passes(comparison):
        interval = comparison["session_ratio_ci95"]
        return interval is not None and interval[0] > limits["ratio_ci95_lower_exceeds"]

    gates = {
        "entropy_latency_reduction": control["mean_latency_reduction"] >= limits["minimum_mean_latency_reduction"],
        "entropy_interval": interval_passes(control),
        "entropy_each_session": all(value > 1 for value in control["session_speed_ratios"]),
        "nonrepetitive_latency_reduction": greedy["mean_latency_reduction"] >= limits["minimum_nonrepetitive_latency_reduction"],
        "nonrepetitive_interval": interval_passes(greedy),
        "nonrepetitive_each_session": all(value > 1 for value in greedy["session_speed_ratios"]),
        "nonrepetitive_hf_baselines": all(nonrepetitive["mean_latency_ms"][candidate] < nonrepetitive["mean_latency_ms"][name]
                                         for name in ("replay_hf_greedy", "replay_hf_lookup")),
        "nonrepetitive_workload_regression": all(group["mean_latency_ms"][candidate] <= group["mean_latency_ms"]["replay_greedy"]
                                                * (1 + limits["maximum_nonrepetitive_workload_regression"])
                                                for name, group in workloads.items() if name != "repetition"),
        "memory": all(report["peak_allocated_bytes"] <= limits["maximum_peak_allocated_bytes"] for report in reports),
        "extra_learned_parameters": all(report["extra_learned_parameters"] <= limits["maximum_extra_learned_parameters"] for report in reports),
        "complete_outputs": True, "complete_protocol": True,
    }
    savings = {name: len(group["cases"]) * (group["mean_latency_ms"]["replay_greedy"] - group["mean_latency_ms"][candidate])
               for name, group in workloads.items()}
    total_saving = sum(savings.values())
    return {
        "schema": "dart-feedback-analysis-v1", "protocol_hash": plan["protocol_hash"], "split": plan["split"],
        "provenance": {key: plan[key] for key in ("control_commit", "corpus_hash", "tokenizer_hash", "model_hashes", "source_hashes", "packages", "settings")},
        "timing_scope": plan["timing_scope"], "memory_scope": plan["memory_scope"], "uncertainty": plan["uncertainty"],
        "uncertainty_limits": "Three processes on one host and fixed inputs; intervals are descriptive repeatability evidence. Workload and case intervals have no multiplicity correction. No independent reproduction or production-population claim.",
        "aggregate": aggregate, "nonrepetitive": nonrepetitive, "workloads": workloads, "budgets": budgets, "cases": per_case,
        "savings_concentration": {"repetition_fraction": savings["repetition"] / total_saving if total_saving > 0 else None,
                                  "scope": "share of equal-case total time savings against direct replay greedy; negative workload savings are retained"},
        "slower_than_greedy": [name for name, group in per_case.items() if group["mean_latency_ms"][candidate] > group["mean_latency_ms"]["replay_greedy"]],
        "full_output_matches": sum(len(report[key]) for report in reports for key in ("checks", "warmups", "samples")),
        "unique_cases": len(cases), "natural_eos_cases": sum(row["eos"] for row in reports[0]["references"].values()),
        "peak_allocated_bytes": max(report["peak_allocated_bytes"] for report in reports),
        "elapsed_seconds": status["elapsed_seconds"], "gates": gates,
        "confirmation_passed": plan["split"] == "test" and all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new analysis JSON file")
    args = parser.parse_args()
    paths = {"protocol": args.run / "protocol.json", "status": args.run / "status.json"}
    plan, status = (json.loads(paths[key].read_text(encoding="utf-8")) for key in ("protocol", "status"))
    validate_archive(plan)
    reports = []
    for session in range(plan["sessions"]):
        path = args.run / f"session-{session}.json"
        paths[f"session-{session}"] = path
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    result = analyze(plan, status, reports)
    result["artifact_hashes"] = {path.name: file_hash(path) for path in paths.values()}
    write_protocol(args.out, result)
    print(json.dumps({"confirmation_passed": result["confirmation_passed"], "gates": result["gates"],
                      "full_output_matches": result["full_output_matches"]}))


if __name__ == "__main__":
    main()
