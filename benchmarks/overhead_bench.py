"""Compare host-overhead changes against cached, fingerprinted HF outputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace

from benchmarks.execution_analysis import summarize_group
from benchmarks.execution_bench import (
    measure_session,
    run_process,
    run_study,
    validate_session,
    write_json,
)
from benchmarks.execution_protocol import (
    PACKAGES,
    digest,
    model_hashes,
    session_checks,
    source_hashes,
)
from dart.drafters.ngram import NGramDrafter

METHODS = ("prior_dart", "guard_only", "current_dart", "direct_greedy", "hf_greedy", "hf_lookup")
CONTROL_COMMIT = "feaad1ce9d5541831adaec40a0dbc6777c987151"
GATES = {"maximum_peak_allocated_bytes": 6 * 1024**3, "minimum_mean_latency_reduction": 0.05,
         "each_session_improves": True, "maximum_workload_regression": 0.05}


def prior_signature(target):
    return tuple((tensor.data_ptr(), tensor.device, tensor.dtype, tuple(tensor.shape), tensor.stride())
                 for tensor in (*target.model.parameters(), *target.model.buffers()))


class PriorNGram(NGramDrafter):
    def _continuations(self, committed, max_tokens, limit):
        seq_len = len(committed)
        scan_from = 0 if self.max_scan is None else max(0, seq_len - self.max_scan)
        found = []
        for n in range(min(self.max_ngram, seq_len - 1), self.min_ngram - 1, -1):
            pattern = committed[seq_len - n:]
            for start in range(seq_len - n - 1, scan_from - 1, -1):
                if committed[start:start + n] == pattern:
                    cont = committed[start + n:start + n + max_tokens]
                    if cont and cont not in found:
                        found.append(cont)
                        if len(found) >= limit:
                            return found
            if found:
                break
        return found


def make_protocol(reference_plan, reference_session, model):
    body = {key: value for key, value in reference_plan.items() if key != "protocol_hash"}
    if reference_plan.get("split") != "validation" or reference_plan.get("protocol_hash") != digest(body):
        raise ValueError("use a fingerprinted validation protocol")
    expected = {"device": "cuda", "dtype": "float32", "attention": "eager", "native_rms_norm": True,
                "reference_native_rms_norm": False, "threads": 1, "matmul_precision": "highest"}
    if any(reference_plan["settings"].get(key) != value for key, value in expected.items()):
        raise ValueError("cached reference execution settings changed")
    validate_session(reference_session, reference_plan, 0)
    if model_hashes(model) != reference_plan["model_hashes"]:
        raise ValueError("cached reference model files changed")
    if {name: version(name) for name in PACKAGES} != reference_plan["packages"]:
        raise ValueError("cached reference package versions changed")
    plan = {"schema": "dart-overhead-protocol-v1", "split": "validation", "context_limit": reference_plan["context_limit"],
            "cases": reference_plan["cases"], "timed_ids": reference_plan["timed_ids"], "methods": list(METHODS),
            "candidate": "current_dart", "control": "prior_dart", "control_commit": CONTROL_COMMIT,
            "baselines": ["direct_greedy", "hf_greedy", "hf_lookup"],
            "control_scope": "prior storage guard and suffix scan in the same executor; guard_only isolates storage traversal",
            "sessions": 3, "warmups": 1, "repeats": 2, "timeout_seconds": 120, "seed": 317,
            "model_hashes": reference_plan["model_hashes"], "packages": reference_plan["packages"],
            "source_hashes": source_hashes(), "reference_protocol_hash": reference_plan["protocol_hash"],
            "reference_session_hash": digest(reference_session),
            "cached_outputs": {name: row["token_ids"] for name, row in reference_session["references"].items()},
            "settings": reference_plan["settings"],
            "scope": "Development validation on previously exposed inputs; not fresh held-out confirmation or independent reproduction",
            "reference_scope": "Cached original-norm HF token IDs; reference latency records are retrieval overhead, not model latency",
            "memory_scope": "One model copy, replay workspace and request caches; not comparable to the earlier two-model harness",
            "gates": dict(GATES)}
    return {**plan, "protocol_hash": digest(plan)}


def read_protocol(path, *, check_source=True):
    plan = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in plan.items() if key != "protocol_hash"}
    if plan.get("schema") != "dart-overhead-protocol-v1" or plan.get("protocol_hash") != digest(body):
        raise ValueError("overhead protocol fingerprint changed")
    fixed = {"split": "validation", "sessions": 3, "warmups": 1, "repeats": 2, "timeout_seconds": 120,
             "methods": list(METHODS), "candidate": "current_dart", "control": "prior_dart",
             "control_commit": CONTROL_COMMIT, "gates": GATES}
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("overhead execution settings changed")
    if check_source and plan["source_hashes"] != source_hashes():
        raise ValueError("overhead source changed")
    return plan


def build_methods(graph, budgets, stack):
    from benchmarks.run_bench import build_hf_method, build_target_method
    from dart.runtime.engine import DartEngine
    from dart.runtime.types import DecodeConfig

    old_guard, new_guard = MethodType(prior_signature, graph), graph._signature
    old_signature, new_signature = old_guard(), new_guard()
    prior = stack.enter_context(DartEngine(graph, PriorNGram()))
    current = stack.enter_context(DartEngine(graph, NGramDrafter()))

    def wrap(operation, old=False):
        def run(prompt):
            graph._signature = old_guard if old else new_guard
            graph._model_signature = old_signature if old else new_signature
            return operation(prompt)
        return run

    methods = {}
    for budget in budgets:
        config = DecodeConfig(max_new_tokens=budget, draft_len=4)
        old = lambda prompt, config=config: prior.generate(prompt, config)
        new = lambda prompt, config=config: current.generate(prompt, config)
        methods[budget] = {"prior_dart": wrap(old, True), "guard_only": wrap(old), "current_dart": wrap(new),
                           "direct_greedy": wrap(build_target_method(graph, budget)),
                           "hf_greedy": wrap(build_hf_method(graph, budget)),
                           "hf_lookup": wrap(build_hf_method(graph, budget, prompt_lookup_num_tokens=4))}
    return methods


def run_session(plan, model, output, session):
    import torch

    from dart.models.replay import CudaGraphTarget
    from dart.presets import local_model_path, supra_config
    from dart.runtime.factory import build_target

    session_checks(plan, session)
    path = output / f"session-{session}.json"
    if path.exists():
        raise FileExistsError("session result already exists")
    model = local_model_path(model)
    if model_hashes(model) != plan["model_hashes"] or {name: version(name) for name in PACKAGES} != plan["packages"]:
        raise ValueError("cached reference model or packages changed")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "dart-execution-session-v1", "status": "running", "session": session,
              "protocol_hash": plan["protocol_hash"], "pid": os.getpid(), "parent_pid": os.getppid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "peak_allocated_bytes": 0, "reference_scope": plan["reference_scope"], "memory_scope": plan["memory_scope"],
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                              "device": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
                              "matmul_precision": torch.get_float32_matmul_precision()}}
    started = time.perf_counter()

    def progress(message):
        write_json(path, report)
        print(message, flush=True)

    try:
        base = build_target(supra_config(model, "cuda", native_rms_norm=True)["target"])
        with ExitStack() as stack:
            graph = CudaGraphTarget(base, capacity=plan["context_limit"], max_block_tokens=5)
            stack.callback(graph.close)
            budgets = sorted({case["max_new_tokens"] for case in plan["cases"]})
            methods = build_methods(graph, budgets, stack)
            outputs = {(case["max_new_tokens"], tuple(case["prompt_ids"])): plan["cached_outputs"][case["id"]]
                       for case in plan["cases"]}
            references = {budget: lambda prompt, budget=budget: SimpleNamespace(token_ids=outputs[budget, tuple(prompt)])
                          for budget in budgets}
            report.update(setup_ms=1000 * (time.perf_counter() - started), capture_ms=graph.setup_ms,
                          resident_model_copies=1, extra_learned_parameters=0)
            with torch.inference_mode():
                measure_session(plan, session, references, methods, report, synchronize=torch.cuda.synchronize,
                                memory=torch.cuda.max_memory_allocated, eos_ids=graph.eos_token_ids,
                                vocab_size=graph.model.config.vocab_size, progress=progress)
        if source_hashes() != plan["source_hashes"]:
            raise ValueError("source changed during measurement")
        report["status"] = "complete"
        validate_session(report, plan, session)
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, report)


def retryable_attempt(output, plan):
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    partial = json.loads((output / "session-2.json").read_text(encoding="utf-8"))
    if (status["status"] != "timeout" or status["protocol_hash"] != plan["protocol_hash"]
            or len(status["processes"]) != 3 or not status["processes"][2]["timed_out"]
            or any(p["returncode"] or p["timed_out"] for p in status["processes"][:2])
            or partial["protocol_hash"] != plan["protocol_hash"] or partial["status"] == "complete"
            or partial.get("samples")):
        raise ValueError("retry requires two completed processes and no timed samples from the third")
    for row in partial.get("checks", []) + partial.get("warmups", []):
        if row.get("match") is not True or row["token_ids"] != plan["cached_outputs"][row["case"]]:
            raise ValueError("partial attempt contains an output mismatch")
    return status, partial


def resume_study(model, output):
    plan = read_protocol(output / "protocol.json")
    retryable_attempt(output, plan)
    folder = output / "resume"
    folder.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    for key, name in (("CUDA_CACHE_PATH", "cuda-cache"), ("PYTORCH_KERNEL_CACHE_PATH", "torch-kernel-cache"),
                      ("TEMP", "tmp"), ("TMP", "tmp")):
        env[key] = str((output / name).resolve())
    command = [sys.executable, "-m", "benchmarks.overhead_bench", "--model", str(model.resolve()),
               "--protocol", str((output / "protocol.json").resolve()), "--out", str(folder.resolve()), "--session", "2"]
    process = run_process(command, folder / "session-2.log", 60, env)
    process["session"] = 2
    report_path = folder / "session-2.json"
    if report_path.exists():
        process["worker_pid"] = json.loads(report_path.read_text(encoding="utf-8"))["pid"]
    write_json(output / "extension.json", {"schema": "dart-overhead-extension-v1", "protocol_hash": plan["protocol_hash"],
                                          "session": 2, "timeout_seconds": 60, "process": process})
    if process["returncode"] or process["timed_out"]:
        raise RuntimeError("bounded retry failed; its original and partial records were preserved")
    write_json(output / "analysis.json", analyze(output))


def analyze(output):
    plan = read_protocol(output / "protocol.json", check_source=False)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    reports = [json.loads((output / f"session-{s}.json").read_text(encoding="utf-8")) for s in range(plan["sessions"])]
    elapsed = status["elapsed_seconds"]
    limit = plan["timeout_seconds"]
    extension_used, partial_matches = False, 0
    if (output / "extension.json").exists():
        status, partial = retryable_attempt(output, plan)
        extension = json.loads((output / "extension.json").read_text(encoding="utf-8"))
        if (extension.get("schema") != "dart-overhead-extension-v1" or extension["protocol_hash"] != plan["protocol_hash"]
                or extension["session"] != 2 or extension["timeout_seconds"] != 60
                or not 0 < extension["process"]["elapsed_seconds"] <= 60):
            raise ValueError("overhead retry scope changed")
        reports[2] = json.loads((output / "resume/session-2.json").read_text(encoding="utf-8"))
        status = {**status, "status": "complete", "processes": [*status["processes"][:2], extension["process"]]}
        elapsed += extension["process"]["elapsed_seconds"]
        limit += 60
        extension_used = True
        partial_matches = len(partial.get("checks", [])) + len(partial.get("warmups", []))
    if (status["status"] != "complete" or status["protocol_hash"] != plan["protocol_hash"]
            or len(status["processes"]) != plan["sessions"] or not 0 < elapsed <= limit):
        raise ValueError("overhead study is incomplete")
    for session, (report, process) in enumerate(zip(reports, status["processes"])):
        validate_session(report, plan, session)
        if (process["session"] != session or process["returncode"] != 0 or process["timed_out"]
                or process["worker_pid"] != report["pid"]):
            raise ValueError("overhead process failed or changed")
        if any(row["token_ids"] != plan["cached_outputs"][name] for name, row in report["references"].items()):
            raise ValueError("cached reference outputs changed")
        if (report["resident_model_copies"] != 1 or report["memory_scope"] != plan["memory_scope"]
                or report["environment"] != reports[0]["environment"]):
            raise ValueError("overhead environment or memory scope changed")
    aggregate = summarize_group(plan, reports, plan["timed_ids"])
    workloads = {workload: summarize_group(plan, reports, [case["id"] for case in plan["cases"] if case["workload"] == workload])
                 for workload in sorted({case["workload"] for case in plan["cases"]})}
    comparison = aggregate["candidate_comparisons"]["prior_dart"]
    gates = {"mean_latency": comparison["mean_latency_reduction"] >= plan["gates"]["minimum_mean_latency_reduction"],
             "every_session": all(ratio > 1 for ratio in comparison["session_speed_ratios"]),
             "workloads": all(group["mean_latency_ms"]["current_dart"] <= 1.05 * group["mean_latency_ms"]["prior_dart"]
                              for group in workloads.values())}
    return {"scope": plan["scope"], "protocol_hash": plan["protocol_hash"], "aggregate": aggregate, "workloads": workloads,
            "cases": {name: summarize_group(plan, reports, [name]) for name in plan["timed_ids"]},
            "full_output_matches": sum(len(r["checks"]) + len(r["warmups"]) + len(r["samples"]) for r in reports),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in reports), "memory_scope": plan["memory_scope"],
            "elapsed_seconds": elapsed, "original_limit_seconds": plan["timeout_seconds"],
            "extension_used": extension_used, "total_limit_seconds": limit, "partial_attempt_matches": partial_matches,
            "gates": gates, "development_passed": all(gates.values()),
            "confirmation_passed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--reference-session", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--analyze", action="store_true", help="reanalyze saved records without model inference")
    parser.add_argument("--resume", action="store_true", help="explicitly retry the untimed third process once, for at most 60 seconds")
    parser.add_argument("--session", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.analyze:
        write_json(args.out / "analysis.json", analyze(args.out))
        return
    if args.model is None:
        parser.error("--model is required for execution")
    if args.resume:
        resume_study(args.model, args.out)
        return
    if args.protocol is None:
        if args.reference_protocol is None or args.reference_session is None:
            parser.error("provide --protocol or both cached reference files")
        plan = make_protocol(json.loads(args.reference_protocol.read_text(encoding="utf-8")),
                             json.loads(args.reference_session.read_text(encoding="utf-8")), args.model)
        args.protocol = args.out.with_suffix(".protocol.json")
        if args.protocol.exists():
            raise FileExistsError("protocol already exists")
        args.protocol.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.protocol, plan)
    if args.session is None:
        run_study(args.protocol, args.model, args.out, protocol_reader=read_protocol, worker_module="benchmarks.overhead_bench")
        write_json(args.out / "analysis.json", analyze(args.out))
    else:
        run_session(read_protocol(args.protocol), args.model, args.out, args.session)


if __name__ == "__main__":
    main()
