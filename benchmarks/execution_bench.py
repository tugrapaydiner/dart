"""Run a frozen execution protocol in separately launched CUDA processes."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.execution_protocol import (
    METHODS,
    SCHEMA,
    digest,
    schedule,
    selected_ids,
    session_checks,
    source_hashes,
    validate_protocol,
)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_protocol(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("protocol exceeds 2 MiB")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("unsupported execution protocol")
    body = {key: value for key, value in plan.items() if key != "protocol_hash"}
    if plan.get("protocol_hash") != digest(body):
        raise ValueError("protocol fingerprint changed")
    checks, timed = selected_ids(plan.get("split"))
    final = plan["split"] == "test"
    expected = {"sessions": 3 if final else 1, "timeout_seconds": 3600 if final else 300,
                "repeats": 3 if final else 2, "warmups": 1, "methods": list(METHODS), "timed_ids": timed}
    if digest({key: plan.get(key) for key in expected}) != digest(expected):
        raise ValueError("protocol execution settings changed")
    if [case["id"] for case in plan["cases"]] != checks or plan["source_hashes"] != source_hashes():
        raise ValueError("protocol cases or source changed")
    return plan


def measure_session(plan, session, references, methods, report, *, synchronize, memory, eos_ids, vocab_size, progress):
    cases = {case["id"]: case for case in plan["cases"]}
    report.update(references={}, checks=[], warmups=[], samples=[])
    report.update(eos_token_ids=sorted(eos_ids), vocab_size=vocab_size)

    def observe(method, case):
        prompt = list(case["prompt_ids"])
        report["peak_allocated_bytes"] = max(report["peak_allocated_bytes"], memory())
        synchronize()
        started = time.perf_counter()
        output = method(prompt)
        synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        report["peak_allocated_bytes"] = max(report["peak_allocated_bytes"], memory())
        if prompt != case["prompt_ids"]:
            raise ValueError("a generation method changed its input prompt")
        ids = list(output.token_ids)
        if not ids or len(ids) > case["max_new_tokens"] or any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError("generation returned invalid token IDs or length")
        if any(token in eos_ids for token in ids[:-1]):
            raise ValueError("generation continued after EOS")
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("request latency must be positive and finite")
        if report["peak_allocated_bytes"] > plan["gates"]["maximum_peak_allocated_bytes"]:
            raise ValueError("execution exceeded the protocol GPU memory limit")
        return {"token_ids": ids, "output_hash": digest(ids), "latency_ms": elapsed,
                "eos": ids[-1] in eos_ids, "early_eos": len(ids) < case["max_new_tokens"] and ids[-1] in eos_ids}

    for name in session_checks(plan, session):
        case = cases[name]
        report["references"][name] = observe(references[case["max_new_tokens"]], case)
        progress(f"Reference {len(report['references'])}/{len(session_checks(plan, session))}")
    for name in session_checks(plan, session):
        case = cases[name]
        row = {"case": name, **observe(methods[case["max_new_tokens"]][plan["candidate"]], case)}
        row["match"] = row["token_ids"] == report["references"][name]["token_ids"]
        report["checks"].append(row)
        progress(f"Checked {len(report['checks'])}/{len(report['references'])}")
        if not row["match"]:
            raise ValueError(f"{name}: candidate output differs from the full reference")
    for call in schedule(plan, session):
        case = cases[call["case"]]
        row = {**call, **observe(methods[case["max_new_tokens"]][call["method"]], case)}
        row["match"] = row["token_ids"] == report["references"][call["case"]]["token_ids"]
        destination = report["samples"] if call["phase"] == "timed" else report["warmups"]
        destination.append(row)
        if not row["match"] or call["position"] == len(plan["methods"]) - 1:
            progress(f"{call['phase']}: {call['case']}, repeat {call['repeat'] + 1}")
        if not row["match"]:
            raise ValueError(f"{call['method']}: complete token mismatch on {call['case']}")


def validate_session(report: dict, plan: dict, session: int) -> None:
    if (report.get("schema") != "dart-execution-session-v1" or report.get("status") != "complete"
            or report.get("protocol_hash") != plan["protocol_hash"] or type(report.get("session")) is not int
            or report["session"] != session):
        raise ValueError("session is incomplete or belongs to another protocol")
    expected_ids = session_checks(plan, session)
    if set(report["references"]) != set(expected_ids) or [row["case"] for row in report["checks"]] != expected_ids:
        raise ValueError("session correctness coverage is incomplete")
    expected = schedule(plan, session)
    for phase, name in (("warmup", "warmups"), ("timed", "samples")):
        expected_calls = [call for call in expected if call["phase"] == phase]
        actual = report[name]
        if digest([{key: row.get(key) for key in call} for row, call in zip(actual, expected_calls)]) != digest(expected_calls) or len(actual) != len(expected_calls):
            raise ValueError("session measurement matrix is incomplete or reordered")
    for row in [*report["checks"], *report["warmups"], *report["samples"]]:
        if row.get("match") is not True or row["token_ids"] != report["references"][row["case"]]["token_ids"]:
            raise ValueError("session contains an output mismatch")
    cases = {case["id"]: case for case in plan["cases"]}
    all_outputs = [(name, row) for name, row in report["references"].items()]
    all_outputs += [(row["case"], row) for row in [*report["checks"], *report["warmups"], *report["samples"]]]
    eos_ids = report["eos_token_ids"]
    for name, row in all_outputs:
        ids = row["token_ids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= cases[name]["max_new_tokens"]
                or any(type(token) is not int or not 0 <= token < report["vocab_size"] for token in ids)
                or any(token in eos_ids for token in ids[:-1])):
            raise ValueError("session contains invalid token IDs, lengths or EOS behavior")
        if (row["output_hash"] != digest(ids) or type(row["latency_ms"]) not in (int, float)
                or not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0):
            raise ValueError("session contains invalid hashes or latencies")
        if (row.get("eos") is not (ids[-1] in eos_ids)
                or row.get("early_eos") is not (len(ids) < cases[name]["max_new_tokens"] and ids[-1] in eos_ids)):
            raise ValueError("session EOS metadata disagrees with its tokens")
    if (type(report["peak_allocated_bytes"]) is not int
            or not 0 <= report["peak_allocated_bytes"] <= plan["gates"]["maximum_peak_allocated_bytes"]):
        raise ValueError("session GPU memory limit exceeded")


def run_session(plan: dict, model: Path, output: Path, session: int) -> None:
    path = output / f"session-{session}.json"
    if path.exists():
        raise FileExistsError("session result already exists")
    report = {"schema": "dart-execution-session-v1", "status": "running", "session": session,
              "protocol_hash": plan["protocol_hash"], "pid": os.getpid(), "parent_pid": os.getppid(), "peak_allocated_bytes": 0,
              "started_at": datetime.now(timezone.utc).isoformat(), "output_hash_schema": "canonical-json-sha256"}
    started = time.perf_counter()

    def progress(message):
        write_json(path, report)
        print(message, flush=True)

    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from benchmarks.run_bench import build_hf_method, build_target_method
        from benchmarks.verification_workloads import load_frozen
        from dart.drafters.ngram import NGramDrafter
        from dart.models.replay import CudaGraphTarget
        from dart.presets import local_model_path, supra_config
        from dart.runtime.engine import DartEngine
        from dart.runtime.factory import build_target
        from dart.runtime.types import DecodeConfig

        model = local_model_path(model)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        rows = load_frozen(tokenizer, config.max_position_embeddings)
        validate_protocol(plan, rows, tokenizer, config.max_position_embeddings, model)
        session_checks(plan, session)
        if not torch.cuda.is_available():
            raise ValueError("execution benchmarks require CUDA")
        torch.set_num_threads(1)
        torch.manual_seed(plan["seed"])
        torch.set_float32_matmul_precision("highest")
        report["environment"] = {"python": platform.python_version(), "system": platform.system(),
                                  "torch": torch.__version__, "cuda": torch.version.cuda,
                                  "device": torch.cuda.get_device_name(), "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                                  "threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision()}
        setup = report["setup_ms"] = {}

        def prepare(name, factory):
            before = time.perf_counter()
            value = factory()
            torch.cuda.synchronize()
            setup[name] = 1000 * (time.perf_counter() - before)
            report["peak_allocated_bytes"] = max(report["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
            progress(f"Prepared {name}")
            return value

        reference = prepare("reference_model", lambda: build_target(supra_config(model, "cuda")["target"]))
        base = prepare("native_model", lambda: build_target(supra_config(model, "cuda", native_rms_norm=True)["target"]))
        with ExitStack() as stack:
            graph = prepare("replay", lambda: CudaGraphTarget(base, max_block_tokens=5, capacity=plan["context_limit"]))
            stack.callback(graph.close)
            static = prepare("static", lambda: CudaGraphTarget(base, max_block_tokens=5, capacity=plan["context_limit"], replay=False))
            stack.callback(static.close)
            report["capture_and_warmup_ms"] = graph.setup_ms
            report["target_parameters"] = sum(parameter.numel() for parameter in base.model.parameters())
            report["extra_learned_parameters"] = 0
            report["resident_model_copies"] = 2
            report["memory_scope"] = plan["memory_scope"]
            report["setup_scope"] = "model loading and backend preparation measured separately; request timings are warmed, not cold-start latency"
            engines = {name: stack.enter_context(DartEngine(target, NGramDrafter(max_ngram=4, min_ngram=1)))
                       for name, target in (("current_dart", base), ("static_dart", static), ("replay_dart", graph))}
            references, methods = {}, {}
            for budget in sorted({case["max_new_tokens"] for case in plan["cases"]}):
                decode = DecodeConfig(max_new_tokens=budget, draft_len=4)
                references[budget] = build_hf_method(reference, budget)
                methods[budget] = {name: (lambda prompt, engine=engine, decode=decode: engine.generate(prompt, decode))
                                   for name, engine in engines.items()}
                methods[budget].update(replay_greedy=build_target_method(graph, budget),
                                       replay_hf_greedy=build_hf_method(graph, budget),
                                       replay_hf_lookup=build_hf_method(graph, budget, prompt_lookup_num_tokens=4))
            with torch.inference_mode():
                measure_session(plan, session, references, methods, report, synchronize=torch.cuda.synchronize,
                                memory=torch.cuda.max_memory_allocated, eos_ids=reference.eos_token_ids,
                                vocab_size=reference.model.config.vocab_size, progress=progress)
            validate_protocol(plan, rows, tokenizer, config.max_position_embeddings, model)
        report["status"] = "complete"
        validate_session(report, plan, session)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, report)


def run_process(command: list[str], log: Path, timeout: float, env: dict) -> dict:
    started = time.perf_counter()
    with log.open("x", encoding="utf-8") as stream, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env) as process:
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            timed_out = True
    return {"pid": process.pid, "returncode": code, "timed_out": timed_out,
            "elapsed_seconds": time.perf_counter() - started}


def run_study(plan_path: Path, model: Path, output: Path) -> dict:
    started = time.perf_counter()
    plan = read_protocol(plan_path)
    model = model.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    for key, folder in (("CUDA_CACHE_PATH", "cuda-cache"), ("PYTORCH_KERNEL_CACHE_PATH", "torch-kernel-cache"), ("TEMP", "tmp"), ("TMP", "tmp")):
        directory = output / folder
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    result = {"schema": "dart-execution-run-v1", "status": "running", "protocol_hash": plan["protocol_hash"],
              "timeout_seconds": plan["timeout_seconds"], "processes": []}
    try:
        frozen_references = None
        for session in range(plan["sessions"]):
            remaining = plan["timeout_seconds"] - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("execution study reached its total time limit")
            print(f"Session {session + 1}/{plan['sessions']}; {remaining:.0f}s left", flush=True)
            command = [sys.executable, "-m", "benchmarks.execution_bench", "--model", str(model),
                       "--protocol", str(output / "protocol.json"), "--out", str(output), "--session", str(session)]
            process = run_process(command, output / f"session-{session}.log", remaining, env)
            result["processes"].append({"session": session, **process})
            write_json(output / "status.json", result)
            if process["timed_out"]:
                raise TimeoutError("execution study reached its total time limit")
            if process["returncode"] != 0:
                raise RuntimeError(f"session {session} failed; see its saved log and partial results")
            report = json.loads((output / f"session-{session}.json").read_text(encoding="utf-8"))
            validate_session(report, plan, session)
            if type(report.get("pid")) is not int or report["pid"] < 1:
                raise ValueError("session worker PID is invalid")
            if report["pid"] != process["pid"] and (
                type(report.get("parent_pid")) is not int or report["parent_pid"] != process["pid"]
            ):  # Windows venvs may start a child interpreter.
                raise ValueError("session process identity changed")
            result["processes"][-1]["worker_pid"] = report["pid"]
            references = {name: row["token_ids"] for name, row in report["references"].items()}
            if frozen_references is None:
                frozen_references = references
            elif any(tokens != frozen_references[name] for name, tokens in references.items()):
                raise ValueError("reference outputs changed between processes")
        read_protocol(output / "protocol.json")
        if time.perf_counter() - started > plan["timeout_seconds"]:
            raise TimeoutError("execution study reached its total time limit")
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "timeout" if isinstance(error, TimeoutError) else "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new results directory")
    parser.add_argument("--session", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.session is None:
        run_study(args.protocol, args.model, args.out)
    else:
        run_session(read_protocol(args.protocol), args.model, args.out, args.session)


if __name__ == "__main__":
    main()
