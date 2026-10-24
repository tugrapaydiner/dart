from __future__ import annotations

import argparse
import cProfile
import json
import platform
import pstats
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch
import transformers

from benchmarks.profiling import profile_methods
from benchmarks.run_bench import build_hf_method
from benchmarks.timing import synchronize
from benchmarks.workloads import load_frozen
from dart.data.corpus import encode_prompt
from dart.data.teacher import target_fingerprint
from dart.presets import supra_config
from dart.runtime.factory import build_engine
from dart.runtime.metrics import output_hash


def trace_decode(engine, prompt, config, trace_path: str | Path) -> dict:
    """One warm request with operator tracing; these times are diagnostic only."""
    config.validate()
    if config.mode != "exact" or engine.verifier_name != "block":
        raise ValueError("decode tracing requires exact mode and the block verifier")
    if not 1 <= config.max_new_tokens <= 32:
        raise ValueError("trace output budget must be from 1 to 32 tokens")
    trace_path = Path(trace_path)
    if trace_path.exists():
        raise FileExistsError(f"trace already exists: {trace_path}")
    target = engine.target
    target.validate_generation(len(prompt), config.max_new_tokens)
    reference = build_hf_method(target, config.max_new_tokens, record_hidden=False)
    expected = reference(prompt).token_ids
    generate = lambda ids: engine.generate(ids, config)
    if generate(prompt).token_ids != expected:
        raise ValueError("complete token mismatch during trace warmup")
    phases = profile_methods(
        {"hf_greedy": reference, "dart": generate}, [prompt], target, {"dart": engine}, repeats=1,
    )

    def labelled(call, name):
        @wraps(call)
        def run(*args, **kwargs):
            with torch.profiler.record_function(name):
                return call(*args, **kwargs)
        return run

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(target.device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(target.device)
    with ExitStack() as stack:
        for obj, attr, label in (
            (target, "prefill", "dart.prefill"),
            (engine, "_propose", "dart.draft"),
            (engine, "verify", "dart.verify"),
        ):
            stack.enter_context(patch.object(obj, attr, labelled(getattr(obj, attr), label)))
        with torch.profiler.profile(activities=activities, profile_memory=True) as profiler:
            start = time.perf_counter()
            with torch.profiler.record_function("dart.request"):
                output = generate(prompt)
                synchronize(target.device)
            request_ms = (time.perf_counter() - start) * 1_000
    if output.token_ids != expected:
        raise ValueError("complete token mismatch during operator tracing")
    with cProfile.Profile() as python_profiler:
        python_output = generate(prompt)
        synchronize(target.device)
    if python_output.token_ids != expected:
        raise ValueError("complete token mismatch during Python profiling")
    python_calls = [{
        "file": Path(location[0]).name, "line": location[1], "name": location[2],
        "calls": values[1], "self_ms": values[2] * 1_000, "total_ms": values[3] * 1_000,
    } for location, values in pstats.Stats(python_profiler).stats.items()]
    operations = [{
        "name": event.key, "calls": event.count,
        "self_cpu_ms": event.self_cpu_time_total / 1_000,
        "self_device_ms": event.self_device_time_total / 1_000,
        "cpu_total_ms": event.cpu_time_total / 1_000,
        "self_cpu_bytes": event.self_cpu_memory_usage,
        "self_device_bytes": event.self_device_memory_usage,
    } for event in profiler.key_averages()]
    gpu_events = sum(event.device_type == torch.autograd.DeviceType.CUDA for event in profiler.events())
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    return {
        "schema": "dart-decode-profile-v1",
        "scope": "one warmed, instrumented request; phase diagnostics are separate; not headline latency",
        "time_scope": "CPU and device activity can overlap; total and self times must not be added together",
        "memory_scope": "operator allocation deltas, including frees; not peak memory or retained storage",
        "python_scope": "a separate warmed request; host elapsed time includes device waits and profiler overhead",
        "prompt_tokens": len(prompt), "decode": asdict(config),
        "generated_tokens": len(output.token_ids), "output_hash": output_hash(output.token_ids),
        "finished_eos": output.metadata["finished_eos"], "rounds": len(output.rounds),
        "request_ms": request_ms, "gpu_events": gpu_events,
        "gpu_timeline_available": gpu_events > 0, "trace_bytes": trace_path.stat().st_size,
        "phases": phases,
        "operations": sorted(operations, key=lambda row: row["self_cpu_ms"], reverse=True),
        "python_calls": sorted(python_calls, key=lambda row: row["self_ms"], reverse=True),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture a bounded decode trace from a frozen pilot or validation prompt.")
    parser.add_argument("--model", required=True, help="existing local Supra-50M directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--case", default="v03-train-code-00")
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--out", required=True, help="new JSON report; a .trace.json file is saved beside it")
    args = parser.parse_args(argv)
    output = Path(args.out)
    trace = output.with_suffix(".trace.json")
    if not 1 <= args.max_new <= 32:
        parser.error("max-new must be from 1 to 32")
    if output.exists() or trace.exists():
        parser.error("report and trace paths must be new")
    with build_engine(supra_config(args.model, args.device)) as engine:
        records = load_frozen(engine.target.tokenizer, engine.target.model.config.max_position_embeddings)
        selected = [row for row in records if row.id == args.case and row.split != "test"]
        if len(selected) != 1:
            parser.error("case must identify one frozen pilot or validation prompt")
        row = selected[0]
        prompt = encode_prompt(row, engine.target.tokenizer, engine.target.model.config.max_position_embeddings)
        config = replace(engine.default_decode_config, max_new_tokens=args.max_new)
        report = trace_decode(engine, prompt, config, trace)
        report.update({
            "prompt_id": row.id, "split": row.split, "target": target_fingerprint(engine.target),
            "environment": {
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda": torch.version.cuda,
                "threads": torch.get_num_threads(),
                "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
            },
        })
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(f"Wrote {output} and {trace}; GPU timeline available: {report['gpu_timeline_available']}")


if __name__ == "__main__":
    main()
