"""Run a frozen feedback protocol in separate CUDA processes."""

from __future__ import annotations

import argparse
import os
import platform
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.execution_bench import (
    measure_session,
    run_study,
    validate_session,
    write_json,
)
from benchmarks.execution_protocol import session_checks
from benchmarks.feedback_protocol import read_protocol, validate_protocol


def build_methods(graph, budgets, stack):
    from benchmarks.run_bench import build_hf_method, build_target_method
    from dart.drafters.base import NullDrafter
    from dart.drafters.ngram import NGramDrafter
    from dart.runtime.engine import DartEngine
    from dart.runtime.scheduler import SchedulerConfig
    from dart.runtime.time_scheduler import TimeScheduler
    from dart.runtime.types import DecodeConfig

    engines = {}
    for name in ("entropy_on4", "entropy_off4", "entropy_off2"):
        drafter = NGramDrafter(max_ngram=4, min_ngram=1)
        drafter.requires_entropy = name == "entropy_on4"
        engines[name] = stack.enter_context(DartEngine(graph, drafter))
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=["target_only", "ngram"], draft_len=4),
                              retry_interval=16, warmup_rounds=2, min_time_saving=0.05)
    engines["time_scheduler"] = stack.enter_context(DartEngine(
        graph, lanes={"target_only": NullDrafter(), "ngram": NGramDrafter()}, scheduler=scheduler,
    ))
    methods = {}
    for budget in budgets:
        methods[budget] = {}
        for name, engine in engines.items():
            config = DecodeConfig(max_new_tokens=budget, draft_len=2 if name == "entropy_off2" else 4)
            methods[budget][name] = lambda prompt, engine=engine, config=config: engine.generate(prompt, config)
        methods[budget].update(replay_greedy=build_target_method(graph, budget),
                               replay_hf_greedy=build_hf_method(graph, budget),
                               replay_hf_lookup=build_hf_method(graph, budget, prompt_lookup_num_tokens=4))
    return methods


def run_session(plan, model, output, session):
    session_checks(plan, session)
    path = output / f"session-{session}.json"
    if path.exists():
        raise FileExistsError("session result already exists")
    report = {"schema": "dart-execution-session-v1", "status": "running", "session": session,
              "protocol_hash": plan["protocol_hash"], "pid": os.getpid(), "parent_pid": os.getppid(),
              "started_at": datetime.now(timezone.utc).isoformat(), "peak_allocated_bytes": 0,
              "output_hash_schema": "canonical-json-sha256"}
    started = time.perf_counter()

    def progress(message):
        write_json(path, report)
        print(message, flush=True)

    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from benchmarks.run_bench import build_hf_method
        from dart.models.replay import CudaGraphTarget
        from dart.presets import local_model_path, supra_config
        from dart.runtime.factory import build_target

        model = local_model_path(model)
        config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        validate_protocol(plan, tokenizer, config.max_position_embeddings, model)
        if not torch.cuda.is_available():
            raise ValueError("feedback benchmarks require CUDA")
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
            report.update(capture_and_warmup_ms=graph.setup_ms,
                          target_parameters=sum(parameter.numel() for parameter in base.model.parameters()),
                          extra_learned_parameters=0, resident_model_copies=2, memory_scope=plan["memory_scope"],
                          setup_scope="model loading and replay preparation excluded from warmed request timings")
            budgets = sorted({case["max_new_tokens"] for case in plan["cases"]})
            references = {budget: build_hf_method(reference, budget) for budget in budgets}
            methods = build_methods(graph, budgets, stack)
            with torch.inference_mode():
                measure_session(plan, session, references, methods, report, synchronize=torch.cuda.synchronize,
                                memory=torch.cuda.max_memory_allocated, eos_ids=reference.eos_token_ids,
                                vocab_size=reference.model.config.vocab_size, progress=progress)
            validate_protocol(plan, tokenizer, config.max_position_embeddings, model)
        report["status"] = "complete"
        validate_session(report, plan, session)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new results directory")
    parser.add_argument("--session", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.session is None:
        run_study(args.protocol, args.model, args.out, protocol_reader=read_protocol, worker_module="benchmarks.feedback_bench")
    else:
        run_session(read_protocol(args.protocol), args.model, args.out, args.session)


if __name__ == "__main__":
    main()
