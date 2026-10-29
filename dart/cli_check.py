"""Local synthetic correctness checks without a checkpoint or default output file."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from dart import __version__


def _output_summary(report, seed):
    total, failed = len(report.results), len(report.failures)
    print(f"Synthetic CPU check: {total - failed}/{total} comparisons passed (seed {seed}).")
    for row in report.failures[:5]:
        index = "" if row.failure.index is None else f" at token {row.failure.index}"
        print(f"Failed: {row.backend} / {row.case.name}: {row.failure.kind}{index}.")
    if failed > 5:
        print(f"{failed - 5} more failures; use --out to save the full report.")
    return {
        "schema": "dart-synthetic-check-v1", "scope": "tiny CPU eager output checks",
        "reference": "native HF greedy",
        "summary": {"cases": len({row.case.name for row in report.results}),
                    "comparisons": total, "passed": total - failed, "failed": failed},
        "results": [asdict(row) for row in report.results],
    }


def _state_summary(report, seed):
    total, failed = len(report.steps), sum(not step.passed for step in report.steps)
    print(f"Synthetic CPU state check: {total - failed}/{total} steps passed (seed {seed}).")
    failures = [(index, step.operation.request, item) for index, step in enumerate(report.steps, 1)
                for item in step.failures]
    for index, actor, item in failures[:5]:
        print(f"Failed: step {index} / {actor} -> {item.request}: {item.failure.kind}.")
    if len(failures) > 5:
        print(f"{len(failures) - 5} more failures; use --out to save the full report.")
    return {
        "schema": "dart-synthetic-state-check-v1", "scope": "tiny CPU eager request-state checks",
        "reference": "native HF forward and greedy",
        "summary": {"steps": total, "passed": total - failed, "failed": failed,
                    "cache_checks": sum(len(step.checked_requests) for step in report.steps)},
        "steps": [asdict(step) for step in report.steps],
    }


def run_synthetic(args) -> int:
    if args.device != "cpu":
        raise ValueError("synthetic checks currently support CPU only")
    if args.corpus is not None or args.split != "validation" or args.native_rms_norm or args.norm_control:
        raise ValueError("synthetic checks do not use corpus, split or norm-control options")
    if not 0 <= args.seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    output = Path(args.out) if args.out else None
    if output is not None and os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")

    import torch

    from dart.verification.synthetic import (
        run_synthetic_check,
        run_synthetic_state_check,
        synthetic_model_config,
    )

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        check = run_synthetic_state_check if args.state else run_synthetic_check
        report = check(args.seed)
    finally:
        torch.set_num_threads(threads)
    summary = _state_summary(report, args.seed) if args.state else _output_summary(report, args.seed)
    if output is not None:
        data = {
            **summary, "status": "passed" if report.passed else "failed", "seed": args.seed,
            "model": {"architecture": "LlamaForCausalLM", "config": synthetic_model_config(),
                      "device": "cpu", "dtype": "float32", "threads": 1},
            "versions": {"dart": __version__, "python": platform.python_version(),
                         "torch": version("torch"), "transformers": version("transformers")},
            "reproduce": ["dart", "check", "--synthetic", *(["--state"] if args.state else []),
                          "--seed", str(args.seed)],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as destination:
            json.dump(data, destination, indent=2)
            destination.write("\n")
        print(f"Wrote {output}")
    return 0 if report.passed else 1
