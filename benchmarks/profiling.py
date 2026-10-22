from __future__ import annotations

import time
from contextlib import ExitStack
from functools import wraps
from statistics import mean
from unittest.mock import patch

from benchmarks.timing import Method, synchronize, tokens
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import GenerationOutput


def profile_methods(
    methods: dict[str, Method],
    prompts: list[list[int]],
    target: TargetModel,
    engines: dict[str, DartEngine] | None = None,
    reference: str = "hf_greedy",
    repeats: int = 2,
) -> dict:
    """Separate diagnostics; phase barriers add overhead to these requests."""
    engines = engines or {}
    if reference not in methods or not prompts or type(repeats) is not int or repeats < 1:
        raise ValueError("profiling needs a reference, prompts, and repeats >= 1")
    if any(engine.target is not target or engine.verifier_name != "block" for engine in engines.values()):
        raise ValueError("profiling requires block engines sharing the target")

    def stamp():
        synchronize(target.device)
        return time.perf_counter()

    expected = [list(tokens(methods[reference](prompt))) for prompt in prompts]
    samples = []
    for repeat in range(repeats):
        for index, prompt in enumerate(prompts):
            for name, method in methods.items():
                prefill = []
                phases = {"draft_ms": 0.0, "verify_ms": 0.0}

                def before_forward(module, args, marks=prefill):
                    if not marks:
                        marks.append(stamp())

                def after_forward(module, args, output, marks=prefill):
                    if len(marks) == 1:
                        marks.append(stamp())

                def timed(call, phase, totals=phases):
                    @wraps(call)
                    def run(*args, **kwargs):
                        start = stamp()
                        try:
                            return call(*args, **kwargs)
                        finally:
                            totals[phase] += (stamp() - start) * 1_000
                    return run

                with ExitStack() as stack:
                    stack.callback(target.model.register_forward_pre_hook(before_forward).remove)
                    stack.callback(target.model.register_forward_hook(after_forward).remove)
                    engine = engines.get(name)
                    if engine is not None:
                        stack.enter_context(patch.object(engine, "_propose", timed(engine._propose, "draft_ms")))
                        stack.enter_context(patch.object(engine, "verify", timed(engine.verify, "verify_ms")))
                    start = stamp()
                    output = method(prompt)
                    end = stamp()
                if tokens(output) != expected[index]:
                    raise ValueError(f"{name}: complete token mismatch on prompt {index}")
                if len(prefill) != 2:
                    raise ValueError(f"{name}: target prefill was not observed")
                if isinstance(output, GenerationOutput) and output.metadata.get("mode", "exact") != "exact":
                    raise ValueError("phase profiling supports exact greedy mode only")
                samples.append({
                    "method": name, "prompt_index": index, "repeat": repeat,
                    "tokens": len(tokens(output)), "output_hash": output_hash(tokens(output)),
                    "request_ms": (end - start) * 1_000,
                    "setup_ms": (prefill[0] - start) * 1_000,
                    "prefill_ms": (prefill[1] - prefill[0]) * 1_000,
                    "decode_ms": (end - prefill[1]) * 1_000,
                    **(phases if engine is not None else {key: None for key in phases}),
                })
    fields = ("request_ms", "setup_ms", "prefill_ms", "decode_ms", "draft_ms", "verify_ms")
    summary = {}
    for name in methods:
        rows = [row for row in samples if row["method"] == name]
        summary[name] = {
            key: mean(row[key] for row in rows) if rows[0][key] is not None else None
            for key in fields
        }
    return {
        "scope": "separate diagnostic requests with synchronized phase barriers; excluded from headline latency",
        "prefill_scope": "first target model forward, including cache construction inside the model",
        "decode_scope": "everything after that forward, including first-token selection and output decoding",
        "draft_verify_scope": "engine proposal and verification calls; subsets of decode time, null for baselines without an engine",
        "repeats": repeats, "samples": samples, "summary": summary,
    }
