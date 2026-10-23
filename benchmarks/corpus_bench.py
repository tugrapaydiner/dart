from __future__ import annotations

from dataclasses import replace

from benchmarks.corpus_eval import evaluate_corpus
from benchmarks.headroom import BASELINES
from benchmarks.profiling import profile_methods
from benchmarks.run_bench import (
    build_hf_method,
    build_target_method,
    with_hidden_recording,
)
from benchmarks.timing import measure_methods, summarize_samples
from dart.data.corpus import Prompt, encode_prompt
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig


def select_records(records: list[Prompt], per_workload: int) -> list[Prompt]:
    if type(per_workload) is not int or not 1 <= per_workload <= 32:
        raise ValueError("per_workload must be an integer from 1 to 32")
    selected = []
    for workload in sorted({row.workload for row in records}):
        group = sorted((row for row in records if row.workload == workload), key=lambda row: row.id)
        if len(group) < per_workload:
            raise ValueError(f"{workload}: fewer than {per_workload} prompts available")
        selected.extend(group[:per_workload])
    if not selected:
        raise ValueError("benchmark corpus must not be empty")
    return sorted(selected, key=lambda row: row.id)


def _summarize(samples: list[dict], seed: int) -> dict:
    timing = summarize_samples(samples, "hf_greedy", seed)
    fastest = min(BASELINES, key=lambda name: timing[name]["mean_ms"])
    comparison = summarize_samples(samples, fastest, seed)
    return {
        "timing": timing, "fastest_baseline": fastest,
        "vs_fastest": {name: {key: row[key] for key in ("speedup", "ci95_speedup")}
                       for name, row in comparison.items()},
    }


def benchmark_corpus(
    engine: DartEngine, records: list[Prompt], config: DecodeConfig | None = None, *,
    warmup: int = 2, repeats: int = 10, seed: int = 17, progress=None,
) -> dict:
    """Repeated comparisons with separate correctness and phase diagnostics."""
    if type(warmup) is not int or type(repeats) is not int or warmup < 1 or repeats < 2:
        raise ValueError("use warmup >= 1 and repeats >= 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if engine.verifier_name != "block":
        raise ValueError("benchmark phase profiling requires the block verifier")
    target = engine.target
    config = config or engine.default_decode_config or DecodeConfig()
    checks = evaluate_corpus(engine, records, config)
    if progress:
        progress("Correctness checks passed; starting warm-up and repeated timing")
    prompts = [encode_prompt(row, target.tokenizer, target.model.config.max_position_embeddings) for row in records]
    keys = [tuple(prompt) for prompt in prompts]
    configs = {key: replace(config, max_new_tokens=row.max_new_tokens, prompt_kind=output["prompt_kind"])
               for key, row, output in zip(keys, records, checks["outputs"])}
    hf = {key: build_hf_method(target, row.max_new_tokens, record_hidden=False) for key, row in zip(keys, records)}
    direct = {key: build_target_method(target, row.max_new_tokens) for key, row in zip(keys, records)}
    with DartEngine(target) as baseline:
        methods = {
            "hf_greedy": lambda prompt: hf[tuple(prompt)](prompt),
            "target_greedy": lambda prompt: direct[tuple(prompt)](prompt),
            "target_only": lambda prompt: baseline.generate(prompt, configs[tuple(prompt)]),
            "dart": lambda prompt: engine.generate(prompt, configs[tuple(prompt)]),
        }
        methods = {name: with_hidden_recording(target, target.record_hidden if name == "dart" else False, method)
                   for name, method in methods.items()}
        timing = measure_methods(methods, prompts, target.device, warmup=warmup, repeats=repeats, seed=seed)
        if progress:
            progress("Repeated timing passed; starting separate phase profiling")
        profile = profile_methods(methods, prompts, target, {"target_only": baseline, "dart": engine})
    for sample in [*timing["samples"], *profile["samples"]]:
        if sample["output_hash"] != checks["outputs"][sample["prompt_index"]]["output_hash"]:
            raise ValueError("reference output changed between correctness, timing and profiling")
    return {
        "schema": "dart-corpus-benchmark-v1", "split": checks["split"],
        "selected_corpus_hash": checks["selected_corpus_hash"], "seed": seed,
        "correctness": checks, "timing": timing, "profile": profile,
        "timing_scope": "token ids through generation and output decoding; loading, tokenization, correctness and warm-up excluded",
        "acceptance_scope": "one separate full-output correctness pass",
        "memory_scope": "allocator peak with one shared target and both engine instances resident",
        "interval_scope": "paired repeat intervals for fixed prompts; each comparison uses its observed fastest baseline",
        "record_hidden_by_method": {name: target.record_hidden if name == "dart" else False for name in methods},
        "summary": _summarize(timing["samples"], seed),
        "by_workload": {
            workload: _summarize([sample for sample in timing["samples"]
                                  if records[sample["prompt_index"]].workload == workload], seed)
            for workload in sorted({row.workload for row in records})
        },
        "by_budget": {
            str(budget): _summarize([sample for sample in timing["samples"]
                                    if records[sample["prompt_index"]].max_new_tokens == budget], seed)
            for budget in sorted({row.max_new_tokens for row in records})
        },
    }
