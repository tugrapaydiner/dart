from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import asdict
from statistics import mean

import torch

from benchmarks.corpus_eval import output_stats
from benchmarks.headroom import BASELINES
from benchmarks.profiling import profile_methods
from benchmarks.run_bench import (
    build_hf_method,
    build_target_method,
    with_hidden_recording,
)
from benchmarks.timing import measure_methods, summarize_samples
from dart.data.corpus import Prompt, corpus_hash, encode_prompt, validate_records
from dart.data.teacher import target_fingerprint
from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.base import NullDrafter
from dart.drafters.compact_diffusion import (
    CompactDiffusionDrafter,
    CompactDiffusionModel,
)
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig


def build_reconciliation_arms(
    model: CompactDiffusionModel, *, canvas_len: int = 4,
    full_steps: int = 2, repair_steps: int = 1,
    keep_confidence_above: float = 0.5, max_keep_fraction: float = 0.75,
    seed: int = 17,
) -> dict[str, CompactDiffusionDrafter]:
    """Compare reuse and pass budgets with one shared set of frozen weights."""
    for name, value in (("canvas_len", canvas_len), ("full_steps", full_steps), ("repair_steps", repair_steps)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if repair_steps >= full_steps:
        raise ValueError("repair_steps must be smaller than full_steps")
    if canvas_len > model.cfg.max_canvas:
        raise ValueError("canvas_len exceeds model capacity")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for name, value in (("keep_confidence_above", keep_confidence_above), ("max_keep_fraction", max_keep_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    settings = {
        "fresh_one_pass": ("discard", 1, 1),
        "fresh_multi_pass": ("discard", full_steps, full_steps),
        "reuse_equal_budget": ("reconcile", full_steps, full_steps),
        "reuse_reduced_budget": ("reconcile", full_steps, repair_steps),
        "adaptive_repair": ("adaptive", full_steps, repair_steps),
    }
    return {
        name: CompactDiffusionDrafter(
            model, canvas_len=canvas_len, denoise_steps=fresh,
            adaptive_stop=False, seed=seed,
            reconcile=ReconcilePolicy(
                mode=mode, repair_steps=repair,
                keep_confidence_above=keep_confidence_above,
                max_keep_fraction=max_keep_fraction,
            ),
        ) for name, (mode, fresh, repair) in settings.items()
    }


def _summarize_group(samples, acceptance, indices, seed, profile):
    selected = [row for row in samples if row["prompt_index"] in indices]
    timing = summarize_samples(selected, "hf_greedy", seed)
    fastest = min(BASELINES, key=lambda name: timing[name]["mean_ms"])
    fresh = min(("fresh_one_pass", "fresh_multi_pass"), key=lambda name: timing[name]["mean_ms"])
    comparisons = {}
    for reference in dict.fromkeys((fastest, "ngram", "fresh_one_pass", "fresh_multi_pass")):
        comparisons[reference] = {
            name: {key: row[key] for key in ("speedup", "ci95_speedup")}
            for name, row in summarize_samples(selected, reference, seed).items()
        }
    counts = {}
    for name, rows in acceptance.items():
        group = [row for row in rows if row["prompt_index"] in indices]
        fields = ("draft_tokens_proposed", "draft_tokens_accepted", "rounds", "diffusion_forwards", "warm_rounds", "warm_rounds_recovered", "restart_no_anchors", "restart_failed_repair")
        counts[name] = totals = {key: sum(row[key] for row in group) for key in fields}
        proposed = totals["draft_tokens_proposed"]
        accepted = totals["draft_tokens_accepted"]
        forwards = totals["diffusion_forwards"]
        totals["acceptance_rate"] = accepted / proposed if proposed else None
        totals["accepted_tokens_per_diffusion_forward"] = accepted / forwards if forwards else None
    phases = None
    if profile is not None:
        phases = {}
        for name in timing:
            group = [row for row in profile["samples"] if row["method"] == name and row["prompt_index"] in indices]
            phases[name] = {
                key: mean(row[key] for row in group) if group[0][key] is not None else None
                for key in ("request_ms", "setup_ms", "prefill_ms", "decode_ms", "draft_ms", "verify_ms")
            }
    return {"prompts": len(indices), "timing": timing, "fastest_baseline": fastest,
            "fastest_fresh": fresh, "speedups": comparisons, "acceptance": counts, "profile": phases}


def measure_reconciliation(
    target: TargetModel, model: CompactDiffusionModel, records: list[Prompt], *,
    canvas_len: int = 4, full_steps: int = 2, repair_steps: int = 1,
    keep_confidence_above: float = 0.5, max_keep_fraction: float = 0.75,
    warmup: int = 2, repeats: int = 10, seed: int = 17, profile: bool = False,
) -> dict:
    """Measure fixed repair policies with matched per-prompt generation budgets."""
    validate_records(records)
    if len({record.split for record in records}) != 1 or target.tokenizer is None:
        raise ValueError("reconciliation needs one corpus split and a tokenizer")
    if model._target() is not target:
        raise ValueError("compact model and benchmark must share the same target")
    if type(warmup) is not int or type(repeats) is not int or warmup < 1 or repeats < 2:
        raise ValueError("use warmup >= 1 and repeats >= 2")
    prompts = [encode_prompt(record, target.tokenizer, target.model.config.max_position_embeddings) for record in records]
    keys = [tuple(prompt) for prompt in prompts]
    if len(set(keys)) != len(keys):
        raise ValueError("corpus contains duplicate tokenized prompts")
    arms = build_reconciliation_arms(
        model, canvas_len=canvas_len, full_steps=full_steps, repair_steps=repair_steps,
        keep_confidence_above=keep_confidence_above, max_keep_fraction=max_keep_fraction, seed=seed,
    )
    rows = {"target_only": NullDrafter(), "ngram": NGramDrafter(max_ngram=4), **arms}
    kinds = {"instruction": "chat", "code": "code", "structured": "json", "repetition": "chat"}
    configs = {key: DecodeConfig(max_new_tokens=record.max_new_tokens, draft_len=canvas_len, canvas_len=canvas_len,
                                 prompt_kind=kinds[record.workload]) for key, record in zip(keys, records)}
    hf = {key: build_hf_method(target, record.max_new_tokens, record_hidden=False) for key, record in zip(keys, records)}
    direct = {key: build_target_method(target, record.max_new_tokens) for key, record in zip(keys, records)}
    target_info = target_fingerprint(target)
    acceptance = {name: [] for name in rows}
    prompt_info = []
    with ExitStack() as stack:
        engines = {name: stack.enter_context(DartEngine(target, drafter)) for name, drafter in rows.items()}
        methods = {"hf_greedy": lambda prompt: hf[tuple(prompt)](prompt),
                   "target_greedy": lambda prompt: direct[tuple(prompt)](prompt)}
        methods.update({name: (lambda prompt, e=engine: e.generate(prompt, configs[tuple(prompt)])) for name, engine in engines.items()})
        methods = {name: with_hidden_recording(target, name in arms, method) for name, method in methods.items()}
        for index, (record, prompt) in enumerate(zip(records, prompts)):
            expected = methods["hf_greedy"](prompt).token_ids
            prompt_info.append({**asdict(record), "prompt_index": index, "prompt_tokens": len(prompt),
                                "prompt_token_ids": prompt, "output_hash": output_hash(expected),
                                "token_ids": expected, **output_stats(expected, target.eos_token_ids, record.max_new_tokens)})
            for name, drafter in rows.items():
                before = dict(getattr(drafter, "stats", {}))
                output = methods[name](prompt)
                if output.token_ids != expected:
                    raise ValueError(f"{name}: complete token mismatch on {record.id}")
                counters = getattr(drafter, "stats", {})
                acceptance[name].append({
                    "prompt_index": index, "output_hash": output_hash(output.token_ids),
                    **{key: output.metadata[key] for key in ("draft_tokens_proposed", "draft_tokens_accepted", "rounds")},
                    **{key: counters.get(key, 0) - before.get(key, 0) for key in ("diffusion_forwards", "warm_rounds", "warm_rounds_recovered", "restart_no_anchors", "restart_failed_repair")},
                })
        timing = measure_methods(methods, prompts, target.device, warmup=warmup, repeats=repeats, seed=seed)
        phases = profile_methods(methods, prompts, target, engines) if profile else None
    groups = {workload: {i for i, record in enumerate(records) if record.workload == workload} for workload in sorted({record.workload for record in records})}
    return {
        "schema": "dart-reconciliation-v1", "split": records[0].split,
        "selected_corpus_hash": corpus_hash(records), "target": target_info,
        "seed": seed, "threads": torch.get_num_threads(), "torch": str(torch.__version__),
        "reference": "plain HF greedy", "prompts": prompt_info,
        "budget_scope": "each prompt uses its recorded max_new_tokens",
        "timing_scope": "token ids through generation and output decoding; loading, tokenization, correctness and warm-up excluded",
        "acceptance_scope": "one separate full-output correctness pass",
        "memory_scope": "all methods resident with one shared compact model",
        "interval_scope": "paired repeat intervals for these fixed prompts; fastest references are selected by observed mean latency",
        "pass_scope": "configured upper limits with confidence-based early filling disabled; actual forwards are reported",
        "model_config": asdict(model.cfg), "learned_parameters": sum(p.numel() for p in model.parameters()),
        "record_hidden_by_method": {name: name in arms for name in methods},
        "policies": {name: {"canvas_len": drafter.canvas_len, "full_steps": drafter.denoise_steps,
                             "adaptive_stop": drafter.adaptive_stop, **asdict(drafter.policy)} for name, drafter in arms.items()},
        "timing": timing, "profile": phases, "acceptance_samples": acceptance,
        "summary": _summarize_group(timing["samples"], acceptance, set(range(len(records))), seed, phases),
        "by_workload": {name: _summarize_group(timing["samples"], acceptance, indices, seed, phases) for name, indices in groups.items()},
        "by_prompt": [{"id": record.id, "prompt_index": index, **_summarize_group(timing["samples"], acceptance, {index}, seed, phases)}
                      for index, record in enumerate(records)],
    }
