from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

import numpy as np
import torch

from dart.runtime.metrics import output_hash
from dart.runtime.types import GenerationOutput

Method = Callable[[list[int]], list[int] | GenerationOutput]


def synchronize(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tokens(output: list[int] | GenerationOutput) -> list[int]:
    return output.token_ids if isinstance(output, GenerationOutput) else output


def summarize_samples(samples: list[dict], reference: str, seed: int = 17) -> dict:
    methods = list(dict.fromkeys(sample["method"] for sample in samples))
    repeats = sorted({sample["repeat"] for sample in samples})
    if not samples or reference not in methods or len(repeats) < 2:
        raise ValueError("samples need a reference and at least two repeats")
    if any(not np.isfinite(sample["latency_ms"]) or sample["latency_ms"] <= 0 for sample in samples):
        raise ValueError("latencies must be positive finite values")
    counts = Counter((sample["method"], sample["repeat"]) for sample in samples)
    if len(counts) != len(methods) * len(repeats) or len(set(counts.values())) != 1:
        raise ValueError("every method and repeat must have the same number of samples")
    means = {}
    grouped = {}
    for name in methods:
        rows = [sample for sample in samples if sample["method"] == name]
        grouped[name] = rows
        means[name] = np.array([
            np.mean([sample["latency_ms"] for sample in rows if sample["repeat"] == repeat])
            for repeat in repeats
        ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(repeats), size=(2_000, len(repeats)))
    bootstrap = {name: values[indices].mean(axis=1) for name, values in means.items()}
    cases = _case_means(samples, methods)
    case_bootstrap = None
    if cases is not None and cases[reference].size > 1:  # Drawn after repeats so their interval is unchanged.
        draws = rng.integers(0, cases[reference].size, size=(2_000, cases[reference].size))
        case_bootstrap = {name: values[draws].mean(axis=1) for name, values in cases.items()}
    summary = {}
    for name, rows in grouped.items():
        latency = [sample["latency_ms"] for sample in rows]
        summary[name] = {
            "mean_ms": float(means[name].mean()),
            "median_ms": float(np.median(latency)),
            "ci95_ms": np.quantile(bootstrap[name], [0.025, 0.975]).tolist(),
            "tokens_per_second": sum(sample["tokens"] for sample in rows) * 1_000 / sum(latency),
            "peak_gpu_bytes": max(sample["peak_gpu_bytes"] for sample in rows),
            "speedup": float(means[reference].mean() / means[name].mean()),
            "ci95_speedup": np.quantile(bootstrap[reference] / bootstrap[name], [0.025, 0.975]).tolist(),
            "case_speedup_geomean": None if cases is None else float(np.exp(np.log(cases[reference] / cases[name]).mean())),
            "case_ci95_speedup": None if case_bootstrap is None else
            np.quantile(case_bootstrap[reference] / case_bootstrap[name], [0.025, 0.975]).tolist(),
        }
    return summary


def _case_means(samples: list[dict], methods: list[str]) -> dict | None:
    """Mean latency per method and prompt, or None without a complete prompt grid."""
    if any("prompt_index" not in sample for sample in samples):
        return None
    prompts = sorted({sample["prompt_index"] for sample in samples})
    counts = Counter((sample["method"], sample["prompt_index"]) for sample in samples)
    if len(counts) != len(methods) * len(prompts) or len(set(counts.values())) != 1:
        return None
    totals = Counter()
    for sample in samples:
        totals[sample["method"], sample["prompt_index"]] += sample["latency_ms"]
    return {name: np.array([totals[name, prompt] for prompt in prompts]) / counts[name, prompts[0]]
            for name in methods}


def measure_methods(
    methods: dict[str, Method],
    prompts: list[list[int]],
    device: str | torch.device,
    reference: str = "hf_greedy",
    warmup: int = 2,
    repeats: int = 10,
    seed: int = 17,
    progress: Callable[[str], None] | None = None,
) -> dict:
    if reference not in methods:
        raise ValueError("reference method is missing")
    if not prompts or type(warmup) is not int or type(repeats) is not int or warmup < 1 or repeats < 2:
        raise ValueError("use nonempty prompts, warmup >= 1, and repeats >= 2")
    rng = np.random.default_rng(seed)
    expected = []
    for index, prompt in enumerate(prompts):
        expected.append(list(tokens(methods[reference](prompt))))
        if progress:
            progress(f"Reference outputs: {index + 1}/{len(prompts)} prompts")
    synchronize(device)

    def check(name, index, output):
        if tokens(output) != expected[index]:
            raise ValueError(f"{name}: complete token mismatch on prompt {index}")

    for index, prompt in enumerate(prompts):
        for name, method in methods.items():
            check(name, index, method(prompt))
        if progress:
            progress(f"Preflight checks: {index + 1}/{len(prompts)} prompts")
    synchronize(device)
    for iteration in range(warmup):
        for index, prompt in enumerate(prompts):
            for name, method in methods.items():
                check(name, index, method(prompt))
                synchronize(device)
            if progress:
                progress(f"Warm-up {iteration + 1}/{warmup}: {index + 1}/{len(prompts)} prompts")

    samples = []
    names = list(methods)
    cuda = torch.device(device).type == "cuda"
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if (repeat // len(names)) % 2:
            order = order[::-1]
        for index in rng.permutation(len(prompts)).tolist():
            for name in order:
                synchronize(device)
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                output = methods[name](prompts[index])
                synchronize(device)
                elapsed = (time.perf_counter() - start) * 1_000
                check(name, index, output)
                samples.append({
                    "method": name, "prompt_index": index, "repeat": repeat,
                    "latency_ms": elapsed, "tokens": len(tokens(output)),
                    "output_hash": output_hash(tokens(output)),
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
                })
            if progress:
                progress(f"Timed {len(samples)}/{repeats * len(prompts) * len(names)} samples")
    return {
        "reference": reference, "warmup": warmup, "repeats": repeats,
        "samples": samples, "summary": summarize_samples(samples, reference, seed),
        "confidence_method": "paired bootstrap over repeat means, 2000 resamples, 95% percentile interval; "
                             "case interval resamples prompts, case speedup is the geometric mean of prompt ratios",
    }
