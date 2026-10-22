from __future__ import annotations

from contextlib import ExitStack

import numpy as np

from benchmarks.timing import Method, measure_methods, tokens
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig

BASELINES = ("hf_greedy", "target_greedy", "target_only")


class PrecomputedDrafter:
    """Offline diagnostic only; continuation generation is excluded from timing."""

    name = "perfect_draft"

    def __init__(self, prompt: list[int], continuation: list[int]):
        self.prompt_len = len(prompt)
        self.sequence = list(prompt) + list(continuation)

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        if len(committed) < self.prompt_len or committed != self.sequence[:len(committed)]:
            raise ValueError("precomputed continuation does not match the committed prefix")
        return self.sequence[len(committed):len(committed) + max(0, max_tokens)]

    def observe(self, result) -> None:
        pass


def draft_budget(samples: list[dict], proposal_rounds: list[int], seed: int = 17) -> dict:
    repeats = sorted({row["repeat"] for row in samples})
    means = {
        name: np.array([
            np.mean([row["latency_ms"] for row in samples if row["method"] == name and row["repeat"] == repeat])
            for repeat in repeats
        ]) for name in (*BASELINES, "perfect_draft")
    }
    fastest = min(BASELINES, key=lambda name: means[name].mean())
    indices = np.random.default_rng(seed).integers(0, len(repeats), size=(2_000, len(repeats)))
    baseline = np.min([means[name][indices].mean(axis=1) for name in BASELINES], axis=0)
    perfect = means["perfect_draft"][indices].mean(axis=1)
    extra_ms = float(means[fastest].mean() - means["perfect_draft"].mean())
    rounds = float(np.mean(proposal_rounds))
    interval = np.quantile(baseline - perfect, [0.025, 0.975])
    return {
        "fastest_baseline": fastest,
        "perfect_speedup": float(means[fastest].mean() / means["perfect_draft"].mean()),
        "ci95_perfect_speedup": np.quantile(baseline / perfect, [0.025, 0.975]).tolist(),
        "extra_draft_ms_per_request": extra_ms,
        "ci95_extra_draft_ms_per_request": interval.tolist(),
        "mean_proposal_rounds": rounds,
        "extra_draft_ms_per_round": extra_ms / rounds if rounds else None,
        "ci95_extra_draft_ms_per_round": (interval / rounds).tolist() if rounds else None,
    }


def measure_headroom(
    baselines: dict[str, Method], prompts: list[list[int]], target: TargetModel,
    config: DecodeConfig, warmup: int = 2, repeats: int = 10, seed: int = 17,
) -> dict:
    if set(baselines) != set(BASELINES) or config.mode != "exact" or config.draft_len < 1:
        raise ValueError("headroom needs three target-only baselines, exact mode, and draft_len >= 1")
    continuations = [list(tokens(baselines["hf_greedy"](prompt))) for prompt in prompts]
    with ExitStack() as stack:
        engines = {
            tuple(prompt): stack.enter_context(DartEngine(target, PrecomputedDrafter(prompt, continuation)))
            for prompt, continuation in zip(prompts, continuations)
        }

        def perfect(prompt):
            return engines[tuple(prompt)].generate(prompt, config)

        rounds = []
        for prompt, continuation in zip(prompts, continuations):
            output = perfect(prompt)
            if output.token_ids != continuation or any(row.n_draft_proposed != row.n_draft_accepted for row in output.rounds):
                raise ValueError("perfect-draft diagnostic failed complete acceptance")
            rounds.append(sum(row.n_draft_proposed > 0 for row in output.rounds))
        timing = measure_methods({**baselines, "perfect_draft": perfect}, prompts, target.device, warmup=warmup, repeats=repeats, seed=seed)
    return {
        "scope": "offline perfect-draft diagnostic; not deployable performance",
        "excluded_cost": "precomputing correct continuations; replay lookup and verification remain timed",
        "budget_scope": "additional drafting time that could fit before losing to the fastest baseline, assuming perfect acceptance",
        "draft_len": config.draft_len, "timing": timing,
        "budget": draft_budget(timing["samples"], rounds, seed),
        "per_prompt": [
            {"prompt_index": index, **draft_budget([row for row in timing["samples"] if row["prompt_index"] == index], [count], seed)}
            for index, count in enumerate(rounds)
        ],
    }
