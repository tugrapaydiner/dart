"""Runtime exactness self-test (docs/design.md §3 lists this file).

The pytest suite proves the contract on the tiny seeded model; this module
proves it on **whatever model you actually deployed**, at runtime, in a
few seconds.  Run it after pointing DART at a new target/checkpoint:

    python -m dart.verification.tests --config configs/dart_exact.yaml

or from code:

    from dart.verification.tests import run_self_test
    report = run_self_test(engine, n_prompts=8)
    assert report["exact_match_rate"] == 1.0
"""

from __future__ import annotations

import argparse

import torch

from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig


def run_self_test(
    engine,
    n_prompts: int = 8,
    prompt_len: int = 24,
    max_new_tokens: int = 32,
    draft_len: int = 4,
    seed: int = 1789,
) -> dict:
    """Generate with the engine vs the trusted baseline on random prompts.

    Uses token-id prompts drawn from the target's own vocab, so it works
    with or without a tokenizer.  Returns a report dict; exact_match_rate
    must be 1.0 - anything else means the deployment is broken.
    """
    target = engine.target
    vocab = target.model.config.vocab_size
    g = torch.Generator().manual_seed(seed)
    lo = min(5, vocab - 1)

    results = []
    for i in range(n_prompts):
        plen = int(torch.randint(4, max(5, prompt_len), (1,), generator=g).item())
        prompt = torch.randint(lo, vocab, (plen,), generator=g).tolist()
        base = target.greedy_generate(prompt, max_new_tokens=max_new_tokens)
        out = engine.generate(
            prompt, DecodeConfig(max_new_tokens=max_new_tokens, draft_len=draft_len)
        )
        results.append(
            {
                "prompt_idx": i,
                "exact": out.token_ids == base,
                "baseline_hash": output_hash(base),
                "dart_hash": out.metadata["output_hash"],
                "acceptance_rate": out.metadata["acceptance_rate"],
            }
        )

    exact = sum(r["exact"] for r in results)
    return {
        "prompts": n_prompts,
        "exact_match_rate": exact / n_prompts,
        "failures": [r for r in results if not r["exact"]],
        "mean_acceptance": sum(r["acceptance_rate"] for r in results) / n_prompts,
    }


def main() -> None:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    from dart.runtime.factory import engine_from_yaml

    report = run_self_test(
        engine_from_yaml(args.config),
        n_prompts=args.prompts,
        max_new_tokens=args.max_new,
    )
    print(report)
    if report["exact_match_rate"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
