"""Canvas reconciliation ablation - DART's novel contribution (docs/design.md §7).

This is the experiment the whole "research" framing of docs/design.md §7.3 rests on:
after a verifier rejection, does re-noising and repairing only the unverified
suffix beat throwing the canvas away and re-drafting from scratch?

Arms (docs/design.md §7.3):
  * ``discard``     - fresh fully-masked canvas + full denoise budget next round.
  * ``full_redraft``- listed separately in the doc, but in DART's round-based
                      engine it is byte-for-byte the same operation as discard
                      (a fresh canvas at the full step budget), so it is reported
                      as an alias, not a third independent number.
  * ``reconcile``   - keep the verified anchors, re-noise only the low-confidence
                      suffix slots, repair with a reduced step budget.

Headline metric (docs/design.md §7.3/§13.4): accepted tokens per diffusion forward.
Reconciliation should raise it by >= 20% while preserving exact greedy output.

Statistical protocol: both arms run the *same* drafter weights (identical
checkpoint, or identically-seeded random init) over the same prompt set, and
the whole comparison is repeated across several prompt seeds.  The artifact
reports the per-seed numbers and the mean +/- sample std of the improvement,
so the headline claim carries its own uncertainty.

    python scripts/run_reconciliation_ablation.py --prompt-seeds 17,18,19,20,21
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.exact_match import run_exact_match
from benchmarks.run_bench import synthetic_prompts
from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_tiny_target
from dart.runtime.types import DecodeConfig


def _drafter(model, mode: str, canvas_len: int, denoise_steps: int, repair_steps: int):
    return DiffusionDrafter(
        model, canvas_len=canvas_len, denoise_steps=denoise_steps,
        reconcile=ReconcilePolicy(mode=mode, repair_steps=repair_steps),
    )


def run_one_seed(args, cfg, ckpt, trained, prompt_seed: int, target) -> dict:
    """Both arms on one prompt set; identical drafter weights per arm."""
    vocab = target.model.config.vocab_size
    prompts = synthetic_prompts(args.prompts, vocab, seed=prompt_seed)

    def fresh_model():
        # Identical weights for BOTH arms: from the checkpoint when trained,
        # otherwise from an identically-seeded random init.  Without the
        # reseed the two arms of the untrained comparison would run
        # different networks - a confound, not an ablation.
        torch.manual_seed(args.init_seed)
        m = DiffusionDrafterModel(cfg)
        if trained:
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return m

    arms = {
        "discard": ("discard", args.denoise_steps),
        "reconcile": ("reconcile", args.repair_steps),
    }
    rows = {}
    for arm, (mode, repair) in arms.items():
        drafter = _drafter(
            fresh_model(), mode, args.canvas_len, args.denoise_steps, repair
        )
        engine = DartEngine(target, drafter=drafter, verifier="block")
        dcfg = DecodeConfig(max_new_tokens=args.max_new, draft_len=args.draft_len)
        t0 = time.perf_counter()
        report = run_exact_match(target, engine, prompts, args.max_new, dcfg)
        fw = drafter.stats["diffusion_forwards"]
        rows[arm] = {
            "exact_match_rate": report.summary()["exact_match_rate"],
            "draft_tokens_accepted": report.draft_tokens_accepted,
            "diffusion_forwards": fw,
            "warm_rounds": drafter.stats["warm_rounds"],
            "rounds": drafter.stats["rounds"],
            "accepted_per_forward": round(report.draft_tokens_accepted / max(fw, 1), 4),
            "recovery_success_rate": (
                round(drafter.recovery_success_rate, 4)
                if drafter.recovery_success_rate is not None else None
            ),
            "wall_s": round(time.perf_counter() - t0, 2),
        }

    d = rows["discard"]["accepted_per_forward"]
    r = rows["reconcile"]["accepted_per_forward"]
    rows["improvement_pct"] = round((r - d) / d * 100.0, 2) if d > 0 else float("nan")
    rows["prompt_seed"] = prompt_seed
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--draft-len", type=int, default=8)
    ap.add_argument("--canvas-len", type=int, default=8)
    ap.add_argument("--denoise-steps", type=int, default=4)
    ap.add_argument("--repair-steps", type=int, default=2)
    ap.add_argument("--prompt-seeds", default="17,18,19,20,21",
                    help="comma-separated prompt-set seeds; the improvement is "
                         "reported as mean +/- sample std across them")
    ap.add_argument("--init-seed", type=int, default=0,
                    help="drafter init seed (untrained runs; both arms share it)")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_drafter_tiny.pt")
    ap.add_argument("--out", default="benchmarks/results/reconciliation_ablation.md")
    args = ap.parse_args()

    target = build_tiny_target(seed=1234, spec={"record_hidden": False})
    cfg = DiffusionDrafterConfig(
        vocab_size=target.model.config.vocab_size,
        hidden_size=128, num_layers=2, num_heads=4,
        ctx_window=32, max_canvas=16,
    )
    ckpt = Path(args.checkpoint)
    trained = ckpt.exists()
    seeds = [int(s) for s in args.prompt_seeds.split(",") if s.strip()]

    per_seed: list[dict] = []
    print(f"{'seed':>5s} {'discard':>9s} {'reconcile':>10s} {'improvement':>12s} "
          f"{'exact':>6s}")
    for seed in seeds:
        rows = run_one_seed(args, cfg, ckpt, trained, seed, target)
        per_seed.append(rows)
        both_exact = (
            rows["discard"]["exact_match_rate"] == 1.0
            and rows["reconcile"]["exact_match_rate"] == 1.0
        )
        print(f"{seed:>5d} {rows['discard']['accepted_per_forward']:>9.4f} "
              f"{rows['reconcile']['accepted_per_forward']:>10.4f} "
              f"{rows['improvement_pct']:>+11.1f}% "
              f"{'1.000' if both_exact else 'FAIL':>6s}")
        if not both_exact:
            print("EXACTNESS FAILURE", file=sys.stderr)
            sys.exit(1)

    imps = [r["improvement_pct"] for r in per_seed]
    mean_imp = statistics.mean(imps)
    std_imp = statistics.stdev(imps) if len(imps) > 1 else 0.0
    rec_rates = [
        r["reconcile"]["recovery_success_rate"]
        for r in per_seed
        if r["reconcile"]["recovery_success_rate"] is not None
    ]
    print(f"\nreconcile vs discard (accepted tokens / diffusion forward): "
          f"{mean_imp:+.1f}% +/- {std_imp:.1f}% (mean +/- sample std, "
          f"n={len(imps)} prompt seeds; min {min(imps):+.1f}%, "
          f"max {max(imps):+.1f}%)  "
          f"({'>=20% target MET on every seed' if min(imps) >= 20 else 'below 20% target on some seed'})")
    if rec_rates:
        print(f"recovery success rate (warm rounds with >=1 accepted token): "
              f"mean {statistics.mean(rec_rates):.3f}")

    note = "trained drafter" if trained else "UNTRAINED drafter (run train_drafter.py)"
    lines = [
        "# Canvas reconciliation ablation (docs/design.md §7)",
        "",
        f"Target: tiny seeded Llama  ·  drafter: {note} (identical weights per "
        f"arm)  ·  {args.prompts} prompts × {args.max_new} tokens per seed  ·  "
        f"{len(seeds)} prompt seeds",
        "",
        "| prompt seed | discard acc/fwd | reconcile acc/fwd | improvement | exact (both arms) |",
        "|---|---|---|---|---|",
    ]
    for r in per_seed:
        lines.append(
            f"| {r['prompt_seed']} | {r['discard']['accepted_per_forward']:.4f} "
            f"| {r['reconcile']['accepted_per_forward']:.4f} "
            f"| {r['improvement_pct']:+.1f}% | 1.000 |"
        )
    lines += [
        "",
        f"**Reconciliation improves accepted-tokens-per-diffusion-forward by "
        f"{mean_imp:+.1f}% ± {std_imp:.1f}% (mean ± sample std over "
        f"{len(seeds)} prompt seeds; min {min(imps):+.1f}%)** while preserving "
        "100% exact greedy output in every run. Both arms share identical "
        "drafter weights and prompts within each seed. `full_redraft` is "
        "identical to `discard` in the round-based engine (fresh canvas + "
        "full step budget) and is therefore not a separate row.",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "mean_improvement_pct": round(mean_imp, 2),
        "std_improvement_pct": round(std_imp, 2),
        "min_improvement_pct": round(min(imps), 2),
        "max_improvement_pct": round(max(imps), 2),
        "n_prompt_seeds": len(seeds),
        "trained": trained,
        "per_seed": per_seed,
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
