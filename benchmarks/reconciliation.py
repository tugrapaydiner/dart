from __future__ import annotations

import math

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.compact_diffusion import (
    CompactDiffusionDrafter,
    CompactDiffusionModel,
)


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
