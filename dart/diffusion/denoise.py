"""Iterative parallel denoising (MaskGIT-style confidence scheduling).

Each step, the drafter model predicts every masked slot in parallel; the
most confident predictions are committed and the rest stay masked for the
next step.  A cosine schedule guarantees termination; adaptive stopping
commits any prediction above ``min_confidence`` early, which is where the
"few steps for easy spans" speedup comes from (DiffusionGemma-style
adaptive stop, docs/design.md §6.3).

Filled slots (including reconciliation anchors) are frozen: they condition
the model but are never re-predicted in v0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from dart.diffusion.canvas import Canvas
from dart.diffusion.noise import cosine_mask_schedule

# (context_ids [C], canvas) -> logits [canvas_len, V]
CanvasModelFn = Callable[[Tensor, Canvas], Tensor]


@dataclass
class DenoiseStats:
    steps_used: int = 0
    slots_filled: int = 0


def denoise(
    model_fn: CanvasModelFn,
    context_ids: Tensor,
    canvas: Canvas,
    num_steps: int,
    min_confidence: float = 0.65,
    adaptive_stop: bool = True,
) -> DenoiseStats:
    """Fill every masked slot of ``canvas`` in place; return step stats."""
    stats = DenoiseStats()
    if canvas.num_masked == 0:
        return stats

    schedule = cosine_mask_schedule(num_steps)
    total = canvas.length

    for keep_frac in schedule:
        masked_idx = torch.nonzero(canvas.masked, as_tuple=False).flatten()
        if masked_idx.numel() == 0:
            break

        logits = model_fn(context_ids, canvas)  # [B, V]
        stats.steps_used += 1
        probs = torch.softmax(logits[masked_idx].float(), dim=-1)
        conf, pred = probs.max(dim=-1)

        # How many slots may remain masked after this step.
        remaining_target = int(keep_frac * total)
        n_fill = max(1, masked_idx.numel() - remaining_target)

        order = torch.argsort(conf, descending=True)
        chosen = order[:n_fill]
        if adaptive_stop:
            confident = torch.nonzero(conf >= min_confidence, as_tuple=False).flatten()
            chosen = torch.unique(torch.cat([chosen, confident]))

        fill_positions = masked_idx[chosen]
        canvas.fill(fill_positions, pred[chosen], conf[chosen])
        stats.slots_filled += int(fill_positions.numel())

    # The cosine schedule ends at 0 so this should be unreachable; keep a
    # hard guarantee anyway.
    if canvas.num_masked > 0:  # pragma: no cover
        masked_idx = torch.nonzero(canvas.masked, as_tuple=False).flatten()
        logits = model_fn(context_ids, canvas)
        stats.steps_used += 1
        probs = torch.softmax(logits[masked_idx].float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        canvas.fill(masked_idx, pred, conf)
        stats.slots_filled += int(masked_idx.numel())

    return stats
