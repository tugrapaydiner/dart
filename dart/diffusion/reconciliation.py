"""Reuse an unverified suffix after confidence filtering.

Adaptive repair restarts when no anchors remain or the last repair accepted
nothing. Every proposed token still passes through the target verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from dart.diffusion.canvas import Canvas
from dart.diffusion.noise import renoise_suffix
from dart.runtime.types import VerifyResult


@dataclass
class ReconcilePolicy:
    mode: Literal["discard", "reconcile", "adaptive"] = "reconcile"
    keep_confidence_above: float = 0.5
    max_keep_fraction: float = 0.75
    repair_steps: int = 4


def make_warm_canvas(
    last_canvas: Optional[Canvas],
    result: VerifyResult,
    policy: ReconcilePolicy,
    generator: Optional[torch.Generator] = None,
) -> Optional[Canvas]:
    """Build the warm-start canvas for the next round, or None for fresh.

    Slot accounting: the old canvas held drafts d_1..d_K.  With m accepted,
    slots 0..m-1 were committed and slot m was replaced by the target's
    correction token (already in context).  Slots m+1.. are the unverified
    suffix available for reuse.
    """
    if policy.mode == "discard" or last_canvas is None:
        return None
    if result.finished or result.all_accepted:
        return None  # nothing rejected, nothing to repair
    m = result.n_draft_accepted
    suffix = last_canvas.sliced(m + 1, last_canvas.length)
    if suffix.length == 0:
        return None
    return renoise_suffix(
        suffix,
        keep_confidence_above=policy.keep_confidence_above,
        max_keep_fraction=policy.max_keep_fraction,
        generator=generator,
    )
