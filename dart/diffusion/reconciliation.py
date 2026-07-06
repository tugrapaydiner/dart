"""Canvas reconciliation - DART's novel contribution (docs/design.md §7).

After the verifier rejects a diffusion block at position j:

  * the accepted prefix and the target's correction token are already
    committed - they become ordinary context for the next round;
  * the unverified suffix of the old canvas is NOT discarded: it is
    partially re-noised (low-confidence slots re-masked) and used to
    warm-start the next round's canvas, repaired with fewer denoise steps.

Hypothesis: because diffusion refines canvases jointly, the stale suffix
is often mostly right, so repairing beats re-drafting from scratch in
accepted-tokens-per-diffusion-forward.  Correctness never depends on this:
every proposed token still goes through the exact verifier.

Why the ablation has two arms, not three: docs/design.md §7.3 lists
"discard" and "full re-draft" baselines separately, but in DART's
round-based engine both manifest identically - a fresh fully-masked canvas
with the full step budget on the next round.  They are one baseline here
(``mode="discard"``), compared against ``mode="reconcile"``.
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
    mode: Literal["discard", "reconcile"] = "reconcile"
    keep_confidence_above: float = 0.5
    max_keep_fraction: float = 0.75
    repair_steps: int = 4  # < full denoise steps: this is the saving


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
