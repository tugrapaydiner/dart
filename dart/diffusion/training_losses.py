"""Training losses for the DART diffusion drafter (docs/design.md §8.3).

The drafter is NOT trained to be a good language model.  It is trained to
be *accepted by the target verifier*: a token that is plausible but not
the target argmax is worthless in exact mode (docs/design.md §8.4).  Hence the
loss mixes plain denoising with target distillation, argmax imitation, and
confidence calibration (calibration drives adaptive early stopping).

    L = L_denoise + 0.3 * L_kl + 0.5 * L_accept + 0.1 * L_cal
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossWeights:
    denoise: float = 1.0
    kl: float = 0.3
    accept: float = 0.5
    calibration: float = 0.1
    kl_temperature: float = 1.0


def denoise_ce(logits: Tensor, x0: Tensor, masked: Tensor) -> Tensor:
    """CE(drafter(x_t) -> x_0) over *masked* positions only.

    logits [N, B, V], x0 [N, B], masked [N, B] bool.
    """
    if not bool(masked.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[masked], x0[masked])


def target_kl(logits: Tensor, target_logits: Tensor, temperature: float = 1.0) -> Tensor:
    """KL(target / T || drafter / T), averaged over all positions."""
    t = temperature
    log_p_drafter = F.log_softmax(logits / t, dim=-1)
    log_p_target = F.log_softmax(target_logits / t, dim=-1)
    return F.kl_div(log_p_drafter, log_p_target, log_target=True, reduction="batchmean")


def acceptance_ce(logits: Tensor, target_argmax: Tensor) -> Tensor:
    """CE toward the target's argmax tokens - the exact-mode acceptance objective."""
    return F.cross_entropy(logits.flatten(0, -2), target_argmax.flatten())


def calibration_bce(logits: Tensor, target_argmax: Tensor) -> Tensor:
    """BCE(confidence, drafter_argmax == target_argmax).

    Confidence = max softmax probability.  A calibrated drafter knows when
    its fill is likely to be accepted, which is what makes adaptive
    stopping and confidence-based re-noising work.
    """
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    hit = (pred == target_argmax).float()
    return F.binary_cross_entropy(conf.clamp(1e-6, 1 - 1e-6), hit)


def dart_drafter_loss(
    logits: Tensor,
    x0: Tensor,
    masked: Tensor,
    target_logits: Tensor,
    weights: LossWeights | None = None,
    valid: Tensor | None = None,
) -> dict[str, Tensor]:
    """Combined loss; ``x0`` must be the target greedy continuation so that
    denoising and acceptance objectives agree.

    ``valid`` [N, B] bool excludes padding slots (short continuations) from
    every term; defaults to all-valid.
    """
    w = weights or LossWeights()
    if valid is None:
        valid = torch.ones_like(masked, dtype=torch.bool)
    v = valid
    target_argmax = target_logits.argmax(dim=-1)
    parts = {
        "denoise": denoise_ce(logits, x0, masked & v),
        "kl": target_kl(logits[v], target_logits[v], w.kl_temperature),
        "accept": acceptance_ce(logits[v], target_argmax[v]),
        "calibration": calibration_bce(logits[v], target_argmax[v]),
    }
    parts["total"] = (
        w.denoise * parts["denoise"]
        + w.kl * parts["kl"]
        + w.accept * parts["accept"]
        + w.calibration * parts["calibration"]
    )
    return parts
