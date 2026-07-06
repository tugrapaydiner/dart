"""Corruption processes and schedules for masked discrete diffusion.

Used in two places:
  * training (docs/design.md §8.2): corrupt teacher continuations
    into noisy canvases the drafter learns to denoise;
  * reconciliation (docs/design.md §7): partially re-noise the unverified suffix of
    a rejected canvas before repair.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from dart.diffusion.canvas import Canvas


def cosine_mask_schedule(num_steps: int) -> list[float]:
    """Fraction of slots that should remain masked after each step.

    MaskGIT-style cosine schedule: starts at ~1, ends at exactly 0, so the
    final step always fills every remaining slot.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    return [math.cos(math.pi / 2 * (k + 1) / num_steps) for k in range(num_steps)]


def mask_corrupt(
    tokens: Tensor,
    noise_level: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Mask each position independently with probability ``noise_level``.

    Returns (corrupted_is_masked [B] bool, noise_draw [B] float).  Token
    values are left untouched; consumers treat masked slots as unknown.
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be in [0, 1]")
    draw = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    return draw < noise_level, draw


def random_token_corrupt(
    tokens: Tensor,
    noise_level: float,
    vocab_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Replace each position w.p. ``noise_level`` with a uniform random token."""
    hit, _ = mask_corrupt(tokens, noise_level, generator)
    rand = torch.randint(
        0, vocab_size, tokens.shape, generator=generator, device=tokens.device
    )
    return torch.where(hit, rand, tokens)


def entropy_weighted_corrupt(
    tokens: Tensor,
    base_level: float,
    entropy: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Mask with per-position probability scaled by normalized entropy.

    High-entropy (hard) positions receive more noise so the drafter trains
    hardest where the target is least predictable (docs/design.md §8.2).
    """
    if entropy.shape != tokens.shape:
        raise ValueError("entropy must align with tokens")
    e = entropy.clamp_min(0.0)
    scale = e / e.max() if float(e.max()) > 0 else torch.zeros_like(e)
    level = (base_level * (0.5 + scale)).clamp(0.0, 1.0)
    draw = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    return draw < level


def span_permute_corrupt(
    tokens: Tensor,
    span_len: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Chop the sequence into contiguous spans and permute their order.

    The code-infill corruption from docs/design.md §8.2: unlike per-token
    noise, this preserves each span's *internal* token order (so local
    syntax stays intact) while scrambling *where* spans live, which trains
    the bidirectional drafter to re-order blocks - the operation code infill
    actually needs.  Returns (permuted_tokens, moved_mask) where moved_mask
    marks positions whose token changed.
    """
    if span_len < 1:
        raise ValueError("span_len must be >= 1")
    n = tokens.numel()
    n_spans = (n + span_len - 1) // span_len
    if n_spans < 2:
        return tokens.clone(), torch.zeros_like(tokens, dtype=torch.bool)
    perm = torch.randperm(n_spans, generator=generator, device=tokens.device)
    pieces = [tokens[i * span_len : (i + 1) * span_len] for i in range(n_spans)]
    out = torch.cat([pieces[i] for i in perm.tolist()])
    return out, out != tokens


def syntax_aware_corrupt(
    tokens: Tensor,
    noise_level: float,
    protected_ids: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Mask w.p. ``noise_level`` but never mask structural tokens.

    ``protected_ids`` lists token ids that carry code/JSON structure -
    braces, brackets, indentation, tool-call delimiters (docs/design.md
    11.2 / 14.3).  Keeping them un-noised stabilizes training on structured
    output: the drafter learns to fill *inside* a scaffold rather than
    having to re-derive the scaffold itself.  Returns the masked-positions
    bool tensor (same contract as ``mask_corrupt``).
    """
    hit, _ = mask_corrupt(tokens, noise_level, generator)
    protected = torch.isin(tokens, protected_ids.to(tokens.device))
    return hit & ~protected


def renoise_suffix(
    canvas: Canvas,
    keep_confidence_above: float,
    max_keep_fraction: float = 0.75,
    generator: Optional[torch.Generator] = None,
) -> Canvas:
    """Partially re-mask a filled canvas for reconciliation repair.

    Slots whose previous denoising confidence is below the threshold are
    re-masked; if too few slots would be re-masked (the old canvas was
    overconfident), the lowest-confidence slots are re-masked until at most
    ``max_keep_fraction`` of the canvas survives.  Returns a new Canvas.
    """
    out = canvas.clone()
    remask = out.confidence < keep_confidence_above

    max_keep = int(max_keep_fraction * out.length)
    kept = int((~remask).sum().item())
    if kept > max_keep:
        # Re-mask the least confident among the kept slots.
        kept_idx = torch.nonzero(~remask, as_tuple=False).flatten()
        order = torch.argsort(out.confidence[kept_idx])  # ascending
        to_remask = kept_idx[order[: kept - max_keep]]
        remask[to_remask] = True

    out.masked |= remask
    out.confidence[remask] = 0.0
    return out
