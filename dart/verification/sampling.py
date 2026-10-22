"""Exact speculative sampling (docs/design.md §5.3).

Greedy verification compares argmaxes; preserving the target's *sampling*
distribution needs the speculative-sampling acceptance rule:

  * draft token d (proposed with probability q(d)) is accepted with
    probability min(1, p(d) / q(d)), where p is the target distribution;
  * on rejection, the correction token is sampled from the residual
    distribution (p - q)^+ / Z;
  * if every draft is accepted, a bonus token is sampled from the
    target's next-position distribution.

This composition emits tokens *exactly* distributed as p - the standard
speculative sampling result; ``test_sampling_math`` checks the identity
min(p,q) + (p-q)^+ = p and Monte-Carlos the actual code path.

Drafters that only return tokens (n-gram, diffusion, ...) are treated as
one-hot proposals: q(d) = 1, so acceptance is p(d) and the residual is p
with d zeroed.  This is still exact; richer q only improves acceptance.

The guarantee is DISTRIBUTIONAL, not trace-level: a sampled DART run will
not reproduce some other run's token sequence, and metadata reports
``exact_greedy_equivalent: false`` accordingly.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from dart.models.target import TargetModel, TargetState
from dart.runtime.types import VerifyResult


def speculative_accept_step(
    p: Tensor,
    q_row: Optional[Tensor],
    draft_token: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[int, bool]:
    """One accept/resample decision.  Returns (emitted_token, accepted).

    ``p`` [V] target probabilities at this position; ``q_row`` [V] the
    drafter's proposal distribution, or None for a one-hot proposal.
    Exactness requires only that ``draft_token`` was actually sampled
    from ``q_row`` (trivially true for one-hot).
    """
    p_d = float(p[draft_token])
    q_d = 1.0 if q_row is None else float(q_row[draft_token])
    if q_d <= 0.0:
        # q underflowed (or the drafter lied about its proposal).  The
        # p/q -> inf limit says accept - but only while p(d) > 0; a token
        # outside the target's support must never be emitted, whatever the
        # drafter claims.  Rejecting it falls through to residual
        # resampling, which draws from p, so exactness is preserved even
        # against a misbehaving drafter.
        accept_prob = 1.0 if p_d > 0.0 else 0.0
    else:
        accept_prob = min(1.0, p_d / q_d)
    u = float(torch.rand((), device=p.device, generator=generator))
    if u < accept_prob:
        return draft_token, True

    if q_row is None:
        residual = p.clone()
        residual[draft_token] = 0.0
    else:
        residual = (p - q_row.to(device=p.device)).clamp_min(0.0)
    z = float(residual.sum())
    if z <= 0.0:  # p == q (or numerically so): any p-sample is exact
        residual = p
        z = 1.0
    token = int(torch.multinomial(residual / z, 1, generator=generator).item())
    return token, False


def _softmax_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits.float() / t, dim=-1)


@torch.inference_mode()
def verify_block_sampling(
    target: TargetModel,
    state: TargetState,
    pending: int | list[int],
    draft_tokens: list[int],
    draft_probs: Optional[Tensor] = None,  # [K, V] rows the drafts were sampled from
    generator: Optional[torch.Generator] = None,
    temperature: float = 1.0,
) -> VerifyResult:
    """Sampling analogue of verify_block_greedy: one target pass checks the
    whole block; KV for rejected suffixes is cropped.  Same state contract
    as the greedy verifiers (see verification/greedy.py)."""
    pending_block = [pending] if isinstance(pending, int) else list(pending)
    drafts = list(draft_tokens)
    k = len(drafts)
    base_len = state.seq_len

    logits = target.forward_block(state, pending_block + drafts)
    probs = _softmax_with_temperature(logits, temperature)
    off = len(pending_block) - 1  # row `off + i` is the distribution for drafts[i]

    committed: list[int] = []
    n_accepted = 0
    finished = False
    for i in range(k):
        q_row = draft_probs[i] if draft_probs is not None else None
        token, accepted = speculative_accept_step(
            probs[off + i], q_row, drafts[i], generator
        )
        committed.append(token)
        if accepted:
            n_accepted += 1
        if token in target.eos_token_ids:
            finished = True
            break
        if not accepted:
            break
    else:
        bonus = int(torch.multinomial(probs[off + k], 1, generator=generator).item())
        committed.append(bonus)
        finished = bonus in target.eos_token_ids

    target.crop(state, base_len + len(pending_block) - 1 + len(committed))
    return VerifyResult(
        committed=committed,
        n_draft_proposed=k,
        n_draft_accepted=n_accepted,
        all_accepted=(n_accepted == k),
        finished=finished,
    )
