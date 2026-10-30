"""Block greedy verifier: check K draft tokens with one target forward.

Implements docs/design.md §5.2.  Feeding ``[pending block, d_1..d_K]``
through the target in a single causal pass yields logits at every position;
the logits after position ``i`` are the target's prediction for position
``i+1``, so the longest matching draft prefix and the correction/bonus
token all come out of one pass.  KV entries written for rejected draft
tokens are rolled back with ``TargetModel.crop``.

The pending block is normally a single token (exact mode invariant).  In
approx mode it may contain additional already-committed-but-unverified
tokens; those are *not* re-decided (they are committed), but each one is
checked against the target argmax and counted in ``pending_divergence`` so
approx metadata can report honestly whether output is target-identical.

Contract: identical to verify_greedy_reference (see verification/greedy.py)
- same VerifyResult, same final KV length, for every input.  Tests enforce
this equivalence.
"""

from __future__ import annotations

import torch

from dart.models.target import TargetModel, TargetState
from dart.runtime.types import VerifyResult


def verify_block_greedy(
    target: TargetModel,
    state: TargetState,
    pending: int | list[int],
    draft_tokens: list[int],
    *,
    compute_entropy: bool = True,
) -> VerifyResult:
    """Set compute_entropy=False when no consumer needs the entropy signal."""
    pending_block = [pending] if isinstance(pending, int) else list(pending)
    if not pending_block:
        raise ValueError("pending block must contain at least one token")
    p_len = len(pending_block)
    k = len(draft_tokens)
    base_len = state.seq_len  # cache length before this round

    tokens = pending_block + list(draft_tokens)
    if not compute_entropy and getattr(target, "capture_predictions", False):
        preds = target.forward_predictions(state, tokens)
    else:
        logits = target.forward_block(state, tokens)
        preds = torch.argmax(logits, dim=-1).tolist()  # preds[i] follows input[i]

    # Divergence audit of unverified pending tokens (approx mode only).
    divergence = sum(
        1 for j in range(1, p_len) if pending_block[j] != preds[j - 1]
    )

    off = p_len - 1  # preds[off + i] is the target's choice for drafts[i]
    m = 0
    accepted_eos = False
    while m < k and draft_tokens[m] == preds[off + m]:
        is_eos = draft_tokens[m] in target.eos_token_ids
        m += 1
        if is_eos:
            accepted_eos = True
            break

    if accepted_eos:
        committed = list(draft_tokens[:m])
        finished = True
        n_accepted = m
    else:
        committed = list(draft_tokens[:m]) + [preds[off + m]]
        finished = committed[-1] in target.eos_token_ids
        n_accepted = m

    # Predictive entropy at the decided (on-trajectory) position - drives the
    # scheduler's entropy gate.  Row off+m produced the correction/bonus token;
    # rows after it are off-trajectory (conditioned on a rejected draft).
    if accepted_eos or not compute_entropy:
        entropy = 0.0
    else:
        dist = torch.softmax(logits[off + m].float(), dim=-1)
        entropy = float(-(dist * dist.clamp_min(1e-12).log()).sum())

    # Roll back KV entries of rejected draft tokens.  Postcondition: the
    # cache covers all committed tokens except the new pending one.
    target.crop(state, base_len + p_len - 1 + len(committed))

    return VerifyResult(
        committed=committed,
        n_draft_proposed=k,
        n_draft_accepted=n_accepted,
        all_accepted=(n_accepted == k),
        finished=finished,
        pending_divergence=divergence,
        entropy=entropy,
        hidden_index=p_len + len(committed) - 2,
    )
