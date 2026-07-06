"""Reference greedy verifier: one target forward per checked token.

This is the slow, obviously-correct implementation of the exactness
contract (docs/design.md §2 and §5.1).  It exists so the fast block
verifier (block_verify.py) can be tested against it: both must produce
identical VerifyResults and leave the KV state at identical lengths for any
(prefix, pending block, drafts) triple.

Shared verifier contract
------------------------
Precondition:
  * the sequence so far has N committed tokens,
  * ``state`` covers the first N - P of them, where P = len(pending block),
  * the pending block holds the last P committed tokens (KV not computed).

Postcondition (L = len(result.committed)):
  * the sequence now has N + L committed tokens,
  * ``state`` covers the first N + L - 1 of them,
  * ``result.committed[-1]`` is the new pending token,
  * every committed token equals the target greedy argmax given its prefix
    (pending tokens beyond the first are audited, not re-decided - see
    ``pending_divergence``).
"""

from __future__ import annotations

import torch

from dart.models.target import TargetModel, TargetState
from dart.runtime.types import VerifyResult


def verify_greedy_reference(
    target: TargetModel,
    state: TargetState,
    pending: int | list[int],
    draft_tokens: list[int],
) -> VerifyResult:
    """Sequentially check each draft token with single-token forwards."""
    pending_block = [pending] if isinstance(pending, int) else list(pending)
    if not pending_block:
        raise ValueError("pending block must contain at least one token")
    k = len(draft_tokens)

    # Feed the pending block one token at a time, auditing divergence of
    # the unverified tail (approx mode); the final pred seeds draft checks.
    divergence = 0
    pred: int | None = None
    for j, tok in enumerate(pending_block):
        if j >= 1 and tok != pred:
            divergence += 1  # unverified token was not the target argmax
        logits = target.forward_block(state, [tok])
        pred = int(torch.argmax(logits[-1]).item())

    accepted: list[int] = []
    for i in range(k):
        if draft_tokens[i] != pred:
            break
        accepted.append(draft_tokens[i])
        if draft_tokens[i] in target.eos_token_ids:
            # EOS accepted: stop here; no correction/bonus token needed.
            return VerifyResult(
                committed=list(accepted),
                n_draft_proposed=k,
                n_draft_accepted=len(accepted),
                all_accepted=(len(accepted) == k),
                finished=True,
                pending_divergence=divergence,
            )
        logits = target.forward_block(state, [draft_tokens[i]])
        pred = int(torch.argmax(logits[-1]).item())

    m = len(accepted)
    committed = accepted + [pred]  # correction (m < k) or bonus (m == k) token
    finished = committed[-1] in target.eos_token_ids
    return VerifyResult(
        committed=committed,
        n_draft_proposed=k,
        n_draft_accepted=m,
        all_accepted=(m == k),
        finished=finished,
        pending_divergence=divergence,
    )
