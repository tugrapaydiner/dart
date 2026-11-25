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

from dart.models.replay import CudaGraphTarget
from dart.models.target import TargetModel, TargetState, _native_method
from dart.runtime.types import VerifyResult
from dart.verification.greedy import _cpu_greedy_indices, _restore_failed_round


def _draft_before_eos(target, base_len: int, pending_len: int, draft_tokens: list[int]) -> list[int]:
    for index, token in enumerate(draft_tokens):
        if type(token) is int and token in target.eos_token_ids:
            config = getattr(getattr(target, "model", None), "config", None)
            vocab_size = getattr(config, "vocab_size", None)
            context = getattr(target, "context_limit", getattr(config, "max_position_embeddings", None))
            block = getattr(target, "max_block_tokens", None)
            width = pending_len + len(draft_tokens)
            fits_context = context is None or (type(context) is int and base_len + width <= context)
            fits_block = block is None or (type(block) is int and width <= block)
            if fits_context and fits_block and type(vocab_size) is int and all(
                type(value) is int and 0 <= value < vocab_size for value in draft_tokens[index:]
            ):
                return draft_tokens[:index]  # EOS and its tail need no forward.
            break  # The full forward handles bounds and invalid IDs.
    return draft_tokens


def verify_block_greedy(
    target: TargetModel,
    state: TargetState,
    pending: int | list[int],
    draft_tokens: list[int],
    *,
    compute_entropy: bool = True,
) -> VerifyResult:
    """Set compute_entropy=False when no consumer needs the entropy signal."""
    tokens = [pending] if isinstance(pending, int) else list(pending)
    if not tokens:
        raise ValueError("pending block must contain at least one token")
    p_len = len(tokens)
    base_len = state.seq_len  # cache length before this round

    forward_drafts = _draft_before_eos(target, base_len, p_len, draft_tokens)
    tokens.extend(forward_drafts)
    compact = not compute_entropy and getattr(target, "capture_predictions", False)
    forward = target.forward_predictions if compact else target.forward_block

    def decide(logits, predictions):
        if predictions is None:
            predictions = _block_predictions(logits, p_len, forward_drafts)
        result = _block_result(tokens, p_len, draft_tokens, predictions, target.eos_token_ids,
                               logits if compute_entropy else None)
        return p_len - 1 + len(result.committed), result

    if (type(target) is CudaGraphTarget and base_len and (forward_drafts or not compact)
            and type(draft_tokens) is list and all(type(token) is int for token in tokens)
            and _native_method(forward, target, CudaGraphTarget._native_predictions if compact else CudaGraphTarget._native_block)
            and _native_method(target._forward, target, CudaGraphTarget._native_forward)
            and _native_method(target.crop, target, CudaGraphTarget._native_crop)):
        return target._verify_block(state, tokens, decide)

    try:
        output = forward(state, tokens)
        kept, result = decide(None, output) if compact else decide(output, None)
        target.crop(state, base_len + kept)
        return result
    except BaseException:
        _restore_failed_round(target, state, base_len)
        raise


def _block_predictions(logits, p_len, drafts):
    if logits.device.type != "cpu" or logits.size(-1) < 32_000 or torch.get_num_threads() != 1:
        return torch.argmax(logits, dim=-1).tolist()

    if len(drafts) >= 3 and type(drafts[0]) is int:
        predictions = _cpu_greedy_indices(logits[:p_len]).tolist()
        if predictions[-1] == drafts[0]:  # A first rejection leaves the remaining rows unused.
            predictions.extend(_cpu_greedy_indices(logits[p_len:]).tolist())
        return predictions
    return _cpu_greedy_indices(logits).tolist()


def _block_result(tokens, p_len, draft_tokens, preds, eos_ids, logits=None):
    k = len(draft_tokens)
    # Divergence audit of unverified pending tokens (approx mode only).
    divergence = sum(
        1 for j in range(1, p_len) if tokens[j] != preds[j - 1]
    )

    off = p_len - 1  # preds[off + i] is the target's choice for drafts[i]
    m = 0
    accepted_eos = False
    while m < k and draft_tokens[m] == preds[off + m]:
        is_eos = draft_tokens[m] in eos_ids
        m += 1
        if is_eos:
            accepted_eos = True
            break

    if accepted_eos:
        committed = list(draft_tokens[:m])
        finished = True
    else:
        committed = list(draft_tokens[:m]) + [preds[off + m]]
        finished = committed[-1] in eos_ids

    # Predictive entropy at the decided (on-trajectory) position - drives the
    # scheduler's entropy gate.  Row off+m produced the correction/bonus token;
    # rows after it are off-trajectory (conditioned on a rejected draft).
    if accepted_eos or logits is None:
        entropy = 0.0
    else:
        dist = torch.softmax(logits[off + m].float(), dim=-1)
        entropy = float(-(dist * dist.clamp_min(1e-12).log()).sum())

    return VerifyResult(
        committed=committed,
        n_draft_proposed=k,
        n_draft_accepted=m,
        all_accepted=(m == k),
        finished=finished,
        pending_divergence=divergence,
        entropy=entropy,
        hidden_index=p_len + len(committed) - 2,
    )
