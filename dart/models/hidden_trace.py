"""Teacher-trace capture for drafter training (docs/design.md §8.1).

Runs the target over prompts and records, per continuation token:
  * the greedy token id (x_0 for the diffusion drafter),
  * the target's full logits row (for KL distillation),
  * entropy (for entropy-weighted corruption),
  * optionally the last hidden state (for conditioned drafters / MTP).

Traces are saved with torch.save as a list of dicts - small-scale,
filesystem-friendly; swap for a sharded format when scaling up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from dart.models.target import TargetModel


@dataclass
class TeacherTrace:
    prompt_ids: list[int]
    continuation_ids: list[int]  # greedy tokens
    logits: Tensor  # [T, V] fp16 (storage-friendly)
    entropy: Tensor  # [T]
    hidden: Optional[Tensor] = None  # [T, H] fp16 if recorded


@torch.inference_mode()
def collect_trace(
    target: TargetModel, prompt_ids: list[int], max_new_tokens: int
) -> TeacherTrace:
    state, logits = target.prefill(prompt_ids)
    rows: list[Tensor] = []
    hiddens: list[Tensor] = []
    cont: list[int] = []

    next_logits = logits
    for _ in range(max_new_tokens):
        rows.append(next_logits)
        if target.record_hidden and target.last_hidden_block is not None:
            hiddens.append(target.last_hidden_block[-1])
        tok = int(torch.argmax(next_logits).item())
        cont.append(tok)
        if tok in target.eos_token_ids:
            break
        step = target.forward_block(state, [tok])
        next_logits = step[-1]

    all_logits = torch.stack(rows)  # [T, V]
    logp = torch.log_softmax(all_logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    return TeacherTrace(
        prompt_ids=list(prompt_ids),
        continuation_ids=cont,
        logits=all_logits.to(torch.float16),
        entropy=entropy.to(torch.float32),
        hidden=torch.stack(hiddens).to(torch.float16) if hiddens else None,
    )
