"""Multi-token prediction (MTP) drafter.

A bank of small heads reads the target's final hidden state at the last
on-trajectory position and predicts tokens at offsets +2, +3, ... (+1 is
the pending token, already known from the target's own argmax).  This
costs *zero extra target forwards*: the hidden state is a byproduct of the
verify pass that just ran.

Heads start random (useless drafts, exactness unaffected - the verifier
guarantees that) and are trained on teacher traces via
scripts/train_drafter.py / dart.diffusion.training_losses.

Position bookkeeping (see TargetModel.record_hidden):
  the verify pass fed [pending, d_1 .. d_K]; with m accepted drafts, row m
  of the hidden block is the last on-trajectory position.  Its +1
  prediction is the new pending token, so heads +2.. propose the drafts
  for the next round.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from dart.models.target import TargetModel
from dart.runtime.types import VerifyResult


class MTPHeads(nn.Module):
    """Offset heads: hidden [H] -> logits [num_offsets, V] (offsets +2..)."""

    def __init__(self, hidden_size: int, vocab_size: int, num_offsets: int = 3) -> None:
        super().__init__()
        self.num_offsets = num_offsets
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(
            nn.Linear(hidden_size, vocab_size) for _ in range(num_offsets)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        z = self.trunk(hidden)
        return torch.stack([head(z) for head in self.heads], dim=-2)


class MTPDrafter:
    name = "mtp"

    def __init__(self, target: TargetModel, heads: MTPHeads) -> None:
        if not target.record_hidden:
            raise ValueError("MTPDrafter requires TargetModel(record_hidden=True)")
        self.target = target
        self.heads = heads.to(device=target.device).eval()
        self.heads.requires_grad_(False)
        self._hidden: Optional[Tensor] = None

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        if max_tokens <= 0 or self._hidden is None:
            return []
        logits = self.heads(self._hidden.to(dtype=next(self.heads.parameters()).dtype))
        drafts = torch.argmax(logits, dim=-1).tolist()
        return drafts[:max_tokens]

    def observe(self, result: VerifyResult) -> None:
        block = self.target.last_hidden_block
        if block is None:
            self._hidden = None
            return
        # Row of the last on-trajectory token (see module docstring).  After
        # prefill the engine has not called observe yet, so the first round
        # is handled by after_prefill().
        idx = min(result.n_draft_accepted, block.size(0) - 1)
        self._hidden = block[idx]

    def after_prefill(self) -> None:
        """Use the last prompt position's hidden state for round one."""
        block = self.target.last_hidden_block
        self._hidden = block[-1] if block is not None else None
