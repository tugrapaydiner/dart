"""Sparse-attention draft lane: sink + sliding-window token-drop drafting.

The cheapest long-context lane: the draft model only ever attends over
sink + recent tokens (compacted sequence), so draft cost stops growing
with context length.  The full verifier still sees everything, so output
is exact (docs/design.md §6.5).

Compaction invalidates incremental-cache reuse whenever the window slides,
so v0.1 re-prefills the compacted sequence when its compacted view drifts;
with small windows this is still far cheaper than full-context drafting at
128K+.  Block-sparse selection (MiniMax-M3-style) is the planned upgrade
in dart/attention/block_sparse.py.
"""

from __future__ import annotations

from typing import Optional

import torch

from dart.drafters.ar_draft import ARDrafter
from dart.kv.compressors.token_drop import TokenDropSpec, compact_sequence
from dart.models.target import TargetModel


class SparseAttentionDrafter(ARDrafter):
    name = "sparse_kv"

    def __init__(self, target: TargetModel, spec: Optional[TokenDropSpec] = None) -> None:
        super().__init__(draft_model=target)
        self.spec = spec or TokenDropSpec()
        self.dropped_last_round = False

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        compacted, dropped = compact_sequence(committed, self.spec)
        self.dropped_last_round = dropped
        if dropped:
            # The compacted view is not an extension of what we processed
            # before (the window slid), so reset and let ARDrafter prefill
            # the short sequence fresh.
            self.reset()
        return super().propose(compacted, max_tokens)
