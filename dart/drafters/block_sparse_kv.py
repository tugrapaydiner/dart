"""Block-sparse KV draft lane (docs/design.md §6.5).

MiniMax-MSA-style: the draft model attends only over a *selected* subset of
context blocks - the sink, the recent window, and the top-k blocks most
relevant to the current query - instead of the whole context.  Relevance is
scored cheaply from the target's own input embeddings (one embedding lookup,
no forward) so no per-layer attention internals are needed:

    block score = <mean embedding of block, mean embedding of recent window>

Like the sink+window sparse lane this is *sequence compaction* (keep the
selected blocks' tokens, drop the rest, prefill the compacted sequence), so
it stays independent of transformers cache internals.  The full verifier
sees everything, so output is exact; block selection only moves acceptance
(docs/design.md §9.3).
"""

from __future__ import annotations

import torch

from dart.attention.block_sparse import block_sparse_mask, score_blocks, select_blocks
from dart.drafters.ar_draft import ARDrafter
from dart.models.target import TargetModel


class BlockSparseKVDrafter(ARDrafter):
    name = "block_sparse_kv"

    def __init__(
        self,
        target: TargetModel,
        block_size: int = 16,
        top_k: int = 4,
        sink_blocks: int = 1,
        recent_blocks: int = 2,
    ) -> None:
        super().__init__(draft_model=target)
        self.block_size = block_size
        self.top_k = top_k
        self.sink_blocks = sink_blocks
        self.recent_blocks = recent_blocks
        self.compacted_last_round = False
        # Compaction only helps once context exceeds what we'd always keep.
        self._min_len = (sink_blocks + recent_blocks + top_k) * block_size

    @torch.inference_mode()
    def _select(self, committed: list[int]) -> tuple[list[int], bool]:
        n = len(committed)
        if n <= self._min_len:
            return list(committed), False
        emb = self.dm.model.get_input_embeddings()
        ids = torch.tensor(committed, dtype=torch.long, device=self.dm.device)
        vecs = emb(ids).detach().float()  # [N, D] - lookup only, no forward
        recent = vecs[-self.recent_blocks * self.block_size :]
        scores = score_blocks(recent, vecs, self.block_size)
        kept = select_blocks(
            scores, top_k=self.top_k,
            always_first=self.sink_blocks, always_last=self.recent_blocks,
        )
        mask = block_sparse_mask(n, self.block_size, kept, device=str(self.dm.device))
        compacted = [committed[i] for i in range(n) if bool(mask[i])]
        return compacted, True

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        compacted, compacted_now = self._select(committed)
        self.compacted_last_round = compacted_now
        if compacted_now:
            # Selection changes round to round, so the compacted view is not a
            # prefix-extension of the previous one: re-prefill it fresh.
            self.reset()
        return super().propose(compacted, max_tokens)
