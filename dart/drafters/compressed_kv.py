"""Compressed-KV draft lane (docs/design.md §6.4 and §9).

The *same* target model drafts ahead using a quantized KV cache, then the
verifier checks those tokens against the full-precision cache.  The
compressed cache is never trusted: aggressive quantization can only lower
acceptance, not correctness.

Compressor objective (docs/design.md §9.3): maximize accepted draft length per
unit verifier cost - NOT perplexity.  ``kv_stats`` exposes the memory side
of that trade-off.

Precision policy (14.3): the drafter is the per-position metadata source
for the cache's PrecisionPolicy.  It always supplies token ids (so syntax
tokens - braces, JSON punctuation, tool-call delimiters - can be kept at
higher precision at zero extra cost), and, only when the policy actually
uses an entropy threshold, it supplies per-token predictive entropy from a
single full forward.  Either way the verifier runs on full KV, so precision
choices move acceptance, never correctness.

Inherits the incremental sync/roll/crop machinery from ARDrafter; the only
difference is the cache the drafts run on.
"""

from __future__ import annotations

from typing import Optional

import torch

from dart.drafters.ar_draft import ARDrafter
from dart.kv.cache import QuantizedKVCache, QuantSpec
from dart.models.target import TargetModel


class CompressedKVDrafter(ARDrafter):
    name = "compressed_kv"

    def __init__(
        self,
        target: TargetModel,
        spec: Optional[QuantSpec] = None,
        store_factory=None,
    ) -> None:
        super().__init__(draft_model=target)  # same weights, different cache
        self.spec = spec or QuantSpec()
        self.store_factory = store_factory
        self._committed_ids: list[int] = []
        self._entropy: Optional[list[float]] = None
        self._entropy_len = -1
        self.last_kv_stats: dict = {}

    # ---- PrecisionPolicy metadata provider (docs/design.md §9.4) ------ #
    def token_id_at(self, pos: int) -> Optional[int]:
        ids = self._committed_ids
        return ids[pos] if 0 <= pos < len(ids) else None

    def entropy_at(self, pos: int) -> Optional[float]:
        e = self._entropy
        return e[pos] if e is not None and 0 <= pos < len(e) else None

    def _uses_entropy(self) -> bool:
        p = self.spec.policy
        return p is not None and p.high_entropy_threshold != float("inf")

    def _make_cache(self) -> QuantizedKVCache:
        return QuantizedKVCache(self.spec, store_factory=self.store_factory, meta=self)

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        # Refresh per-position metadata *before* drafting so the cache sees it
        # while aging tokens out during this round's forward.
        self._committed_ids = list(committed)
        if self._uses_entropy():
            if len(committed) != self._entropy_len:
                logits = self.dm.all_logits(committed)  # [T, V]
                logp = torch.log_softmax(logits, dim=-1)
                h = -(logp.exp() * logp).sum(dim=-1)  # [T]
                # entropy_at(p) = uncertainty of the distribution that produced
                # token p (row p-1); position 0 has no predecessor.
                self._entropy = [0.0] + h[:-1].tolist()
                self._entropy_len = len(committed)
        else:
            self._entropy = None
        return super().propose(committed, max_tokens)

    def kv_stats(self) -> dict:
        if self._state is None or not isinstance(self._state.cache, QuantizedKVCache):
            return {}
        return self._state.cache.stats()

    def finish_request(self) -> None:
        try:
            self.last_kv_stats = self.kv_stats()
        finally:
            super().finish_request()
            self._committed_ids = []
            self._entropy = None
            self._entropy_len = -1
