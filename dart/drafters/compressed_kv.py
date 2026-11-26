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
uses an entropy threshold, it supplies per-token predictive entropy from the
committed prefix when tokens can leave the full-precision window. The verifier
uses full KV for every precision setting.

Inherits the incremental sync/roll/crop machinery from ARDrafter; the only
difference is the cache the drafts run on.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn.modules import module as nn_module
from transformers.cache_utils import DynamicCache, DynamicLayer

from dart.drafters.ar_draft import ARDrafter
from dart.kv.cache import QuantizedKVCache, QuantSpec
from dart.models.target import (
    _TARGET_FORWARD,
    _TARGET_PREFILL,
    TargetModel,
    TargetState,
    _native_method,
)

_TARGET_BLOCK = TargetModel.forward_block
_TARGET_RUN = TargetModel._run_model


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
        self.last_kv_stats: dict = {}

    # ---- PrecisionPolicy metadata provider (docs/design.md §9.4) ------ #
    def token_id_at(self, pos: int) -> Optional[int]:
        ids = self._committed_ids
        return ids[pos] if 0 <= pos < len(ids) else None

    def entropy_at(self, pos: int) -> Optional[float]:
        if self._entropy is None or not 0 <= pos < len(self._committed_ids):
            return None
        if pos >= len(self._entropy):
            hidden = self.dm.last_hidden_block
            try:
                with torch.inference_mode():
                    logits = self.dm.all_logits(self._committed_ids)
                    logp = torch.log_softmax(logits, dim=-1)
                    self._entropy = [0.0] + (-(logp.exp() * logp).sum(dim=-1)).tolist()
            finally:
                self.dm.last_hidden_block = hidden  # Keep the live request's hidden states.
        return self._entropy[pos]

    def _uses_entropy(self) -> bool:
        p = self.spec.policy
        return p is not None and p.high_entropy_threshold != float("inf")

    def _make_cache(self) -> QuantizedKVCache:
        return QuantizedKVCache(self.spec, store_factory=self.store_factory, meta=self)

    @torch.inference_mode()
    def _reuse_verified_prefix(self, source: TargetState, committed: list[int], width: int) -> None:
        length = len(committed) - 1  # The pending token still needs a draft forward.
        target = self.dm
        if (self._state is not None or width < 1 or length < 1 or length + width > self.spec.window_len
                or self.store_factory is not None or type(target) is not TargetModel or target.record_hidden
                or target.dtype != torch.float32 or not target._native_cpu_llama()
                or type(source) is not TargetState or source._failed or type(source.cache) is not DynamicCache):
            return
        for owner, method, original in (
            (self, self.propose, _COMPRESSED_PROPOSE), (self, self._sync, _AR_SYNC),
            (self, self._make_cache, _MAKE_CACHE), (target, target.prefill, _TARGET_PREFILL),
            (target, target.forward_block, _TARGET_BLOCK), (target, target._forward, _TARGET_FORWARD),
            (target, target._run_model, _TARGET_RUN),
        ):
            if not _native_method(getattr(method, "_profiled_method", method), owner, original):
                return
        if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
            return
        for module in target.model.modules():
            if (module.training or "forward" in vars(module)
                    or any(not getattr(hook, "_dart_observer", False) for hooks in
                           (module._forward_hooks, module._forward_pre_hooks) for hook in hooks.values())):
                return

        layers = source.cache.layers
        if len(layers) != target.model.config.num_hidden_layers or any(
            type(layer) is not DynamicLayer or layer.get_seq_length() != length
            or layer.keys.dtype != torch.float32 or layer.values.dtype != torch.float32 for layer in layers
        ):
            return
        cache = self._make_cache()
        for i, layer in enumerate(layers):
            cache.update(layer.keys.clone(), layer.values.clone(), i)
        self._state, self._processed = TargetState(cache), committed[:-1]

    def _prepare_metadata(self, committed: list[int], max_tokens: int) -> None:
        ids = list(committed)
        if not self._uses_entropy():
            self._entropy = None
        elif self._entropy is None or ids[:len(self._committed_ids)] != self._committed_ids:
            self._entropy = [0.0]
        self._committed_ids = ids
        if self._entropy is not None:
            width = self._budget(committed, max_tokens)
            if width > 0:
                aged = min(len(ids), len(ids) + width - 1 - self.spec.window_len)
                if aged > len(self._entropy):
                    self.entropy_at(aged - 1)  # Fill missing rows before the cache can age them out.

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        self._prepare_metadata(committed, max_tokens)
        return super().propose(committed, max_tokens)

    def propose_with_probs(self, committed: list[int], max_tokens: int) -> tuple[list[int], Optional[torch.Tensor]]:
        self._prepare_metadata(committed, max_tokens)
        return super().propose_with_probs(committed, max_tokens)

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


_COMPRESSED_PROPOSE = CompressedKVDrafter.propose
_AR_SYNC = ARDrafter._sync
_MAKE_CACHE = CompressedKVDrafter._make_cache
