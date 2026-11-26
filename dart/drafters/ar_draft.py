"""Small autoregressive draft model lane (classic speculative decoding).

A cheaper AR model rolls out K tokens; the target verifies.  The draft
model keeps its own incremental KV cache, synchronized to the committed
sequence: committed tokens are append-only, so a cached prefix never
invalidates - after each round we just crop away the speculative tokens.

Two proposal styles:

  * ``propose``            - greedy roll (argmax); used by exact greedy mode.
  * ``propose_with_probs`` - temperature-sampled roll returning the q-row
    each draft token was sampled from; used by exact_sampling mode, where
    acceptance is min(1, p/q).  For exactness the q rows must be the
    distributions actually sampled from - which they are by construction.
    Match ``sample_temperature`` to the engine's DecodeConfig.temperature
    for best acceptance (any mismatch costs speed, never correctness).

The draft model is wrapped in TargetModel purely for its cache plumbing;
"target" in that class name means "the model whose KV this state tracks",
not "the trusted model".
"""

from __future__ import annotations

from copy import copy
from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules import module as nn_module
from transformers.cache_utils import DynamicCache, DynamicLayer

from dart.models.target import (
    _TARGET_FORWARD,
    _TARGET_PREFILL,
    TargetModel,
    TargetState,
    _native_method,
)
from dart.runtime.types import VerifyResult
from dart.verification.sampling import _softmax_with_temperature


class ARDrafter:
    name = "ar_draft"

    def __init__(
        self,
        draft_model: TargetModel,
        sample_temperature: float = 1.0,
        seed: int = 0,
        *,
        owns_model: bool = False,
    ) -> None:
        if type(owns_model) is not bool:
            raise ValueError("owns_model must be a boolean")
        self.dm = draft_model
        self._owns_model = owns_model
        self._model_closed = False
        self.sample_temperature = sample_temperature
        self.gen = torch.Generator(device=draft_model.device).manual_seed(seed)
        self._state: Optional[TargetState] = None
        self._processed: list[int] = []

    def reset(self) -> None:
        self._state = None
        self._processed = []

    def finish_request(self) -> None:
        self.reset()
        self.dm.last_hidden_block = None

    def seed_request(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def close(self) -> None:
        """Close the draft model when this drafter owns it; shared models stay open."""
        if self._owns_model and not self._model_closed:
            close = getattr(self.dm, "close", None)
            if close is not None:
                close()
            self._model_closed = True

    def _make_cache(self):
        """Cache for fresh prefills; subclasses substitute compressed caches."""
        return None  # TargetModel.prefill defaults to a full DynamicCache

    def _verified_layers(self, source, length, propose, make_cache):
        target = self.dm
        if (type(target) is not TargetModel or target.record_hidden or target.dtype != torch.float32
                or not target._native_cpu_llama() or type(source) is not TargetState or source._failed
                or type(source.cache) is not DynamicCache or source.cache.offloading):
            return None
        for owner, method, original in (
            (self, self.propose, propose), (self, self._sync, _AR_SYNC),
            (self, self._make_cache, make_cache), (target, target.prefill, _TARGET_PREFILL),
            (target, target.forward_block, _TARGET_BLOCK), (target, target._forward, _TARGET_FORWARD),
            (target, target._run_model, _TARGET_RUN),
        ):
            if not _native_method(getattr(method, "_profiled_method", method), owner, original):
                return None
        if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
            return None
        for module in target.model.modules():
            if (module.training or "forward" in vars(module)
                    or any(not getattr(hook, "_dart_observer", False) for hooks in
                           (module._forward_hooks, module._forward_pre_hooks) for hook in hooks.values())):
                return None

        layers = source.cache.layers
        if len(layers) != target.model.config.num_hidden_layers or any(
            type(layer) is not DynamicLayer or layer.get_seq_length() != length
            or layer.keys.dtype != torch.float32 or layer.values.dtype != torch.float32 for layer in layers
        ):
            return None
        return layers

    @torch.inference_mode()
    def _reuse_verified_prefix(self, source: TargetState, committed: list[int], width: int) -> None:
        length = len(committed) - 1
        if width < 1 or length < 1:
            return
        if self._state is not None and (length <= len(self._processed) or not self._is_extension(committed)):
            return
        layers = self._verified_layers(source, length, _AR_PROPOSE, _AR_CACHE)
        if not layers:
            return
        cache = DynamicCache()
        for layer in layers:
            owned = copy(layer)
            owned.keys = layer.keys.clone()
            owned.values = layer.values.clone()
            cache.layers.append(owned)
        self._state, self._processed = TargetState(cache), committed[:-1]  # The final token stays pending.

    def _is_extension(self, committed: list[int]) -> bool:
        n = len(self._processed)
        return len(committed) >= n and committed[:n] == self._processed

    def _budget(self, committed: list[int], max_tokens: int) -> int:
        """Drafts that fit the draft model; the final draft needs no forward."""
        limit = self.dm.context_limit
        return max_tokens if limit is None else min(max_tokens, limit - len(committed) + 1)

    @torch.inference_mode()
    def _sync(self, committed: list[int]) -> Tensor:
        """Bring the draft cache up to ``committed``; return next-token logits."""
        delta = committed[len(self._processed):]
        width = getattr(self.dm, "max_block_tokens", len(delta))
        overflow = len(delta) > width
        refill = overflow and (len(delta) > 2 * width or len(delta) > len(self._processed))
        if self._state is None or not self._processed or refill or not self._is_extension(committed):
            self.reset()
            self._state, logits = self.dm.prefill(committed, cache=self._make_cache())
            return logits
        if delta:
            try:
                if overflow:
                    self.dm.forward_block(self._state, delta[:width])  # Keep the longer cached prefix.
                    delta = delta[width:]
                return self.dm.forward_block(self._state, delta)[-1]
            except BaseException:
                self.reset()  # A failed append may have changed the cache.
                raise
        # Engine never calls propose twice without committing, but stay
        # correct if it does: re-derive logits from the last token.
        self.dm.crop(self._state, len(committed) - 1)
        return self.dm.forward_block(self._state, [committed[-1]])[-1]

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        max_tokens = self._budget(committed, max_tokens)
        if max_tokens <= 0:
            return []
        last_logits = self._sync(committed)
        drafts: list[int] = []
        for _ in range(max_tokens):
            tok = int(torch.argmax(last_logits).item())
            drafts.append(tok)
            if tok in self.dm.eos_token_ids or len(drafts) == max_tokens:
                break
            last_logits = self.dm.forward_block(self._state, [tok])[-1]
        self.dm.crop(self._state, len(committed))
        self._processed = list(committed)
        return drafts

    @torch.inference_mode()
    def propose_with_probs(
        self, committed: list[int], max_tokens: int
    ) -> tuple[list[int], Optional[Tensor]]:
        max_tokens = self._budget(committed, max_tokens)
        if max_tokens <= 0:
            return [], None
        temperature = self.sample_temperature
        last_logits = self._sync(committed)
        drafts: list[int] = []
        rows: list[Tensor] = []
        for _ in range(max_tokens):
            q = _softmax_with_temperature(last_logits, temperature)
            tok = int(torch.multinomial(q, 1, generator=self.gen).item())
            drafts.append(tok)
            rows.append(q)
            if tok in self.dm.eos_token_ids or len(drafts) == max_tokens:
                break
            last_logits = self.dm.forward_block(self._state, [tok])[-1]
        self.dm.crop(self._state, len(committed))
        self._processed = list(committed)
        return drafts, torch.stack(rows)

    def observe(self, result: VerifyResult) -> None:
        pass


_TARGET_BLOCK = TargetModel.forward_block
_TARGET_RUN = TargetModel._run_model
_AR_PROPOSE = ARDrafter.propose
_AR_CACHE = ARDrafter._make_cache
_AR_SYNC = ARDrafter._sync
