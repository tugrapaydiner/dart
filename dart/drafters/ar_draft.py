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

from typing import Optional

import torch
from torch import Tensor

from dart.models.target import TargetModel, TargetState
from dart.runtime.types import VerifyResult


class ARDrafter:
    name = "ar_draft"

    def __init__(
        self,
        draft_model: TargetModel,
        sample_temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.dm = draft_model
        self.sample_temperature = sample_temperature
        self.gen = torch.Generator(device=draft_model.device).manual_seed(seed)
        self._state: Optional[TargetState] = None
        self._processed: list[int] = []

    def reset(self) -> None:
        self._state = None
        self._processed = []

    def _make_cache(self):
        """Cache for fresh prefills; subclasses substitute compressed caches."""
        return None  # TargetModel.prefill defaults to a full DynamicCache

    def _is_extension(self, committed: list[int]) -> bool:
        n = len(self._processed)
        return len(committed) >= n and committed[:n] == self._processed

    @torch.inference_mode()
    def _sync(self, committed: list[int]) -> Tensor:
        """Bring the draft cache up to ``committed``; return next-token logits."""
        if self._state is None or not self._is_extension(committed) or not self._processed:
            self.reset()
            self._state, logits = self.dm.prefill(committed, cache=self._make_cache())
            return logits
        delta = committed[len(self._processed):]
        if delta:
            return self.dm.forward_block(self._state, delta)[-1]
        # Engine never calls propose twice without committing, but stay
        # correct if it does: re-derive logits from the last token.
        self.dm.crop(self._state, len(committed) - 1)
        return self.dm.forward_block(self._state, [committed[-1]])[-1]

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        if max_tokens <= 0:
            return []
        last_logits = self._sync(committed)
        drafts: list[int] = []
        for _ in range(max_tokens):
            tok = int(torch.argmax(last_logits).item())
            drafts.append(tok)
            if tok in self.dm.eos_token_ids:
                break
            last_logits = self.dm.forward_block(self._state, [tok])[-1]
        self.dm.crop(self._state, len(committed))
        self._processed = list(committed)
        return drafts

    @torch.inference_mode()
    def propose_with_probs(
        self, committed: list[int], max_tokens: int
    ) -> tuple[list[int], Optional[Tensor]]:
        if max_tokens <= 0:
            return [], None
        t = max(self.sample_temperature, 1e-6)
        last_logits = self._sync(committed)
        drafts: list[int] = []
        rows: list[Tensor] = []
        for _ in range(max_tokens):
            q = torch.softmax(last_logits.float() / t, dim=-1)
            tok = int(torch.multinomial(q, 1, generator=self.gen).item())
            drafts.append(tok)
            rows.append(q)
            if tok in self.dm.eos_token_ids:
                break
            last_logits = self.dm.forward_block(self._state, [tok])[-1]
        self.dm.crop(self._state, len(committed))
        self._processed = list(committed)
        return drafts, torch.stack(rows)

    def observe(self, result: VerifyResult) -> None:
        pass
