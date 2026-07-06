"""Canvas state for discrete masked-diffusion drafting.

A canvas is a fixed-length block of token slots.  Each slot is either
filled (a concrete token id with a confidence) or masked (still unknown).
Denoising progressively fills masked slots; reconciliation re-masks part
of a previously filled canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Canvas:
    tokens: Tensor  # [B] long; value is meaningless where masked
    masked: Tensor  # [B] bool; True = slot not yet filled
    confidence: Tensor  # [B] float in [0, 1]; 0 where masked

    def __post_init__(self) -> None:
        if not (self.tokens.shape == self.masked.shape == self.confidence.shape):
            raise ValueError("canvas field shapes must match")

    @property
    def length(self) -> int:
        return self.tokens.numel()

    @property
    def num_masked(self) -> int:
        return int(self.masked.sum().item())

    def clone(self) -> "Canvas":
        return Canvas(
            self.tokens.clone(), self.masked.clone(), self.confidence.clone()
        )

    @staticmethod
    def empty(length: int, device: str = "cpu") -> "Canvas":
        """Fully masked canvas of the given length."""
        return Canvas(
            tokens=torch.zeros(length, dtype=torch.long, device=device),
            masked=torch.ones(length, dtype=torch.bool, device=device),
            confidence=torch.zeros(length, dtype=torch.float, device=device),
        )

    @staticmethod
    def from_tokens(
        tokens: list[int], confidence: list[float] | None = None, device: str = "cpu"
    ) -> "Canvas":
        """Fully filled canvas (used to warm-start reconciliation)."""
        t = torch.tensor(tokens, dtype=torch.long, device=device)
        c = (
            torch.tensor(confidence, dtype=torch.float, device=device)
            if confidence is not None
            else torch.ones(len(tokens), dtype=torch.float, device=device)
        )
        return Canvas(
            tokens=t,
            masked=torch.zeros(len(tokens), dtype=torch.bool, device=device),
            confidence=c,
        )

    def fill(self, positions: Tensor, values: Tensor, confidence: Tensor) -> None:
        """Commit predictions at ``positions`` (bool mask or index tensor)."""
        self.tokens[positions] = values
        self.confidence[positions] = confidence
        self.masked[positions] = False

    def as_draft(self) -> list[int]:
        """Export as a draft block; canvas must be fully filled."""
        if bool(self.masked.any()):
            raise ValueError("canvas still has masked slots")
        return self.tokens.tolist()

    def sliced(self, start: int, end: int) -> "Canvas":
        return Canvas(
            self.tokens[start:end].clone(),
            self.masked[start:end].clone(),
            self.confidence[start:end].clone(),
        )

    def padded_to(self, length: int) -> "Canvas":
        """Extend with masked slots up to ``length`` (no-op if already there)."""
        if length <= self.length:
            return self.clone()
        pad = length - self.length
        device = self.tokens.device
        return Canvas(
            tokens=torch.cat(
                [self.tokens, torch.zeros(pad, dtype=torch.long, device=device)]
            ),
            masked=torch.cat(
                [self.masked, torch.ones(pad, dtype=torch.bool, device=device)]
            ),
            confidence=torch.cat(
                [self.confidence, torch.zeros(pad, dtype=torch.float, device=device)]
            ),
        )
