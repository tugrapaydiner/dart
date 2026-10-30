"""Small trainable diffusion canvas drafter (docs/design.md §6.3 and §16).

A compact bidirectional transformer fills a masked canvas conditioned on a
window of committed context.  It is the local, trainable stand-in for a
DiffusionGemma-class block-diffusion drafter, and the model that
scripts/train_drafter.py optimizes for *acceptance by the target*, not for
its own fluency (docs/design.md §8.4).

Layout fed to the encoder (single sequence, bidirectional attention):

    [ ctx token window, right-aligned | canvas slots (MASK where unfilled) ]

Canvas slots always start at index ``ctx_window``, so position embeddings
are stable regardless of how much context exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from dart.diffusion.canvas import Canvas
from dart.diffusion.denoise import DenoiseStats, denoise
from dart.diffusion.reconciliation import ReconcilePolicy, make_warm_canvas
from dart.runtime.types import VerifyResult


@dataclass
class DiffusionDrafterConfig:
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_mult: int = 4
    ctx_window: int = 64
    max_canvas: int = 64
    dropout: float = 0.0


class DiffusionDrafterModel(nn.Module):
    """Bidirectional canvas denoiser.  MASK id == vocab_size."""

    def __init__(self, cfg: DiffusionDrafterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = cfg.vocab_size
        total_len = cfg.ctx_window + cfg.max_canvas

        self.tok_emb = nn.Embedding(cfg.vocab_size + 1, cfg.hidden_size)
        self.pos_emb = nn.Embedding(total_len, cfg.hidden_size)
        self.type_emb = nn.Embedding(2, cfg.hidden_size)  # 0 = ctx, 1 = canvas
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_mult * cfg.hidden_size,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(
        self,
        context_ids: Tensor,  # [N, C] (C <= ctx_window), right-aligned content
        context_pad_mask: Tensor,  # [N, C] bool, True where PAD
        canvas_tokens: Tensor,  # [N, B]
        canvas_masked: Tensor,  # [N, B] bool, True where slot unfilled
    ) -> Tensor:
        """Return logits [N, B, vocab] for every canvas slot."""
        n, c = context_ids.shape
        b = canvas_tokens.size(1)
        if c > self.cfg.ctx_window or b > self.cfg.max_canvas:
            raise ValueError("context/canvas exceed configured windows")

        canvas_in = torch.where(
            canvas_masked,
            torch.full_like(canvas_tokens, self.mask_token_id),
            canvas_tokens,
        )
        ids = torch.cat([context_ids, canvas_in], dim=1)  # [N, C+B]

        # Right-align context positions so canvas always starts at ctx_window.
        ctx_pos = torch.arange(
            self.cfg.ctx_window - c, self.cfg.ctx_window, device=ids.device
        )
        can_pos = torch.arange(
            self.cfg.ctx_window, self.cfg.ctx_window + b, device=ids.device
        )
        pos = torch.cat([ctx_pos, can_pos]).unsqueeze(0).expand(n, -1)
        types = torch.cat(
            [
                torch.zeros(c, dtype=torch.long, device=ids.device),
                torch.ones(b, dtype=torch.long, device=ids.device),
            ]
        ).unsqueeze(0).expand(n, -1)

        x = self.tok_emb(ids) + self.pos_emb(pos) + self.type_emb(types)
        pad_mask = torch.cat(
            [context_pad_mask, torch.zeros_like(canvas_masked, dtype=torch.bool)],
            dim=1,
        )
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(self.norm(h[:, c:]))


class DiffusionDrafter:
    """Draft lane wrapping the canvas model with denoising + reconciliation."""

    name = "diffusion"
    uses_canvas = True

    def __init__(
        self,
        model: DiffusionDrafterModel,
        canvas_len: int = 32,
        denoise_steps: int = 8,
        min_confidence: float = 0.65,
        adaptive_stop: bool = True,
        reconcile: Optional[ReconcilePolicy] = None,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if canvas_len > model.cfg.max_canvas:
            raise ValueError("canvas_len exceeds model.cfg.max_canvas")
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.canvas_len = canvas_len
        self.denoise_steps = denoise_steps
        self.min_confidence = min_confidence
        self.adaptive_stop = adaptive_stop
        self.policy = reconcile or ReconcilePolicy(mode="discard")
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        self._last_canvas: Optional[Canvas] = None
        self._warm: Optional[Canvas] = None
        self._used_warm = False
        self._context: list[int] = []
        self.stats = {
            "diffusion_forwards": 0,
            "rounds": 0,
            "warm_rounds": 0,  # rounds that drafted from a reconciled canvas
            "warm_rounds_recovered": 0,  # ...of which >=1 repaired token was accepted
            "restart_no_anchors": 0,
            "restart_failed_repair": 0,
            "proposed_tokens": 0,
        }

    @property
    def recovery_success_rate(self) -> Optional[float]:
        """Fraction of reconciled (warm) rounds that recovered >=1 accepted
        token - the docs/design.md §13.2 'recovery success rate after canvas
        reconciliation' metric.  None until a warm round has occurred."""
        w = self.stats["warm_rounds"]
        return self.stats["warm_rounds_recovered"] / w if w else None

    # ------------------------------------------------------------------ #
    def _model_fn(self, context_ids: Tensor, canvas: Canvas) -> Tensor:
        ctx = context_ids[-self.model.cfg.ctx_window :]
        c = ctx.numel()
        pad = self.model.cfg.ctx_window - c
        ctx_in = torch.cat(
            [torch.zeros(pad, dtype=torch.long, device=self.device), ctx]
        ).unsqueeze(0)
        pad_mask = torch.cat(
            [
                torch.ones(pad, dtype=torch.bool, device=self.device),
                torch.zeros(c, dtype=torch.bool, device=self.device),
            ]
        ).unsqueeze(0)
        logits = self.model(
            ctx_in,
            pad_mask,
            canvas.tokens.unsqueeze(0),
            canvas.masked.unsqueeze(0),
        )
        return logits[0]

    @torch.inference_mode()
    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        b = min(self.canvas_len, max_tokens)
        if b <= 0:
            return []

        warm = self._warm if committed == self._context else None
        self._warm = None
        self._context = list(committed)
        if warm is not None:
            warm = warm.sliced(0, b) if warm.length > b else warm.padded_to(b)
            if self.policy.mode == "adaptive" and warm.num_masked == warm.length:
                warm = None  # Trimming can remove the last reusable token.
                self.stats["restart_no_anchors"] += 1
        self._used_warm = warm is not None
        if warm is not None:
            canvas = warm
            steps = self.policy.repair_steps
            self.stats["warm_rounds"] += 1
        else:
            canvas = Canvas.empty(b, device=self.device)
            steps = self.denoise_steps

        ctx = torch.tensor(committed, dtype=torch.long, device=self.device)
        stats: DenoiseStats = denoise(
            self._model_fn,
            ctx,
            canvas,
            num_steps=steps,
            min_confidence=self.min_confidence,
            adaptive_stop=self.adaptive_stop,
        )
        self.stats["diffusion_forwards"] += stats.steps_used
        self.stats["rounds"] += 1
        self.stats["proposed_tokens"] += canvas.length

        self._last_canvas = canvas.clone()
        return canvas.as_draft()

    def observe(self, result: VerifyResult) -> None:
        self._context.extend(result.committed)
        # Credit a reconciled round when its repaired canvas yielded >=1 accept.
        if self._used_warm and result.n_draft_accepted > 0:
            self.stats["warm_rounds_recovered"] += 1
        self._warm = make_warm_canvas(
            self._last_canvas, result, self.policy, generator=self.gen
        )
        if self.policy.mode == "adaptive" and self._used_warm and result.n_draft_accepted == 0 and self._warm is not None:
            self._warm = None  # Retry reuse after a fresh block.
            self.stats["restart_failed_repair"] += 1

    def reset(self) -> None:
        self._last_canvas = None
        self._warm = None
        self._used_warm = False
        self._context = []

    def seed_request(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def finish_request(self) -> None:
        self.reset()
