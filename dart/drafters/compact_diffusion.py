from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dart.diffusion.canvas import Canvas
from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.diffusion_small import DiffusionDrafter
from dart.models.target import TargetModel
from dart.runtime.types import VerifyResult

MAX_LEARNED_PARAMETERS = 5_000_000


@dataclass(frozen=True)
class CompactDrafterConfig:
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 6
    ffn_mult: int = 4
    ctx_window: int = 32
    max_canvas: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("hidden_size", "num_layers", "num_heads", "ffn_mult", "ctx_window", "max_canvas"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")

    def parameter_count(self, target_hidden_size: int) -> int:
        d, m = self.hidden_size, self.ffn_mult
        layer = (4 + 2 * m) * d * d + (m + 9) * d
        return self.num_layers * layer + 3 * target_hidden_size * d + (self.ctx_window + self.max_canvas + 9) * d + target_hidden_size


class CompactDiffusionModel(nn.Module):
    """Canvas denoiser sharing the frozen target's embedding and output weights."""

    def __init__(self, target: TargetModel, cfg: CompactDrafterConfig | None = None):
        super().__init__()
        self.cfg = cfg or CompactDrafterConfig()
        inputs = target.model.get_input_embeddings().weight
        outputs = target.model.get_output_embeddings()
        if inputs.ndim != 2 or outputs.weight.shape != inputs.shape:
            raise ValueError("target input and output projections must have matching vocabulary and hidden sizes")
        if inputs.requires_grad or any(parameter.requires_grad for parameter in outputs.parameters()):
            raise ValueError("shared target projections must be frozen")
        self.vocab_size, self.target_hidden_size = inputs.shape
        self.mask_token_id = self.vocab_size
        if self.cfg.parameter_count(self.target_hidden_size) > MAX_LEARNED_PARAMETERS:
            raise ValueError("compact drafter exceeds the 5M learned-parameter limit")
        self._target = weakref.ref(target)  # Borrowed weights stay out of state_dict and module.to().
        d, device = self.cfg.hidden_size, inputs.device
        self.input_proj = nn.Linear(self.target_hidden_size, d, device=device)
        self.feature_proj = nn.Linear(self.target_hidden_size, d, device=device)
        self.output_proj = nn.Linear(d, self.target_hidden_size, device=device)
        self.mask_emb = nn.Parameter(torch.empty(d, device=device))
        nn.init.normal_(self.mask_emb, std=0.02)
        self.pos_emb = nn.Embedding(1 + self.cfg.ctx_window + self.cfg.max_canvas, d, device=device)
        self.type_emb = nn.Embedding(3, d, device=device)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=self.cfg.num_heads, dim_feedforward=d * self.cfg.ffn_mult,
                dropout=self.cfg.dropout, activation="gelu", batch_first=True,
                norm_first=True, device=device,
            ) for _ in range(self.cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(d, device=device)

    def forward(
        self, context_ids: Tensor, context_pad_mask: Tensor,
        canvas_tokens: Tensor, canvas_masked: Tensor, target_hidden: Tensor,
    ) -> Tensor:
        target = self._target()
        if target is None:
            raise RuntimeError("the shared target must remain alive")
        if context_ids.ndim != 2 or canvas_tokens.ndim != 2:
            raise ValueError("context and canvas must have shape [batch, tokens]")
        n, c = context_ids.shape
        b = canvas_tokens.size(1)
        if n < 1 or not 1 <= c <= self.cfg.ctx_window or not 1 <= b <= self.cfg.max_canvas:
            raise ValueError("context or canvas exceeds the configured window")
        if canvas_tokens.size(0) != n or context_pad_mask.shape != context_ids.shape or canvas_masked.shape != canvas_tokens.shape:
            raise ValueError("context, canvas, and mask shapes do not match")
        if target_hidden.shape != (n, self.target_hidden_size):
            raise ValueError("target_hidden must have shape [batch, target_hidden_size]")
        if context_pad_mask.dtype != torch.bool or canvas_masked.dtype != torch.bool:
            raise ValueError("padding and canvas masks must be boolean")
        inputs = target.model.get_input_embeddings().weight
        outputs = target.model.get_output_embeddings()
        device, dtype = self.mask_emb.device, self.mask_emb.dtype
        if any(tensor.device != device for tensor in (context_ids, context_pad_mask, canvas_tokens, canvas_masked, target_hidden, inputs, outputs.weight)):
            raise ValueError("compact model, target projections, and inputs must share a device")

        features = target_hidden.detach().to(dtype=dtype)
        if torch.is_grad_enabled() and torch.is_inference(features):
            features = features.clone()
        context = torch.where(context_pad_mask, 0, context_ids)
        canvas = torch.where(canvas_masked, 0, canvas_tokens)
        embedded = self.input_proj(F.embedding(torch.cat([context, canvas], dim=1), inputs.detach()).to(dtype=dtype))
        canvas_emb = torch.where(canvas_masked.unsqueeze(-1), self.mask_emb, embedded[:, c:])
        x = torch.cat([self.feature_proj(features).unsqueeze(1), embedded[:, :c], canvas_emb], dim=1)
        positions = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(1 + self.cfg.ctx_window - c, 1 + self.cfg.ctx_window + b, device=device),
        ])
        types = torch.tensor([0] + [1] * c + [2] * b, device=device)
        x = x + self.pos_emb(positions) + self.type_emb(types)
        padding = torch.cat([
            torch.zeros(n, 1, dtype=torch.bool, device=device), context_pad_mask,
            torch.zeros_like(canvas_masked),
        ], dim=1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding)
        future = features.unsqueeze(1) + self.output_proj(self.norm(x[:, 1 + c:]))
        bias = getattr(outputs, "bias", None)
        return F.linear(future.to(dtype=outputs.weight.dtype), outputs.weight.detach(), bias.detach() if bias is not None else None).float()


class CompactDiffusionDrafter(DiffusionDrafter):
    """Inference lane using target features from prefill and verification."""

    def __init__(
        self, model: CompactDiffusionModel, *, canvas_len: int = 4,
        denoise_steps: int = 1, min_confidence: float = 0.65,
        adaptive_stop: bool = True, reconcile: ReconcilePolicy | None = None,
        seed: int = 0,
    ):
        target = model._target()
        if target is None or not target.record_hidden:
            raise ValueError("compact drafting requires a live TargetModel(record_hidden=True)")
        for name, value in (("canvas_len", canvas_len), ("denoise_steps", denoise_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        super().__init__(
            model, canvas_len=canvas_len, denoise_steps=denoise_steps,
            min_confidence=min_confidence, adaptive_stop=adaptive_stop,
            reconcile=reconcile, device=target.device, seed=seed,
        )
        self.target = target
        self._hidden: Tensor | None = None

    def _model_fn(self, context_ids: Tensor, canvas: Canvas) -> Tensor:
        if self._hidden is None:
            raise RuntimeError("call after_prefill before compact drafting")
        context = context_ids[-self.model.cfg.ctx_window:].unsqueeze(0)
        return self.model(
            context, torch.zeros_like(context, dtype=torch.bool),
            canvas.tokens.unsqueeze(0), canvas.masked.unsqueeze(0),
            self._hidden.unsqueeze(0),
        )[0]

    def _read_hidden(self, index: int) -> None:
        block = self.target.last_hidden_block
        if block is None:
            raise RuntimeError("target hidden states are unavailable")
        self._hidden = block[index].detach()

    def after_prefill(self) -> None:
        self.reset()
        self._read_hidden(-1)

    def observe(self, result: VerifyResult) -> None:
        super().observe(result)
        self._read_hidden(result.hidden_index)

    def after_verify(self, result: VerifyResult) -> None:
        self.reset()  # Another lane consumed the old repair context.
        self._read_hidden(result.hidden_index)

    def reset(self) -> None:
        super().reset()
        self._hidden = None
