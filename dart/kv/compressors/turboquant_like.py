"""TurboQuant-inspired extreme KV compression (docs/design.md §9.1).

A simplified PolarQuant/QJL-flavor scheme, implemented honestly rather
than name-dropped:

  * each K/V vector x [D] is rotated by a fixed random orthogonal matrix R
    (seeded; Johnson-Lindenstrauss-style coordinate mixing so no single
    coordinate dominates);
  * the rotated vector is stored as 1 SIGN BIT per dimension (packed - the
    one compressor in DART v0.1 with real bit-packing) plus one fp32
    magnitude scalar per vector: polar split of direction and length;
  * dequantization: x_hat = R^T (sign * mean|y|), the least-squares scale
    for a sign-only direction code.

Rate: D/8 bytes + 4 bytes per (token, head) vs 2*D bytes bf16 - about 13x
for D=64.  Accuracy is far below int4; that is the point: this lane maps
the far end of the acceptance-vs-bits curve (docs/design.md §9.3), and the exact
verifier makes it safe to even try.

Plugs into QuantizedKVCache via ``store_factory`` so the aging-window /
sink machinery is shared; sinks stay at int8 inside this store for the
same reason hybrid_policy keeps them high precision.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from dart.kv.cache import QuantSpec, QuantStore


def random_orthogonal(dim: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, generator=g)
    q, r = torch.linalg.qr(a)
    # Fix signs so the decomposition is unique/deterministic.
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def pack_signs(signs: Tensor) -> Tensor:
    """bool [..., D] -> uint8 [..., D/8] (D must be a multiple of 8)."""
    *lead, d = signs.shape
    if d % 8:
        raise ValueError("head_dim must be a multiple of 8 for sign packing")
    bits = signs.to(torch.uint8).reshape(*lead, d // 8, 8)
    weights = (1 << torch.arange(8, dtype=torch.uint8, device=signs.device))
    return (bits * weights).sum(dim=-1, dtype=torch.uint8)


def unpack_signs(packed: Tensor, dim: int) -> Tensor:
    """uint8 [..., D/8] -> float {-1,+1} [..., D]."""
    weights = (1 << torch.arange(8, dtype=torch.uint8, device=packed.device))
    bits = packed.unsqueeze(-1).bitwise_and(weights).ne(0)
    out = bits.reshape(*packed.shape[:-1], dim).float()
    return out * 2.0 - 1.0


class PolarVQStore:
    """Sign-bit + magnitude store; same interface as QuantStore.

    ``append(x, bits)``: bits >= 8 routes the block to an embedded int8
    QuantStore (sink tokens); anything lower takes the 1-bit polar path.
    """

    def __init__(self, rotation_seed: int = 0) -> None:
        self._rot: Optional[Tensor] = None
        self._seed = rotation_seed
        self.packed: Optional[Tensor] = None  # [B,H,T,D/8] uint8
        self.scale: Optional[Tensor] = None  # [B,H,T,1] fp32 (mean |y|)
        self._sink = QuantStore()
        self._order: list[tuple[str, int]] = []  # ("polar"|"sink", count) runs

    # -- helpers ----------------------------------------------------- #
    def _rotation(self, d: int, device, dtype) -> Tensor:
        if self._rot is None or self._rot.size(0) != d:
            self._rot = random_orthogonal(d, self._seed)
        return self._rot.to(device=device, dtype=dtype)

    @property
    def length(self) -> int:
        polar = 0 if self.packed is None else self.packed.size(-2)
        return polar + self._sink.length

    def append(self, x: Tensor, bits: int) -> None:
        if bits >= 8:
            self._sink.append(x, bits)
            self._push_order("sink", x.size(-2))
            return
        r = self._rotation(x.size(-1), x.device, torch.float32)
        y = x.float() @ r  # [B,H,T,D] rotated
        signs = y > 0
        scale = y.abs().mean(dim=-1, keepdim=True)  # least-squares magnitude
        packed = pack_signs(signs)
        if self.packed is None:
            self.packed, self.scale = packed, scale
        else:
            self.packed = torch.cat([self.packed, packed], dim=-2)
            self.scale = torch.cat([self.scale, scale], dim=-2)
        self._push_order("polar", x.size(-2))

    def _push_order(self, kind: str, n: int) -> None:
        if self._order and self._order[-1][0] == kind:
            self._order[-1] = (kind, self._order[-1][1] + n)
        else:
            self._order.append((kind, n))

    def dequantize(self, dtype) -> Optional[Tensor]:
        if self.packed is None and self._sink.length == 0:
            return None
        parts: list[Tensor] = []
        polar_used = sink_used = 0
        for kind, n in self._order:
            if kind == "sink":
                full = self._sink.dequantize(dtype)
                parts.append(full[..., sink_used : sink_used + n, :])
                sink_used += n
            else:
                packed = self.packed[..., polar_used : polar_used + n, :]
                scale = self.scale[..., polar_used : polar_used + n, :]
                d = packed.size(-1) * 8
                r = self._rotation(d, packed.device, torch.float32)
                y_hat = unpack_signs(packed, d) * scale
                parts.append((y_hat @ r.T).to(dtype))
                polar_used += n
        return torch.cat(parts, dim=-2)

    def crop(self, length: int) -> None:
        if length >= self.length:
            return
        # Walk runs from the front, keeping `length` tokens total.
        kept_polar = kept_sink = 0
        remaining = length
        new_order: list[tuple[str, int]] = []
        for kind, n in self._order:
            take = min(n, remaining)
            if take > 0:
                new_order.append((kind, take))
                if kind == "polar":
                    kept_polar += take
                else:
                    kept_sink += take
            remaining -= take
        self._order = new_order
        self._sink.crop(kept_sink)
        if self.packed is not None:
            if kept_polar == 0:
                self.packed = self.scale = None
            else:
                self.packed = self.packed[..., :kept_polar, :]
                self.scale = self.scale[..., :kept_polar, :]

    def logical_bytes(self) -> int:
        polar = 0
        if self.packed is not None:
            polar = self.packed.numel() + self.scale.numel() * 4
        return polar + self._sink.logical_bytes()


def polar_vq_spec(sink_len: int = 16, window_len: int = 32) -> QuantSpec:
    """Spec preset: 1-bit polar VQ for aged tokens, int8 sinks, fp window.

    Use with ``QuantizedKVCache(spec, store_factory=PolarVQStore)``.
    ``bits=1`` is a routing label here (any value < 8 takes the polar
    path); QuantSpec validation is bypassed by the store, not the spec.
    """
    return QuantSpec(bits=2, sink_len=sink_len, sink_bits=8, window_len=window_len)
