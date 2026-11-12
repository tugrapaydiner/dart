"""Quantized KV cache for the compressed-KV draft lane.

Design (KIVI-style aging buffer, append-only - no re-quantization):

  * the most recent ``window_len`` tokens stay in full precision in a ring
    buffer (recent tokens matter most for next-token prediction);
  * tokens aging out of the window are quantized once and appended to the
    quantized store;
  * the first ``sink_len`` tokens ("attention sinks" / global anchors) are
    quantized at ``sink_bits`` instead of ``bits`` (docs/design.md §9.4).

Quantization is asymmetric min/max per (head, token) vector over head_dim:
append-friendly, and the standard reference scheme for KV quantization.
v0.1 stores quantized values unpacked in uint8 for clarity; ``stats()``
reports both the logical (packed) and physical sizes so memory claims stay
honest.  Nibble packing is a roadmap kernel optimization (docs/design.md §16).

This cache is for the *draft* path only.  The verifier always runs on the
full-precision cache, which is what makes aggressive bits safe
(docs/design.md §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

try:
    from transformers.cache_utils import DynamicCache
except ImportError as e:  # pragma: no cover
    raise ImportError("dart requires `transformers>=5.16.1`") from e

_IMPL_VERSION = 2  # bumped when cache semantics change (sync sanity marker)


# --------------------------------------------------------------------- #
# Quantization primitives (per head, per token, over head_dim)
# --------------------------------------------------------------------- #
def quantize_blockwise(x: Tensor, bits: int) -> tuple[Tensor, Tensor, Tensor]:
    """x [B, H, T, D] -> (q uint8 [B,H,T,D], scale [B,H,T,1], zero [B,H,T,1])."""
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    levels = (1 << bits) - 1
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / levels
    q = torch.round((x - mn) / scale).clamp(0, levels).to(torch.uint8)
    return q, scale.to(torch.float32), mn.to(torch.float32)


def dequantize_blockwise(q: Tensor, scale: Tensor, zero: Tensor, dtype) -> Tensor:
    return (q.to(torch.float32) * scale + zero).to(dtype)


@runtime_checkable
class KVStore(Protocol):
    """Pluggable codec for aged-out KV tokens (the KVCompressor backend of
    docs/design.md §9).

    QuantizedKVCache is DART's realization of the section-12 ``KVCompressor``
    interface (compress = quantize-on-aging, append = ``update``,
    draft_attention = standard/triton attention over the dequantized K/V,
    stats = ``stats``).  The part that is genuinely swappable - and therefore
    worth pinning to a formal contract - is the per-layer store: any class
    satisfying ``KVStore`` can be passed as ``store_factory``.  ``QuantStore``
    (min/max integer) and ``turboquant_like.PolarVQStore`` (1-bit polar VQ)
    both conform; ``isinstance(store, KVStore)`` is enforced in tests.
    """

    @property
    def length(self) -> int: ...
    def append(self, x: Tensor, bits: int) -> None: ...
    def dequantize(self, dtype) -> Optional[Tensor]: ...
    def crop(self, length: int) -> None: ...
    def logical_bytes(self) -> int: ...


@dataclass
class QuantStore:
    """Append-only quantized token store for one layer's K or V."""

    q: Optional[Tensor] = None  # [B, H, T, D] uint8
    scale: Optional[Tensor] = None  # [B, H, T, 1] fp32
    zero: Optional[Tensor] = None  # [B, H, T, 1] fp32
    bits_per_token: Optional[Tensor] = None  # [T] uint8 (for stats/policy)

    @property
    def length(self) -> int:
        return 0 if self.q is None else self.q.size(-2)

    def append(self, x: Tensor, bits: int) -> None:
        q, scale, zero = quantize_blockwise(x, bits)
        b = torch.full((x.size(-2),), bits, dtype=torch.uint8, device=x.device)
        if self.q is None:
            self.q, self.scale, self.zero, self.bits_per_token = q, scale, zero, b
        else:
            self.q = torch.cat([self.q, q], dim=-2)
            self.scale = torch.cat([self.scale, scale], dim=-2)
            self.zero = torch.cat([self.zero, zero], dim=-2)
            self.bits_per_token = torch.cat([self.bits_per_token, b])

    def dequantize(self, dtype) -> Optional[Tensor]:
        if self.q is None:
            return None
        return dequantize_blockwise(self.q, self.scale, self.zero, dtype)

    def crop(self, length: int) -> None:
        if self.q is None or length >= self.length:
            return
        if length == 0:
            self.q = self.scale = self.zero = self.bits_per_token = None
            return
        self.q = self.q[..., :length, :]
        self.scale = self.scale[..., :length, :]
        self.zero = self.zero[..., :length, :]
        self.bits_per_token = self.bits_per_token[:length]

    def logical_bytes(self) -> int:
        if self.q is None:
            return 0
        b, h, t, d = self.q.shape
        data_bits = int(self.bits_per_token.to(torch.int64).sum().item()) * b * h * d
        meta = b * h * t * 2 * 4  # scale + zero, fp32
        return data_bits // 8 + meta


@dataclass
class PrecisionPolicy:
    """Per-token / per-layer bit-width policy (docs/design.md §9.4).

    ``bits_for`` returns the precision a token should be aged out at.  Rules
    compose by *taking the maximum* - precision is only ever raised above the
    default, never lowered - so a token that is both a sink and a syntax
    delimiter simply gets the highest applicable bit-width:

      * sink prefix          - first ``sink_len`` tokens (attention sinks);
      * syntax tokens        - ids in ``syntax_token_ids`` (braces, JSON
                               punctuation, tool-call delimiters);
      * high-entropy tokens  - predictive entropy >= ``high_entropy_threshold``
                               (the model was unsure here, so keep KV crisp);
      * sensitive layers     - the first and/or last ``last_layers_hi`` layers.

    Token-id and entropy rules are inert when that metadata is unavailable
    (e.g. during the prompt prefill, before logits exist), so the policy
    degrades gracefully to sink+layer+default precision.
    """

    default_bits: int = 4
    sink_len: int = 16
    sink_bits: int = 8
    syntax_token_ids: frozenset = frozenset()
    syntax_bits: int = 8
    high_entropy_threshold: float = float("inf")  # nats; inf disables the rule
    high_entropy_bits: int = 8
    num_layers: Optional[int] = None  # required for the last-N-layers rule
    last_layers_hi: int = 0
    first_layer_hi: bool = False
    layer_hi_bits: int = 8

    def bits_for(
        self,
        position: int,
        layer_idx: int,
        token_id: Optional[int] = None,
        entropy: Optional[float] = None,
    ) -> int:
        bits = self.default_bits
        if position < self.sink_len:
            bits = max(bits, self.sink_bits)
        if token_id is not None and token_id in self.syntax_token_ids:
            bits = max(bits, self.syntax_bits)
        if entropy is not None and entropy >= self.high_entropy_threshold:
            bits = max(bits, self.high_entropy_bits)
        if self.num_layers is not None:
            if self.first_layer_hi and layer_idx == 0:
                bits = max(bits, self.layer_hi_bits)
            if layer_idx >= self.num_layers - self.last_layers_hi:
                bits = max(bits, self.layer_hi_bits)
        return max(2, min(8, bits))  # quantize_blockwise domain


@dataclass
class QuantSpec:
    bits: int = 4  # tokens older than the window
    sink_len: int = 16  # leading tokens treated as attention sinks
    sink_bits: int = 8
    window_len: int = 32  # trailing tokens kept in full precision
    # Optional fine-grained precision policy (14.3).  When None the cache
    # builds the default sink+middle policy below, preserving v0.1 behaviour.
    policy: Optional[PrecisionPolicy] = None


class QuantizedKVCache(DynamicCache):
    """transformers-compatible cache holding quantized old KV + fp window.

    Storage lives in ``_kq/_vq`` (quantized) and ``_kw/_vw`` (fp window);
    the inherited DynamicCache layer list stays empty, so every interface
    method the model calls (update / get_seq_length / get_mask_sizes /
    crop) is overridden here.  ``update`` returns fully dequantized K/V so
    any HF attention implementation works unchanged - correctness-first
    reference semantics, kernels later (docs/design.md §16).
    """

    def __init__(self, spec: QuantSpec, store_factory=None, meta=None) -> None:
        """``store_factory`` swaps the aged-token codec: any class satisfying
        ``KVStore`` (append/dequantize/crop/length/logical_bytes), e.g.
        kv.compressors.turboquant_like.PolarVQStore.

        ``meta`` is an optional per-position metadata provider exposing
        ``token_id_at(pos)`` and ``entropy_at(pos)`` (both may return None);
        it lets the PrecisionPolicy raise bits for syntax / high-entropy
        tokens.  The compressed-KV drafter supplies it; without it the policy
        uses only sink + layer + default precision.
        """
        super().__init__()
        self.spec = spec
        self._policy = spec.policy or PrecisionPolicy(
            default_bits=spec.bits, sink_len=spec.sink_len, sink_bits=spec.sink_bits
        )
        self._meta = meta
        self._store_factory = store_factory if store_factory is not None else QuantStore
        self._kq: list = []
        self._vq: list = []
        self._kw: list[Optional[Tensor]] = []
        self._vw: list[Optional[Tensor]] = []

    # -------------------------------------------------------------- #
    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._kq) <= layer_idx:
            self._kq.append(self._store_factory())
            self._vq.append(self._store_factory())
            self._kw.append(None)
            self._vw.append(None)

    def _layer_len(self, layer_idx: int) -> int:
        if layer_idx >= len(self._kq):
            return 0
        w = self._kw[layer_idx]
        return self._kq[layer_idx].length + (0 if w is None else w.size(-2))

    def _bits_at(self, position: int, layer_idx: int) -> int:
        tid = self._meta.token_id_at(position) if self._meta is not None else None
        ent = self._meta.entropy_at(position) if self._meta is not None else None
        return self._policy.bits_for(position, layer_idx, tid, ent)

    def _age_out(self, layer_idx: int) -> None:
        """Quantize tokens that overflow the fp window.

        Each aged token's bit-width comes from the PrecisionPolicy (sink /
        syntax / high-entropy / sensitive-layer); tokens keep their absolute
        position.  Consecutive tokens of equal precision are batched into one
        append, so the common case (uniform precision, or a contiguous sink
        prefix) is still one or two appends.
        """
        spec = self.spec
        kw, vw = self._kw[layer_idx], self._vw[layer_idx]
        if kw is None or kw.size(-2) <= spec.window_len:
            return
        n_out = kw.size(-2) - spec.window_len
        k_out, self._kw[layer_idx] = kw[..., :n_out, :], kw[..., n_out:, :]
        v_out, self._vw[layer_idx] = vw[..., :n_out, :], vw[..., n_out:, :]

        start = self._kq[layer_idx].length  # absolute position of first aged token
        bits = [self._bits_at(start + i, layer_idx) for i in range(n_out)]
        i = 0
        while i < n_out:
            j = i + 1
            while j < n_out and bits[j] == bits[i]:
                j += 1
            self._kq[layer_idx].append(k_out[..., i:j, :], bits[i])
            self._vq[layer_idx].append(v_out[..., i:j, :], bits[i])
            i = j

    # -------------------------------------------------------------- #
    # transformers Cache API (v4.40+ and v5 names)
    # -------------------------------------------------------------- #
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._ensure_layer(layer_idx)
        if self._kw[layer_idx] is None:
            self._kw[layer_idx] = key_states
            self._vw[layer_idx] = value_states
        else:
            self._kw[layer_idx] = torch.cat([self._kw[layer_idx], key_states], dim=-2)
            self._vw[layer_idx] = torch.cat([self._vw[layer_idx], value_states], dim=-2)
        self._age_out(layer_idx)

        dtype = key_states.dtype
        k_old = self._kq[layer_idx].dequantize(dtype)
        v_old = self._vq[layer_idx].dequantize(dtype)
        k = self._kw[layer_idx] if k_old is None else torch.cat(
            [k_old, self._kw[layer_idx]], dim=-2
        )
        v = self._vw[layer_idx] if v_old is None else torch.cat(
            [v_old, self._vw[layer_idx]], dim=-2
        )
        return k, v

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        return self._layer_len(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx: int = 0):
        """v5 mask sizing: (kv_length, kv_offset), called *before* this
        layer's update(), so kv_length = stored tokens + incoming block.

        Some call sites pass cache_position tensors instead of an int;
        normalize defensively.
        """
        if isinstance(query_length, Tensor):
            query_length = int(query_length.numel())
        return self._layer_len(layer_idx) + int(query_length), 0

    def get_max_length(self) -> Optional[int]:  # pragma: no cover
        return None

    def get_max_cache_shape(self, layer_idx: int = 0):  # pragma: no cover
        return None

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self.get_seq_length() + max_length)
        for li in range(len(self._kq)):
            qlen = self._kq[li].length
            if max_length <= qlen:
                self._kq[li].crop(max_length)
                self._vq[li].crop(max_length)
                self._kw[li] = None
                self._vw[li] = None
            else:
                keep_w = max_length - qlen
                if self._kw[li] is not None and self._kw[li].size(-2) > keep_w:
                    self._kw[li] = self._kw[li][..., :keep_w, :]
                    self._vw[li] = self._vw[li][..., :keep_w, :]

    # -------------------------------------------------------------- #
    def stats(self) -> dict:
        logical = sum(s.logical_bytes() for s in self._kq + self._vq)
        fp_bytes = sum(
            t.numel() * t.element_size()
            for t in self._kw + self._vw
            if t is not None
        )
        out = {
            "seq_len": self.get_seq_length(),
            "store": type(self._kq[0]).__name__ if self._kq else None,
            "quantized_tokens": self._kq[0].length if self._kq else 0,
            "window_tokens": (
                0 if not self._kw or self._kw[0] is None else self._kw[0].size(-2)
            ),
            "logical_compressed_bytes": logical + fp_bytes,
            # JSON-native view of the spec; the PrecisionPolicy object is
            # summarized by a flag rather than embedded as an object.
            "spec": {
                k: v for k, v in vars(self.spec).items() if k != "policy"
            },
            "precision_policy": self.spec.policy is not None,
        }
        # Unpacked-physical accounting only applies to the minmax QuantStore
        # (its v0 storage is unpacked uint8); packed stores ARE their
        # logical size.
        if all(hasattr(s, "q") for s in self._kq + self._vq):
            physical = sum(
                s.q.numel() + s.scale.numel() * 4 + s.zero.numel() * 4
                for s in self._kq + self._vq
                if s.q is not None
            )
            out["physical_bytes_v0_unpacked"] = physical + fp_bytes
        return out
