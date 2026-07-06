"""Fused dequantize-and-attend decode kernel for the quantized KV store.

Motivation (docs/design.md §9 and §16): the v0.1 QuantizedKVCache dequantizes
the whole store into fp tensors on every update - correct, but it gives
back the memory-bandwidth win at the worst moment.  This kernel reads the
uint8 codes + per-token scale/zero directly and dequantizes in registers
inside the attention loop, so HBM traffic stays at the compressed rate.

Scope: single query position (decode step), one program per head, fp32
accumulation, minmax-asymmetric codes as produced by quantize_blockwise.
The torch reference below is the contract; the Triton path must match it
to ~1e-3 (fp32 vs fused rounding) - see tests.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from dart.attention.triton_kernels import HAS_TRITON

if HAS_TRITON:  # pragma: no cover - GPU only
    import triton
    import triton.language as tl


def dequant_attention_reference(
    q: Tensor,  # [H, D] fp32 query at the current decode position
    k_q: Tensor,  # [H, T, D] uint8 quantized keys
    k_scale: Tensor,  # [H, T, 1] fp32
    k_zero: Tensor,  # [H, T, 1] fp32
    v_q: Tensor,  # [H, T, D] uint8 quantized values
    v_scale: Tensor,  # [H, T, 1] fp32
    v_zero: Tensor,  # [H, T, 1] fp32
) -> Tensor:
    """Pure-torch contract for the fused kernel.  Returns [H, D]."""
    k = k_q.float() * k_scale + k_zero  # [H, T, D]
    v = v_q.float() * v_scale + v_zero
    scores = torch.einsum("hd,htd->ht", q.float(), k) / math.sqrt(q.size(-1))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,htd->hd", attn, v)


if HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _dequant_attn_kernel(
        q_ptr, kq_ptr, ks_ptr, kz_ptr, vq_ptr, vs_ptr, vz_ptr, out_ptr,
        T, D: tl.constexpr, BLOCK_T: tl.constexpr,
        stride_qh, stride_kh, stride_kt,
        stride_sh, stride_st,
        sm_scale,
    ):
        h = tl.program_id(0)
        d_idx = tl.arange(0, D)
        q = tl.load(q_ptr + h * stride_qh + d_idx).to(tl.float32)

        m_i = -float("inf")  # running max (online softmax)
        l_i = 0.0  # running denominator
        acc = tl.zeros([D], dtype=tl.float32)

        for t0 in range(0, T, BLOCK_T):
            t_idx = t0 + tl.arange(0, BLOCK_T)
            t_mask = t_idx < T

            k_off = h * stride_kh + t_idx[:, None] * stride_kt + d_idx[None, :]
            kq = tl.load(kq_ptr + k_off, mask=t_mask[:, None], other=0).to(tl.float32)
            ks = tl.load(ks_ptr + h * stride_sh + t_idx * stride_st,
                         mask=t_mask, other=1.0)
            kz = tl.load(kz_ptr + h * stride_sh + t_idx * stride_st,
                         mask=t_mask, other=0.0)
            k = kq * ks[:, None] + kz[:, None]  # dequant in registers

            s = tl.sum(k * q[None, :], axis=1) * sm_scale
            s = tl.where(t_mask, s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)

            vq = tl.load(vq_ptr + k_off, mask=t_mask[:, None], other=0).to(tl.float32)
            vs = tl.load(vs_ptr + h * stride_sh + t_idx * stride_st,
                         mask=t_mask, other=1.0)
            vz = tl.load(vz_ptr + h * stride_sh + t_idx * stride_st,
                         mask=t_mask, other=0.0)
            v = vq * vs[:, None] + vz[:, None]

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        tl.store(out_ptr + h * stride_qh + d_idx, acc / l_i)


def dequant_attention(
    q: Tensor, k_q: Tensor, k_scale: Tensor, k_zero: Tensor,
    v_q: Tensor, v_scale: Tensor, v_zero: Tensor,
    block_t: int = 64,
) -> Tensor:
    """Fused path; falls back loudly (raises) rather than silently slowly."""
    if not (HAS_TRITON and q.is_cuda):  # pragma: no cover - GPU only
        raise RuntimeError(
            "dequant_attention requires triton + CUDA tensors; use "
            "dequant_attention_reference for the correctness path"
        )
    h, t, d = k_q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    _dequant_attn_kernel[(h,)](
        q, k_q, k_scale.squeeze(-1).contiguous(), k_zero.squeeze(-1).contiguous(),
        v_q, v_scale.squeeze(-1).contiguous(), v_zero.squeeze(-1).contiguous(),
        out,
        t, D=d, BLOCK_T=block_t,
        stride_qh=q.stride(0),
        stride_kh=k_q.stride(0), stride_kt=k_q.stride(1),
        stride_sh=t, stride_st=1,
        sm_scale=1.0 / (d ** 0.5),
    )
    return out
