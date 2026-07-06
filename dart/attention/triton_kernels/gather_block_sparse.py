"""KV block-gather kernel for block-sparse attention (MiniMax-MSA-style).

After dart/attention/block_sparse.py selects which context blocks a query
should see, the KV rows of those blocks must be gathered into a dense
buffer for the attention kernel.  On GPU this gather is bandwidth-bound
and worth fusing; the torch reference is a strided index_select.

Same validation status as dequant_attention.py: reference is tested on
CPU; the Triton path is GPU-gated and unvalidated in this environment.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dart.attention.triton_kernels import HAS_TRITON

if HAS_TRITON:  # pragma: no cover - GPU only
    import triton
    import triton.language as tl


def gather_blocks_reference(
    kv: Tensor,  # [H, T, D]
    kept_blocks: Tensor,  # [N] long, block indices
    block_size: int,
) -> Tensor:
    """Concatenate the kept blocks' rows -> [H, N*block_size(<=), D]."""
    t = kv.size(1)
    rows = []
    for b in kept_blocks.tolist():
        rows.append(torch.arange(b * block_size, min((b + 1) * block_size, t)))
    idx = torch.cat(rows).to(kv.device)
    return kv.index_select(1, idx)


if HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _gather_kernel(
        kv_ptr, idx_ptr, out_ptr,
        D: tl.constexpr,
        stride_h, stride_t, out_stride_h, out_stride_t,
        n_rows,
    ):
        h = tl.program_id(0)
        r = tl.program_id(1)
        if r < n_rows:
            src_t = tl.load(idx_ptr + r)
            d_idx = tl.arange(0, D)
            row = tl.load(kv_ptr + h * stride_h + src_t * stride_t + d_idx)
            tl.store(out_ptr + h * out_stride_h + r * out_stride_t + d_idx, row)


def gather_blocks(kv: Tensor, kept_blocks: Tensor, block_size: int) -> Tensor:
    if not (HAS_TRITON and kv.is_cuda):  # pragma: no cover - GPU only
        raise RuntimeError(
            "gather_blocks requires triton + CUDA; use gather_blocks_reference"
        )
    t = kv.size(1)
    rows = []
    for b in kept_blocks.tolist():
        rows.append(torch.arange(b * block_size, min((b + 1) * block_size, t)))
    idx = torch.cat(rows).to(kv.device)
    h, _, d = kv.shape
    out = torch.empty(h, idx.numel(), d, device=kv.device, dtype=kv.dtype)
    _gather_kernel[(h, idx.numel())](
        kv, idx, out, D=d,
        stride_h=kv.stride(0), stride_t=kv.stride(1),
        out_stride_h=out.stride(0), out_stride_t=out.stride(1),
        n_rows=idx.numel(),
    )
    return out
