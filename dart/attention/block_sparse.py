"""Block-sparse attention layout (docs/design.md §6.5, MiniMax-M3-style).

Context is divided into fixed blocks; each query block attends to the
top-k context blocks by a relevance score plus its own and adjacent
blocks.  v0.1 provides the layout/selection math in pure torch (testable
on CPU); the gather kernel that makes it fast is roadmap Triton work (docs/design.md §16).
"""

from __future__ import annotations

import torch
from torch import Tensor


def score_blocks(q: Tensor, k: Tensor, block_size: int) -> Tensor:
    """Mean-pooled block relevance scores.

    q [Tq, D] (current queries), k [Tk, D] -> scores [num_blocks].
    """
    num_blocks = (k.size(0) + block_size - 1) // block_size
    pad = num_blocks * block_size - k.size(0)
    if pad:
        k = torch.cat([k, torch.zeros(pad, k.size(1), dtype=k.dtype, device=k.device)])
    block_means = k.view(num_blocks, block_size, -1).mean(dim=1)  # [N, D]
    q_mean = q.mean(dim=0)  # [D]
    return block_means @ q_mean


def select_blocks(
    scores: Tensor, top_k: int, always_first: int = 1, always_last: int = 1
) -> Tensor:
    """Indices of blocks to keep: top-k by score plus mandatory sink/recent."""
    n = scores.numel()
    keep = set(range(min(always_first, n))) | {
        n - 1 - i for i in range(min(always_last, n))
    }
    k = min(top_k, n)
    keep |= set(torch.topk(scores, k).indices.tolist())
    return torch.tensor(sorted(keep), dtype=torch.long, device=scores.device)


def block_sparse_mask(
    kv_len: int, block_size: int, kept_blocks: Tensor, device: str = "cpu"
) -> Tensor:
    """Boolean [kv_len]; True where the token's block was selected."""
    mask = torch.zeros(kv_len, dtype=torch.bool, device=device)
    for b in kept_blocks.tolist():
        mask[b * block_size : min((b + 1) * block_size, kv_len)] = True
    return mask
