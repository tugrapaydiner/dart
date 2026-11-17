"""Verifier attention masks.

The v0.1 block verifier rides on the model's standard causal mask (one
linear block of [pending, drafts] per round), so it needs nothing custom.
This module provides the mask algebra for the planned upgrades:

  * batched verification of multiple candidate blocks,
  * tree/branch verification (multiple drafts sharing a prefix),

both of which need a block-causal mask over a packed token layout.
"""

from __future__ import annotations

import torch
from torch import Tensor


def linear_block_mask(prefix_len: int, block_len: int, device: str = "cpu") -> Tensor:
    """Boolean [block_len, prefix_len + block_len] causal mask for one
    draft block appended after a committed prefix (the v0.1 layout)."""
    total = prefix_len + block_len
    q_pos = torch.arange(prefix_len, total, device=device).unsqueeze(1)
    k_pos = torch.arange(total, device=device).unsqueeze(0)
    return k_pos <= q_pos


def tree_mask(
    parent: list[int], prefix_len: int, device: str = "cpu", *, dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Attention mask for tree-structured drafts.

    ``parent[i]`` is the in-tree parent index of packed draft token i
    (-1 = attaches directly to the committed prefix).  Token i may attend
    to the full prefix and to its ancestor chain (and itself).
    Parents must precede their children. By default, True marks visible
    positions. A floating dtype uses zero for visible positions and the
    dtype's minimum for blocked positions.
    """
    if type(prefix_len) is not int or prefix_len < 0:
        raise ValueError("prefix_len must be a nonnegative integer")
    if dtype == torch.bool:
        visible, blocked = True, False
    elif isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        visible, blocked = 0.0, torch.finfo(dtype).min
    else:
        raise ValueError("tree mask dtype must be boolean or floating point")
    n = len(parent)
    rows = []
    for i, p in enumerate(parent):
        if type(p) is not int or not -1 <= p < i:
            raise ValueError("tree parents must be -1 or a preceding index")
        row = [blocked] * n if p < 0 else rows[p].copy()
        row[i] = visible
        rows.append(row)
    mask = torch.full((n, prefix_len + n), visible, dtype=dtype, device=device)
    if n > 1:
        mask[:, prefix_len:] = torch.tensor(rows, dtype=dtype, device=device)
    return mask
