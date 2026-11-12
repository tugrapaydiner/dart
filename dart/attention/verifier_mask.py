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


def tree_mask(parent: list[int], prefix_len: int, device: str = "cpu") -> Tensor:
    """Boolean attention mask for tree-structured drafts.

    ``parent[i]`` is the in-tree parent index of packed draft token i
    (-1 = attaches directly to the committed prefix).  Token i may attend
    to the full prefix and to its ancestor chain (and itself).
    Parents must precede their children.
    """
    if type(prefix_len) is not int or prefix_len < 0:
        raise ValueError("prefix_len must be a nonnegative integer")
    n = len(parent)
    rows: list[list[bool]] = []
    for i, p in enumerate(parent):
        if type(p) is not int or not -1 <= p < i:
            raise ValueError("tree parents must be -1 or a preceding index")
        row = [False] * n if p < 0 else rows[p].copy()
        row[i] = True
        rows.append(row)
    allow = torch.ones(n, prefix_len + n, dtype=torch.bool, device=device)
    if n > 1:
        allow[:, prefix_len:] = torch.tensor(rows, dtype=torch.bool, device=device)  # One write for all ancestors.
    return allow
