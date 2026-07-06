"""Sliding-window + global-anchor attention mask (docs/design.md §6.5).

Pure-torch mask constructor: position i may attend to (a) the first
``num_global`` tokens, (b) the last ``window`` tokens before it, (c) itself.
Usable as an additive bias for eager attention or as a boolean layout for
future Triton kernels - the simple baseline the doc says to build first.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sliding_global_mask(
    q_len: int,
    kv_len: int,
    window: int,
    num_global: int,
    device: str = "cpu",
) -> Tensor:
    """Boolean [q_len, kv_len]; True = may attend.  Causal by construction.

    Query position i corresponds to absolute position kv_len - q_len + i
    (standard incremental-decode alignment).
    """
    if kv_len < q_len:
        raise ValueError("kv_len must be >= q_len")
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    causal = k_pos <= q_pos
    in_window = k_pos > (q_pos - window)
    is_global = k_pos < num_global
    return causal & (in_window | is_global)
