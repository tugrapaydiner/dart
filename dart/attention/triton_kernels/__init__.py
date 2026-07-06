"""Triton kernels (docs/design.md §16).

STATUS - read this before benchmarking:
  * kernel sources are written and shipped here;
  * every kernel has a pure-torch reference implementation, and the CPU
    test suite proves the reference against eager attention math;
  * the Triton paths themselves are gated on CUDA and have NOT been
    compiled/validated in this repository's CPU-only development
    environment.  Run tests/test_kv_and_components.py -k triton on a GPU
    box before trusting them; Nsight traces are the roadmap deliverable (docs/design.md §16).
"""

from __future__ import annotations

try:  # pragma: no cover - environment probe
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
except Exception:  # ImportError or driver issues
    HAS_TRITON = False

__all__ = ["HAS_TRITON"]
