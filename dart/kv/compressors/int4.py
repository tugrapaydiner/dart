"""INT4 KV preset: the aggressive draft-lane compression level.

Safe in exact mode only because the verifier runs on full KV
(docs/design.md §9.1: "Exact mode can use aggressive KV quantization for
drafting because the verifier catches mistakes").
"""

from __future__ import annotations

from dart.kv.cache import QuantSpec


def int4_spec(sink_len: int = 16, window_len: int = 32) -> QuantSpec:
    return QuantSpec(bits=4, sink_len=sink_len, sink_bits=8, window_len=window_len)


def int3_spec(sink_len: int = 16, window_len: int = 32) -> QuantSpec:
    """Experimental tier; expect acceptance to degrade measurably."""
    return QuantSpec(bits=3, sink_len=sink_len, sink_bits=8, window_len=window_len)
