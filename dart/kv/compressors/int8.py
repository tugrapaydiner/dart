"""INT8 KV preset: the safe-fast compression level (docs/design.md §9.1).

Stage 1 of the compressor ladder - used to validate infrastructure before
pushing to aggressive bit-widths.
"""

from __future__ import annotations

from dart.kv.cache import QuantSpec


def int8_spec(sink_len: int = 16, window_len: int = 32) -> QuantSpec:
    return QuantSpec(bits=8, sink_len=sink_len, sink_bits=8, window_len=window_len)
