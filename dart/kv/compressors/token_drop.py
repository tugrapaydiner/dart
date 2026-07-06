"""Token-drop compression: sink + recent-window sequence compaction.

StreamingLLM-style sparsity for the draft path: keep the first
``sink_len`` tokens and the most recent ``window_len`` tokens, drop the
middle entirely.  Implemented as *sequence* compaction rather than a Cache
subclass: the drafter prefills a compacted sequence, which keeps it
independent of transformers cache internals and version churn.

Positional caveat (documented, deliberate): compaction shifts the window
tokens' positions left, so the draft model sees slightly different RoPE
geometry than the full model.  That is a fidelity cost of this lane,
visible directly in its acceptance rate - which is exactly the metric the
compressor ladder optimizes (docs/design.md §9.3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenDropSpec:
    sink_len: int = 16
    window_len: int = 256

    @property
    def budget(self) -> int:
        return self.sink_len + self.window_len


def compact_sequence(tokens: list[int], spec: TokenDropSpec) -> tuple[list[int], bool]:
    """Return (compacted tokens, whether anything was dropped)."""
    if len(tokens) <= spec.budget:
        return list(tokens), False
    return tokens[: spec.sink_len] + tokens[-spec.window_len :], True
