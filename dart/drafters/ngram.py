"""N-gram / prompt-lookup drafter.

Zero-model speculative decoding: find the most recent earlier occurrence of
the current suffix n-gram in the sequence and propose the tokens that
followed it.  Surprisingly strong on code, RAG, and editing workloads,
where output repeats spans of the context - and completely free, which
makes it the canonical "cheap lane" for the scheduler.

With the tree verifier (verification/tree_verify.py), ``propose_tree``
returns *several* historical continuations of the suffix as candidate
branches - when the context contains more than one way the span continued,
the single verify pass can accept whichever one the target picks.
"""

from __future__ import annotations

from dart.runtime.types import VerifyResult
from dart.verification.tree_verify import DraftTree


class NGramDrafter:
    name = "ngram"
    requires_entropy = False

    def __init__(
        self,
        max_ngram: int = 4,
        min_ngram: int = 1,
        tree_width: int = 3,
        max_scan: int | None = None,
    ) -> None:
        """``max_scan`` bounds how far back (in tokens) the suffix search
        looks.  The scan is O(context x n) per round - linear-in-context
        drafting is fine at chat lengths but becomes the round bottleneck
        in the long-context regime DART targets, so cap it there (recency
        also correlates with match quality on code/edit workloads).  None
        scans the full context (v0.1 behavior)."""
        if min_ngram < 1 or max_ngram < min_ngram:
            raise ValueError("require 1 <= min_ngram <= max_ngram")
        if tree_width < 1:
            raise ValueError("tree_width must be >= 1")
        if max_scan is not None and max_scan < 1:
            raise ValueError("max_scan must be >= 1 (or None for unbounded)")
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.tree_width = tree_width
        self.max_scan = max_scan

    def _continuations(
        self, committed: list[int], max_tokens: int, limit: int
    ) -> list[list[int]]:
        """Up to ``limit`` distinct continuations of the current suffix,
        longest n-gram first, most recent occurrence first within an n."""
        seq_len = len(committed)
        scan_from = 0 if self.max_scan is None else max(0, seq_len - self.max_scan)
        candidates = committed[scan_from:seq_len - 1]
        candidates.reverse()
        found: list[list[int]] = []
        for n in range(min(self.max_ngram, seq_len - 1), self.min_ngram - 1, -1):
            pattern = committed[seq_len - n :]
            cursor, stop = 0, len(candidates) - n + 1
            while cursor < stop:
                try:
                    offset = candidates.index(pattern[-1], cursor, stop)  # Skip impossible suffix endings.
                except ValueError:
                    break
                start = seq_len - n - 1 - offset
                cursor = offset + 1
                if committed[start : start + n] == pattern:
                    cont = committed[start + n : start + n + max_tokens]
                    if cont and cont not in found:
                        found.append(cont)
                        if len(found) >= limit:
                            return found
            if found:
                break  # branches from one n keep the recency semantics clean
        return found

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        if max_tokens <= 0:
            return []
        found = self._continuations(committed, max_tokens, limit=1)
        return found[0] if found else []

    def propose_tree(self, committed: list[int], max_tokens: int) -> DraftTree:
        if max_tokens <= 0:
            return DraftTree([], [])
        branches = self._continuations(committed, max_tokens, self.tree_width)
        return DraftTree.from_branches(branches)

    def observe(self, result: VerifyResult) -> None:
        pass
