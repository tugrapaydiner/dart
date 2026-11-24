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

from collections.abc import Collection, Iterator
from numbers import Integral

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
        if isinstance(tree_width, bool) or not isinstance(tree_width, Integral) or tree_width < 1:
            raise ValueError("tree_width must be an integer >= 1")
        if max_scan is not None and max_scan < 1:
            raise ValueError("max_scan must be >= 1 (or None for unbounded)")
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.tree_width = int(tree_width)
        self.max_scan = max_scan

    def _continuations(
        self, committed: list[int], max_tokens: int, limit: int, eos_token_ids: Collection[int] = ()
    ) -> Iterator[list[int]]:
        """Up to ``limit`` distinct continuations of the current suffix,
        longest n-gram first, most recent occurrence first within an n."""
        if max_tokens <= 0:
            return
        seq_len = len(committed)
        scan_from = 0 if self.max_scan is None else max(0, seq_len - self.max_scan)
        stop = seq_len - scan_from - self.min_ngram
        if stop <= 0:
            return
        candidates = committed[scan_from:seq_len - 1]
        candidates.reverse()
        last = committed[-1]
        try:
            first = candidates.index(last, 0, stop)  # Every suffix ends in this token.
        except ValueError:
            return
        found: list[list[int]] = []
        longest = min(self.max_ngram, len(candidates) - first)
        for n in range(longest, self.min_ngram - 1, -1):
            pattern = committed[seq_len - n :]
            offset, stop = first, len(candidates) - n + 1
            while True:
                start = seq_len - n - 1 - offset
                if committed[start : start + n] == pattern:
                    cont = committed[start + n : start + n + max_tokens]
                    if eos_token_ids:
                        for i, token in enumerate(cont):
                            if int(token) in eos_token_ids:
                                cont = cont[:i + 1]  # Keep EOS as the last candidate.
                                break
                    if cont and cont not in found:
                        found.append(cont)
                        yield cont
                        if len(found) >= limit:
                            return
                if offset + 1 >= stop:
                    break
                try:
                    offset = candidates.index(last, offset + 1, stop)
                except ValueError:
                    break
            if found:
                break  # branches from one n keep the recency semantics clean

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        return next(self._continuations(committed, max_tokens, limit=1), [])

    def propose_tree(
        self, committed: list[int], max_tokens: int, *, max_nodes: int | None = None,
        eos_token_ids: Collection[int] = (),
    ) -> DraftTree:
        branches = self._continuations(committed, max_tokens, self.tree_width, eos_token_ids)
        return DraftTree.from_branches(branches, max_nodes=max_nodes)

    def observe(self, result: VerifyResult) -> None:
        pass
