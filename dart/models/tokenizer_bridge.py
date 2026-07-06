"""Tokenizer bridge: turn external drafter text into target-vocab drafts.

Remote drafters (e.g. a DiffusionGemma vLLM server) return *text*.  To
verify it we must re-tokenize under the target tokenizer (docs/design.md
§6.3).  The classic pitfall is the junction: encoding
``context_text + draft_text`` can merge tokens across the boundary, so the
encoded result need not extend the committed ids exactly.

DART's safety property makes this benign: the bridge keeps only the part
of the re-encoding that strictly extends the committed ids and returns it
as draft tokens.  A broken boundary can only reduce acceptance - it can
never corrupt output, because every draft goes through the exact verifier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeResult:
    draft_ids: list[int]
    boundary_intact: bool  # False when junction merging trimmed the drafts


class TokenizerBridge:
    def __init__(self, target_tokenizer) -> None:
        self.tok = target_tokenizer

    def drafts_from_text(
        self, committed_ids: list[int], draft_text: str, max_tokens: int
    ) -> BridgeResult:
        if not draft_text:
            return BridgeResult([], True)
        committed_text = self.tok.decode(committed_ids, skip_special_tokens=False)
        full_ids = self.tok.encode(
            committed_text + draft_text, add_special_tokens=False
        )
        # Longest common prefix between committed ids and the re-encoding.
        lcp = 0
        for a, b in zip(committed_ids, full_ids):
            if a != b:
                break
            lcp += 1
        boundary_intact = lcp == len(committed_ids)
        if not boundary_intact:
            # Junction merged or context re-encoded differently: drop drafts
            # whose alignment we cannot trust beyond the divergence point.
            return BridgeResult([], False)
        return BridgeResult(full_ids[lcp : lcp + max_tokens], True)
