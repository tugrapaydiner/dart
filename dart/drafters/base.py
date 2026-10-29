"""Drafter interface.

A drafter is *advisory only*.  It sees the committed sequence (prompt plus
generated tokens, including the pending one) and proposes up to
``max_tokens`` future tokens.  Nothing a drafter returns can affect exact
mode output - only acceptance rate and therefore speed.  This is the
load-bearing safety property of DART: drafters may be lossy, compressed,
diffused, or simply wrong.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dart.runtime.types import VerifyResult


@runtime_checkable
class Drafter(Protocol):
    """Protocol implemented by every draft lane.

    An optional finish_request() hook releases request data after execution,
    including errors, without resetting statistics or the sampling generator.
    """

    name: str

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        """Propose up to ``max_tokens`` continuation tokens for ``committed``.

        Returning fewer tokens (or none) is always legal; the engine falls
        back to a plain target step.
        """
        ...

    def observe(self, result: VerifyResult) -> None:
        """Feedback hook called after each verify round (optional).

        Set requires_entropy=False only when feedback does not use entropy.
        Drafters without this flag keep receiving the signal.
        """
        ...


class NullDrafter:
    """Proposes nothing: engine degenerates to plain incremental greedy."""

    name = "target_only"
    requires_entropy = False

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        return []

    def observe(self, result: VerifyResult) -> None:
        pass
