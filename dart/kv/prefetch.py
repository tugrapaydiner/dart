"""Full-KV prefetch for the cpu_pinned residency path (docs/design.md §9.2).

The long-context exact pipeline wants the verifier-KV offload (park to host)
to happen *while the draft lane is proposing the next block*, not on the
critical path.  Two implementations behind one interface:

  * ``SyncPrefetcher``   - resolves immediately; the overlap-free reference.
  * ``ThreadPrefetcher`` - runs the move on a background thread so it overlaps
    with drafting; the engine waits on it just before the next verify pass.

On a CPU target the move is a no-op either way, so the overlap only does real
work on a GPU target - but the wiring and ordering are exercised everywhere.
A GPU verifier pool (loading a full-KV chunk into a dedicated device buffer)
is a remaining roadmap item (docs/design.md §16) and is not implemented here.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Prefetched(Generic[T]):
    _value: Optional[T] = None
    _thunk: Optional[Callable[[], T]] = None
    _future: Optional["concurrent.futures.Future"] = field(default=None)

    def wait(self) -> T:
        """Block until the value is ready; idempotent."""
        if self._future is not None:
            return self._future.result()
        if self._value is None and self._thunk is not None:
            self._value = self._thunk()
        return self._value


class SyncPrefetcher:
    """Resolves immediately; the overlap-free reference implementation."""

    def prefetch(self, thunk: Callable[[], T]) -> Prefetched[T]:
        return Prefetched(_value=thunk())

    def close(self) -> None:  # symmetry with ThreadPrefetcher
        pass


class ThreadPrefetcher:
    """Runs ``thunk`` on a single background thread for compute/transfer
    overlap.  One worker preserves submission order (FIFO), which the engine
    relies on for correct KV residency state."""

    def __init__(self) -> None:
        self._ex = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dart-prefetch"
        )

    def prefetch(self, thunk: Callable[[], T]) -> Prefetched[T]:
        return Prefetched(_future=self._ex.submit(thunk))

    def close(self) -> None:
        self._ex.shutdown(wait=True)
