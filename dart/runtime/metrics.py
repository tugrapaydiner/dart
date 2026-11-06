"""Measurement primitives: output hashing, timing, memory, trace logging.

Measurement requirements (docs/design.md §13.2): exact output hashing,
token-level trace logging, latency and memory measurement.  Every benchmark
claim in DART must be reproducible from these records.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch

from dart.runtime.types import RoundRecord

HASH_SCHEMA = "dart-tokens-v1"


def output_hash(token_ids: list[int]) -> str:
    """Stable hash of a token sequence; equality of hashes == exactness."""
    payload = HASH_SCHEMA + ":" + ",".join(str(t) for t in token_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def wall_timer() -> Iterator[list[float]]:
    """``with wall_timer() as t: ...`` then ``t[0]`` is elapsed milliseconds."""
    out = [0.0]
    start = time.perf_counter()
    try:
        yield out
    finally:
        out[0] = (time.perf_counter() - start) * 1000.0


def gpu_memory_gb(device: str | torch.device | None = None) -> float:
    """Peak process allocation on one CUDA device; zero for other devices."""
    if device is not None and torch.device(device).type != "cuda":
        return 0.0
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 1e9
    return 0.0


def reset_gpu_memory_stats(device: str | torch.device | None = None) -> None:
    """Reset one CUDA device's peak; None selects the current CUDA device."""
    if device is not None and torch.device(device).type != "cuda":
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


@dataclass
class DecodeMetrics:
    """Aggregated per-generation metrics; serialized into output metadata."""

    mode: str = "exact"
    target_model: str = ""
    lanes_used: set[str] = field(default_factory=set)
    verifiers_used: set[str] = field(default_factory=set)
    tokens_generated: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    rounds: int = 0
    rejections: int = 0
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    wall_time_ms: float = 0.0
    gpu_peak_gb: float = 0.0

    def record_round(self, rec: RoundRecord) -> None:
        self.rounds += 1
        self.lanes_used.add(rec.lane)
        if rec.verifier != "none":
            self.verifiers_used.add(rec.verifier)
        self.tokens_generated += rec.n_committed
        self.draft_tokens_proposed += rec.n_draft_proposed
        self.draft_tokens_accepted += rec.n_draft_accepted
        if rec.n_draft_proposed > rec.n_draft_accepted:
            self.rejections += 1
        self.draft_ms += rec.draft_ms
        self.verify_ms += rec.verify_ms

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def avg_accept_run(self) -> float:
        if self.rounds == 0:
            return 0.0
        return self.tokens_generated / self.rounds

    def to_metadata(self, **extra) -> dict:
        meta = {
            "mode": self.mode,
            "target_model": self.target_model,
            "lanes_used": sorted(self.lanes_used),
            "verifiers_used": sorted(self.verifiers_used),
            "tokens_generated": self.tokens_generated,
            "draft_tokens_proposed": self.draft_tokens_proposed,
            "draft_tokens_accepted": self.draft_tokens_accepted,
            "rounds": self.rounds,
            "rejections": self.rejections,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_accept_run": round(self.avg_accept_run, 2),
            "draft_ms": round(self.draft_ms, 2),
            "verify_ms": round(self.verify_ms, 2),
            "wall_time_ms": round(self.wall_time_ms, 2),
            "gpu_peak_gb": round(self.gpu_peak_gb, 3),
        }
        meta.update(extra)
        return meta


class TraceLogger:
    """JSONL token-level trace writer (one record per verify round)."""

    def __init__(self, path: Optional[str | Path]) -> None:
        self._fh = open(path, "a", encoding="utf-8") if path else None

    def log_round(self, run_id: str, rec: RoundRecord, committed: list[int]) -> None:
        if self._fh is None:
            return
        entry = {"run_id": run_id, **asdict(rec), "committed_ids": committed}
        self._fh.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
