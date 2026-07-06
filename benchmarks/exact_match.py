"""Exact-match evaluator: the benchmark-grade form of the exactness contract.

For each prompt: run the trusted baseline, run DART with the lane under
test, compare token ids and hashes.  Any mismatch is a correctness bug -
not a quality regression, a bug (docs/design.md §2 and §13.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig


@dataclass
class ExactMatchReport:
    total: int = 0
    exact: int = 0
    mismatches: list[dict] = field(default_factory=list)
    acceptance_rates: list[float] = field(default_factory=list)
    avg_accept_runs: list[float] = field(default_factory=list)
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    baseline_ms: float = 0.0
    dart_ms: float = 0.0
    baseline_tokens: int = 0
    dart_tokens: int = 0
    # Accepted-prefix length at each *rejected* round: the rejection-position
    # distribution (docs/design.md §13.2); early-skewed -> entropy-gate shorter
    # blocks (docs/design.md §10), late-skewed -> the drafter is tracking the target well.
    rejection_positions: list[int] = field(default_factory=list)

    @property
    def exact_rate(self) -> float:
        return self.exact / self.total if self.total else 0.0

    def summary(self) -> dict:
        n = max(1, len(self.acceptance_rates))
        hist: dict[int, int] = {}
        for p in self.rejection_positions:
            hist[p] = hist.get(p, 0) + 1
        return {
            "prompts": self.total,
            "exact_match_rate": self.exact_rate,
            "mean_acceptance_rate": sum(self.acceptance_rates) / n,
            "mean_accept_run": sum(self.avg_accept_runs) / n,
            "baseline_ms_total": round(self.baseline_ms, 1),
            "dart_ms_total": round(self.dart_ms, 1),
            "baseline_tok_per_s": (
                round(self.baseline_tokens / (self.baseline_ms / 1000), 2)
                if self.baseline_ms else None
            ),
            "dart_tok_per_s": (
                round(self.dart_tokens / (self.dart_ms / 1000), 2)
                if self.dart_ms else None
            ),
            "speedup_wall": (
                round(self.baseline_ms / self.dart_ms, 3) if self.dart_ms else None
            ),
            "mismatch_count": len(self.mismatches),
            "rejections": len(self.rejection_positions),
            "rejection_position_hist": {str(k): hist[k] for k in sorted(hist)},
            "mean_rejection_position": (
                round(sum(self.rejection_positions) / len(self.rejection_positions), 3)
                if self.rejection_positions else None
            ),
        }


def run_exact_match(
    target: TargetModel,
    engine: DartEngine,
    prompts: list[list[int]],
    max_new_tokens: int,
    config: DecodeConfig | None = None,
) -> ExactMatchReport:
    import time

    report = ExactMatchReport()
    base_cfg = config or DecodeConfig(max_new_tokens=max_new_tokens)

    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        base = target.greedy_generate(prompt, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()
        out = engine.generate(prompt, base_cfg)
        t2 = time.perf_counter()

        report.total += 1
        report.baseline_ms += (t1 - t0) * 1000
        report.dart_ms += (t2 - t1) * 1000
        report.baseline_tokens += len(base)
        report.dart_tokens += len(out.token_ids)
        report.acceptance_rates.append(out.metadata["acceptance_rate"])
        report.avg_accept_runs.append(out.metadata["avg_accept_run"])
        report.draft_tokens_proposed += out.metadata["draft_tokens_proposed"]
        report.draft_tokens_accepted += out.metadata["draft_tokens_accepted"]
        for rec in out.rounds:
            if 0 < rec.n_draft_proposed and rec.n_draft_accepted < rec.n_draft_proposed:
                report.rejection_positions.append(rec.n_draft_accepted)

        if out.token_ids == base:
            report.exact += 1
        else:
            div = next(
                (j for j, (a, b) in enumerate(zip(base, out.token_ids)) if a != b),
                min(len(base), len(out.token_ids)),
            )
            report.mismatches.append(
                {
                    "prompt_idx": i,
                    "diverged_at": div,
                    "baseline_hash": output_hash(base),
                    "dart_hash": output_hash(out.token_ids),
                    "baseline_len": len(base),
                    "dart_len": len(out.token_ids),
                }
            )
    return report
