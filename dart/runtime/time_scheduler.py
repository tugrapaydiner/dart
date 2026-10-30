from __future__ import annotations

import math

from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig, entropy_gated
from dart.runtime.types import DecodeAction, DecodeState, VerifyResult


class TimeScheduler(HeuristicScheduler):
    """Choose lanes by measured time per committed token, with periodic probes."""

    requires_round_timing = True

    @property
    def requires_entropy(self) -> bool:
        return self.config.entropy_gate_threshold != float("inf")

    def __init__(self, config: SchedulerConfig, *, retry_interval: int = 16,
                 warmup_rounds: int = 2, min_time_saving: float = 0.05) -> None:
        for name, value in (("retry_interval", retry_interval), ("warmup_rounds", warmup_rounds)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if (isinstance(min_time_saving, bool) or not isinstance(min_time_saving, (int, float))
                or not math.isfinite(min_time_saving) or not 0 <= min_time_saving < 1):
            raise ValueError("min_time_saving must be finite and in [0, 1)")
        if (isinstance(config.ewma_alpha, bool) or not isinstance(config.ewma_alpha, (int, float))
                or not math.isfinite(config.ewma_alpha) or not 0 < config.ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be finite and in (0, 1]")
        if (isinstance(config.max_gpu_kv_gb, bool) or not isinstance(config.max_gpu_kv_gb, (int, float))
                or math.isnan(config.max_gpu_kv_gb) or config.max_gpu_kv_gb < 0):
            raise ValueError("max_gpu_kv_gb must be >= 0")
        super().__init__(config)
        self.retry_interval = retry_interval
        self.warmup_rounds = warmup_rounds
        self.min_time_saving = min_time_saving
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.round_ms: dict[str, float] = {}
        self.committed_tokens: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.last_played: dict[str, int] = {}
        self.kv_gb: dict[str, float] = {}
        self.decisions = 0
        self._last_arm: tuple[str, str] | None = None  # (key, lane) of the latest decision

    def observe(self, lane: str, result: VerifyResult, *, round_ms: float,
                kv_gb: float = 0.0, **_timing) -> None:
        if (isinstance(round_ms, bool) or not isinstance(round_ms, (int, float))
                or not math.isfinite(round_ms) or round_ms <= 0):
            raise ValueError("round_ms must be finite and > 0")
        if (isinstance(kv_gb, bool) or not isinstance(kv_gb, (int, float))
                or not math.isfinite(kv_gb) or kv_gb < 0):
            raise ValueError("kv_gb must be finite and >= 0")
        super().observe(lane, result)
        key = self._last_arm[0] if self._last_arm is not None and self._last_arm[1] == lane else lane
        alpha = self.config.ewma_alpha
        for values, value in ((self.round_ms, round_ms), (self.committed_tokens, len(result.committed))):
            values[key] = (1 - alpha) * values.get(key, value) + alpha * value
        self.counts[key] = self.counts.get(key, 0) + 1
        self.last_played[key] = self.decisions
        self.kv_gb[key] = kv_gb

    def _candidates(self, allowed) -> dict[str, tuple[str, int, str | None]]:
        """Map each drafting arm's key to its (lane, draft length, verifier); None keeps the engine's."""
        return {lane: (lane, self.config.draft_len, None) for lane in dict.fromkeys(self.config.available_lanes)
                if lane != "target_only" and (allowed is None or lane in allowed)}

    def _choose_arm(self, allowed) -> tuple[str, str, int, str | None]:
        arms = {key: arm for key, arm in self._candidates(allowed).items()
                if self.kv_gb.get(key, 0.0) <= self.config.max_gpu_kv_gb}
        baseline = ("target_only", "target_only", 0, None)
        if self.counts.get("target_only", 0) < self.warmup_rounds:
            return baseline
        for key, arm in arms.items():
            if key not in self.counts:
                return (key, *arm)
        stale = [key for key in ["target_only", *arms]
                 if self.decisions - self.last_played[key] >= self.retry_interval]
        if stale:
            key = min(stale, key=self.last_played.get)  # Refresh the baseline too.
            return baseline if key == "target_only" else (key, *arms[key])
        target_cost = self.round_ms["target_only"] / self.committed_tokens["target_only"]
        costs = {key: self.round_ms[key] / self.committed_tokens[key] for key in arms}
        useful = [key for key in arms if costs[key] < target_cost * (1 - self.min_time_saving)]
        if not useful:
            return baseline
        key = min(useful, key=costs.get)
        return (key, *arms[key])

    def choose_action(self, state: DecodeState, allowed=None) -> DecodeAction:
        self.decisions += 1
        key, lane, draft_len, verifier = self._choose_arm(allowed)
        self._last_arm = (key, lane)
        config = self.config
        draft_len, canvas_len = entropy_gated(
            draft_len, config.canvas_len if lane == "diffusion" else 0, state, config,
        )
        return DecodeAction(
            lane=lane, draft_len=draft_len, canvas_len=canvas_len, kv_bits=config.kv_bits,
            verify_every=1, denoise_steps=config.denoise_steps if lane == "diffusion" else 0,
            exact_mode=state.exact_required, verifier=verifier,
        )


class CostTableScheduler(TimeScheduler):
    """Choose (lane, draft length, verifier) arms by measured time per committed token.

    Every lane is paired with each length in ``draft_lengths``; statistics are
    kept per arm under ``"lane:length"`` keys. With a ``verifiers`` menu such as
    ``("block", "tree")`` each length is also paired with each verifier, under
    ``"lane:length:verifier"`` keys. Target-only steps remain the baseline an arm
    must beat by ``min_time_saving``.
    """

    def __init__(self, config: SchedulerConfig, *, draft_lengths: tuple[int, ...] = (2, 4, 8),
                 verifiers: tuple[str, ...] | None = None, **options) -> None:
        if (not isinstance(draft_lengths, tuple) or not draft_lengths
                or len(set(draft_lengths)) != len(draft_lengths)
                or any(type(length) is not int or not 1 <= length <= 64 for length in draft_lengths)):
            raise ValueError("draft_lengths must be a tuple of distinct integers from 1 to 64")
        if verifiers is not None and (not isinstance(verifiers, tuple) or not verifiers
                                      or len(set(verifiers)) != len(verifiers)
                                      or any(name not in ("block", "tree") for name in verifiers)):
            raise ValueError("verifiers must be a tuple of distinct names from 'block' and 'tree'")
        self.draft_lengths = draft_lengths
        self.verifiers = verifiers
        super().__init__(config, **options)

    def _candidates(self, allowed) -> dict[str, tuple[str, int, str | None]]:
        lanes = super()._candidates(allowed)
        if self.verifiers is None:
            return {f"{lane}:{length}": (lane, length, None) for lane in lanes for length in self.draft_lengths}
        return {f"{lane}:{length}:{verifier}": (lane, length, verifier)
                for lane in lanes for length in self.draft_lengths for verifier in self.verifiers}
