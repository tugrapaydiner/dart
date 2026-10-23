from __future__ import annotations

import math

from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig, entropy_gated
from dart.runtime.types import DecodeAction, DecodeState, VerifyResult


class TimeScheduler(HeuristicScheduler):
    """Choose lanes by measured time per committed token, with periodic probes."""

    requires_round_timing = True

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

    def observe(self, lane: str, result: VerifyResult, *, round_ms: float,
                kv_gb: float = 0.0, **_timing) -> None:
        if (isinstance(round_ms, bool) or not isinstance(round_ms, (int, float))
                or not math.isfinite(round_ms) or round_ms <= 0):
            raise ValueError("round_ms must be finite and > 0")
        if (isinstance(kv_gb, bool) or not isinstance(kv_gb, (int, float))
                or not math.isfinite(kv_gb) or kv_gb < 0):
            raise ValueError("kv_gb must be finite and >= 0")
        super().observe(lane, result)
        alpha = self.config.ewma_alpha
        for values, value in ((self.round_ms, round_ms), (self.committed_tokens, len(result.committed))):
            values[lane] = (1 - alpha) * values.get(lane, value) + alpha * value
        self.counts[lane] = self.counts.get(lane, 0) + 1
        self.last_played[lane] = self.decisions
        self.kv_gb[lane] = kv_gb

    def _choose_lane(self, allowed) -> str:
        lanes = [lane for lane in dict.fromkeys(self.config.available_lanes)
                 if lane != "target_only" and (allowed is None or lane in allowed)
                 and self.kv_gb.get(lane, 0.0) <= self.config.max_gpu_kv_gb]
        if self.counts.get("target_only", 0) < self.warmup_rounds:
            return "target_only"
        for lane in lanes:
            if lane not in self.counts:
                return lane
        stale = [lane for lane in ["target_only", *lanes]
                 if self.decisions - self.last_played[lane] >= self.retry_interval]
        if stale:
            return min(stale, key=self.last_played.get)  # Refresh the baseline too.
        baseline = self.round_ms["target_only"] / self.committed_tokens["target_only"]
        costs = {lane: self.round_ms[lane] / self.committed_tokens[lane] for lane in lanes}
        useful = [lane for lane in lanes if costs[lane] < baseline * (1 - self.min_time_saving)]
        return min(useful, key=costs.get) if useful else "target_only"

    def choose_action(self, state: DecodeState, allowed=None) -> DecodeAction:
        self.decisions += 1
        lane = self._choose_lane(allowed)
        config = self.config
        draft_len, canvas_len = entropy_gated(
            config.draft_len if lane != "target_only" else 0,
            config.canvas_len if lane == "diffusion" else 0, state, config,
        )
        return DecodeAction(
            lane=lane, draft_len=draft_len, canvas_len=canvas_len, kv_bits=config.kv_bits,
            verify_every=1, denoise_steps=config.denoise_steps if lane == "diffusion" else 0,
            exact_mode=state.exact_required,
        )
