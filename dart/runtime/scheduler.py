"""Lane scheduler (docs/design.md §10).

Two policies ship here: ``HeuristicScheduler`` (acceptance EWMAs + fallback
rules) and ``BanditScheduler`` (UCB1 contextual bandit optimizing
accepted_tokens/sec under the memory / rejection constraints of config
24.4).  ``scheduler.type: contextual_bandit`` selects the latter.

The scheduler is advisory, like everything outside the verifier: a bad
lane choice costs speed, never correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dart.runtime.types import DecodeAction, DecodeState, VerifyResult


@dataclass
class SchedulerConfig:
    available_lanes: list[str] = field(default_factory=lambda: ["target_only"])
    long_context_threshold: int = 128_000
    min_lane_acceptance: float = 0.30  # EWMA floor before falling back
    mtp_acceptance_gate: float = 0.65  # doc heuristic: prefer MTP when hot
    ewma_alpha: float = 0.2
    draft_len: int = 4
    canvas_len: int = 32
    denoise_steps: int = 8
    repair_steps: int = 4
    kv_bits: int = 4
    # Bandit reward shaping / constraints (docs/design.md §11).
    memory_penalty: float = 0.0  # reward -= memory_penalty * observed_kv_gb
    max_gpu_kv_gb: float = float("inf")  # bench lanes whose KV exceeds this
    max_rejection_rate: float = 1.0  # bench lanes rejecting more than this
    # Entropy gate (docs/design.md §10): when the running
    # predictive-entropy EWMA exceeds the threshold, shorten drafts - high
    # entropy means early rejection is likely, so a long block wastes work.
    # Disabled by default (inf threshold).
    entropy_gate_threshold: float = float("inf")
    entropy_gate_factor: float = 0.5  # multiply draft_len/canvas by this when gated


def entropy_gated(draft_len: int, canvas: int, s: DecodeState, c: SchedulerConfig):
    """Shrink draft_len/canvas when the entropy EWMA is over the gate."""
    if s.entropy_ewma <= c.entropy_gate_threshold:
        return draft_len, canvas
    f = c.entropy_gate_factor
    if draft_len:
        draft_len = max(1, int(draft_len * f))
    if canvas:
        canvas = max(1, int(canvas * f))
    return draft_len, canvas


class HeuristicScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.acceptance_ewma: dict[str, float] = {}

    def reset(self) -> None:
        self.acceptance_ewma.clear()

    # -------------------------------------------------------------- #
    def observe(self, lane: str, result: VerifyResult, **_timing) -> None:
        if result.n_draft_proposed == 0:
            return
        rate = result.n_draft_accepted / result.n_draft_proposed
        prev = self.acceptance_ewma.get(lane, rate)
        a = self.config.ewma_alpha
        self.acceptance_ewma[lane] = (1 - a) * prev + a * rate

    def _usable(self, lane: str, allowed=None) -> bool:
        if lane not in self.config.available_lanes:
            return False
        if allowed is not None and lane not in allowed:
            return False  # caller restricted lanes (DecodeConfig.allowed_lanes)
        ewma = self.acceptance_ewma.get(lane)  # Benched until the next request.
        return ewma is None or ewma >= self.config.min_lane_acceptance

    # -------------------------------------------------------------- #
    def choose_action(self, s: DecodeState, allowed=None) -> DecodeAction:
        c = self.config
        usable = lambda lane: self._usable(lane, allowed)  # noqa: E731

        def action(lane: str, draft_len: int = 0, canvas: int = 0, steps: int = 0):
            draft_len, canvas = entropy_gated(draft_len, canvas, s, c)
            return DecodeAction(
                lane=lane,
                draft_len=draft_len,
                canvas_len=canvas,
                kv_bits=c.kv_bits,
                verify_every=1,
                denoise_steps=steps,
                exact_mode=s.exact_required,
            )

        # Long context: KV bandwidth dominates; draft on compressed KV.
        if s.context_len > c.long_context_threshold:
            for lane in ("compressed_kv", "sparse_kv"):
                if usable(lane):
                    return action(lane, draft_len=max(c.draft_len, 16))

        # Deterministic-ish domains: MTP when it is hot, else diffusion.
        if s.prompt_kind in {"code", "json", "agent"}:
            if (
                usable("mtp")
                and self.acceptance_ewma.get("mtp", 0.0) > c.mtp_acceptance_gate
            ):
                return action("mtp", draft_len=c.draft_len)
            if usable("diffusion"):
                return action("diffusion", canvas=c.canvas_len, steps=c.denoise_steps)

        # Interactive low batch with memory headroom: big diffusion canvas.
        if s.batch_size <= 4 and s.gpu_free_gb > 8 and usable("diffusion"):
            return action(
                "diffusion", canvas=min(4 * c.canvas_len, 128), steps=c.denoise_steps
            )

        # Anything usable beats target-only; cheap lanes first.
        for lane in ("ngram", "mtp", "diffusion", "compressed_kv", "ar_draft"):
            if usable(lane):
                return action(
                    lane,
                    draft_len=c.draft_len,
                    canvas=c.canvas_len if lane == "diffusion" else 0,
                    steps=c.denoise_steps if lane == "diffusion" else 0,
                )

        return action("target_only")


class BanditScheduler:
    """Contextual-bandit lane selection (docs/design.md §10 "later" /
    config 24.4 ``scheduler.type: contextual_bandit``).

    UCB1 over lanes with reward

        reward = committed_tokens / round_wall_ms
                 - rejection_penalty * rejected_fraction
                 - memory_penalty   * observed_kv_gb

    which is the doc's accepted-tokens-per-second objective (24.4) with the
    rejection and memory terms folded in.  The hard constraints from 24.4 -
    ``max_gpu_kv_gb`` and ``max_rejection_rate`` - bench any lane that
    violates them, so the bandit never *keeps* picking a lane that blows the
    KV budget or rejects too often.  "Contextual" is deliberately shallow in
    v0.1: long contexts hard-route to KV-compressed lanes (the one context
    feature with an unambiguous mechanism); everything else is learned
    from observed rewards.  Like every scheduler decision, a bad arm costs
    speed, never exactness.
    """

    def __init__(self, config: SchedulerConfig, ucb_c: float = 0.5,
                 rejection_penalty: float = 0.2) -> None:
        self.config = config
        self.ucb_c = ucb_c
        self.rejection_penalty = rejection_penalty
        self.counts: dict[str, int] = {}
        self.reward_sums: dict[str, float] = {}
        self.acceptance_ewma: dict[str, float] = {}  # engine/DecodeState compat
        self.kv_gb: dict[str, float] = {}  # last observed KV footprint per lane
        self._last_lane: str | None = None

    def reset(self) -> None:
        self.counts.clear()
        self.reward_sums.clear()
        self.acceptance_ewma.clear()
        self.kv_gb.clear()
        self._last_lane = None

    # -------------------------------------------------------------- #
    def observe(self, lane: str, result: VerifyResult,
                draft_ms: float = 0.0, verify_ms: float = 0.0,
                kv_gb: float = 0.0) -> None:
        ms = max(draft_ms + verify_ms, 1e-3)
        reject_frac = 0.0
        if result.n_draft_proposed > 0:
            reject_frac = 1.0 - result.n_draft_accepted / result.n_draft_proposed
            a = self.config.ewma_alpha
            prev = self.acceptance_ewma.get(lane, 1.0 - reject_frac)
            self.acceptance_ewma[lane] = (1 - a) * prev + a * (1.0 - reject_frac)
        self.kv_gb[lane] = kv_gb
        reward = (
            len(result.committed) / ms
            - self.rejection_penalty * reject_frac
            - self.config.memory_penalty * kv_gb
        )
        self.counts[lane] = self.counts.get(lane, 0) + 1
        self.reward_sums[lane] = self.reward_sums.get(lane, 0.0) + reward

    def _within_constraints(self, lane: str) -> bool:
        """False once a lane has *observed* a budget/rejection violation.

        Unexplored lanes (no observation yet) always pass so the bandit can
        still try them; benching is based on measured behaviour, not guesses.
        """
        c = self.config
        ewma = self.acceptance_ewma.get(lane)
        if ewma is not None and (1.0 - ewma) > c.max_rejection_rate:
            return False
        if self.kv_gb.get(lane, 0.0) > c.max_gpu_kv_gb:
            return False
        return True

    def _ucb_pick(self, lanes: list[str]) -> str:
        for lane in lanes:  # play every arm once first
            if self.counts.get(lane, 0) == 0:
                return lane
        total = sum(self.counts[lane] for lane in lanes)
        best, best_score = lanes[0], float("-inf")
        for lane in lanes:
            n = self.counts[lane]
            mean = self.reward_sums[lane] / n
            score = mean + self.ucb_c * math.sqrt(2.0 * math.log(total) / n)
            if score > best_score:
                best, best_score = lane, score
        return best

    # -------------------------------------------------------------- #
    def choose_action(self, s: DecodeState, allowed=None) -> DecodeAction:
        c = self.config
        lanes = [ln for ln in c.available_lanes if ln != "target_only"]
        if allowed is not None:  # DecodeConfig.allowed_lanes restriction
            lanes = [ln for ln in lanes if ln in allowed]
        if not lanes:
            lanes = ["target_only"]
        # Hard context routing: at long context only KV-light lanes make sense.
        if s.context_len > c.long_context_threshold:
            kv_lanes = [ln for ln in lanes if ln in ("compressed_kv", "sparse_kv")]
            if kv_lanes:
                lanes = kv_lanes
        # Hard constraints (24.4): drop lanes that violated KV/rejection budgets.
        ok = [ln for ln in lanes if self._within_constraints(ln)]
        lanes = ok or ["target_only"]
        lane = self._ucb_pick(lanes)
        self._last_lane = lane
        draft_len, canvas = entropy_gated(
            c.draft_len, c.canvas_len if lane == "diffusion" else 0, s, c
        )
        return DecodeAction(
            lane=lane,
            draft_len=draft_len,
            canvas_len=canvas,
            kv_bits=c.kv_bits,
            verify_every=1,
            denoise_steps=c.denoise_steps if lane == "diffusion" else 0,
            exact_mode=s.exact_required,
        )
