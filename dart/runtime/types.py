"""Core datatypes shared across DART.

Decode-loop invariant (used by the engine and all verifiers)
------------------------------------------------------------
At the start of every round:

  * ``committed`` is the full token sequence so far (prompt + generated).
  * The target KV cache covers everything except the pending block.
  * ``pending_block`` holds the most recently committed tokens whose KV
    entries have not been computed yet (length 1 in exact modes; approx
    mode grows it between verify rounds).

A verify round feeds ``[pending block, d_1 .. d_K]`` through the target in
one pass.  The logits after consuming position ``i`` give the target's
prediction for the *next* token, so one forward pass simultaneously
(a) computes the missing KV entries, (b) checks every draft token against
the target argmax, and (c) yields the correction/bonus token.  Each round
therefore commits between 1 and K+1 tokens, and plain greedy decoding is
exactly the K=0 case of the same code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

Lane = Literal[
    "target_only",
    "mtp",
    "ngram",
    "ar_draft",
    "diffusion",
    "compressed_kv",
    "sparse_kv",
    "block_sparse_kv",
]

# exact          - greedy verification; numerical equality needs a reference check
# exact_sampling - speculative sampling; target *distribution* preserved
# approx         - greedy commits with verification skipped some rounds
Mode = Literal["exact", "exact_sampling", "approx"]


@dataclass
class DecodeConfig:
    """User-facing decoding configuration (docs/design.md §11)."""

    mode: Mode = "exact"
    max_new_tokens: int = 256
    # Optional label for the trusted target (docs/design.md §11); when
    # set it is reported in metadata["target_model"] instead of the autodetected
    # class name. Purely informational - the engine's actual target is fixed.
    target_model: Optional[str] = None
    # Restrict which lanes DART may draft with (docs/design.md §11).
    # None = no restriction; target_only is always permitted as the fallback.
    allowed_lanes: Optional[list[str]] = None
    # Draft sizing knobs (lane-specific code may override per round).
    draft_len: int = 4
    canvas_len: int = 32
    # Sampling mode (exact_sampling).
    temperature: float = 1.0
    # Approx mode: verify every Nth round, commit drafts unverified between.
    verify_every: int = 1
    # Scheduler hint (exactness is derived from ``mode``, not set here).
    prompt_kind: Literal["code", "json", "chat", "math", "rag", "agent"] = "chat"
    seed: Optional[int] = None
    # Where the full-precision verifier KV lives (reported in metadata, docs/design.md §2.4).
    full_kv_residency: Literal["gpu", "cpu_pinned"] = "gpu"
    # Tree verification packs at most this many draft nodes per round, besides the pending token.
    max_tree_nodes: int = 15

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in ("exact", "exact_sampling", "approx"):
            raise ValueError(f"unknown mode {self.mode!r}")
        for name, minimum in (
            ("max_new_tokens", 0), ("draft_len", 0), ("canvas_len", 0), ("verify_every", 1), ("max_tree_nodes", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be finite and >= 0")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        if self.full_kv_residency not in ("gpu", "cpu_pinned"):
            raise ValueError("full_kv_residency must be 'gpu' or 'cpu_pinned'")
        if self.allowed_lanes is not None and (
            not isinstance(self.allowed_lanes, list)
            or any(not isinstance(lane, str) for lane in self.allowed_lanes)
        ):
            raise ValueError("allowed_lanes must be a list of names or None")


@dataclass
class VerifyResult:
    """Outcome of one verify round.

    ``committed`` always holds at least one token: the accepted draft
    prefix plus the target's correction token (on mismatch) or bonus token
    (on full acceptance).  In both cases the final element came from the
    target's own computation, so it becomes the next round's pending token.
    """

    committed: list[int]
    n_draft_proposed: int
    n_draft_accepted: int
    all_accepted: bool
    finished: bool  # an EOS token was committed (committed is truncated at it)
    # Approx mode: how many previously-committed unverified tokens turned
    # out NOT to be the target argmax (0 in exact mode by construction).
    pending_divergence: int = 0
    # Predictive entropy (nats) of the target at the last on-trajectory
    # position this round - a scheduler difficulty signal, NOT part of the
    # verification contract, so it is excluded from equality (the block and
    # reference verifiers may differ here by fp rounding).
    entropy: float = field(default=0.0, compare=False)
    hidden_index: int = field(default=-1, compare=False)  # Row that predicted the new pending token.

    def __post_init__(self) -> None:
        if not self.committed:
            raise ValueError("a verify round must commit at least one token")
        if self.n_draft_accepted > self.n_draft_proposed:
            raise ValueError("accepted drafts cannot exceed proposed drafts")


@dataclass
class RoundRecord:
    """Per-round trace entry for metrics/debugging."""

    round_idx: int
    lane: str
    n_draft_proposed: int
    n_draft_accepted: int
    n_committed: int
    draft_ms: float
    verify_ms: float
    verifier: str = "block"  # "none" for approx rounds that skip verification


@dataclass
class GenerationOutput:
    """Result of ``DartEngine.generate``.

    Metadata separates ``greedy_verified`` from ``exact_greedy_equivalent``.
    The latter is None because the engine does not run a reference comparison.
    Floating-point results can depend on block shape, precision and backend.
    """

    token_ids: list[int]  # generated tokens only (no prompt)
    text: Optional[str]
    metadata: dict
    rounds: list[RoundRecord] = field(default_factory=list)


@dataclass
class DecodeState:
    """Scheduler input state (docs/design.md §10)."""

    context_len: int
    generated_len: int
    gpu_free_gb: float
    batch_size: int
    prompt_kind: Literal["code", "json", "chat", "math", "rag", "agent"]
    exact_required: bool
    entropy_ewma: float
    acceptance_ewma: dict[str, float]
    kv_transfer_ms: float
    target_step_ms: float


@dataclass
class DecodeAction:
    """Scheduler output action (docs/design.md §10)."""

    lane: Lane
    draft_len: int
    canvas_len: int
    kv_bits: int
    verify_every: int
    denoise_steps: int
    exact_mode: bool
    verifier: Optional[str] = None  # None keeps the engine's verifier
