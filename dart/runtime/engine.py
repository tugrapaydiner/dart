"""DART decode engine: the draft -> verify -> commit loop.

Modes (docs/design.md §2):

  * ``exact``          - greedy; output is byte-identical to
                         ``TargetModel.greedy_generate`` by construction.
  * ``exact_sampling`` - speculative sampling; the emitted distribution is
                         exactly the target's sampling distribution (see
                         verification/sampling.py).  Trace-level identity
                         with any particular baseline run is NOT claimed.
  * ``approx``         - greedy commits, but verification may be skipped
                         (``verify_every``); skipped drafts are committed
                         unverified and audited at the next verify round.
                         Metadata reports divergence and risk honestly -
                         modes are never silently mixed (docs/design.md §2.4).

Engine state invariant: ``pending_block`` holds the most recently
committed tokens whose KV entries have not been computed yet.  In exact
modes its length is always 1; approx mode grows it between verify rounds.
"""

from __future__ import annotations

import time
import uuid
from functools import partial
from typing import Callable, Optional

import torch

from dart.drafters.base import Drafter, NullDrafter
from dart.kv.prefetch import SyncPrefetcher, ThreadPrefetcher
from dart.kv.residency import ResidencyStats, move_cache
from dart.models.target import TargetModel
from dart.runtime.metrics import (
    DecodeMetrics,
    TraceLogger,
    gpu_memory_gb,
    output_hash,
    reset_gpu_memory_stats,
)
from dart.runtime.types import (
    DecodeConfig,
    DecodeState,
    GenerationOutput,
    RoundRecord,
    VerifyResult,
)
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from dart.verification.sampling import verify_block_sampling
from dart.verification.tree_verify import DraftTree, verify_tree_greedy

VerifyFn = Callable[..., VerifyResult]

_VERIFIERS: dict[str, VerifyFn] = {
    "block": verify_block_greedy,
    "reference": verify_greedy_reference,
    "tree": verify_tree_greedy,
}


class DartEngine:
    """Single-sequence DART engine (batch 1).

    Two construction styles:
      * single lane:   DartEngine(target, drafter=some_drafter)
      * scheduled:     DartEngine(target, lanes={...}, scheduler=...)
        where lanes maps lane name -> Drafter and the scheduler picks a
        lane every round (docs/design.md §10).
    """

    def __init__(
        self,
        target: TargetModel,
        drafter: Optional[Drafter] = None,
        verifier: str = "block",
        trace_path: Optional[str] = None,
        lanes: Optional[dict[str, Drafter]] = None,
        scheduler=None,
        default_decode_config: Optional[DecodeConfig] = None,
    ) -> None:
        self.target = target
        # Decode defaults captured from a YAML config (mode, residency, draft
        # sizing, ...) so from_config(...).generate(prompt) honors the file.
        self.default_decode_config = default_decode_config
        if (lanes is None) != (scheduler is None):
            raise ValueError("lanes and scheduler must be provided together")
        if lanes is not None and drafter is not None:
            raise ValueError("pass either a single drafter or lanes+scheduler")
        self.lanes = lanes
        self.scheduler = scheduler
        self._null = NullDrafter()
        self.drafter: Drafter = drafter if drafter is not None else self._null
        if verifier not in _VERIFIERS:
            raise ValueError(f"unknown verifier {verifier!r}; use {list(_VERIFIERS)}")
        self.verify: VerifyFn = _VERIFIERS[verifier]
        self.verifier_name = verifier
        self.trace = TraceLogger(trace_path)

    def close(self) -> None:
        """Flush and close the trace log (no-op without a trace_path)."""
        self.trace.close()

    def __enter__(self) -> "DartEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, path: str) -> "DartEngine":
        """Build an engine from a YAML config (docs/design.md §11)."""
        from dart.runtime.factory import engine_from_yaml

        return engine_from_yaml(path)

    def _all_drafters(self) -> list[Drafter]:
        return list(self.lanes.values()) if self.lanes is not None else [self.drafter]

    @staticmethod
    def _gpu_free_gb() -> float:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9
        return float("inf")  # CPU hosts: memory is not the binding constraint

    @staticmethod
    def _drafter_kv_gb(drafter: Drafter) -> float:
        """Compressed-KV footprint of a lane in GB, or 0.0 if it has none.

        Drives both the bandit's memory term and the gpu_kv_gb metadata
        field; lanes without a quantized cache (n-gram, diffusion, ...)
        report 0.0.
        """
        if not hasattr(drafter, "kv_stats"):
            return 0.0
        st = drafter.kv_stats()
        return st.get("logical_compressed_bytes", 0) / 1e9 if st else 0.0

    def _pick_lane(
        self, config: DecodeConfig, prompt_len: int, generated_len: int,
        entropy_ewma: float = 0.0,
    ):
        """Return (drafter, per-round draft budget) for this round."""
        allowed = set(config.allowed_lanes) if config.allowed_lanes is not None else None
        if self.scheduler is None:
            d = self.drafter
            # Honor allowed_lanes: a disallowed single drafter -> target-only step.
            if allowed is not None and d.name not in allowed and d.name != "target_only":
                return self._null, 0
            n = config.canvas_len if "diffusion" in d.name else config.draft_len
            return d, n
        s = DecodeState(
            context_len=prompt_len + generated_len,
            generated_len=generated_len,
            gpu_free_gb=self._gpu_free_gb(),
            batch_size=1,
            prompt_kind=config.prompt_kind,
            exact_required=(config.mode == "exact"),
            entropy_ewma=entropy_ewma,
            acceptance_ewma=dict(self.scheduler.acceptance_ewma),
            kv_transfer_ms=0.0,
            target_step_ms=0.0,
        )
        action = self.scheduler.choose_action(s, allowed=allowed)
        d = self.lanes.get(action.lane, self._null)
        n = action.canvas_len if action.canvas_len > 0 else action.draft_len
        return d, n

    # ------------------------------------------------------------------ #
    @staticmethod
    def _propose(drafter: Drafter, committed: list[int], budget: int, sampling: bool):
        """Returns (draft_tokens, draft_probs_or_None).

        q-distributions are only meaningful (and only requested) in
        sampling mode; greedy modes always use the drafter's argmax
        proposals so exact greedy acceptance is not diluted.
        """
        if budget <= 0:
            return [], None
        if sampling and hasattr(drafter, "propose_with_probs"):
            drafts, probs = drafter.propose_with_probs(committed, budget)
            return list(drafts), probs
        return list(drafter.propose(committed, budget)), None

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int] | str,
        config: Optional[DecodeConfig] = None,
    ) -> GenerationOutput:
        # Accept a text prompt (docs/design.md §11) when a tokenizer exists.
        if isinstance(prompt_ids, str):
            if self.target.tokenizer is None:
                raise ValueError("text prompt requires a tokenizer; pass token ids")
            prompt_ids = self.target.tokenizer.encode(prompt_ids)
        config = config or self.default_decode_config or DecodeConfig()
        config.validate()
        if self.verifier_name == "tree" and config.mode != "exact":
            raise ValueError(
                "the tree verifier supports exact (greedy) mode only; "
                "use verifier='block' for sampling/approx"
            )
        sampling = config.mode == "exact_sampling"
        generator = None
        if sampling:
            generator = torch.Generator(device=self.target.device)
            generator.manual_seed(config.seed if config.seed is not None else 0)
            verify_fn: VerifyFn = partial(
                verify_block_sampling,
                generator=generator,
                temperature=config.temperature,
            )
        else:
            verify_fn = self.verify

        run_id = uuid.uuid4().hex[:12]
        metrics = DecodeMetrics(
            mode=config.mode,
            target_model=config.target_model
            or getattr(self.target.model.config, "name_or_path", "")
            or type(self.target.model).__name__,
        )
        reset_gpu_memory_stats()  # so gpu_peak_gb reflects only this generation
        t_start = time.perf_counter()

        prompt = list(prompt_ids)
        state, logits = self.target.prefill(prompt)
        if self.scheduler is not None and hasattr(self.scheduler, "reset"):
            self.scheduler.reset()
        for d in self._all_drafters():
            if hasattr(d, "after_prefill"):
                d.after_prefill()
        if sampling:
            probs = torch.softmax(logits.float() / max(config.temperature, 1e-6), -1)
            first = int(torch.multinomial(probs, 1, generator=generator).item())
        else:
            first = int(torch.argmax(logits).item())

        generated: list[int] = [first]
        pending_block: list[int] = [first]
        finished = first in self.target.eos_token_ids
        rounds: list[RoundRecord] = []
        unverified_tokens = 0
        divergence_total = 0
        round_idx = 0

        # KV residency (docs/design.md §9.2): in cpu_pinned mode the full
        # verifier KV is parked in (pinned) host memory between rounds and
        # staged back to the device per verify pass.  On CPU-only hosts /
        # CPU targets every move is a no-op, so this is correct everywhere
        # and only does real transfers on a GPU target.  Drafters keep their
        # own caches, so parking `state` between rounds is always safe.
        park_kv = config.full_kv_residency == "cpu_pinned"
        # ThreadPrefetcher overlaps the host-offload with the next round's
        # drafting; SyncPrefetcher is the no-overlap reference.
        prefetcher = ThreadPrefetcher() if park_kv else SyncPrefetcher()
        res_stats = ResidencyStats()
        pending_park = None  # in-flight offload of the verifier KV to host
        # Running predictive-entropy EWMA from verify rounds; gates draft
        # length in the scheduler (docs/design.md §10).  Seeded
        # from the prompt's last-token distribution.
        ent_ewma = 0.0
        if sampling:
            ent_ewma = 0.0
        else:
            d0 = torch.softmax(logits.float(), dim=-1)
            ent_ewma = float(-(d0 * d0.clamp_min(1e-12).log()).sum())

        while not finished and len(generated) < config.max_new_tokens:
            committed_seq = prompt + generated
            drafter, draft_len = self._pick_lane(
                config, len(prompt), len(generated), entropy_ewma=ent_ewma
            )

            t0 = time.perf_counter()
            budget = config.max_new_tokens - len(generated)
            per_round = min(draft_len, budget - 1)  # Leave room for the verifier's token.
            if self.verifier_name == "tree":
                # Tree-capable drafters propose several candidate branches;
                # plain drafters degrade to a chain, for which tree and
                # block verification coincide decision-for-decision.
                if per_round <= 0:
                    drafts = DraftTree.chain([])
                elif hasattr(drafter, "propose_tree"):
                    drafts = drafter.propose_tree(committed_seq, per_round)
                else:
                    linear = list(drafter.propose(committed_seq, per_round))
                    drafts = DraftTree.chain(linear[:per_round])
                if not isinstance(drafts, DraftTree):
                    drafts = DraftTree.chain(list(drafts))
                drafts = drafts.truncate(per_round)
                draft_probs = None
            else:
                drafts, draft_probs = self._propose(
                    drafter, committed_seq, per_round, sampling
                )
                # Clamp to the round budget even if a drafter over-returns:
                # tokens past the budget would be verified (wasted target
                # compute) only to be truncated at the end.
                drafts = drafts[:per_round] if per_round > 0 else []
            t1 = time.perf_counter()

            skip_verify = (
                config.mode == "approx"
                and config.verify_every > 1
                and (round_idx % config.verify_every) != 0
                and len(drafts) > 0
            )

            if skip_verify:
                # Commit drafts WITHOUT any target forward.  They join the
                # pending block and get audited at the next verify round.
                kept: list[int] = []
                for tok in drafts:
                    kept.append(int(tok))
                    if tok in self.target.eos_token_ids:
                        finished = True
                        break
                generated.extend(kept)
                pending_block.extend(kept)
                unverified_tokens += len(kept)
                rec = RoundRecord(
                    round_idx=round_idx, lane=drafter.name,
                    n_draft_proposed=0, n_draft_accepted=0,
                    n_committed=len(kept),
                    draft_ms=(t1 - t0) * 1000.0, verify_ms=0.0,
                )
            else:
                if park_kv:
                    # Wait for the previous round's offload (overlapped with
                    # this round's drafting), then stage the KV back on-device.
                    if pending_park is not None:
                        parked = pending_park.wait()
                        res_stats.moved_tensors += parked.moved_tensors
                        res_stats.moved_bytes += parked.moved_bytes
                        pending_park = None
                    staged = move_cache(state.cache, self.target.device)
                    res_stats.moved_tensors += staged.moved_tensors
                    res_stats.moved_bytes += staged.moved_bytes
                if sampling:
                    result = verify_fn(
                        self.target, state, pending_block, drafts,
                        draft_probs=draft_probs,
                    )
                else:
                    result = verify_fn(self.target, state, pending_block, drafts)
                t2 = time.perf_counter()

                drafter.observe(result)
                if self.scheduler is not None:
                    self.scheduler.observe(
                        drafter.name, result,
                        draft_ms=(t1 - t0) * 1000.0,
                        verify_ms=(t2 - t1) * 1000.0,
                        kv_gb=self._drafter_kv_gb(drafter),
                    )
                generated.extend(result.committed)
                pending_block = [result.committed[-1]]
                divergence_total += result.pending_divergence
                ent_ewma = 0.7 * ent_ewma + 0.3 * result.entropy
                finished = result.finished
                rec = RoundRecord(
                    round_idx=round_idx, lane=drafter.name,
                    n_draft_proposed=result.n_draft_proposed,
                    n_draft_accepted=result.n_draft_accepted,
                    n_committed=len(result.committed),
                    draft_ms=(t1 - t0) * 1000.0,
                    verify_ms=(t2 - t1) * 1000.0,
                )
                self.trace.log_round(run_id, rec, result.committed)
                if park_kv:
                    # Offload the verifier KV to host, overlapped with the next
                    # round's drafting.  Capture the cache object now so a later
                    # forward rebinding state.cache can't race the worker.
                    cache_ref = state.cache
                    pending_park = prefetcher.prefetch(
                        lambda c=cache_ref: move_cache(c, "cpu", pin=True)
                    )

            rounds.append(rec)
            metrics.record_round(rec)
            round_idx += 1

        if pending_park is not None:  # drain the final in-flight offload
            parked = pending_park.wait()
            res_stats.moved_tensors += parked.moved_tensors
            res_stats.moved_bytes += parked.moved_bytes
        prefetcher.close()

        generated = generated[: config.max_new_tokens]

        trailing_unverified = max(0, len(pending_block) - 1)
        if config.mode == "exact":
            exact_equiv = True
        elif config.mode == "approx":
            exact_equiv = divergence_total == 0 and trailing_unverified == 0
        else:
            exact_equiv = False  # sampling: distributional, not trace-level

        metrics.tokens_generated = len(generated)
        metrics.wall_time_ms = (time.perf_counter() - t_start) * 1000.0
        metrics.gpu_peak_gb = gpu_memory_gb()
        text = (
            self.target.tokenizer.decode(generated, skip_special_tokens=False)
            if self.target.tokenizer is not None
            else None
        )
        kv_gb = max((self._drafter_kv_gb(d) for d in self._all_drafters()), default=0.0)
        metadata = metrics.to_metadata(
            run_id=run_id,
            verifier=("sampling" if sampling else self.verifier_name),
            exact_greedy_equivalent=exact_equiv,
            output_hash=output_hash(generated),
            finished_eos=bool(generated and generated[-1] in self.target.eos_token_ids),
            gpu_kv_gb=round(kv_gb, 6),
            full_kv_residency=config.full_kv_residency,
            tokens_verified=max(0, len(generated) - unverified_tokens),
        )
        if park_kv:
            metadata["kv_residency_moves"] = res_stats.moved_tensors
            metadata["kv_residency_moved_mb"] = round(res_stats.moved_bytes / 1e6, 3)
        if sampling:
            metadata["distribution_exact"] = True
            metadata["temperature"] = config.temperature
            metadata["seed"] = config.seed
        if config.mode == "approx":
            metadata["verify_skip_policy"] = {"verify_every": config.verify_every}
            metadata["unverified_tokens"] = unverified_tokens
            metadata["trailing_unverified"] = trailing_unverified
            metadata["pending_divergence"] = divergence_total
            metadata["quality_risk"] = bool(
                unverified_tokens and (divergence_total or trailing_unverified)
            )
        return GenerationOutput(
            token_ids=generated, text=text, metadata=metadata, rounds=rounds
        )
