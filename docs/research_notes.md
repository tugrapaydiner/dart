# Research notes / open questions

## The bet (docs/design.md §1)

Let internal compute be approximate, parallel, compressed - guarantee the
committed output exactly. v0.1 implements the full skeleton; the open
research questions are now empirical:

1. **Canvas reconciliation (design.md §7).** Does warm-starting the
   unverified suffix beat fresh canvases in accepted tokens per diffusion
   forward at equal exactness?
   **Current measurement** (the reproducible
   `scripts/run_reconciliation_ablation.py` protocol; tiny target, trained
   drafter, identical weights per arm, 16 prompts x 64 tokens per seed,
   5 prompt seeds): **+55% ± 8%** accepted tokens per diffusion forward
   (mean ± sample std; min +46%, max +64%), every run 1.000 exact; per-seed
   table at `benchmarks/results/reconciliation_ablation.md`. An earlier
   exploratory run (different prompt mix) showed ~1.7x; the committed
   protocol number is the multi-seed one. Above the >=20% pre-registered
   target in docs/design.md §13.4 on every seed; still needs replication on
   a real target/corpus.
   Knobs: `keep_confidence_above`, `max_keep_fraction`, `repair_steps`.
2. **Acceptance-optimized distillation (design.md §8.4).** How much of the
   gap between L_denoise-only and the full loss mix is the acceptance CE
   term? Run scripts/train_drafter.py with weight ablations.
3. **Acceptance-vs-bits curves (design.md §9.3).** Accepted draft length as
   a function of KV bits / sink size / window size, per task family. The
   hypothesis worth testing: curves are task-dependent enough that the
   scheduler should pick bits per request.
4. **Scheduler upgrade (design.md §10).** Replace the heuristic with a
   contextual bandit on reward = accepted_tokens/wall_s - penalties.
   The EWMA plumbing and DecodeState features already exist.

## Shipped since v0.1-draft (formerly deferred)

* Exact sampling (speculative-sampling math, distribution-exact) -
  verification/sampling.py, MC-tested.
* Approx mode with verify_every skip policy + honest divergence/risk
  metadata - engine.
* TurboQuant-like 1-bit polar VQ KV store (real bit packing) -
  kv/compressors/turboquant_like.py.
* UCB bandit scheduler (accepted-tokens/sec objective) -
  runtime/scheduler.BanditScheduler.
* MTP head training - scripts/train_mtp.py (fresh-prompt acceptance
  0.00 -> ~0.12 on the tiny reference; conditions on target hidden state,
  which is why it generalizes where the token-conditioned diffusion
  drafter could not on a structureless random target).
* Triton kernel sources + torch references - attention/triton_kernels/
  (CPU-validated references; GPU compilation/Nsight still pending).
* Tree verification (design.md §5.4) - verification/tree_verify.py:
  multi-branch drafts in one forward via 4D tree mask + depth position
  ids + accepted-path KV gather; chain case proven equivalent to the
  block verifier; n-gram lane proposes multi-continuation trees.
* Measured GPU run (benchmarks/results/gpu_bench_rtx5060.md): gpt2 fp32
  on an RTX 5060 Laptop GPU, all lanes 1.000 exact on CUDA; ngram lane
  2.20x wall-clock over the same-run greedy baseline on code prompts.
  Compressed-KV lanes accept well (int8 ~0.98) but lose wall-clock on a
  124M compute-bound target - their regime is long-context memory-bound
  decoding, which needs the Triton kernels above.

## Deferred engineering (tracked, not forgotten)

* GPU validation of the Triton kernels; nibble-packing inside QuantStore;
  async KV prefetch overlap (kv/prefetch.py interface is ready).
* Sampling-mode tree verification (SpecInfer-style) + learned branch
  scoring on top of verification/tree_verify.py.
* vLLM/SGLang backends behind TargetModel's interface.
* Batched serving (>1 sequence) after re-validating the FP caveat.
* External harnesses (HumanEval/SWE-bench/LongBench) on a network-open box.

## Reading list anchors

DiffusionGemma + DFlash (block-diffusion drafting + AR verification),
VeriCache (lossy-KV draft / full-KV verify), TurboQuant (KV compression),
MaskGIT (confidence-scheduled parallel decode), KIVI (recent-window fp +
aged quantized KV), StreamingLLM (attention sinks), classic speculative
decoding + EAGLE/Medusa-style MTP heads. Verify availability/claims again
before publishing (docs/design.md §14 warning).
