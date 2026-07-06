# DART design document

This is the engineering design for DART (Diffusion-Augmented Reliable
Transformer Serving). It states the system's contract, the design of each
component, and the evaluation protocol. Code comments cite sections of this
document as `docs/design.md §N`. For the result figures and headline claims,
see the [README](../README.md); for the formal exactness statement and its
caveats, see [exactness_contract.md](exactness_contract.md).

---

## §1 Motivation

The classic efficiency project asks "can we make a smaller model that is
almost as good?" DART asks a different question:

> Can a serving system use approximate, parallel, or compressed compute
> internally, while guaranteeing that the committed output is exactly the
> trusted model's output?

Three trends make this the right framing for single-sequence serving:

1. **Non-autoregressive drafting is practical.** Block-diffusion models
   (DiffusionGemma, DFlash lineage) produce whole token canvases per forward
   pass — a natural high-throughput proposal machine, even where their
   standalone quality trails autoregressive models.
2. **Lossy KV compression becomes safe under verification.** Quantized or
   sparse KV caches drift from the target when trusted directly
   (VeriCache-style systems demonstrate the wrapped alternative); used only
   to *draft*, their error costs throughput, never correctness.
3. **Speculative decoding is single-technique.** Published systems pick one
   drafter. A verifier that is agnostic to where proposals come from can
   compose many — and schedule between them online.

The bet: let every fast path be aggressive and lossy, gate everything behind
one exact verifier, and make the whole system's fidelity a property of the
verifier rather than of any drafter. Drafter quality then becomes a pure
speed lever, which changes how drafters should be built and trained (§8.4).

## §2 The exactness contract

DART's modes, from strongest to weakest guarantee. The formal statement
lives in [exactness_contract.md](exactness_contract.md); this section is the
design rationale.

### §2.1 Exact mode

For greedy decoding, the engine's output must be **byte-identical** to the
target model's own greedy decode — including whitespace, JSON punctuation,
tool-call delimiters, and the stop token. The property is verifier-induced,
not weight-induced: every committed token is the target's full-precision
argmax at its position, so the guarantee holds for arbitrary weights and
transfers across checkpoints without retuning. Exact mode is the default and
the required mode for code patches, shell commands, structured outputs, tool
calls, and any benchmark claim.

### §2.2 Distribution-exact sampling

`mode="exact_sampling"` preserves the target's full sampling distribution
(not just its mode) via speculative sampling (§5.3). The guarantee is
distributional: emitted tokens are distributed exactly as the target's, but
no particular baseline trace is reproduced. Metadata reports
`exact_greedy_equivalent: false`, `distribution_exact: true`.

### §2.3 Approximate mode

`mode="approx"` may skip verification on a schedule (`verify_every=N`),
committing drafts unverified and auditing them at the next verify round.
Metadata must report the skip policy, the number of unverified tokens, the
audited divergence count, and a quality-risk flag.

### §2.4 The rule: never silently mix modes

Every response carries metadata sufficient to know what guarantee it has:

```json
{
  "mode": "exact",
  "exact_greedy_equivalent": true,
  "target_model": "gpt2",
  "lanes_used": ["diffusion", "ngram"],
  "tokens_generated": 1024,
  "tokens_verified": 1024,
  "output_hash": "…",
  "gpu_kv_gb": 0.41,
  "full_kv_residency": "cpu_pinned"
}
```

An output that was not fully verified must say so in machine-readable form.
This rule is not negotiable anywhere in the codebase.

## §3 System overview

Decoding proceeds in rounds over a single sequence (batch 1):

```text
prompt ──► scheduler picks a lane (§10)
              │
              ▼
        draft lane proposes K tokens (§6)      lossy, cheap
              │
              ▼
        block verifier: one target forward (§5)  full precision
              │
              ├── commit longest argmax-matching prefix + 1 target token
              ├── crop KV entries of the rejected suffix
              └── diffusion lane: reconcile the rejected canvas suffix (§7)
```

The engine (`dart/runtime/engine.py`) maintains one invariant between
rounds: *the target KV cache covers every committed token except the pending
one*, which is re-folded into the next verification pass. Plain greedy
decoding is the K=0 degenerate case of the same code path, so the baseline
and the accelerated system cannot diverge by implementation.

Repository map:

```text
dart/
  runtime/       engine, scheduler, OpenAI-style server, metrics, types
  verification/  reference + block verifiers, speculative sampling
  drafters/      n-gram, MTP, diffusion, compressed-KV, sparse lanes
  diffusion/     canvas, noise schedules, denoise loop, reconciliation, losses
  kv/            quantized cache, codecs, residency, prefetch
  attention/     sliding-window+global, block-sparse, verifier masks, Triton
  models/        target wrapper, tokenizer bridge, teacher-trace capture
configs/  benchmarks/  scripts/  tests/  docs/
```

## §4 Target model

The target is the model whose output is trusted; DART reproduces it, never
improves on it. `dart/models/target.py` wraps a Hugging Face causal LM (or
the seeded tiny reference model used by the test suite) behind a minimal
interface: `prefill`, `forward_block`, `crop`, `greedy_generate`. Because
exactness is weight-independent (§2.1), the entire verification suite runs
on the tiny CPU target with no downloads; `scripts/run_real_model.py`
validates the same property against real checkpoints. Model-specific
serving details (chat templates, EOS conventions, attention backends) are
config concerns, not engine concerns.

## §5 Verification

Two independent verifier implementations bound the system. They must
produce identical commit sequences and identical KV lengths on every input;
the test suite enforces this equivalence across adversarial, constant,
oracle, and real drafters.

### §5.1 Reference verifier

`verification/greedy.py` decides one token per single-token forward — the
same computation shape as the baseline greedy decode, so it bit-matches the
baseline by construction. It exists as the trust anchor and the paranoid
fallback (`DartEngine(verifier="reference")`); it yields no speedup.

### §5.2 Block greedy verification

`verification/block_verify.py` decides K+1 tokens per forward. Feeding
`[pending, d_1 … d_K]` through the target in one causal pass yields the
target's next-token distribution at every position; the longest
argmax-matching draft prefix is committed, plus the target's own correction
token (on mismatch) or a free bonus token (on full acceptance). An accepted
EOS terminates the round without a correction. KV entries written for
rejected draft tokens are rolled back by cropping the cache, which
re-establishes the §3 invariant. The floating-point caveat on
reduction-order differences between chunk shapes is documented in
[exactness_contract.md](exactness_contract.md).

### §5.3 Speculative sampling

`verification/sampling.py` implements the standard speculative-sampling
rule: accept draft `d ~ q` with probability `min(1, p(d)/q(d))`; on
rejection resample from the normalized positive residual `(p-q)^+`; on full
acceptance draw a bonus from `p`. Token-only lanes are treated as one-hot
proposals, for which acceptance reduces to `p(d)`. The identity
`min(p,q) + (p-q)^+ = p` gives marginal correctness. Monte-Carlo tests
exercise the production code path, not a copy of the math.

### §5.4 Tree verification

`verification/tree_verify.py` packs several candidate branches that share
prefixes into one forward, using a tree-structured attention mask
(`attention/verifier_mask.tree_mask` algebra) and depth-based position ids
(`TargetModel.forward_tree`). The target's argmax walk selects at most one
root-to-leaf path; the longest matching path is committed plus the usual
correction or bonus token. Accepted-path KV entries are gathered out of the
packed tail (`TargetModel.gather_tree_tail`) — valid because a packed
node's K/V depend only on the prefix and its ancestor chain. A chain-shaped
tree coincides with block verification decision-for-decision, which the
test suite enforces; that equivalence anchors the tree path to the §5.1/
§5.2 pair. Exact (greedy) mode only; requires a model that honors custom 4D
attention masks (Llama-family, eager attention). Enabled with
`DartEngine(verifier="tree")`; drafters may expose `propose_tree`
(the n-gram lane proposes multiple historical continuations), and plain
drafters degrade to chains.

## §6 Draft lanes

Lanes are interchangeable behind the verifier; they differ only in how
cheaply they approximate the target's argmax continuation. The operative
metric is acceptance (§13.2), never the drafter's own fluency.

### §6.1 n-gram retrieval

Longest-suffix match in the existing context; the historical continuation
is copied forward. Draft cost is retrieval only — no forward passes. On
repetitive spans (code, structured output) this is the cheapest possible
lane; on novel text it proposes nothing and the engine falls back to a
plain target step at zero extra cost.

### §6.2 Multi-token prediction (MTP)

Learned heads predict offsets +2, +3, … from a single target hidden state
(`drafters/mtp.py`, trained by `scripts/train_mtp.py`). Conditioning on the
target's own hidden state is what lets small heads generalize; this is the
low-risk lane and the standard speculative-decoding baseline.

### §6.3 Diffusion canvas

The flagship lane (`drafters/diffusion_small.py`, `dart/diffusion/`). A
compact bidirectional encoder denoises a fixed-length masked canvas
conditioned on a right-aligned context window, MaskGIT-style: cosine
schedule, confidence-thresholded commits per step, adaptive early stop.
Unlike an autoregressive drafter, a canvas is refined *jointly* — which is
what makes rejected suffixes repairable (§7). An off-the-shelf
block-diffusion model can serve as this lane via
`drafters/diffusiongemma.py` with the tokenizer bridge of
`models/tokenizer_bridge.py`; drafts that do not re-tokenize cleanly at the
junction are dropped and counted.

### §6.4 Compressed-KV self-drafting

The target model drafts against its *own* KV cache, compressed (INT8, INT4,
1-bit polar VQ — §9). Per-forward memory traffic drops; the verifier still
runs on full-precision KV, which is precisely why aggressive quantization is
admissible: dequantization error perturbs acceptance, not output.

### §6.5 Sparse-attention lanes

Two token-selection lanes with cost flat (or sublinear) in context length:
`sparse_kv` — attention-sink + sliding-window token drop
(`attention/sliding_global.py`, StreamingLLM-style); and `block_sparse_kv`
— top-k context blocks by embedding relevance (`attention/block_sparse.py`,
MSA-style). Both mispredict exactly where dropped context mattered, which
the verifier converts from a correctness bug into a measurable acceptance
dip (§15).

## §7 Canvas reconciliation

DART's novel contribution: recovery of rejected diffusion work.

### §7.1 Rationale

After a rejection at slot m, an autoregressive draft suffix is conditioned
on an overruled token and is worthless. A diffusion canvas is refined
jointly, so the unverified suffix is often *mostly right* — a strong
initialization rather than garbage.

### §7.2 Algorithm

With m of K slots accepted, the accepted prefix and the target's correction
enter ordinary context. The unverified suffix `d_{m+1} … d_K` is re-masked
at its low-confidence slots (below `keep_confidence_above`, with a
`max_keep_fraction` cap so enough slots always reopen) and repaired under a
reduced step budget `repair_steps « denoise_steps`
(`diffusion/reconciliation.py`, `diffusion/noise.renoise_suffix`). The
verifier still gates every repaired token; reconciliation is throughput
optimization only.

### §7.3 Evaluation protocol

Two arms at equal canvas length and step budgets — `discard` (fresh
fully-masked canvas each round; in a round-based engine the "discard" and
"full re-draft" baselines coincide) and `reconcile`. Both arms run
identical drafter weights (same checkpoint, or identically-seeded init) on
the same prompt set, and the comparison repeats over several prompt seeds;
the claim is the mean ± sample std of the per-seed improvement. Headline
metric: **accepted tokens per diffusion forward**, with exact-match rate as
a gate on both arms and recovery success rate (warm rounds with ≥1 accepted
token) as a diagnostic. `scripts/run_reconciliation_ablation.py` implements
the protocol; the pre-registered success threshold was ≥20% improvement.

## §8 Training the diffusion drafter

### §8.1 Teacher traces

Supervision comes from the target itself (`models/hidden_trace.py`): greedy
continuations, per-token logits, entropies, and hidden states captured over
a prompt corpus. Training contexts must include *mid-generation* positions,
not only prompt-boundary positions — training only at the boundary is the
measured "offset degeneracy" failure of
[failure_cases.md](failure_cases.md).

### §8.2 Corruption

Teacher continuations are corrupted into noisy canvases: masking, random
token replacement, span permutation, entropy-weighted noise (high-entropy
positions get more noise), and syntax-aware noise that sometimes preserves
braces/indentation so code training stays stable (`diffusion/noise.py`).
High noise floors matter: at decode time the canvas starts fully masked, and
training distributions that are too clean produce exposure bias (measured;
see [failure_cases.md](failure_cases.md)).

### §8.3 Losses

```text
L = L_denoise + 0.3·L_kl + 0.5·L_accept + 0.1·L_cal
```

masked-denoising cross-entropy; temperature-scaled KL to the target's
distribution; token-level cross-entropy to the target *argmax*; and a
confidence-calibration BCE making per-slot confidence a usable acceptance
estimator (`diffusion/training_losses.py`).

### §8.4 The acceptance objective is the trick

A normal diffusion LM optimizes the quality of its own output. A DART
drafter optimizes *expected consecutive tokens accepted by the verifier*. A
token that is fluent but not the target argmax has zero value in exact mode.
This is why the acceptance CE carries the largest auxiliary weight, and why
calibration matters: confidence drives both adaptive early stopping (§6.3)
and the re-masking policy of §7.2.

## §9 KV compression

### §9.1 Codec interface and stages

`kv/cache.py` implements a KIVI-style aging buffer: the most recent
`window_len` tokens stay full-precision in a ring buffer; tokens aging out
are quantized once (asymmetric min/max per head/token vector) and appended —
no re-quantization. The per-layer store is pluggable behind the `KVStore`
protocol; implemented codecs, in order of aggression: INT8, INT4, token
dropping, and a TurboQuant-like 1-bit polar vector quantizer with real bit
packing (`kv/compressors/`). `stats()` reports logical (packed) and
physical bytes separately so memory claims stay honest.

### §9.2 Residency

In exact mode the system may hold two caches: the compressed cache
on-device for drafting, and the full-precision verifier cache parked in
pinned host memory between rounds (`kv/residency.py`), staged deviceward per
verify pass and overlapped with the next round's drafting
(`kv/prefetch.py`). Device residency is then the compressed cache alone.
Measured quantities: moved tensors/bytes, transfer time, verifier stall.

### §9.3 Compression objective

Do not optimize perplexity. Optimize **accepted draft length per unit
verifier cost**. A codec that slightly worsens perplexity may be excellent
if it preserves the target argmax over long spans; the acceptance-vs-bits
curve per task family is the publishable object (§13.2), and the measured
collapse point at low bits is a feature of the report, not a secret
([failure_cases.md](failure_cases.md)).

### §9.4 Sensitive-position precision policy

Some positions disproportionately flip downstream argmaxes. The hybrid
policy (`kv/compressors/hybrid_policy.py`) raises per-token or per-layer
bit-width for: attention sinks, first/last layers, structural delimiters
(JSON punctuation, code braces/indentation), and high-entropy positions —
taking the maximum applicable width. Config surface:

```yaml
precision_policy:
  default_kv_bits: 4
  sink_tokens_bits: 8
  last_2_layers_bits: 8
  syntax_tokens_bits: 8
  high_entropy_bits: 8
```

## §10 Scheduling

A scheduler maps decode state to a lane each round
(`runtime/scheduler.py`). Inputs: context length, generated length, device
headroom, prompt kind, exactness requirement, predictive-entropy EWMA, and
per-lane acceptance EWMAs. The default is a heuristic with sticky
rejection-based fallback; a UCB contextual bandit is also provided with
per-lane reward

```text
reward(lane) = committed_tokens / round_latency
               − λ_rej · rejection_fraction
               − λ_mem · lane_KV_footprint
```

and hard constraints benching any lane that breaches a rejection-rate or
KV-budget ceiling. Scheduling is advisory: a poor arm costs latency, never
correctness. Entropy gating shortens draft budgets where the target is
uncertain (drafts would mostly be rejected) and lengthens them where it is
confident.

## §11 Runtime API and configuration

The user-facing call:

```python
from dart import DartEngine, DecodeConfig

engine = DartEngine.from_config("configs/dart_exact.yaml")
out = engine.generate(
    "def binary_search(arr, target):\n",
    DecodeConfig(mode="exact", max_new_tokens=128),
)
print(out.text, out.metadata)
```

`DartEngine.from_config` (backed by `runtime/factory.py`) builds the
target, lanes, scheduler, and decode defaults from YAML; `generate` accepts
either token ids or text (when a tokenizer exists). Shipped presets under
`configs/`: `dart_exact.yaml` (n-gram lane, lowest risk),
`dart_diffusion.yaml` (diffusion + reconciliation), `dart_long_context.yaml`
(compressed-KV + residency + precision policy), `dart_research_max.yaml`
(all lanes + bandit scheduler), plus `target_*.yaml` presets for real
checkpoints. Every config keys the same `DecodeConfig` surface; metadata
(§2.4) is uniform across presets.

## §12 Server and demo

`runtime/server.py` exposes an OpenAI-style completions API plus a `/demo`
page rendering the decode side-by-side: baseline target decoding against
DART exact mode, with matching output hashes, per-round accept/reject
visualization, lane-switching decisions, and KV memory. The demo exists to
make the contract *visible*: identical bytes, fewer target forwards.

## §13 Benchmark protocol

### §13.1 Workloads

Prompt families under `benchmarks/prompts/`: code completion, JSON/tool
calling, chat, math, long-context. These stress respectively: syntax
exactness, structured-output exactness, generality, entropy, and KV
pressure. External harnesses (HumanEval/SWE-bench/LongBench) are roadmap
items (§16) and must run on a network-open box.

### §13.2 Metrics

System: tokens/sec, per-phase wall time (draft vs verify), GPU peak memory,
KV bytes (logical vs physical). Verification: exact-match rate vs target
greedy (a **gate**, not a metric — any row below 1.000 fails the run);
acceptance rate by lane; tokens committed per verify round; rejection
position distribution; accepted tokens per diffusion forward (§7.3).
Acceptance metrics are hardware-independent; wall-clock claims are only
reported from the hardware that produced them.

### §13.3 Required comparisons

`benchmarks/run_bench.py` produces: target-only (ground truth), n-gram,
AR-draft, compressed-KV at each bit-width, sparse, diffusion-discard,
diffusion-reconcile, MTP. Untrained drafters are labelled `_untrained`;
their rows validate machinery, not performance. The lossy direct-decode
baseline (compressed KV trusted, no verification) is included to show what
the verifier is buying.

### §13.4 Success criteria

Pre-registered tiers. Minimum credible: 100% exact-match on the
deterministic suite, ≥1.3× measured speedup on at least one real workload,
clear long-context memory reduction, complete ablations and failure
analysis. Strong: ≥2× batch-1 decode throughput on selected workloads, ≥50%
KV memory reduction at 128K context, reconciliation ≥+20% accepted tokens
per diffusion forward. Claims must always distinguish measured numbers from
roofline projections.

## §14 Honest-reporting rules

Never claim "always faster" or "quality unchanged." Claim: measured
exact-match rate, acceptance and throughput on named workloads and named
hardware, and where the approach fails. Projections must be labelled as
projections everywhere they appear, including figures. Comparisons against
published systems use their papers' numbers on their hardware and say so —
positioning, not head-to-head. Never compare DART to hosted chat models on
answer quality; DART is a serving layer and inherits quality from its
target. Verify external claims and availability again before publishing
them.

## §15 Failure analysis

Failures are investigated and published, not hidden
([failure_cases.md](failure_cases.md) is the live document). Every failure
entry carries cause and mitigation; measured failures (offset degeneracy,
exposure bias, low-bit acceptance collapse) include the numbers. The sparse
lanes' failure mode — mispredicting exactly where dropped context mattered —
is the canonical example of the verifier converting a correctness bug into a
visible acceptance dip.

## §16 Roadmap

In rough order of value:

* **GPU kernel validation.** The Triton sources under
  `attention/triton_kernels/` ship with CPU-tested torch references;
  compile, validate, and profile them on GPU (Nsight), then wire them into
  the compressed-KV lane.
* **Tree verification extensions.** §5.4 ships greedy tree verification;
  sampling-mode trees (SpecInfer-style multi-branch speculative sampling)
  and learned branch scoring remain open.
* **Nibble packing** inside `QuantStore`; async KV prefetch overlap is
  interface-ready (`kv/prefetch.py`).
* **vLLM/SGLang backends** behind the `TargetModel` interface, including
  off-the-shelf block-diffusion checkpoints as the §6.3 lane.
* **Batched serving** (>1 sequence) after re-validating the floating-point
  chunk-shape caveat under batching.
* **External harnesses** (HumanEval, SWE-bench Lite, LongBench/RULER) for
  workload-level acceptance curves.
