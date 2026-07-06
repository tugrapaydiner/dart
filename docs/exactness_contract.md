# The DART exactness contract

## The promise

For greedy decoding:

```
DartEngine.generate(prompt, mode="exact").token_ids
    == TargetModel.greedy_generate(prompt)
```

token-for-token, including whitespace, JSON punctuation, tool-call
delimiters, and stop tokens. Draft lanes (diffusion canvas, MTP, n-gram,
compressed KV, sparse attention) affect only *how fast* that sequence is
produced, never *which* sequence is produced.

## How it is enforced

Every committed token equals a target argmax computed by the target model
itself:

1. The engine maintains the invariant that the target KV cache covers all
   committed tokens except the last ("pending") one (see
   `dart/runtime/types.py`).
2. A verify round feeds `[pending, d_1 .. d_K]` in one causal pass. The
   logits after position `i` give the target's prediction for position
   `i+1`, so each draft token is compared against the argmax the target
   would have produced at exactly that point.
3. The longest matching draft prefix is committed, plus the target's own
   correction (on mismatch) or bonus token (on full acceptance). KV
   entries for rejected drafts are cropped away.
4. Plain greedy decoding is the `K=0` case of the same code path, so the
   baseline and DART share their forward machinery by construction.

Two independent verifier implementations (`verification/greedy.py`,
sequential; `verification/block_verify.py`, single-pass) must produce
identical results and KV lengths on every input; the test suite enforces
this equivalence and end-to-end output identity under adversarial,
constant, oracle, n-gram, diffusion, compressed-KV and sparse drafters.
A third, the tree verifier (`verification/tree_verify.py`, design.md
§5.4), checks several candidate branches in one pass; it is anchored to
the pair above by an enforced equivalence on chain-shaped trees and by
the same end-to-end byte-identity tests. Greedy/exact mode only.

## Scope and caveats — read before quoting numbers

* **Sampling is distribution-exact, not trace-exact.**
  `mode="exact_sampling"` implements speculative sampling (accept draft d
  w.p. `min(1, p(d)/q(d))`, residual resampling, bonus sampling); the
  emitted distribution equals the target's exactly - verified by
  Monte-Carlo against the production code path in
  tests/test_sampling_and_approx.py. A sampled run does NOT reproduce any
  particular baseline trace, and metadata reports
  `exact_greedy_equivalent: false`, `distribution_exact: true`.
* **Floating-point chunk-shape caveat.** Verifying K tokens in one pass
  computes mathematically identical logits to K single-token passes, but
  floating-point reduction order can differ between query-chunk shapes.
  If the target's top-2 logits at some position are within numerical
  noise (~1e-6 relative), the argmax could in principle flip between the
  two computation shapes. Consequences:
    - every committed token is still *a* target argmax under some valid
      forward computation - corrections are always target-computed;
    - empirically, the full test suite and benchmark exact-match checks
      pass at 100% on the reference setup (fp32, eager attention, CPU);
    - for paranoid verification, `DartEngine(verifier="reference")`
      checks tokens with single-token forwards that bit-match the
      baseline's shapes exactly, at the cost of all speedup.
  Run benchmarks with the block verifier and report the exact-match rate
  the harness measures - it is part of the output, not an assumption.
* **Batch 1.** v0.1 targets low-batch interactive serving; batched
  verification changes reduction shapes and must be re-validated.
* **Approximate mode** (`mode="approx"`) commits drafts without
  verification on skipped rounds (`verify_every=N`), then audits them at
  the next verify pass. Metadata reports `verify_skip_policy`,
  `unverified_tokens`, `pending_divergence`, `quality_risk`, and an
  honest `exact_greedy_equivalent` (true only when every committed token
  was eventually argmax-checked and none diverged). Modes are never
  silently mixed (design.md §2.4).

## What would break the contract (and how tests catch it)

| Bug class | Guard |
|---|---|
| KV not rolled back after rejection | `test_crop_restores_exact_state`, verifier equivalence tests |
| Off-by-one in pending-token bookkeeping | engine-vs-baseline identity tests across prompt/budget matrix |
| EOS committed but generation continues | forced-EOS tests (engine + verifier level) |
| Draft tokens leaking into output uncommitted | adversarial/constant drafter identity tests |
| Token budget overshoot | `test_engine_respects_token_budget_exactly` |
