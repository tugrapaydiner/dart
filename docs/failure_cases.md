# Failure cases (live document)

docs/design.md §15 requires investigating failures, not hiding
them. Known/expected entries, to be filled with measurements as they are
reproduced:

| Failure | Cause | Status / mitigation |
|---|---|---|
| Untrained diffusion drafter acceptance ~ 1/vocab | random weights; expected | train with acceptance-optimized losses (`scripts/train_drafter.py`); measured before/after in its output |
| **Offset degeneracy (measured)**: drafter trains to low accept-CE but engine acceptance stays ~0 | `continuation_len == canvas_length` makes the sampled offset always 0, so every training context ends at the prompt boundary; mid-generation contexts are out-of-distribution | fixed: `training.canvas_length` decoupled from `data.continuation_len` (8 vs 32). Effect on tiny setup: engine acceptance 0.004 -> ~0.25, tokens/round 1.04 -> 2.77 |
| **Exposure bias at denoise time (measured)**: training CE improves faster than engine acceptance | training canvases contain ground-truth tokens in unmasked slots; at decode the canvas starts fully masked and conditions on its own fills | raise `noise_min` (0.5+); self-conditioning / scheduled re-noising is the research follow-up (docs/research_notes.md) |
| Early-position rejections dominate a canvas | drafter conditioned on stale suffix after correction | reconciliation re-masks low-confidence slots; tune `keep_confidence_above` upward if stale tokens persist |
| Compressed-KV acceptance collapses at low bits | quantized attention flips target argmax | raise `sink_bits`/`window_len`; verify-more-often policy; this is the acceptance-vs-bits curve worth publishing |
| Sparse (sink+window) lane mispredicts after long-range references | dropped middle tokens carried the answer | expected; the lane is for locally-predictable spans - scheduler should bench it on RAG tasks |
| n-gram lane proposes nothing on novel text | no suffix repeats in context | by design returns []; engine falls back to a plain target step (costless) |
| Tokenizer boundary merge on remote drafts | junction re-tokenization | TokenizerBridge drops un-alignable drafts; counted in `stats["boundary_breaks"]` |
| Speedup disappears at high batch | AR batching saturates compute | out of scope for v0.1 (batch 1); documented, not hidden |
| FP chunk-shape argmax flip on near-ties | reduction-order noise | see exactness_contract.md caveat; `verifier="reference"` paranoid mode; zero observed in suite |
