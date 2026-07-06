# DART lane benchmark — gpt2

Device: cuda (NVIDIA GeForce RTX 5060 Laptop GPU) · dtype torch.float32 · 6 prompts × 64 tokens · draft_len 8 · torch 2.10.0+cu128

`speedup` is wall-clock vs the same-run trusted greedy baseline (same model, device, prompts). Exact match is a gate: any row below 1.000 fails the run.

| lane | exact match | acceptance | tokens/round | tok/s | speedup |
|---|---|---|---|---|---|
| target_only | 1.000 | 0.000 | 1.02 | 59.96 | 1.03x |
| ngram | 1.000 | 0.395 | 3.23 | 162.82 | 2.20x |
| compressed_kv_int8 | 1.000 | 0.976 | 8.76 | 37.15 | 0.56x |
| compressed_kv_int4 | 1.000 | 0.931 | 8.42 | 38.04 | 0.54x |
| sparse_kv | 1.000 | 0.745 | 7.00 | 35.53 | 0.51x |
| block_sparse_kv | 1.000 | 0.351 | 3.65 | 19.95 | 0.30x |
| diffusion_discard | 1.000 | 0.000 | 1.02 | 25.59 | 0.54x |
| diffusion_reconcile | 1.000 | 0.000 | 1.02 | 29.6 | 0.73x |
| compressed_kv_vq1bit | 1.000 | 0.334 | 3.61 | 9.69 | 0.18x |
