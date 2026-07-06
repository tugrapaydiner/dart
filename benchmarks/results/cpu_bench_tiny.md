# DART lane benchmark — tiny-seeded

Device: cpu (CPU) · dtype torch.float32 · 8 prompts × 48 tokens · draft_len 4 · torch 2.12.1+cu130

`speedup` is wall-clock vs the same-run trusted greedy baseline (same model, device, prompts). Exact match is a gate: any row below 1.000 fails the run.

| lane | exact match | acceptance | tokens/round | tok/s | speedup |
|---|---|---|---|---|---|
| target_only | 1.000 | 0.000 | 1.02 | 1193.8 | 0.93x |
| ngram | 1.000 | 0.657 | 1.80 | 1920.93 | 1.47x |
| ar_draft | 1.000 | 0.011 | 1.06 | 241.93 | 0.18x |
| compressed_kv_int8 | 1.000 | 1.000 | 4.80 | 820.72 | 0.61x |
| compressed_kv_int4 | 1.000 | 0.933 | 4.63 | 801.86 | 0.60x |
| sparse_kv | 1.000 | 0.865 | 4.28 | 890.3 | 0.66x |
| block_sparse_kv | 1.000 | 0.494 | 2.92 | 534.92 | 0.40x |
| diffusion_discard | 1.000 | 0.001 | 1.02 | 212.14 | 0.17x |
| diffusion_reconcile | 1.000 | 0.001 | 1.02 | 333.66 | 0.26x |
| compressed_kv_vq1bit | 1.000 | 0.562 | 3.15 | 413.7 | 0.32x |
| mtp_untrained | 1.000 | 0.000 | 1.02 | 995.49 | 0.79x |
| mtp_trained | 1.000 | 0.189 | 1.57 | 1344.52 | 1.04x |
| diffusion_trained_discard | 1.000 | 0.111 | 1.87 | 291.77 | 0.23x |
| diffusion_trained_reconcile | 1.000 | 0.097 | 1.73 | 443.31 | 0.34x |
