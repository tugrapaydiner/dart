# Canvas reconciliation ablation (docs/design.md §7)

Target: tiny seeded Llama  ·  drafter: trained drafter (identical weights per arm)  ·  16 prompts × 64 tokens per seed  ·  5 prompt seeds

| prompt seed | discard acc/fwd | reconcile acc/fwd | improvement | exact (both arms) |
|---|---|---|---|---|
| 17 | 0.0699 | 0.1048 | +49.9% | 1.000 |
| 18 | 0.0631 | 0.0921 | +46.0% | 1.000 |
| 19 | 0.0371 | 0.0567 | +52.8% | 1.000 |
| 20 | 0.0604 | 0.0990 | +63.9% | 1.000 |
| 21 | 0.1017 | 0.1662 | +63.4% | 1.000 |

**Reconciliation improves accepted-tokens-per-diffusion-forward by +55.2% ± 8.1% (mean ± sample std over 5 prompt seeds; min +46.0%)** while preserving 100% exact greedy output in every run. Both arms share identical drafter weights and prompts within each seed. `full_redraft` is identical to `discard` in the round-based engine (fresh canvas + full step budget) and is therefore not a separate row.
