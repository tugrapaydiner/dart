# Exactness vs KV bit-width (README Figure 2)

Target: tiny seeded Llama · 12 prompts × 48 tokens · sink 4 @8-bit · fp window 8 · CPU (exact-match rate is hardware-independent)

| KV bits (aged tokens) | unverified quantized decode: exact rate | DART (same cache, drafts only): exact rate | DART acceptance |
|---|---|---|---|
| 8 | 0.917 | 1.000 | 1.000 |
| 6 | 0.833 | 1.000 | 0.990 |
| 4 | 0.167 | 1.000 | 0.913 |
| 3 | 0.083 | 1.000 | 0.846 |
| 2 | 0.000 | 1.000 | 0.671 |

Unverified decoding on a quantized cache loses byte-exactness as bits fall; the verified path stays at 1.000 at every bit-width because the quantized cache is only ever used to *draft*. Quantization error shows up in the acceptance column (a throughput cost) instead of the output.
