# DART v0.3

**Draft several tokens. Verify them together.**

Exact greedy decoding with n-gram drafts and CUDA replay. No extra learned weights.

**[Run & reproduce →](docs/reproduce.md)** · [Raw results](benchmarks/results/v0.3-feedback) · [How it works](#how-it-works)

### Frozen benchmark

Supra-50M · RTX 5060 Laptop · FP32 · batch one · warm requests.

![DART takes 82.8 ms versus direct greedy 107.1 ms, HF greedy 118.6 ms and HF prompt lookup 178.8 ms, excluding synthetic repetition.](docs/figures/v0.3-feedback/methods.svg)

HF retains its generation loop. **48 cases, seven methods, three processes.**

| Complete output matches | Faster case means | Added parameters |
|:---:|:---:|:---:|
| **4,176 / 4,176** | **46 / 48** | **0** |

Token IDs, lengths and EOS all match. Five cases end naturally at EOS.
[Measured revision](https://github.com/tugrapaydiner/dart/tree/3d651482538e15d427fd8533340e98a53b7eec8b);
[later runtime changes](docs/reproduce.md#later-runtime-changes) are reported separately.

### Where it helps

![DART and direct greedy latency across code, instruction, structured and repetition workloads.](docs/figures/v0.3-feedback/workloads.svg)

All four workload means improve. Including repetition gives **31.9% less latency**;
repetition contributes **47.6% of the saved time**.

![Four measured methods at output budgets of 16, 32 and 64 tokens.](docs/figures/v0.3-feedback/budgets.svg)

Different cases populate each budget. This is not a controlled length sweep;
EOS can stop a request early.

### How it works

For prefix $`x`$ and draft $`d`$, the target checks every position in one causal pass:

$$
y_i = \arg\max_v P_\theta\left(v \mid x, d_{1:i-1}\right).
$$

Keep the longest draft prefix matching these greedy choices. Emit the first
correction, or a bonus token if all match. **Accepted EOS stops immediately**;
output and context limits bound every round.

[CUDA replay](dart/models/replay.py) reuses captured execution, while request-owned
KV snapshots support rollback. Eager execution remains the default.

![Three paired process means with entropy on versus off: 92.0 to 83.9 ms, 78.9 to 73.0 ms, and 70.0 to 65.8 ms.](docs/figures/v0.3-feedback/feedback.svg)

The matched on/off ablation saves **7.6%** by skipping entropy nobody consumes.
Fixed-four drafting was selected on development inputs.

### Every case, including the losses

![All 48 case means; two slower instruction cases are highlighted in pink.](docs/figures/v0.3-feedback/cases.svg)

Two instruction cases take **3.7% and 11.8% longer**. Both case intervals cross
parity. [Per-case timings and intervals →](benchmarks/results/v0.3-feedback/analysis.json)

**Scope:** one 51.8M model and one laptop. Warm timings exclude preparation.
Peak allocation is **0.589 GiB**, including two model copies and shared workspaces.
No larger-model, answer-quality or independent-reproduction result is claimed.

**[Reproduce →](docs/reproduce.md)** · [Development & alternatives](benchmarks/results/v0.3-feedback/development.json) ·
[Apache-2.0](LICENSE) · [Data attribution](benchmarks/fixtures/feedback-sources.json) · [Cite DART](CITATION.cff)
