<h1 align="center">DART v0.2</h1>

<p align="center">
  <strong>Draft several tokens. Verify them together.</strong><br>
  Faster greedy decoding on Supra-50M, with the target deciding what gets emitted.
</p>

<p align="center">
  <a href="#decoding-speed">Results</a> ·
  <a href="#verified-drafting">Method</a> ·
  <a href="docs/reproduce.md">Run &amp; reproduce</a> ·
  <a href="#measurement-notes">Evidence</a>
</p>

| GPU speedup | CPU speedup | Full-output agreement |
|:---:|:---:|:---:|
| **2.18×** | **1.46×** | **128 / 128** |
| RTX 5060 Laptop | One CPU thread | Test prompts, per device |

Speedup is against the fastest of three measured baselines. Agreement compares
complete greedy token sequences, lengths and EOS with Hugging Face.
One model, one machine, fixed procedural prompts.

---

## Decoding speed

![Throughput of HF generate, direct target greedy, target-only engine and DART n-gram on CPU and GPU.](docs/figures/v0.2/throughput.svg)

**DART reaches 39.6 tokens/s on GPU and 27.4 tokens/s on CPU.** All four
methods use the same weights, prompts and output limits. The release preset
copies four-token n-gram drafts from the context; it needs no learned drafter.

$$
S = \frac{\min_{b\in\mathcal{B}}\overline{T}_b}{\overline{T}_{\mathrm{DART}}}
$$

Here, $\mathcal{B}$ contains HF generate, the direct target greedy loop and the
target-only engine. $\overline{T}$ is mean request latency. Paired 95% intervals
are **2.155–2.212× GPU** and **1.459–1.469× CPU**, conditional on these fixed cases.

## Verified drafting

**Draft → verify → keep the matching prefix → correct or continue.**

For a prefix $p$ and draft tokens $d_1,\ldots,d_K$, one causal target pass
computes the next-token choices. Let $m$ be the number of consecutive draft
tokens that match from the start:

$$
t_i = \underset{v}{\arg\max}\;P_\theta(v\mid p,d_{<i})
$$

$$
\text{commit} = (d_1,\ldots,d_m,t_{m+1})
$$

A mismatch supplies the target's correction. A full match supplies a bonus
token: with $K=4$, a verification pass can emit up to five tokens. An accepted
EOS ends the sequence without an extra token; output and context limits also
bound each round. Rejected cache entries are removed before decoding continues.

## Across workloads

![Direct target greedy and DART n-gram latency across code, instruction, repetition and structured workloads on CPU and GPU.](docs/figures/v0.2/workload-comparison.svg)

**Lower mean latency in all four workload groups.** Each pair compares direct
target greedy with DART on the same cases. Bars show mean latency; whiskers show
95% intervals. These are procedural workloads, and repeated output patterns
favor n-gram copying.

## Every test case

![DART speedup against each case's fastest measured baseline. GPU has 24 of 24 cases above 1; CPU has 20 of 24.](docs/figures/v0.2/case-speedups.svg)

**24/24 GPU cases and 20/24 CPU cases have lower observed mean latency.** Each
point uses that case's fastest baseline over ten repeats. Four CPU slowdowns
remain visible. The 1× line means equal latency; these counts do not establish
statistical significance for individual cases.

## Output budgets

![HF generate and DART latency at 32, 64 and 128-token output budgets, with matching CPU and GPU scales.](docs/figures/v0.2/latency-by-budget.svg)

**The gap grows with the output budget.** This chart compares DART with HF
generate at 32, 64 and 128 tokens. Against the fastest measured baseline,
the 32-token CPU result is **1.21×**, below the **1.25×** target.
Actual output length can be shorter when the model emits EOS.

## GPU memory

![Peak allocated GPU memory: HF generate at 215.8 MiB; direct target greedy, target-only engine and DART each round to 227.8 MiB.](docs/figures/v0.2/gpu-memory.svg)

**DART uses 227.8 MiB; HF generate uses 215.8 MiB.** DART and the other two
baselines round to the same peak. This measures PyTorch allocations with the
shared target and both engine instances resident. It excludes total device
memory use and does not demonstrate a memory saving.

## Where request time goes

![Separate diagnostic request times, split into setup and prefill versus decoding, for all four methods on CPU and GPU.](docs/figures/v0.2/request-profile.svg)

$$
T_{\mathrm{request}} = T_{\mathrm{setup}} + T_{\mathrm{prefill}} + T_{\mathrm{decode}}
$$

**The solid bars show decoding; the hatched bars show setup and prefill.**
Drafting and verification are included within decoding. These are separate
diagnostic requests with phase synchronization: 24 cases, two repeats.
They are excluded from the headline speedup calculation.

## Learned drafting

A **2.08M-parameter** compact drafter improved held-out draft acceptance from
**5.76% to 12.45%** after training with one seed:

$$
a = \frac{N_{\mathrm{accepted\ draft\ tokens}}}{N_{\mathrm{proposed\ draft\ tokens}}}
$$

![Nine drafting policies ordered by GPU latency, with aligned CPU and GPU bars. DART n-gram is fastest on both devices.](docs/figures/v0.2/policy-comparison.svg)

**Higher acceptance did not establish a learned-policy speedup.** In this
separate four-case, 32-token validation pilot, no learned policy beat the
fastest target-only baseline in the aggregate. N-gram drafting was fastest
on both devices. Whiskers show 95% intervals; the **10% repair latency target
was missed**.

## Measurement notes

| Setting | Scope |
|:---|:---|
| Target | Supra-50M-Instruct · 51.8M parameters · local HF format |
| Execution | One sequence · FP32 · CPU/SDPA or CUDA/eager · one CPU thread |
| Context | Prompt + output + verification lookahead ≤ 1,024 tokens |
| Timing | 24 test cases · 2 warm-ups · 10 repeats · rotating method order |
| Output budgets | 32 / 64 / 128 tokens; timing prompts are 30–101 tokens |
| Correctness | 128/128 full outputs per device; all 960 timed and 192 profiled outputs also match |

CUDA timings are synchronized. Loading, tokenization, correctness checks and
warm-up are excluded. Only the completed GPU retry contributes latency results;
the interrupted earlier attempt remains in the run history.

The evidence covers one model and machine, not answer quality or general
deployment performance. CUDA bfloat16/eager failed a token/EOS comparison and
is excluded. Learned-policy studies use one training seed; repair and the
scheduler were slower, and failure-mask training reduced acceptance.

[Protocol](benchmarks/results/v0.2/protocol.json) ·
[All results](benchmarks/results/v0.2/summary.json) ·
[Outputs](benchmarks/results/v0.2/correctness.json) ·
[Raw timings](benchmarks/results/v0.2/timings.json) ·
[Pilots & failures](benchmarks/results/v0.2/validation-pilots.json)

**[Run locally, train and reproduce the measurements →](docs/reproduce.md)**

---

Built on [speculative decoding](https://arxiv.org/abs/2211.17192).
Repair uses token anchors; it does not reproduce
[ReTrace's hidden-state repair](https://arxiv.org/abs/2608.29748v2).

[Apache-2.0](LICENSE) · [Cite DART](CITATION.cff) · Model weights and trained
checkpoints are not included in the v0.2 package.
