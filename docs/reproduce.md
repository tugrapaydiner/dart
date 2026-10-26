# Reproduce the feedback study

[Results](../README.md) · [Raw evidence](../benchmarks/results/v0.3-feedback) ·
[Study record](../benchmarks/results/v0.3-feedback/study.json)

Use an existing **HF-format Supra-50M-Instruct** directory as `MODEL_DIR`.
Commands load local files only; GGUF is unsupported. Run from the repository
root with new output paths. The benchmark requires CUDA, batch one, FP32 Llama
with default RoPE and Transformers ≥5.16.1.

## Install

The measured environment used Python 3.13.14, Windows 11 and an RTX 5060 Laptop.
Create a venv without system packages:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell. Install the measured CUDA build,
then the recorded dependencies and project:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-tested.txt -r benchmarks/results/v0.3-execution/requirements.txt ".[dev,figures]"
python -m pip check
dart --version
```

The dependency pins describe this measured environment, not every platform.
The feedback study used a public wheel outside the checkout in a new venv
**reusing existing system dependencies**. It does not establish a new fresh
dependency install or independent reproduction. The earlier execution study
records a separate [fresh-dependency validation](../benchmarks/results/v0.3-execution/reproduction.json).
CPU CI separately builds and installs the current wheel with fresh dependencies
on Python 3.10/3.12. Local CUDA and Supra output checks complement its skips.

## Inspect the published records

No model or GPU is needed to reanalyze records or regenerate figures:

```sh
python -m benchmarks.feedback_analysis --run benchmarks/results/v0.3-feedback --out runs/feedback-analysis.json
python -m benchmarks.feedback_figures --run benchmarks/results/v0.3-feedback --out runs/feedback-figures --previews runs/feedback-previews
```

Both commands validate complete output sequences, hashes, timing matrices,
settings and process records. Figures recompute results from the raw sessions,
ignoring any precomputed analysis. Missing or mismatched records fail.
`confirmation_passed` covers the frozen quantitative gates, not a release or
portfolio-quality judgment. Archive checks do not rerun weights or re-tokenize.

Published figures use Matplotlib 3.11.1, fontTools 4.63.0 and
[Inter](https://github.com/google/fonts/blob/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf)
under the [SIL Open Font License](https://github.com/google/fonts/blob/main/ofl/inter/OFL.txt).
Default typography is DejaVu Sans. Add `--font-file Inter-variable.ttf` for Inter;
its SHA256 is `29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031`
(weights 400/500/600, optical size 18). No font is downloaded by the command.
[Figure data](figures/v0.3-feedback/figure-data.json) records raw evidence and
SVG hashes. The packaged command reproduced all four SVGs byte for byte.
On a restricted host, set `MPLCONFIGDIR` to a writable scratch directory.

## Start with validation

```sh
python -m benchmarks.feedback_protocol --model "MODEL_DIR" --split validation --out runs/feedback-validation-protocol.json
python -m benchmarks.feedback_bench --model "MODEL_DIR" --protocol runs/feedback-validation-protocol.json --out runs/feedback-validation
python -m benchmarks.feedback_analysis --run runs/feedback-validation --out runs/feedback-validation-analysis.json
```

The pilot freezes 16 development cases, seven methods, one warm-up and two timed
repeats in one fresh process, with a **300-second cap**. The recorded public run
took **58.2 seconds**; all **352 outputs** matched. A pilot has no three-session
interval and cannot pass confirmation. Inspect outputs, latency and memory
before expanding. Keep logs and partial records if a run fails.

## Freeze and run confirmation

```sh
python -m benchmarks.feedback_protocol --model "MODEL_DIR" --split test --out runs/feedback-test-protocol.json
python -m benchmarks.feedback_bench --model "MODEL_DIR" --protocol runs/feedback-test-protocol.json --out runs/feedback-test
python -m benchmarks.feedback_analysis --run runs/feedback-test --out runs/feedback-test-analysis.json
python -m benchmarks.feedback_figures --run runs/feedback-test --out runs/feedback-test-figures
```

All **48 fresh cases** run in **three fresh processes**, with one warm-up and
three timed repeats for each method and case. The **1,800-second cap** includes
setup and full-output checks. The recorded run took **573.8 seconds**:
144 DART checks, 1,008 warm-ups and 3,024 timed outputs all matched complete
original-norm HF greedy references. Five unique cases reached EOS. Reference
outputs also match across processes. No case or timing was removed.

Measured code: `3d651482538e15d427fd8533340e98a53b7eec8b`.
Chart tooling: `be81f207f89e81c3ce5c1a52e510a3de06e1cb2a`.
The [protocol](../benchmarks/results/v0.3-feedback/protocol.json) fingerprints
source, weights, tokenizer and core packages. To execute that exact archived
protocol, use its measured code revision in a separate checkout. A newer
checkout must freeze a new protocol; the added chart module changes the source
inventory. Do not edit hashes to make an older protocol pass. Archived analysis
and plotting accept the recorded source fingerprints.

## Inputs and controls

The new final corpus has **12 cases per workload**, with 16/32/64-token budgets
balanced within each workload. All are timed. Prompts contain 39–696 tokens.
Its hash is `e1c8af374753b723a00dc9bd4546394f83daa08e9faac8d30350cf15c1d77f6c`.
The 36-case subset excludes the synthetic repetition workload; it does not
classify generated text as non-repetitive.

[Source revisions and attribution](../benchmarks/fixtures/feedback-sources.json)
cover MBPP by Jacob Austin and colleagues, Google Research (CC-BY-4.0), Dolly by
Databricks and contributors (CC-BY-SA-3.0), and procedural families. Dolly
adaptations retain that license. Input-only source/family and lexical checks
exclude both older corpora. This does not prove semantic independence or absence
from pretraining. `benchmarks.feedback_workloads` exports the fixture; supplying
the pinned MBPP and Dolly source files rebuilds selection. Policy selection
used development inputs before any new final outputs.

| Method | What changes |
|---|---|
| `entropy_on4` | Fixed4 DART, unused entropy explicitly enabled |
| `entropy_off4` | Selected fixed4 DART, unused entropy skipped |
| `entropy_off2` | Fixed2 policy control, same optimization |
| `time_scheduler` | Existing measured-time scheduler; disabled entropy gate gets the same optimization |
| `replay_greedy` | Direct greedy loop |
| `replay_hf_greedy` | HF greedy generation loop |
| `replay_hf_lookup` | HF prompt lookup, draft length four and maximum n-gram four |

All seven paths share one replay executor, weights, FP32, eager attention,
native RMSNorm and one CPU thread. The entropy control retains the previous
arithmetic in the same engine; it is not a separately executed old checkout.
Unknown custom drafters/schedulers keep entropy by default. Fixed2 and
TimeScheduler are DART policy controls, not independent implementations.
HF retains its own generation loop and last-logit prefill. Context admission
reserves three HF lookahead positions; prompts ending in EOS are rejected.

Timing covers token IDs through generation and text decoding, synchronized
around each request. Loading, tokenization, checking, warm-up and writes are
outside the timer. Method order rotates by process, case and repeat. Shared
replay preparation took **15.98 / 11.24 / 0.92 seconds**, outside warm timings;
the processes reuse the host and kernel caches. This is not a per-method
cold-start comparison. Peak allocation was **632,025,600 bytes** with two model
copies, one replay workspace and request caches, not total VRAM or standalone
method memory.

## Mechanism and uncertainty

The initial CPU profile found entropy softmax/log/reduction and a host scalar
conversion on the fixed-drafting path, even without a consumer. Instrumentation
also showed model-storage validation and waits; no GPU kernel attribution was
possible. Skipping unused entropy removes this work while keeping the same
drafting and verification decisions. The capability is checked per request;
active entropy consumers retain the signal.

$$
T_{request}=T_{prefill}+\sum_r(T_{draft,r}+T_{copy,r}+T_{verify,r}+T_{feedback,r}+T_{commit,r})
$$

The matched on/off ablation isolates **7.6% lower mean latency**, with geometric
ratio **1.081 [1.040, 1.123]** and improvements in all three processes. On all
48 cases, fixed4 takes 74.2 ms, fixed2 79.3 ms and TimeScheduler 84.2 ms.
The [development record](../benchmarks/results/v0.3-feedback/development.json)
retains the two-process pilot; the public validation run then applied the same
optimization to TimeScheduler before freezing confirmation.

The frozen targets required ≥5% mean reduction from entropy elision and ≥5%
against direct greedy outside repetition, improvement in every process, both
paired intervals above parity, lower subset means than both HF controls, no
non-repetitive workload regression >5%, complete outputs, ≤6 GiB allocation
and zero added learned parameters. All passed. These are project decision rules,
not employer standards or a universal speedup guarantee.

Latency reductions use equally weighted case arithmetic means. For each
process $s$, let $r_s=\log(\bar T_{control,s}/\bar T_{DART,s})$:

$$
S_g=e^{\bar r},\qquad
\mathrm{CI}_{95}=\exp\left(\bar r\pm4.30265\frac{s_r}{\sqrt3}\right).
$$

This Student-t interval has two degrees of freedom and assumes roughly normal
session log ratios. The uncertainty units are three processes, not 3,024
independent requests. The subset ratio is **1.293 [1.276, 1.309]**, and its mean
latency reduction is **22.7%**. Subgroup intervals have no multiplicity correction.
Bar whiskers show the range of process means, not confidence intervals.

## Losses and limits

Instruction cases **06 and 09** lose to direct greedy by **3.7% and 11.8%**.
Both have a 16-token budget; case 06 has a 677-token prompt, case 09 has 40.
Their case intervals cross parity. Every loss, natural EOS and timing remains
in the published records. Fixed2 and TimeScheduler also beat fixed4 on some
individual cases; the frozen policy was not retuned on these outcomes.

This is one 51.8M model on a shared laptop, batch one and short outputs.
Absolute times decreased across processes. Paired ratios improve in every
process, but the narrow subset interval is same-host repeatability on fixed
inputs, not population uncertainty or independent performance reproduction.
`CUPTI_ERROR_INVALID_DEVICE` prevented a GPU kernel timeline. Compiled-model
and serving-engine comparisons were not measured. Native normalization and
graph execution may change rounding; observed output agreement is not a proof
for every input. No answer-quality or larger-model claim follows.

Run `pytest -ra` and `ruff check .` for project checks. Set `DART_TEST_MODEL` to
the local model for optional real-model tests. Regressions cover full IDs,
length/EOS, context budgets, rejection rollback, graph-input lifetime, stale
logits, interleaved requests, scheduler consumers and incomplete evidence.

## Earlier execution study

[Raw execution records](../benchmarks/results/v0.3-execution) and the
[original guide](https://github.com/tugrapaydiner/dart/blob/2c9a6a8ad32172b3d92a557af575c4c2a71cac79/docs/reproduce.md)
preserve the separate static/replay ablation, architecture, development failures
and fresh-dependency check. The earlier 12 timed cases showed 12.8% less latency
than direct replay greedy, but structured requests lost 9.7%. An exploratory
check assigned 89.7% of saved time to repetition; the other nine cases improved
1.8%, with interval 1.018 [0.963, 1.077]. That motivated the fresh study.

The earlier CUDA executor introduced fixed per-block buffers, request-owned KV
snapshots, valid-prefix masking, rollback and retained graph inputs. Its
static/replay ablation isolates execution replay. The new entropy ablation
isolates only feedback removal. **Different corpora and sessions cannot establish
a direct version-speed comparison.** The old result and thresholds are unchanged.

The existing `benchmarks.execution_analysis` and `benchmarks.execution_figures`
commands still inspect that archive. [v0.2](https://github.com/tugrapaydiner/dart/tree/v0.2)
and the [normalization study](https://github.com/tugrapaydiner/dart/blob/63411869e4a8ed832c0cc4c1ecb560801625d84e/README.md)
also remain separate experiments.
