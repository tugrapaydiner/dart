# Run the feedback benchmark

[Results](../README.md) · [Data attribution](../benchmarks/fixtures/feedback-sources.json)

Raw historical runs are not distributed. These commands generate new records
under `runs/`; they do not reproduce the historical timings exactly.

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
python -m pip install -c constraints-tested.txt ".[dev,figures]"
python -m pip check
dart --version
```

The pins describe this measured environment, not every platform. CPU CI builds
and installs the wheel with fresh dependencies on Python 3.10/3.12. CUDA and
real-model tests require compatible local hardware and weights.

## Start with validation

```sh
python -m benchmarks.feedback_protocol --model "MODEL_DIR" --split validation --out runs/feedback-validation-protocol.json
python -m benchmarks.feedback_bench --model "MODEL_DIR" --protocol runs/feedback-validation-protocol.json --out runs/feedback-validation
python -m benchmarks.feedback_analysis --run runs/feedback-validation --out runs/feedback-validation-analysis.json
```

The pilot freezes 16 development cases, seven methods, one warm-up and two timed
repeats in one fresh process, with a **300-second cap**. A pilot has no
three-session interval and cannot pass confirmation. Inspect outputs, latency
and memory before expanding. Keep logs and partial records if a run fails.

## Run the full comparison

```sh
python -m benchmarks.feedback_protocol --model "MODEL_DIR" --split test --out runs/feedback-test-protocol.json
python -m benchmarks.feedback_bench --model "MODEL_DIR" --protocol runs/feedback-test-protocol.json --out runs/feedback-test
python -m benchmarks.feedback_analysis --run runs/feedback-test --out runs/feedback-test-analysis.json
python -m benchmarks.feedback_figures --run runs/feedback-test --out runs/feedback-test-figures
```

All **48 cases** run in **three fresh processes**, with one warm-up and
three timed repeats for each method and case. The **1,800-second cap** includes
setup and full-output checks. The recorded run took **573.8 seconds**:
144 DART checks, 1,008 warm-ups and 3,024 timed outputs all matched complete
original-norm HF greedy references. Five unique cases reached EOS. Reference
outputs also match across processes. No case or timing was removed.

These published inputs are already exposed; tuning on them does not establish
a new held-out result. Each protocol fingerprints source, weights, tokenizer
and core packages. Freeze a new protocol when the source changes; do not edit
hashes to bypass checks. Analysis validates complete outputs, hashes, timing
matrices, settings and process records. Figures recompute results from the
sessions. Missing or mismatched records fail. Archive analysis does not rerun
weights or re-tokenize. `confirmation_passed` covers quantitative gates,
not a release-quality judgment.

## Benchmark setup

The final corpus has **12 cases per workload**, with 16/32/64-token budgets
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
Their case intervals cross parity. Fixed2 and TimeScheduler also beat fixed4
on some individual cases; the frozen policy was not retuned on these outcomes.

Raw historical runs are not included, so readers can inspect the summary charts
and run the public benchmark, but cannot independently audit those historical
samples here.

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

## Overhead validation

The later bounded comparison uses 16 existing development cases, six methods
and three completed fresh processes. It is separate from the 48-case confirmation
and README charts.
The earlier results are unchanged; these exposed inputs are not fresh held-out
confirmation or independent reproduction.

The replay guard now visits live modules once instead of recursively collecting
named parameters and buffers separately. It still checks storage, device,
dtype, shape and stride before every forward. N-gram lookup uses native list
search to skip impossible suffix endings, preserving match order and scan bounds.
Neither change adds learned parameters or persistent proposal state.

| Matched path | Mean request latency |
|---|---:|
| Prior guard and lookup | 66.34 ms |
| New guard only | 62.24 ms |
| Both changes | **61.67 ms** |
| Direct greedy | 80.76 ms |
| HF greedy | 90.54 ms |
| HF prompt lookup | 129.90 ms |

The prior control reinstates the old guard and lookup in the same executor;
it does not run a separate old checkout. Applicable baselines receive the
new guard. All use the same weights, FP32, replay workspace and warm request
boundaries. Each case has one warm-up and two timed repeats per method/process.

The combined change saves **7.0%** against the prior control. Its geometric
process ratio is **1.075 [1.011, 1.143]**, using the descriptive interval above.
All three process means and all four workload means improve. The separate
lookup increment averages **0.9%**, with interval **1.009 [0.974, 1.045]**;
an additional full-request benefit from lookup alone is not established.

Three case means lose to the prior control: **code 03 by 0.6%, repetition 03
by 11.8%, and structured 03 by 5.5%**. Against direct greedy, code 03,
instruction 01 and repetition 01 lose by 0.2%, 6.4% and 0.5%. These case
intervals cross parity. The lookup-only losses are included in the means.

All **912 complete outputs** match cached original-norm HF references, including
IDs, lengths and EOS; one unique case reaches EOS. Cached references are accepted
only with matching model/tokenizer/config hashes and core package versions.
Their retrieval times are not model inference measurements. The harness keeps
one resident target instead of recomputing a second reference model. Peak
allocation is **400,130,560 bytes**, including replay workspace and request caches;
this is not comparable with the earlier two-model allocation.

The original **120-second cap timed out** during the third process, after
16 matching checks and before any warm-up or timed samples. The archive retains
that attempt and its timeout. A separate **60-second bounded retry** completed
in **44.1 seconds**, reusing the first two completed processes and kernel caches.
Total measured launcher wall time was **164.2 seconds**, including the failed
attempt and setup, not GPU kernel time. No timed samples were discarded.
The 16 partial-attempt matches are reported separately from the 912 completed-run
matches. `development_passed` is true; `confirmation_passed` remains false.

After generating validation records above, run a new comparison:

```sh
python -m benchmarks.overhead_bench --model "MODEL_DIR" --reference-protocol runs/feedback-validation/protocol.json --reference-session runs/feedback-validation/session-0.json --out runs/overhead
python -m benchmarks.overhead_bench --out runs/overhead --analyze
```

An explicit retry is available only when the first two processes completed and
the third timed out before producing any timed samples. It runs that process
once, for at most 60 seconds, retaining the original files:

```sh
python -m benchmarks.overhead_bench --model "MODEL_DIR" --out runs/overhead --resume
```

New runs fingerprint the current source; do not edit an archived protocol
to bypass source checks. Analysis accepts archived fingerprints.

## Later runtime changes

The charts retain the frozen study. The separate
[overhead validation](#overhead-validation) measures the later 7.0% development
improvement; it does not update or compound the 22.7% frozen-study result.

Two subsequent changes improve the request path:

- Worker cleanup closes and drains request workers on failure or interruption.
  Six regressions reproduced the missing cleanup before the fix.
- Token validation checks short Python integer blocks before allocation,
  avoiding tensor reductions. Tensor inputs and longer prompts retain their path.
  All 170 replay, adapter and input checks passed, including CUDA execution.

Neither change adds model parameters. These maintenance changes are not a new
held-out latency study. A prefix-reuse pilot did not improve latency and was
reverted.

## Figure style

Published figures use Matplotlib 3.11.1, fontTools 4.63.0 and
[Inter](https://github.com/google/fonts/blob/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf)
under the [SIL Open Font License](https://github.com/google/fonts/blob/main/ofl/inter/OFL.txt).
Default typography is DejaVu Sans. Add `--font-file Inter-variable.ttf` for Inter;
no font is downloaded by the command. Figure metadata is written beside your
generated SVGs under `runs/`.

Whiskers in the baseline chart span three process means. Workload pairs and
entropy slopes show matched means; the latter share one vertical scale. Case
dots show every mean reduction, including the two losses. Budget panels share
a zero-based scale and contain different cases, so they do not represent a
controlled output-length sweep. On a restricted host, set `MPLCONFIGDIR` to a
writable scratch directory.
