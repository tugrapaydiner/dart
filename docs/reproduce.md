# Reproduce the execution study

[Results](../README.md) · [Raw evidence](../benchmarks/results/v0.3-execution) ·
[Recorded environment](../benchmarks/results/v0.3-execution/reproduction.json)

Use an existing **HF-format Supra-50M-Instruct** directory as `MODEL_DIR`.
The public commands load local files only; GGUF is unsupported. Run from the
repository root and use new output paths. The execution benchmark requires
CUDA, batch one, FP32 Llama with default RoPE, and Transformers ≥5.16.1.

## Install

The measured environment used Python 3.13.14, Windows 11 and an RTX 5060
Laptop GPU. Create a venv without system packages:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell. Install the measured CUDA build
first, then the recorded runtime and plotting dependencies:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-tested.txt -r benchmarks/results/v0.3-execution/requirements.txt ".[dev]"
python -m pip check
dart --version
```

These dependency pins describe the measured environment, not every supported
Python/platform combination. Fresh CPU CI separately builds and installs the
wheel on Python 3.10/3.12 with current dependencies.

The recorded local reproduction installed fresh distributions from cached
wheels, with system and user packages disabled. It ran the public commands
below from the installed wheel outside the checkout. This verified installation
and four validation cases on the same host, not independent performance.

## Inspect the published result

No model is needed to validate the saved records or regenerate charts:

```sh
python -m benchmarks.execution_analysis --run benchmarks/results/v0.3-execution --out runs/execution-analysis.json
python -m benchmarks.execution_figures --run benchmarks/results/v0.3-execution --out runs/execution-figures --previews runs/execution-previews
```

Analysis rejects missing sessions, incomplete timing matrices, changed settings,
output mismatches and inconsistent process records. It checks archived hashes
and outputs; it does not re-tokenize prompts or rerun archived weights. Its
`evaluation_passed` field covers quantitative gates only. `release_complete`
stays false because a timing analyzer cannot establish a release review.

Default typography is DejaVu Sans. Published SVGs used Matplotlib 3.11.1,
fontTools 4.63.0 and [Inter](https://github.com/google/fonts/blob/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf)
under the [SIL Open Font License](https://github.com/google/fonts/blob/main/ofl/inter/OFL.txt).
Add `--font-file Inter-variable.ttf` to match the typography. The file hash is
`29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031`;
weights 400/500/600, optical size 18. The command does not download fonts.
[Figure data](figures/v0.3-execution/figure-data.json) records every value and hash.
The isolated reproduction produced byte-identical SVGs with this font.

## Run a small validation first

```sh
python -m benchmarks.execution_protocol --model "MODEL_DIR" --split validation --out runs/execution-validation-protocol.json
python -m benchmarks.execution_bench --model "MODEL_DIR" --protocol runs/execution-validation-protocol.json --out runs/execution-validation
python -m benchmarks.execution_analysis --run runs/execution-validation --out runs/execution-validation-analysis.json
```

This freezes four development inputs, six methods, one warm-up and two timed
repeats in one fresh process. The run has a **300-second cap**. The fresh-install
check completed in **145.9 seconds**: four DART checks, 24 warm-ups and 48 timed
outputs all matched original-norm HF greedy. No final verdict or multi-session
interval is computed for a pilot. Inspect correctness, setup and memory before
expanding; keep logs and partial results if a run fails.

## Freeze and run the final protocol

```sh
python -m benchmarks.execution_protocol --model "MODEL_DIR" --split test --out runs/execution-test-protocol.json
python -m benchmarks.execution_bench --model "MODEL_DIR" --protocol runs/execution-test-protocol.json --out runs/execution-test
python -m benchmarks.execution_analysis --run runs/execution-test --out runs/execution-test-analysis.json
python -m benchmarks.execution_figures --run runs/execution-test --out runs/execution-test-figures
```

The runner launches **three fresh processes** under one **3,600-second cap**.
The published run took **1,602.3 seconds**. It checks all 48 final cases in the
first process and the 12 timing cases in each later process: 72 candidate checks,
216 warm-ups and 648 timed outputs. Every generated sequence must match the full
original-norm HF reference, including length and EOS. References must also agree
across processes. Five unique cases ended naturally at EOS.

Measured execution commit: `d7f33ee53c6a83d4b405e454ac0505cf637a3fe6`.
Chart command commit: `4b6b3c819e26c8ed31d6d32f2819ef3c17899b01`.
The [protocol](../benchmarks/results/v0.3-execution/protocol.json) records every
source, model, tokenizer and core-package fingerprint. To rerun that exact
archived protocol, use the measured execution revision in a separate checkout.
For a newer checkout, freeze a new protocol with the commands above. The added
chart module changes the source inventory; do not edit hashes to make an old
protocol pass. Archived analysis and plotting accept recorded source fingerprints.

## Inputs and fair controls

The frozen fixture contains **8 development / 16 validation / 48 final** inputs.
The benchmark uses the first three fixed IDs per workload for timing, with
16/32/64-token budgets. Final prompts contain 39–588 tokens. The full corpus hash is
`0c26d7b8db1ab88ee6075ddc73a13dbbb5b38d90ca070be0698571039ae10003`.

[Attribution and source revisions](../benchmarks/fixtures/verification-sources.json)
cover MBPP by Jacob Austin and colleagues, Google Research (CC-BY-4.0), Dolly by
Databricks and contributors (CC-BY-SA-3.0), and procedural families. Dolly
adaptations retain that license. Selection used input-only source/family and
lexical deduplication against the earlier corpus. This does not establish semantic
independence or absence from pretraining. No final output selected a policy or
removed a case. The optional `benchmarks.verification_workloads` command can export
the fixture or rebuild selection from the pinned MBPP/Dolly source files.

| Recorded method | Execution |
|---|---|
| `current_dart` | Optimized eager DART path retained from `6341186` |
| `static_dart` | Same fixed buffers and KV copies as replay, graph disabled |
| `replay_dart` | DART block verification with CUDA graph replay |
| `replay_greedy` | Direct greedy loop with the same replay executor |
| `replay_hf_greedy` | HF generation loop with the same replay executor |
| `replay_hf_lookup` | HF prompt lookup with the same replay executor |

The eager control is the retained path, not an old checkout executed separately.
All six methods use the same weights, FP32, eager attention, native RMSNorm and
one CPU thread. HF retains its own generation loop and last-logit prefill.
Both lookup implementations use draft length four and maximum n-gram four.
Hidden-state recording and scheduling are disabled. Context admission reserves
three lookahead positions for HF; prompts ending in EOS are rejected.

Timing covers token IDs through generation and text decoding, with CUDA
synchronization around each request. Loading, tokenization, reference checks,
warm-up and file writes are outside the timer. Order rotates within each session.
Graph preparation is separate: 19.81/24.45/21.14 seconds for lengths one through
five. Standalone greedy could prepare only length one, so this is not its
cold-start comparison. Peak allocation is 641,507,328 bytes with two model copies,
static/replay workspaces and request caches; it is not total VRAM or per-method memory.

## Why this change

The bounded diagnostic verified CUDA placement, FP32 and warm baselines. Fixed
verification blocks replayed much faster than eager calls with matching probe
logits. That supported investigating dispatch overhead; it did not attribute
individual kernels or establish an end-to-end gain.

$$
T_{request}=T_{prefill}+\sum_r(T_{draft,r}+T_{copy,r}+T_{verify,r}+T_{commit,r}),
\qquad R\approx\frac{G}{a+1}
$$

Here $G$ is the output length and $a$ the mean accepted draft prefix. EOS and
budget boundaries limit the approximation. Replay reduces repeated dispatch;
copies, drafting and host decisions remain. Long contexts add copy/attention
work, while longer outputs amortize preparation. These predictions preceded
final outputs. Repetition's measured gain is consistent with this model, not
a causal isolation of acceptance from all other costs.

[CUDA graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs) are
an existing PyTorch technique. DART adds fixed per-block execution, request-owned
KV snapshots, valid-prefix masking and rejection rollback. Graph inputs stay
alive; returned logits are cloned. Owner, device, stream and model-storage guards
reject incompatible reuse. Factory setting `execution` selects `eager` (default),
`static`, or `cuda_graph`. Replay supports the tested greedy path; sampling,
tree verification, CPU offload and other dtypes are outside its scope.

The static/replay ablation holds buffers and copying fixed. Static takes 2,458 ms
versus replay's 285 ms, supporting replay as the main measured intervention.
The remaining 12.8% aggregate benefit over replay greedy is much smaller than the
88.1% reduction against eager DART. Neither number establishes a larger-model gain.

[Development evidence](../benchmarks/results/v0.3-execution/development.json)
retains the negative six-case prototype, rejected decision capture, its tensor
lifetime failure, and all fixed-draft calibration results. Draft four won the
16-case calibration by a small point estimate; other workload means lost.
The SDPA/HF StaticCache audit timed out during its second pass, so it supplies
no confirmed ranking. A last-logit prefill option was not selected. A tiny Inductor
probe failed with `TritonMissing`; no compiled-model performance was measured.

## Uncertainty, losses and checks

Latency reductions use arithmetic means over equally weighted fixed cases.
For each process $s$, compute $r_s=\log(\bar T_{control,s}/\bar T_{DART,s})$:

$$
S_g=e^{\bar r},\qquad
\mathrm{CI}_{95}=\exp\left(\bar r\pm4.30265\frac{s_r}{\sqrt3}\right).
$$

This paired Student-t interval has two degrees of freedom and assumes roughly
normal session log ratios. Three processes, not 648 requests, are the uncertainty
units. The direct-greedy geometric ratio is 1.1468 [1.1127, 1.1820]; the arithmetic
ratio is 1.1472. Per-case/workload intervals are descriptive, without multiplicity
correction or population coverage. Bar whiskers show the range of session means,
not confidence intervals.

Four case means lose to replay greedy: instruction-02 and structured-00/01/02.
Structured mean latency is 9.7% worse; the code-workload interval includes parity.
The 32-token structured case produces 13 tokens before EOS and remains included.
All workloads improve against eager DART. The predeclared quantitative thresholds
passed; they are project decision rules, not a universal quality standard.

Run `pytest -ra` and `ruff check .` for project checks. `DART_TEST_MODEL` can point
to the existing local model for optional real-model checks. CI skips CUDA and
unconfigured model cases; focused local CUDA tests and Supra checks complement it.
Regression tests cover full IDs/length/EOS, rejection rollback, empty crop,
context/budget boundaries, interleaved requests, stale logits, allocator reuse,
state reset, unsupported settings and process timeouts.

This was a shared laptop, not a quiet dedicated host. Read-only cache inspection
overlapped the end of the first final session; all timings remain included.
`CUPTI_ERROR_INVALID_DEVICE` prevented a GPU kernel timeline. Native normalization
and graph execution can change floating-point rounding: observed full-output
agreement is not universal numerical equivalence. There is no independent
reproduction, answer-quality, serving-throughput or larger-model result.

[Earlier normalization study and commands](https://github.com/tugrapaydiner/dart/blob/63411869e4a8ed832c0cc4c1ecb560801625d84e/docs/reproduce.md)
and [v0.2](https://github.com/tugrapaydiner/dart/tree/v0.2) remain separate experiments.
Their different inputs and methods are not a direct version-speed comparison.
