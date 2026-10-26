# Reproduce DART v0.3

[Results](../README.md) · [Protocol](../benchmarks/results/v0.3/protocol.json)

Run from the repository root. Replace `MODEL_DIR` with an existing
**HF-format Supra-50M-Instruct** directory containing weights, config and
tokenizer. GGUF is unsupported. Model loading is local-only; the commands
do not fetch weights. Use new report and figure paths for each run.

## Install

Create a fresh environment with Python 3.13 for the measured dependency set:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell. Install the CUDA 12.8 build used
for the published GPU run, then the project and its test/plot dependencies:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-tested.txt ".[dev,figures]"
python -m pip check
dart --version
```

The constraints describe the measured environment, not every supported Python
combination. Fresh CPU CI separately builds and installs the wheel on Python
3.10/3.12 with current dependencies. CPU checks can use `--device cpu`, but
the v0.3 latency result is GPU-only. Native RMSNorm requires a PyTorch build
providing `torch.nn.functional.rms_norm` and FP32 Llama norm layers.

## Frozen inputs

```sh
python -m benchmarks.workloads --model "MODEL_DIR" --out runs/v03/corpus
```

This loads the checked-in, licensed prompt fixture and the local tokenizer.
It creates **8 pilot / 24 validation / 24 test** cases. The `train` split is
used only for pilots; no model training is needed.

The corpus hash must be
`0d91404ef9cd6b0f4565689ef28ce2b4f2a3f77aa2d8e18f44cb0e3c0960697e`.
The manifest records tokenizer hash, formatting, token counts and split hashes.
Prompt, output and verification lookahead stay within 1,024 positions.

Each validation/test split has four MBPP, eight Dolly and twelve procedural
cases. Selection uses input-only rules, normalized family hashes and lexical
deduplication. It does not establish semantic independence or absence from
Supra pretraining. No test result was used for tuning or case removal.

[Source revisions and licenses](../benchmarks/fixtures/v0.3-sources.json):
MBPP by Jacob Austin and colleagues, Google Research (CC-BY-4.0); Dolly by
Databricks and its contributors (CC-BY-SA-3.0). Dolly prompt adaptations retain
that license. To rebuild selection from the pinned source files, add
`--mbpp SOURCE.json --dolly SOURCE.jsonl`; their normalized hashes are checked.

## Check, then time

Begin with the small production validation run: four cases, one warm-up,
two repeats, six methods. It took about 3.6 minutes on the tested machine.

```sh
dart bench --model "MODEL_DIR" --device cuda --corpus runs/v03/corpus --split validation --seed 303 --per-workload 1 --warmup 1 --repeats 2 --native-rms-norm --norm-control --skip-profile --out runs/v03/smoke.json
```

Inspect full-output checks, timing and memory before expanding. Then run
the frozen correctness set and timing selection without changing the policy:

```sh
dart check --model "MODEL_DIR" --device cuda --corpus runs/v03/corpus --split test --seed 303 --native-rms-norm --norm-control --out runs/v03/check.json
dart bench --model "MODEL_DIR" --device cuda --corpus runs/v03/corpus --split test --seed 303 --per-workload 3 --warmup 1 --repeats 6 --native-rms-norm --norm-control --skip-profile --out runs/v03/benchmark.json
```

Correctness took **4.3 minutes**, timing **35.3 minutes** on this machine.
The evaluation used a 55-minute timing cap. This cap is an experiment limit,
not a CLI timeout; stop an unexpectedly long run and retain its log.

`--norm-control` loads a second target with original RMSNorm. The 24-case
check compares native DART with **original HF greedy**, including complete
token IDs, lengths and EOS. On the 12 timing cases, every method must match
native HF greedy before timing, during warm-up and on every measured request.
The published timing hashes also match the separate original-reference check.

| Method | Implementation |
|---|---|
| `hf_greedy` | Transformers `generate`, greedy |
| `hf_prompt_lookup` | Transformers prompt lookup; four draft tokens, maximum n-gram four |
| `target_greedy` | Direct cached target loop |
| `target_only` | DART with drafting disabled |
| `dart` | Four-token n-gram drafting and block verification, native RMSNorm |
| `dart_model_norm` | Same DART policy with original model RMSNorm |

All methods except the explicit original-norm control receive native RMSNorm.
They share weights, tokenizer, FP32/eager attention, one CPU thread, EOS and
output budgets. Hidden-state recording and scheduling are disabled.

Timing selects the first three sorted IDs per workload: **9 real-source and
3 procedural cases**, with 16/32/64-token budgets. Six methods × six repeats
× twelve cases give **432 samples**. Method order rotates; prompt order uses
seed 303. CUDA synchronizes before and after each measured call.

The timer covers generation from token IDs through output decoding. Loading,
input tokenization, reference checks, warm-up and progress printing are outside
it. `.timing.json` is saved separately; `--skip-profile` keeps diagnostic
instrumentation out of this run.

Intervals use 2,000 paired bootstrap resamples of repeat means, seed 303.
Speedup is the ratio of mean latencies, not the mean of per-case ratios.
Intervals describe these fixed cases in one run. Fastest-control intervals
condition on the observed winner; six repeats do not measure machine-to-machine
or population uncertainty.

## Profile the change

```sh
python -m benchmarks.profile_decode --model "MODEL_DIR" --device cuda --case v03-train-code-00 --max-new 8 --out runs/v03/profile-original.json
python -m benchmarks.profile_decode --model "MODEL_DIR" --device cuda --case v03-train-code-00 --max-new 8 --native-rms-norm --out runs/v03/profile-native.json
```

The bounded profiler uses a pilot input, separate phase timings, a warmed
operator trace and a separate Python profile. All outputs are checked.
Host profiling found repeated RMSNorm work; entropy and list conversion were
small on this input. The intervention replaces 25 known norm modules with
PyTorch RMSNorm, preserving parameter objects, values and epsilon.

The original profile recorded 200 norm calls and 205.70 ms cumulative host
time; the native profile recorded 200 and 51.76 ms. These are **different
instrumented sessions**, not the end-to-end speedup. CUPTI reported
`CUPTI_ERROR_INVALID_DEVICE`, so no GPU kernel timeline was captured. Zero
device-event values mean missing data. Raw traces may contain local paths.

The release ablation is the paired `dart_model_norm` / `dart` comparison:
**1.186× [1.146, 1.232]**, or **15.7% lower mean latency**.
[Profiles](../benchmarks/results/v0.3/profiles.json) and
[validation / failed pilots](../benchmarks/results/v0.3/validation.json)
retain the evidence. No adaptive policy or learned training was added.

## Rebuild the charts

No model is needed for this step. Statistics are recomputed from the complete
raw timing matrix; missing samples and mismatched output hashes are rejected.

```sh
python -m benchmarks.figures --report benchmarks/results/v0.3/benchmark.json --out runs/v03/figures --previews runs/v03/figure-previews
```

This writes four SVGs, optional PNG previews and `figure-data.json` with
numeric values and source/figure hashes. Default typography is DejaVu Sans.
Published figures used Matplotlib 3.11.1 and fontTools 4.63.0. To match their
typography, add `--font-file Inter-variable.ttf` using
[Inter](https://github.com/google/fonts/blob/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf)
under the [SIL Open Font License](https://github.com/google/fonts/blob/main/ofl/inter/OFL.txt).
The file used has SHA-256
`29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031`;
weights 400/500/600, optical size 18. Fonts are not downloaded by the command.
[Published figure data](figures/v0.3/figure-data.json) records the exact inputs.

## Environment and limits

| Item | Measured environment |
|---|---|
| Target | Supra-50M-Instruct; 51,786,240 parameters; no added learned parameters |
| OS / GPU | Windows 11 / RTX 5060 Laptop, 8 GiB |
| Python / PyTorch | 3.13.14 / 2.11.0+cu128 |
| Transformers / NumPy | 5.16.1 / 2.5.2 |
| PyYAML / safetensors | 6.0.3 / 0.8.0 |
| Mode | Batch one, FP32, CUDA eager, one CPU thread |
| Memory | 550,306,816 bytes peak allocator use; two target models resident |
| Correctness inputs | 24 cases, 46–758 prompt tokens; one natural early EOS |

The [protocol](../benchmarks/results/v0.3/protocol.json) records weight, config
and tokenizer hashes. Check these before treating a new run as a reproduction.
Measurements used commit `73ce9efb751051c9a9e22ac4f5be04229b099459`, before
the package version changed from 0.2.0 to 0.3.0; exported metadata retains that
original version. Export removes local model paths and decoded text, preserving
token IDs, hashes and measurements.

Run `pytest -ra` and `ruff check .` for project checks. Model tests require
`DART_TEST_MODEL` pointing to the existing local model. CPU CI skips CUDA,
optional Triton and unavailable local-model cases; local focused CUDA and
actual Supra checks complement it. Fresh CI tested Torch 2.14.1+cpu and
Transformers 5.20.2; that is compatibility evidence, not GPU performance data.

The cache covers committed tokens except the pending last token. Rejection
crops speculative KV entries; accepted EOS emits no bonus. Tests cover
rollback, length/EOS, context admission and request-state reuse. The greedy
invariant assumes compatible numerical decisions: native norm outputs can
differ by floating-point rounding, so the 24 matches are not a proof for every
input. Sampling distribution tests are separate from these greedy results.

The 16-token structured case is **0.878× versus direct greedy**. Three intervals
against HF prompt lookup include 1×. Memory is allocated tensor memory, not
total VRAM or a claimed saving. This is a small decoding study, not an
answer-quality benchmark, larger-model result or independent reproduction.
The [archived v0.2 guide](https://github.com/tugrapaydiner/dart/blob/v0.2/docs/reproduce.md)
covers its separate learned-drafter experiments.
