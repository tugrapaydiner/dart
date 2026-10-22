"""Repeated latency comparisons against HF greedy and DART target-only.

Correctness checks, loading, tokenization, and warm-up are outside timing.
Each measured run must match the complete HF token sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.exact_match import run_exact_match
from benchmarks.headroom import BASELINES, measure_headroom
from benchmarks.profiling import profile_methods
from benchmarks.timing import measure_methods, summarize_samples
from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.ar_draft import ARDrafter
from dart.drafters.base import NullDrafter
from dart.drafters.block_sparse_kv import BlockSparseKVDrafter
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.drafters.mtp import MTPDrafter, MTPHeads
from dart.drafters.ngram import NGramDrafter
from dart.drafters.sparse_attention import SparseAttentionDrafter
from dart.kv.cache import QuantSpec
from dart.kv.compressors.token_drop import TokenDropSpec
from dart.kv.compressors.turboquant_like import PolarVQStore, polar_vq_spec
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_tiny_target
from dart.runtime.types import DecodeConfig, GenerationOutput


def synthetic_prompts(n: int, vocab: int, seed: int = 17) -> list[list[int]]:
    """Token-id prompts with repeated spans (gives ngram a fair shot,
    mimicking the span-reuse structure of code/RAG workloads)."""
    out = []
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        base_len = int(torch.randint(8, 20, (1,), generator=g).item())
        span = torch.randint(5, vocab, (base_len,), generator=g).tolist()
        filler = torch.randint(5, vocab, (4,), generator=g).tolist()
        out.append(span + filler + span)  # repetition in-context
    return out


def build_rows(
    target, drafter_seed: int = 5, tiny_drafters: bool = True,
    selected: set[str] | None = None,
):
    vocab = target.model.config.vocab_size
    hidden = target.model.config.hidden_size
    checkpoints = Path(__file__).resolve().parents[1] / "checkpoints"

    def diffusion(mode, trained=False):
        checkpoint = checkpoints / "diffusion_drafter_tiny.pt"
        if trained and not checkpoint.exists():
            return None
        torch.manual_seed(drafter_seed)  # Match weights across selected policies.
        model = DiffusionDrafterModel(DiffusionDrafterConfig(
            vocab_size=vocab, hidden_size=128, num_layers=2, num_heads=4,
            ctx_window=32, max_canvas=16 if trained else 32,
        ))
        if trained:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not _state_dict_fits(model, state):
                return None
            model.load_state_dict(state)
        return DiffusionDrafter(
            model, canvas_len=8, denoise_steps=4,
            reconcile=ReconcilePolicy(mode=mode, repair_steps=2), device=target.device,
        )

    def mtp(trained=False):
        checkpoint = checkpoints / "mtp_heads_tiny.pt"
        if trained and not checkpoint.exists():
            return None
        torch.manual_seed(drafter_seed)
        heads = MTPHeads(hidden, vocab, num_offsets=3)
        if trained:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not _state_dict_fits(heads, state):
                return None
            heads.load_state_dict(state)
        return MTPDrafter(target, heads)

    factories = {
        "target_only": NullDrafter,
        "ngram": lambda: NGramDrafter(max_ngram=4),
        "compressed_kv_int8": lambda: CompressedKVDrafter(
            target, QuantSpec(bits=8, sink_len=8, window_len=16)
        ),
        "compressed_kv_int4": lambda: CompressedKVDrafter(
            target, QuantSpec(bits=4, sink_len=8, window_len=16)
        ),
        "sparse_kv": lambda: SparseAttentionDrafter(
            target, TokenDropSpec(sink_len=8, window_len=48)
        ),
        "block_sparse_kv": lambda: BlockSparseKVDrafter(
            target, block_size=8, top_k=2, sink_blocks=1, recent_blocks=2
        ),
        "diffusion_discard": lambda: diffusion("discard"),
        "diffusion_reconcile": lambda: diffusion("reconcile"),
        "diffusion_trained_discard": lambda: diffusion("discard", trained=True),
        "diffusion_trained_reconcile": lambda: diffusion("reconcile", trained=True),
        "compressed_kv_vq1bit": lambda: CompressedKVDrafter(
            target, polar_vq_spec(sink_len=8, window_len=16), store_factory=PolarVQStore,
        ),
    }
    if tiny_drafters:
        factories["ar_draft"] = lambda: ARDrafter(build_tiny_target(seed=4321, spec={
            "device": target.device,
            "precision": {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[target.dtype],
            "attn_implementation": target.model.config._attn_implementation,
        }))
    if target.record_hidden:
        factories["mtp_untrained"] = mtp
        factories["mtp_trained"] = lambda: mtp(trained=True)
    requested = set(factories) if selected is None else set(selected) | {"target_only"}
    unknown = requested - factories.keys()
    if unknown:
        raise ValueError(f"unknown lanes {sorted(unknown)}; have {list(factories)}")
    rows = {}
    for name, build in factories.items():
        if name not in requested:
            continue
        drafter = build()
        if drafter is None:
            if selected is not None:
                raise ValueError(f"{name}: checkpoint is missing or incompatible")
            continue
        rows[name] = drafter
    return rows


def _state_dict_fits(module: torch.nn.Module, state: dict) -> bool:
    own = module.state_dict()
    return set(own) == set(state) and all(
        own[k].shape == state[k].shape for k in own
    )


def build_hf_method(target, max_new_tokens: int):
    eos = sorted(target.eos_token_ids)
    pad = getattr(target.model.config, "pad_token_id", None)
    settings = GenerationConfig().to_dict()
    defaults = getattr(GenerationConfig, "_get_default_generation_params", lambda: {})()
    settings.update(defaults)
    settings = {key: value for key, value in settings.items() if not key.startswith("_") and key not in ("transformers_version", "max_length")}
    settings.update(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=eos or None, pad_token_id=pad if pad is not None else (eos[0] if eos else 0),
        output_hidden_states=target.record_hidden,
    )

    @torch.inference_mode()
    def generate(prompt):
        target.validate_generation(len(prompt), max_new_tokens)
        ids = torch.tensor([prompt], device=target.device)
        output = target.model.generate(ids, attention_mask=torch.ones_like(ids), **settings)  # Every prompt token is real input.
        token_ids = output[0, len(prompt):].tolist()
        text = target.tokenizer.decode(token_ids, skip_special_tokens=False) if target.tokenizer else None
        return GenerationOutput(token_ids, text, {})

    return generate


def build_target_method(target, max_new_tokens: int):
    def generate(prompt):
        token_ids = target.greedy_generate(prompt, max_new_tokens)
        text = target.tokenizer.decode(token_ids, skip_special_tokens=False) if target.tokenizer else None
        return GenerationOutput(token_ids, text, {})

    return generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tiny",
                    help='"tiny" (seeded, no downloads) or "hf:<name>", e.g. hf:gpt2')
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--attention", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--profile", action="store_true", help="run separate synchronized phase diagnostics")
    ap.add_argument("--headroom", action="store_true", help="run an offline perfect-draft diagnostic")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompts-file",
                    help="JSONL of {'prompt': ...} text prompts (hf targets only)")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--draft-len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="runs/benchmarks")
    ap.add_argument("--tag", default=None,
                    help="artifact basename (default: bench_<timestamp>)")
    ap.add_argument("--lanes", default="target_only,ngram",
                    help="comma-separated subset of lanes to run")
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be at least 1")
    if args.warmup < 1 or args.repeats < 2:
        ap.error("use --warmup >= 1 and --repeats >= 2")
    if args.prompts < 1 or args.max_new < 1 or args.draft_len < 0:
        ap.error("use --prompts >= 1, --max-new >= 1, and --draft-len >= 0")
    if args.headroom and args.draft_len < 1:
        ap.error("--headroom requires --draft-len >= 1")
    torch.set_num_threads(args.threads)
    selected = {name.strip() for name in args.lanes.split(",") if name.strip()}
    record_hidden = bool(selected & {"mtp_untrained", "mtp_trained"})

    if args.target == "tiny":
        target = build_tiny_target(seed=args.seed, spec={
            "device": args.device, "record_hidden": record_hidden,
            "precision": {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}[args.dtype],
            "attn_implementation": args.attention,
        })
        target_name = "tiny-seeded"
    elif args.target.startswith("hf:"):
        from dart.models.target import TargetModel

        target_name = args.target[3:]
        target = TargetModel.from_pretrained(
            target_name, device=args.device, dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention, record_hidden=record_hidden,
        )
    else:
        raise SystemExit(f"unknown --target {args.target!r}; use tiny or hf:<name>")

    vocab = target.model.config.vocab_size
    if args.prompts_file:
        if target.tokenizer is None:
            raise SystemExit("--prompts-file requires a target with a tokenizer")
        texts = [
            json.loads(ln)["prompt"]
            for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ][: args.prompts]
        prompts = [target.tokenizer.encode(t) for t in texts]
    else:
        prompts = synthetic_prompts(args.prompts, vocab, seed=args.seed)

    hardware = (
        torch.cuda.get_device_name(0)
        if args.device == "cuda" and torch.cuda.is_available()
        else "CPU"
    )
    print(f"target={target_name}  device={args.device} ({hardware})  "
          f"dtype={target.dtype}  prompts={len(prompts)}  max_new={args.max_new}\n")

    try:
        rows = build_rows(target, tiny_drafters=(args.target == "tiny"), selected=selected)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    results = {}
    cfg = DecodeConfig(max_new_tokens=args.max_new, draft_len=args.draft_len)
    with ExitStack() as stack:
        engines = {
            name: stack.enter_context(DartEngine(target, drafter=drafter))
            for name, drafter in rows.items()
        }
        for name, engine in engines.items():
            report = run_exact_match(target, engine, prompts, args.max_new, cfg)
            if report.exact_rate != 1.0:
                raise ValueError(f"{name}: greedy exactness failed")
            row = report.summary()
            for key in ("baseline_ms_total", "dart_ms_total", "baseline_tok_per_s", "dart_tok_per_s", "speedup_wall"):
                row.pop(key, None)
            drafter = rows[name]
            if hasattr(drafter, "stats"):
                row["drafter_stats"] = dict(drafter.stats)
                forwards = row["drafter_stats"].get("diffusion_forwards", 0)
                if forwards:
                    row["accepted_tokens_per_diffusion_forward"] = report.draft_tokens_accepted / forwards
            if hasattr(drafter, "kv_stats"):
                row["kv_stats"] = drafter.kv_stats()
            results[name] = row
        methods = {"hf_greedy": build_hf_method(target, args.max_new), "target_greedy": build_target_method(target, args.max_new)}
        methods.update({name: (lambda prompt, e=engine: e.generate(prompt, cfg)) for name, engine in engines.items()})
        timings = measure_methods(methods, prompts, target.device, warmup=args.warmup, repeats=args.repeats, seed=args.seed)
        profile = profile_methods(methods, prompts, target, engines) if args.profile else None
        headroom = measure_headroom(
            {name: methods[name] for name in BASELINES}, prompts, target, cfg,
            warmup=args.warmup, repeats=args.repeats, seed=args.seed,
        ) if args.headroom else None

    hf = timings["summary"]["hf_greedy"]
    fastest = min(BASELINES, key=lambda name: timings["summary"][name]["mean_ms"])
    vs_fastest = summarize_samples(timings["samples"], fastest, args.seed)
    vs_target = summarize_samples(timings["samples"], "target_only", args.seed)
    hf_total = sum(sample["latency_ms"] for sample in timings["samples"] if sample["method"] == "hf_greedy")
    for name, row in results.items():
        measured = timings["summary"][name]
        total = sum(sample["latency_ms"] for sample in timings["samples"] if sample["method"] == name)
        row.update(measured)
        row.update({
            "baseline_ms_total": hf_total, "dart_ms_total": total,
            "baseline_tok_per_s": hf["tokens_per_second"], "dart_tok_per_s": measured["tokens_per_second"],
            "speedup_wall": measured["speedup"], "bench_wall_s": total / 1_000,
            "speedup_vs_target_only": vs_target[name]["speedup"],
            "ci95_speedup_vs_target_only": vs_target[name]["ci95_speedup"],
            "speedup_vs_fastest_baseline": vs_fastest[name]["speedup"],
            "ci95_speedup_vs_fastest_baseline": vs_fastest[name]["ci95_speedup"],
        })
    for name, measured in timings["summary"].items():
        low, high = measured["ci95_ms"]
        print(f"[{name:24s}] mean={measured['mean_ms']:.2f} ms  "
              f"95% CI=[{low:.2f}, {high:.2f}]  vs HF={measured['speedup']:.3f}x")

    run_info = {
        "target": target_name,
        "device": args.device,
        "hardware": hardware,
        "dtype": str(target.dtype),
        "attention": target.model.config._attn_implementation,
        "record_hidden": target.record_hidden,
        "threads": torch.get_num_threads(),
        "prompts": len(prompts),
        "prompts_file": args.prompts_file,
        "max_new": args.max_new,
        "draft_len": args.draft_len,
        "torch": torch.__version__,
        "reference": "hf_greedy", "seed": args.seed,
        "fastest_baseline": fastest,
        "warmup": args.warmup, "repeats": args.repeats,
        "timing_scope": "token ids to token ids and decoded text; loading, tokenization, correctness, and warm-up excluded",
        "acceptance_scope": "untimed correctness pass",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.tag or f"bench_{time.strftime('%Y%m%d_%H%M%S')}"
    out_json = out_dir / f"{base}.json"
    out_json.write_text(
        json.dumps({"run": run_info, "lanes": results, "timing": timings, "profile": profile, "headroom": headroom}, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        f"# DART benchmark: {target_name}", "",
        f"{args.device} ({hardware}), {target.dtype}, {args.attention}, {args.threads} threads.", "",
        f"{len(prompts)} prompts, {args.warmup} warm-ups, {args.repeats} measured repeats.", "",
        "Latency covers generation and output decoding. Loading and prompt tokenization are excluded.",
        "Complete token sequences match HF greedy on every measured run.",
        "Intervals use paired bootstrap resampling of repeat means for these fixed prompts.", "",
        "| Method | Mean ms | 95% CI ms | Tokens/s | Speedup vs HF |",
        "|---|---:|---|---:|---:|",
    ]
    for name, measured in timings["summary"].items():
        low, high = measured["ci95_ms"]
        lines.append(
            f"| {name} | {measured['mean_ms']:.2f} | [{low:.2f}, {high:.2f}] "
            f"| {measured['tokens_per_second']:.1f} | {measured['speedup']:.3f}x |"
        )
    (out_dir / f"{base}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if headroom is not None:
        budget = headroom["budget"]
        print(f"Offline perfect draft: {budget['perfect_speedup']:.3f}x vs {budget['fastest_baseline']}; "
              f"additional draft budget={budget['extra_draft_ms_per_request']:.2f} ms/request")
    print(f"\nwrote {out_json}")
    print("all measured outputs match HF greedy")


if __name__ == "__main__":
    main()
