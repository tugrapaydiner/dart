"""DART benchmark runner: the docs/design.md §13.3 comparison table.

Rows (lanes) are built against one shared target so numbers are
comparable; every row also reruns the exact-match check, because a
benchmark row with broken exactness is worthless (docs/design.md §13.4:
"100% exact greedy match" is the minimum bar, not a stretch goal).

Works out of the box on the tiny seeded target (no downloads, CPU-only);
point --target at "hf:<name>" on a real box for meaningful wall-clock.
Wall-clock claims are only meaningful from GPU runs with hardware stated
(docs/benchmark_methodology.md); the markdown artifact records both.

    python benchmarks/run_bench.py --prompts 8 --max-new 48
    python benchmarks/run_bench.py --target hf:gpt2 --device cuda \
        --prompts-file benchmarks/prompts/code.jsonl --max-new 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.exact_match import run_exact_match
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
from dart.runtime.types import DecodeConfig


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


def build_rows(target, drafter_seed: int = 5, tiny_drafters: bool = True):
    vocab = target.model.config.vocab_size
    hidden = target.model.config.hidden_size

    torch.manual_seed(drafter_seed)
    diff_cfg = DiffusionDrafterConfig(
        vocab_size=vocab, hidden_size=128, num_layers=2, num_heads=4,
        ctx_window=32, max_canvas=32,
    )
    diff_model_a = DiffusionDrafterModel(diff_cfg)
    diff_model_b = DiffusionDrafterModel(diff_cfg)
    diff_model_b.load_state_dict(diff_model_a.state_dict())  # identical weights

    rows = {
        "target_only": NullDrafter(),
        "ngram": NGramDrafter(max_ngram=4),
    }
    if tiny_drafters:
        # A different-weight tiny model as a cheap AR drafter only makes
        # sense when the target *is* the tiny model family.
        draft_model = build_tiny_target(seed=4321)
        rows["ar_draft"] = ARDrafter(draft_model)
    rows.update({
        "compressed_kv_int8": CompressedKVDrafter(
            target, QuantSpec(bits=8, sink_len=8, window_len=16)
        ),
        "compressed_kv_int4": CompressedKVDrafter(
            target, QuantSpec(bits=4, sink_len=8, window_len=16)
        ),
        "sparse_kv": SparseAttentionDrafter(
            target, TokenDropSpec(sink_len=8, window_len=48)
        ),
        "block_sparse_kv": BlockSparseKVDrafter(
            target, block_size=8, top_k=2, sink_blocks=1, recent_blocks=2
        ),
        "diffusion_discard": DiffusionDrafter(
            diff_model_a, canvas_len=8, denoise_steps=4,
            reconcile=ReconcilePolicy(mode="discard"),
        ),
        "diffusion_reconcile": DiffusionDrafter(
            diff_model_b, canvas_len=8, denoise_steps=4,
            reconcile=ReconcilePolicy(mode="reconcile", repair_steps=2),
        ),
        "compressed_kv_vq1bit": CompressedKVDrafter(
            target,
            polar_vq_spec(sink_len=8, window_len=16),
            store_factory=PolarVQStore,
        ),
    })
    if target.record_hidden:
        rows["mtp_untrained"] = MTPDrafter(
            target, MTPHeads(hidden, vocab, num_offsets=3)
        )
        mtp_ckpt = Path("checkpoints/mtp_heads_tiny.pt")
        if mtp_ckpt.exists():
            heads = MTPHeads(hidden, vocab, num_offsets=3)
            state = torch.load(mtp_ckpt, map_location="cpu", weights_only=True)
            if _state_dict_fits(heads, state):  # tiny-shaped ckpt, tiny target only
                heads.load_state_dict(state)
                rows["mtp_trained"] = MTPDrafter(target, heads)
    diff_ckpt = Path("checkpoints/diffusion_drafter_tiny.pt")
    if diff_ckpt.exists():
        tcfg = DiffusionDrafterConfig(
            vocab_size=vocab, hidden_size=128, num_layers=2, num_heads=4,
            ctx_window=32, max_canvas=16,
        )
        for mode, steps in (("discard", 4), ("reconcile", 2)):
            m = DiffusionDrafterModel(tcfg)
            state = torch.load(diff_ckpt, map_location="cpu", weights_only=True)
            if not _state_dict_fits(m, state):
                break
            m.load_state_dict(state)
            rows[f"diffusion_trained_{mode}"] = DiffusionDrafter(
                m, canvas_len=8, denoise_steps=4,
                reconcile=ReconcilePolicy(mode=mode, repair_steps=steps),
            )
    return rows


def _state_dict_fits(module: torch.nn.Module, state: dict) -> bool:
    own = module.state_dict()
    return set(own) == set(state) and all(
        own[k].shape == state[k].shape for k in own
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tiny",
                    help='"tiny" (seeded, no downloads) or "hf:<name>", e.g. hf:gpt2')
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default=None,
                    choices=[None, "float32", "bfloat16", "float16"])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompts-file",
                    help="JSONL of {'prompt': ...} text prompts (hf targets only)")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--draft-len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="benchmarks/results")
    ap.add_argument("--tag", default=None,
                    help="artifact basename (default: bench_<timestamp>)")
    ap.add_argument("--lanes", default=None,
                    help="comma-separated subset of lanes to run")
    args = ap.parse_args()

    if args.target == "tiny":
        target = build_tiny_target(seed=args.seed, spec={"record_hidden": True})
        target_name = "tiny-seeded"
    elif args.target.startswith("hf:"):
        from dart.models.target import TargetModel

        target_name = args.target[3:]
        dtype = getattr(torch, args.dtype) if args.dtype else None
        target = TargetModel.from_pretrained(
            target_name, device=args.device, dtype=dtype
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
        prompts = synthetic_prompts(args.prompts, vocab)

    hardware = (
        torch.cuda.get_device_name(0)
        if args.device == "cuda" and torch.cuda.is_available()
        else "CPU"
    )
    print(f"target={target_name}  device={args.device} ({hardware})  "
          f"dtype={target.dtype}  prompts={len(prompts)}  max_new={args.max_new}\n")

    rows = build_rows(target, tiny_drafters=(args.target == "tiny"))
    if args.lanes:
        keep = {s.strip() for s in args.lanes.split(",")}
        unknown = keep - set(rows)
        if unknown:
            raise SystemExit(f"unknown lanes {sorted(unknown)}; have {list(rows)}")
        rows = {n: r for n, r in rows.items() if n in keep or n == "target_only"}

    results = {}
    for name, drafter in rows.items():
        engine = DartEngine(target, drafter=drafter, verifier="block")
        cfg = DecodeConfig(max_new_tokens=args.max_new, draft_len=args.draft_len)
        t0 = time.perf_counter()
        report = run_exact_match(target, engine, prompts, args.max_new, cfg)
        elapsed = time.perf_counter() - t0
        row = report.summary()
        row["bench_wall_s"] = round(elapsed, 2)
        if hasattr(drafter, "stats"):
            row["drafter_stats"] = dict(drafter.stats)
            fw = row["drafter_stats"].get("diffusion_forwards", 0)
            if fw:
                # The reconciliation ablation headline metric (docs/design.md §7.3).
                row["accepted_tokens_per_diffusion_forward"] = round(
                    report.draft_tokens_accepted / fw, 4
                )
        if hasattr(drafter, "kv_stats"):
            row["kv_stats"] = drafter.kv_stats()
        results[name] = row
        apf = row.get("accepted_tokens_per_diffusion_forward")
        apf_str = f" acc/fwd={apf:.3f}" if apf is not None else ""
        spd = row.get("speedup_wall")
        spd_str = f" speedup={spd:.2f}x" if spd else ""
        print(f"[{name:24s}] exact={row['exact_match_rate']:.3f} "
              f"accept={row['mean_acceptance_rate']:.3f} "
              f"run={row['mean_accept_run']:.2f}{apf_str}{spd_str} "
              f"wall={elapsed:.2f}s")

    run_info = {
        "target": target_name,
        "device": args.device,
        "hardware": hardware,
        "dtype": str(target.dtype),
        "prompts": len(prompts),
        "prompts_file": args.prompts_file,
        "max_new": args.max_new,
        "draft_len": args.draft_len,
        "torch": torch.__version__,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.tag or f"bench_{time.strftime('%Y%m%d_%H%M%S')}"
    out_json = out_dir / f"{base}.json"
    out_json.write_text(
        json.dumps({"run": run_info, "lanes": results}, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        f"# DART lane benchmark — {target_name}",
        "",
        f"Device: {args.device} ({hardware}) · dtype {target.dtype} · "
        f"{len(prompts)} prompts × {args.max_new} tokens · draft_len {args.draft_len} · "
        f"torch {torch.__version__}",
        "",
        "`speedup` is wall-clock vs the same-run trusted greedy baseline "
        "(same model, device, prompts). Exact match is a gate: any row "
        "below 1.000 fails the run.",
        "",
        "| lane | exact match | acceptance | tokens/round | tok/s | speedup |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in results.items():
        spd = row.get("speedup_wall")
        tps = row.get("dart_tok_per_s")
        lines.append(
            f"| {name} | {row['exact_match_rate']:.3f} "
            f"| {row['mean_acceptance_rate']:.3f} "
            f"| {row['mean_accept_run']:.2f} "
            f"| {tps if tps is not None else '—'} "
            f"| {f'{spd:.2f}x' if spd else '—'} |"
        )
    (out_dir / f"{base}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_json}")
    bad = {n: r for n, r in results.items() if r["exact_match_rate"] != 1.0}
    if bad:
        print(f"EXACTNESS FAILURES: {list(bad)}", file=sys.stderr)
        sys.exit(1)
    print("all lanes 100% exact greedy match")


if __name__ == "__main__":
    main()
