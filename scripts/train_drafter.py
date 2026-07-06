"""Train the small diffusion drafter for *acceptance*, not fluency.

Pipeline (docs/design.md §8):
  1. collect teacher traces from the target (greedy tokens + logits),
  2. corrupt continuation blocks into noisy canvases,
  3. train the canvas denoiser with denoise CE + target KL + acceptance CE
     + confidence calibration,
  4. evaluate end-to-end engine acceptance before vs after training.

Runs on CPU against the tiny target in a few minutes:

    python scripts/train_drafter.py --config configs/train_diffusion_drafter.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_bench import synthetic_prompts
from dart.diffusion.reconciliation import ReconcilePolicy
from dart.diffusion.training_losses import LossWeights, dart_drafter_loss
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.models.hidden_trace import collect_trace
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_target
from dart.runtime.types import DecodeConfig


def build_dataset(target, data_cfg: dict):
    """Teacher traces (docs/design.md §8.1).

    With a real tokenizer + ``prompt_files`` (the curated code/JSON/math/RAG
    workloads), this trains on the actual domain mixture the doc calls for.
    On the tokenizer-less tiny target it falls back to random-token prompts -
    enough to exercise the acceptance objective on CPU.
    """
    import json

    g = torch.Generator().manual_seed(data_cfg.get("seed", 7))
    vocab = target.model.config.vocab_size

    texts: list[str] = []
    files = data_cfg.get("prompt_files")
    if target.tokenizer is not None and files:
        for f in files:
            for ln in Path(f).read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    texts.append(json.loads(ln)["prompt"])

    traces = []
    for i in range(data_cfg["num_prompts"]):
        if texts:  # domain mixture
            prompt = target.tokenizer.encode(texts[i % len(texts)])
        else:  # random-token fallback (tiny target)
            plen = int(torch.randint(8, data_cfg["prompt_len"], (1,), generator=g).item())
            prompt = torch.randint(5, vocab, (plen,), generator=g).tolist()
        tr = collect_trace(target, prompt, data_cfg["continuation_len"])
        if len(tr.continuation_ids) >= 4:
            traces.append(tr)
    return traces


def sample_batch(traces, batch_size, canvas_len, ctx_window, noise_lo, noise_hi, g):
    ctxs, pads, x0s, masks, valids, tlogits = [], [], [], [], [], []
    vocab = traces[0].logits.size(-1)
    for _ in range(batch_size):
        tr = traces[int(torch.randint(0, len(traces), (1,), generator=g).item())]
        cont = tr.continuation_ids
        b = min(canvas_len, len(cont))
        s_max = len(cont) - b
        s = int(torch.randint(0, s_max + 1, (1,), generator=g).item())

        full = tr.prompt_ids + cont[:s]
        ctx = full[-ctx_window:]
        pad = ctx_window - len(ctx)
        ctxs.append([0] * pad + ctx)
        pads.append([True] * pad + [False] * len(ctx))

        x0 = cont[s : s + b] + [0] * (canvas_len - b)
        row_logits = tr.logits[s : s + b].float()
        if b < canvas_len:  # pad rows; padded slots are always masked
            row_logits = torch.cat(
                [row_logits, torch.zeros(canvas_len - b, vocab)], dim=0
            )
        level = noise_lo + (noise_hi - noise_lo) * torch.rand(1, generator=g).item()
        draw = torch.rand(canvas_len, generator=g)
        masked = draw < level
        masked[b:] = True  # padding slots stay masked in the model input
        valid = torch.zeros(canvas_len, dtype=torch.bool)
        valid[:b] = True  # ...but contribute nothing to any loss term
        # Guarantee at least one supervised slot.
        if not bool(masked[:b].any()):
            masked[int(torch.randint(0, b, (1,), generator=g).item())] = True

        x0s.append(x0)
        masks.append(masked)
        valids.append(valid)
        tlogits.append(row_logits)

    return (
        torch.tensor(ctxs, dtype=torch.long),
        torch.tensor(pads, dtype=torch.bool),
        torch.tensor(x0s, dtype=torch.long),
        torch.stack(masks),
        torch.stack(valids),
        torch.stack(tlogits),
    )


def eval_acceptance(target, model, eval_cfg: dict, seed: int = 23) -> dict:
    drafter = DiffusionDrafter(
        copy.deepcopy(model),
        canvas_len=eval_cfg["canvas_length"],
        denoise_steps=eval_cfg["denoise_steps"],
        reconcile=ReconcilePolicy(mode="reconcile"),
    )
    engine = DartEngine(target, drafter=drafter, verifier="block")
    prompts = synthetic_prompts(
        eval_cfg["num_prompts"], target.model.config.vocab_size, seed=seed
    )
    rates, runs = [], []
    for p in prompts:
        out = engine.generate(
            p,
            DecodeConfig(
                max_new_tokens=eval_cfg["max_new_tokens"],
                canvas_len=eval_cfg["canvas_length"],
            ),
        )
        rates.append(out.metadata["acceptance_rate"])
        runs.append(out.metadata["avg_accept_run"])
    n = len(prompts)
    return {
        "mean_acceptance": sum(rates) / n,
        "mean_accept_run": sum(runs) / n,
        "diffusion_forwards": drafter.stats["diffusion_forwards"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_diffusion_drafter.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    target = build_target(cfg["target"])
    d = cfg["drafter"]
    model = DiffusionDrafterModel(
        DiffusionDrafterConfig(
            vocab_size=target.model.config.vocab_size,
            hidden_size=d["hidden_size"],
            num_layers=d["num_layers"],
            num_heads=d["num_heads"],
            ctx_window=d["ctx_window"],
            max_canvas=d["max_canvas"],
        )
    )

    print("collecting teacher traces...")
    traces = build_dataset(target, cfg["data"])
    print(f"  {len(traces)} traces")

    before = eval_acceptance(target, model, cfg["eval"])
    print(f"acceptance before training: {before}")

    t = cfg["training"]
    w = t.get("weights", {})
    weights = LossWeights(
        denoise=w.get("denoise", 1.0),
        kl=w.get("kl", 0.3),
        accept=w.get("accept", 0.5),
        calibration=w.get("calibration", 0.1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]))
    g = torch.Generator().manual_seed(0)
    # Train canvases must be SHORTER than teacher continuations so the
    # sampled offset s varies - otherwise the model only ever sees
    # prompt-boundary contexts and generalizes nowhere (see
    # docs/failure_cases.md, "offset degeneracy").
    canvas_len = t.get("canvas_length", d["max_canvas"])

    model.train()
    t0 = time.time()
    for step in range(t["steps"]):
        ctx, pad, x0, masked, valid, tlog = sample_batch(
            traces, t["batch_size"], canvas_len, d["ctx_window"],
            t["noise_min"], t["noise_max"], g,
        )
        logits = model(ctx, pad, x0, masked)
        parts = dart_drafter_loss(logits, x0, masked, tlog, weights, valid=valid)
        opt.zero_grad()
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == t["steps"] - 1:
            print(
                f"step {step:4d}  total={parts['total'].item():.4f} "
                f"denoise={parts['denoise'].item():.4f} "
                f"accept={parts['accept'].item():.4f}"
            )
    model.eval()
    print(f"trained {t['steps']} steps in {time.time() - t0:.1f}s")

    after = eval_acceptance(target, model, cfg["eval"])
    print(f"acceptance after training:  {after}")

    ckpt = Path(cfg["output"]["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}")
    print(
        "acceptance delta: "
        f"{after['mean_acceptance'] - before['mean_acceptance']:+.4f}"
    )


if __name__ == "__main__":
    main()
