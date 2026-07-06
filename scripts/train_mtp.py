"""Train MTP heads for acceptance (docs/design.md §6.2 and §16).

Teacher traces supply (hidden state at position i, continuation tokens
i+1..i+k): head j learns to predict offset +2+j from the hidden state
whose +1 prediction is the pending token.  Zero extra target forwards at
decode time - the hidden states are byproducts of verify passes.

    python scripts/train_mtp.py --config configs/train_mtp.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_bench import synthetic_prompts
from dart.drafters.mtp import MTPDrafter, MTPHeads
from dart.models.hidden_trace import collect_trace
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_target
from dart.runtime.types import DecodeConfig


def build_pairs(target, data_cfg: dict, num_offsets: int):
    """Stack (hidden [H], labels [num_offsets]) training pairs."""
    g = torch.Generator().manual_seed(data_cfg.get("seed", 7))
    vocab = target.model.config.vocab_size
    hiddens, labels = [], []
    for _ in range(data_cfg["num_prompts"]):
        plen = int(torch.randint(8, data_cfg["prompt_len"], (1,), generator=g).item())
        prompt = torch.randint(5, vocab, (plen,), generator=g).tolist()
        tr = collect_trace(target, prompt, data_cfg["continuation_len"])
        if tr.hidden is None:
            raise RuntimeError("target must be built with record_hidden: true")
        cont = tr.continuation_ids
        # hidden[i] predicts cont[i] at +1; heads cover +2..+1+num_offsets.
        for i in range(len(cont) - num_offsets - 1):
            hiddens.append(tr.hidden[i].float())
            labels.append([cont[i + 1 + j] for j in range(num_offsets)])
    return torch.stack(hiddens), torch.tensor(labels, dtype=torch.long)


def eval_acceptance(target, heads, eval_cfg: dict, prompts) -> dict:
    drafter = MTPDrafter(target, heads)
    engine = DartEngine(target, drafter=drafter, verifier="block")
    rates, runs = [], []
    for p in prompts:
        out = engine.generate(
            p,
            DecodeConfig(
                max_new_tokens=eval_cfg["max_new_tokens"],
                draft_len=eval_cfg["draft_len"],
            ),
        )
        rates.append(out.metadata["acceptance_rate"])
        runs.append(out.metadata["avg_accept_run"])
    n = max(1, len(prompts))
    return {
        "mean_acceptance": round(sum(rates) / n, 4),
        "mean_accept_run": round(sum(runs) / n, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_mtp.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    target = build_target(cfg["target"])
    num_offsets = cfg["heads"]["num_offsets"]
    heads = MTPHeads(
        hidden_size=target.model.config.hidden_size,
        vocab_size=target.model.config.vocab_size,
        num_offsets=num_offsets,
    )

    print("collecting teacher traces...")
    x, y = build_pairs(target, cfg["data"], num_offsets)
    print(f"  {x.size(0)} training pairs")

    vocab = target.model.config.vocab_size
    eval_prompts_fresh = synthetic_prompts(
        cfg["eval"]["num_prompts"], vocab, seed=23
    )
    before = eval_acceptance(target, heads, cfg["eval"], eval_prompts_fresh)
    print(f"acceptance before training (fresh prompts): {before}")

    t = cfg["training"]
    heads.train().requires_grad_(True)
    opt = torch.optim.AdamW(heads.parameters(), lr=float(t["lr"]))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(t["steps"]):
        idx = torch.randint(0, x.size(0), (t["batch_size"],), generator=g)
        logits = heads(x[idx])  # [B, num_offsets, V]
        loss = F.cross_entropy(logits.flatten(0, 1), y[idx].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == t["steps"] - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y[idx]).float().mean().item()
            print(f"step {step:4d}  ce={loss.item():.4f}  head-top1={acc:.3f}")
    heads.eval().requires_grad_(False)
    print(f"trained {t['steps']} steps in {time.time() - t0:.1f}s")

    after = eval_acceptance(target, heads, cfg["eval"], eval_prompts_fresh)
    print(f"acceptance after training (fresh prompts):  {after}")

    ckpt = Path(cfg["output"]["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), ckpt)
    print(f"saved {ckpt}")
    print(
        "acceptance delta: "
        f"{after['mean_acceptance'] - before['mean_acceptance']:+.4f}"
    )


if __name__ == "__main__":
    main()
