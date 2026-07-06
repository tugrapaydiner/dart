"""Regenerate ALL result figures from committed artifacts and stated formulas.

Every figure under docs/figures/ is produced by this script - none is
hand-made - so figures cannot drift from the data
(docs/benchmark_methodology.md). Data sources per figure:

  01_cost_vs_fidelity            benchmarks/results/gpu_bench_rtx5060.json
  02_exactness_under_kv_compression  benchmarks/results/exactness_vs_bits.json
                                     (scripts/measure_exactness_vs_bits.py)
  03_kv_cache_memory_at_scale    exact byte accounting (formula in-code,
                                     matches QuantStore.logical_bytes)
  04_acceleration_landscape      PUBLISHED numbers below + measured best lane
                                     from gpu_bench_rtx5060.json
  05_speed_vs_fidelity           PUBLISHED numbers below + measured/projected
  06_technique_coverage          capability matrix from the systems' papers
  07_canvas_reconciliation       benchmarks/results/reconciliation_ablation.json
                                     (scripts/run_reconciliation_ablation.py)
  08_draft_acceptance_by_lane    benchmarks/results/cpu_bench_tiny.json
                                     (benchmarks/run_bench.py --tag cpu_bench_tiny)

Competitor numbers are published peaks from each system's own paper/release
(own hardware, models, tasks - a positioning landscape, not a controlled
head-to-head); citations and last-verified date in docs/figures/SOURCES.md.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py 04 07      # a subset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
FIGDIR = ROOT / "docs" / "figures"

INK = "#222222"
GRAY = "#777777"
GRID = "#E6E3DD"
DART = "#C96A2B"
BLUE = "#5B7B99"
GREEN = "#4E8570"
RED = "#B04A5A"
LIGHT = "#B9B2A6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": "#CCC8C0",
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Published peaks (each system's own paper/release; see SOURCES.md,
# last verified 2026-07-05). (name, peak speedup, fidelity tier, color)
PUBLISHED = [
    ("EAGLE-3", 6.47, "lossless", BLUE),
    ("DFlash", 6.08, "lossless", BLUE),
    ("VeriCache", 4.00, "lossless", GREEN),
    ("DiffusionGemma", 4.00, "lossy", RED),
    ("KIVI (2-bit)", 3.47, "lossy", RED),
]
DART_CEILING = 5.00  # roofline projection, always drawn hatched

LANE_LABELS = {
    "target_only": "target only",
    "ngram": "n-gram",
    "ar_draft": "AR draft model",
    "sparse_kv": "sparse KV (sink+window)",
    "block_sparse_kv": "block-sparse KV",
    "compressed_kv_int8": "compressed KV · int8",
    "compressed_kv_int4": "compressed KV · int4",
    "compressed_kv_vq1bit": "compressed KV · 1-bit VQ",
    "mtp_trained": "MTP heads (trained)",
    "mtp_untrained": "MTP heads (untrained)",
    "diffusion_trained_discard": "diffusion (trained, discard)",
    "diffusion_trained_reconcile": "diffusion (trained, reconcile)",
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _fig(w: float, h: float, *, bottom=0.13, left=0.10, right=0.97, top=0.79):
    """Figure with space reserved at the top for a title + provenance line."""
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    return fig, ax


def _header(fig, ax, title: str, subtitle: str):
    """Title + one short provenance line, left-aligned with the plot area.
    Full provenance (data files, protocols) lives in the README captions
    and docs/figures/SOURCES.md - images stay quiet."""
    x0 = ax.get_position().x0
    fig.text(x0, 0.955, title, ha="left", va="top",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(x0, 0.875, subtitle, ha="left", va="top",
             fontsize=9.5, color=GRAY)


def _save(fig, name: str):
    out = FIGDIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ #
def fig_01_cost_vs_fidelity():
    """Target forwards per committed token, from the GPU bench artifact."""
    data = _load("gpu_bench_rtx5060.json")
    lanes = data["lanes"]
    short = {"target_only": "target only", "ngram": "n-gram",
             "sparse_kv": "sparse KV", "compressed_kv_int4": "int4 KV",
             "compressed_kv_int8": "int8 KV"}
    order = ["target_only", "ngram", "sparse_kv",
             "compressed_kv_int4", "compressed_kv_int8"]
    rows = [(short[n], 1.0 / lanes[n]["mean_accept_run"], False)
            for n in order if n in lanes]
    rows.append(("unverified\nint4 decode", 1.0, True))

    fig, ax = _fig(9.6, 5.4, bottom=0.17)
    for i, (_name, fwd, lossy) in enumerate(rows):
        if lossy:  # disqualified: outlined + hatched, never the visual winner
            ax.bar(i, fwd, width=0.6, facecolor="white", edgecolor=RED,
                   hatch="///", linewidth=1.2)
        else:
            ax.bar(i, fwd, width=0.6, color=LIGHT if i == 0 else DART)
        if fwd > 0.55 and not lossy:  # tall solid bars: label inside
            ax.text(i, fwd - 0.045, f"{fwd:.2f}", ha="center", fontsize=10.5,
                    color="white", fontweight="bold")
        else:
            ax.text(i, fwd + 0.028, f"{fwd:.2f}", ha="center", fontsize=10.5,
                    color=RED if lossy else INK)
        tag, tcolor = ("not exact", RED) if lossy else ("exact", GREEN)
        ax.text(i, -0.155, tag, ha="center", fontsize=8.6, color="white",
                fontweight="bold", transform=ax.get_xaxis_transform(),
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=tcolor,
                          edgecolor="none"))
    ax.axhline(1.0, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(2.5, 1.045, "1 forward per token = vanilla AR decoding",
            fontsize=8.8, color=GRAY, ha="center")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.6)
    ax.set_ylabel("target forwards per committed token")
    ax.set_ylim(0, 1.16)
    ax.yaxis.grid(color=GRID, linewidth=0.9)
    run = data["run"]
    _header(fig, ax, "Verification cost vs fidelity",
            f"Measured tokens per round · {run['target']} fp32, "
            f"{run['hardware'].replace('NVIDIA GeForce ', '')} · single run")
    _save(fig, "01_cost_vs_fidelity.png")


def fig_02_exactness_under_kv_compression():
    data = _load("exactness_vs_bits.json")
    rows = data["rows"]
    bits = [r["bits"] for r in rows]
    unver = [r["unverified_exact_rate"] for r in rows]
    dart = [r["dart_exact_rate"] for r in rows]
    acc = [r["dart_acceptance_rate"] for r in rows]

    fig, ax = _fig(9.6, 5.6, bottom=0.12)
    ax.fill_between(bits, unver, dart, color=RED, alpha=0.06, zorder=1)
    ax.plot(bits, dart, "-o", color=DART, linewidth=2.4, markersize=6.5,
            zorder=4, label="DART exact mode (drafts on quantized KV, verifies on full KV)")
    ax.plot(bits, unver, "-s", color=RED, linewidth=2.4, markersize=6.5,
            zorder=3, label="unverified decoding on the same quantized KV")
    ax.plot(bits, acc, "--", color=GRAY, linewidth=1.5, marker="o",
            markersize=3.5, zorder=2,
            label="DART draft acceptance (a throughput cost, not correctness)")
    ax.annotate("1.00 at every bit-width", (bits[2], 1.0),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=9.5, color=DART, fontweight="bold")
    ax.text(4, 0.62, "exactness lost\nwithout verification", ha="center",
            fontsize=9.5, color=RED, style="italic", alpha=0.85)
    for x, y in ((bits[0], unver[0]), (bits[-1], unver[-1])):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(10, -13), ha="left", fontsize=9, color=RED)
    ax.set_xlabel("KV bit-width for aged tokens (sinks stay at 8-bit; small fp window)")
    ax.set_ylabel("exact-match rate vs full-precision greedy")
    ax.set_xticks(bits)
    ax.invert_xaxis()  # aggressiveness increases to the right
    ax.set_ylim(-0.06, 1.13)
    ax.grid(color=GRID, linewidth=0.9)
    ax.legend(loc="lower left", bbox_to_anchor=(0.015, 0.06), framealpha=0)
    run = data["run"]
    _header(fig, ax, "Exactness under aggressive KV compression",
            f"Seeded tiny target · {run['prompts']} prompts × "
            f"{run['max_new']} tokens · exact-match is hardware-independent")
    _save(fig, "02_exactness_under_kv_compression.png")


def kv_bytes(s: int, *, layers=46, kv_heads=16, head_dim=128,
             hi_bytes=2.0, window=1024, sink=256):
    """Exact byte accounting, matching QuantStore.logical_bytes: quantized
    tokens cost head_dim*bits/8 + 8 bytes (fp32 scale+zero) per (head, token)
    per K and per V; window tokens stay at hi_bytes/element."""
    full = 2 * layers * kv_heads * s * head_dim * hi_bytes
    aged = max(0, s - window - sink)
    per_tok_int4 = head_dim * 4 / 8 + 8
    per_tok_int8 = head_dim * 8 / 8 + 8
    dart = 2 * layers * kv_heads * (
        min(s, window) * head_dim * hi_bytes
        + min(max(s - window, 0), sink) * per_tok_int8
        + aged * per_tok_int4
    )
    return full, dart


def fig_03_kv_cache_memory_at_scale():
    ctxs = [4096, 8192, 16384, 32768, 65536, 131072]
    full, dart = [], []
    for s in ctxs:
        f, d = kv_bytes(s)
        full.append(f / 2**30)
        dart.append(d / 2**30)

    fig, ax = _fig(9.6, 5.4, bottom=0.12)
    x = range(len(ctxs))
    ax.bar([i - 0.21 for i in x], full, width=0.42, color=LIGHT,
           label="full-precision KV (bf16)")
    ax.bar([i + 0.21 for i in x], dart, width=0.42, color=DART,
           label="DART draft cache (int4 + int8 sinks + fp window)")
    for i, (f, d) in enumerate(zip(full, dart)):
        ax.text(i + 0.21, d + 0.5, f"−{1 - d / f:.0%}", ha="center",
                fontsize=9.5, color=DART, fontweight="bold")
    ax.text(len(ctxs) - 1 - 0.21, full[-1] + 0.5, f"{full[-1]:.0f} GiB",
            ha="center", fontsize=9.5, color=GRAY)
    ax.text(len(ctxs) - 1 + 0.21, dart[-1] + 2.1, f"{dart[-1]:.0f} GiB",
            ha="center", fontsize=9.5, color=DART)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c // 1024}K" for c in ctxs])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV cache size (GiB)")
    ax.set_ylim(0, max(full) * 1.14)
    ax.yaxis.grid(color=GRID, linewidth=0.9)
    ax.legend(loc="upper left", framealpha=0)
    _header(fig, ax, "GPU KV-cache memory at scale (27B-class)",
            "Exact byte accounting · 46 layers × 16 KV heads × d=128, bf16 · "
            "fp window + int8 sinks + int4 rest")
    _save(fig, "03_kv_cache_memory_at_scale.png")


def fig_04_acceleration_landscape():
    data = _load("gpu_bench_rtx5060.json")
    lanes = data["lanes"]
    name, row = max(
        ((n, r) for n, r in lanes.items() if n != "target_only"),
        key=lambda kv: kv[1].get("speedup_wall") or 0.0,
    )
    speed = float(row["speedup_wall"])
    hw = data["run"]["hardware"].replace("NVIDIA GeForce ", "")

    rows = sorted(
        [(n, v, c) for n, v, _t, c in PUBLISHED]
        + [("DART · projected ceiling", DART_CEILING, DART),
           ("DART · measured best lane", speed, DART)],
        key=lambda r: r[1],
    )
    fig, ax = _fig(9.6, 5.4, bottom=0.11, left=0.26)
    for i, (nm, val, color) in enumerate(rows):
        hatched = "projected" in nm
        ax.barh(i, val, height=0.6, color="white" if hatched else color,
                edgecolor=color, hatch="////" if hatched else None,
                linewidth=1.2 if hatched else 0)
        ax.text(val + 0.09, i, f"{val:.2f}×", va="center", fontsize=10.5,
                color=INK)
    ax.set_yticks(range(len(rows)))
    labels = ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    for lab, (nm, _, _c) in zip(labels, rows):
        if nm.startswith("DART"):
            lab.set_color(DART)
            lab.set_fontweight("bold")
    ax.set_xlabel("peak speedup vs vanilla AR decoding (×)")
    ax.set_xlim(0, 7.3)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(color=GRID, linewidth=0.9)
    ax.legend(handles=[
        Patch(color=BLUE, label="published, lossless"),
        Patch(color=GREEN, label="published, verified KV (lossless)"),
        Patch(color=RED, label="published, lossy"),
        Patch(facecolor="white", edgecolor=DART, hatch="////",
              label="DART projection (roofline)"),
    ], loc="lower right", framealpha=0)
    _header(fig, ax, "Acceleration landscape",
            f"DART measured: {LANE_LABELS.get(name, name)} lane, gpt2 on {hw}, "
            "single run · others: published peaks on their own setups")
    _save(fig, "04_acceleration_landscape.png")


def fig_05_speed_vs_fidelity():
    data = _load("gpu_bench_rtx5060.json")
    lanes = data["lanes"]
    speed = max(
        (r.get("speedup_wall") or 0.0)
        for n, r in lanes.items() if n != "target_only"
    )
    tiers = {"lossy": 0, "lossless": 1}
    offsets = {
        "EAGLE-3": (12, -3, "left"),
        "DFlash": (-12, -3, "right"),
        "VeriCache": (0, 13, "center"),
        "DiffusionGemma": (12, -3, "left"),
        "KIVI (2-bit)": (-12, -3, "right"),
    }
    fig, ax = _fig(9.6, 4.6, bottom=0.15, left=0.16)
    ax.axhspan(0.55, 1.62, color=GREEN, alpha=0.05, zorder=0)
    ax.axhspan(-0.55, 0.45, color=RED, alpha=0.05, zorder=0)
    for nm, val, tier, color in PUBLISHED:
        ax.scatter(val, tiers[tier], s=150, color=color, zorder=3)
        dx, dy, ha = offsets[nm]
        ax.annotate(nm, (val, tiers[tier]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center_baseline",
                    fontsize=10, color=INK)
    ax.annotate("", xy=(DART_CEILING - 0.06, 0.955), xytext=(speed + 0.08, 0.955),
                arrowprops=dict(arrowstyle="->", color=DART, linewidth=1.2,
                                linestyle=(0, (4, 3)), alpha=0.7,
                                connectionstyle="arc3,rad=0.18"))
    ax.text((speed + DART_CEILING) / 2, 0.60, "draft-side kernels pending (§10)",
            ha="center", fontsize=8.8, color=DART, style="italic", alpha=0.9)
    ax.scatter(speed, 1, s=210, color=DART, zorder=4)
    ax.annotate("DART · measured", (speed, 1), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=10.5, color=DART,
                fontweight="bold")
    ax.scatter(DART_CEILING, 1, s=210, facecolors="white", edgecolors=DART,
               linewidths=1.8, zorder=4)
    ax.annotate("DART · projected", (DART_CEILING, 1),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=10.5, color=DART)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["lossy\n(output changes)",
                        "lossless / exact\n(output preserved)"], fontsize=10.5)
    ax.set_ylim(-0.55, 1.62)
    ax.set_xlim(1.5, 7.2)
    ax.set_xlabel("peak speedup vs vanilla AR decoding (×)")
    ax.grid(color=GRID, linewidth=0.9)
    _header(fig, ax, "Speed vs fidelity",
            "The claim is the joint position, not speed alone · published "
            "peaks, each on its own setup")
    _save(fig, "05_speed_vs_fidelity.png")


def fig_06_technique_coverage():
    techniques = [
        "exact output\nverification", "diffusion\ndrafting",
        "MTP / feature\nheads", "n-gram\nretrieval", "compressed-KV\ndrafting",
        "sparse-attn\ndrafting", "tree\nverification",
        "canvas\nreconciliation", "multi-lane\nscheduler",
    ]
    systems = {  # from each system's paper (SOURCES.md)
        "DART":           [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "DFlash":         [1, 1, 0, 0, 0, 0, 0, 0, 0],
        "EAGLE-3":        [1, 0, 1, 0, 0, 0, 1, 0, 0],
        "VeriCache":      [1, 0, 0, 0, 1, 0, 0, 0, 0],
        "DiffusionGemma": [0, 1, 0, 0, 0, 0, 0, 0, 0],
        "KIVI":           [0, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    fig, ax = _fig(11.6, 5.0, bottom=0.15, left=0.13, top=0.80)
    n_sys = len(systems)
    for yi, (nm, rows) in enumerate(systems.items()):
        y = n_sys - 1 - yi
        if nm == "DART":  # band the DART row for emphasis
            ax.axhspan(y - 0.42, y + 0.42, color="#F7EBDF", zorder=0)
        for xi, v in enumerate(rows):
            if v:
                ax.scatter(xi, y, s=190, marker="o",
                           color=DART if nm == "DART" else BLUE, zorder=3)
            else:
                ax.scatter(xi, y, s=45, marker="o", facecolors="white",
                           edgecolors="#D8D2C8", linewidths=1.1, zorder=2)
    ax.set_xticks(range(len(techniques)))
    ax.set_xticklabels(techniques, fontsize=9.3)
    ax.set_yticks(range(n_sys))
    ylabels = ax.set_yticklabels(list(systems)[::-1], fontsize=10.5)
    for lab in ylabels:
        if lab.get_text() == "DART":
            lab.set_color(DART)
            lab.set_fontweight("bold")
    ax.set_xlim(-0.55, len(techniques) - 0.45)
    ax.set_ylim(-0.6, n_sys - 0.4)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(color=GRID, linewidth=0.7)
    _header(fig, ax, "Technique coverage",
            "Capabilities per each system's own paper - a map, not a quality "
            "benchmark")
    _save(fig, "06_technique_coverage.png")


def fig_07_canvas_reconciliation():
    data = _load("reconciliation_ablation.json")
    per_seed = data["per_seed"]
    seeds = [r["prompt_seed"] for r in per_seed]
    disc = [r["discard"]["accepted_per_forward"] for r in per_seed]
    reco = [r["reconcile"]["accepted_per_forward"] for r in per_seed]
    mean, std = data["mean_improvement_pct"], data["std_improvement_pct"]

    fig, ax = _fig(9.6, 5.4, bottom=0.12)
    x = range(len(seeds))
    ax.bar([i - 0.21 for i in x], disc, width=0.42, color=LIGHT,
           label="discard: fresh canvas + full step budget")
    ax.bar([i + 0.21 for i in x], reco, width=0.42, color=DART,
           label="reconcile: re-noise low-confidence suffix, repair")
    for i, (d, r) in enumerate(zip(disc, reco)):
        ax.text(i + 0.21, r + 0.0035, f"+{(r - d) / d * 100:.0f}%",
                ha="center", fontsize=10, color=DART, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"prompt seed {s}" for s in seeds], fontsize=9.8)
    ax.set_ylabel("accepted tokens per diffusion forward")
    ax.set_ylim(0, max(reco) * 1.18)
    ax.yaxis.grid(color=GRID, linewidth=0.9)
    ax.legend(loc="upper left", framealpha=0)
    _header(fig, ax,
            f"Canvas reconciliation: +{mean:.0f}% ± {std:.0f}% accepted tokens "
            "per diffusion forward",
            f"Mean ± sample std, {data['n_prompt_seeds']} prompt seeds · "
            "identical weights per arm · every run 1.000 exact")
    _save(fig, "07_canvas_reconciliation.png")


def fig_08_draft_acceptance_by_lane():
    data = _load("cpu_bench_tiny.json")
    lanes = data["lanes"]
    skip = {"target_only", "diffusion_discard", "diffusion_reconcile",
            "mtp_untrained", "ar_draft"}
    rows = sorted(
        ((LANE_LABELS.get(n, n), r["mean_acceptance_rate"])
         for n, r in lanes.items() if n not in skip),
        key=lambda kv: kv[1],
    )
    family = {
        "n-gram": (BLUE, "retrieval (free)"),
        "compressed KV · int8": (DART, "target self-draft on cheap KV"),
        "compressed KV · int4": (DART, "target self-draft on cheap KV"),
        "compressed KV · 1-bit VQ": (DART, "target self-draft on cheap KV"),
        "sparse KV (sink+window)": (DART, "target self-draft on cheap KV"),
        "block-sparse KV": (DART, "target self-draft on cheap KV"),
        "MTP heads (trained)": (GREEN, "learned drafters (tiny-scale)"),
        "diffusion (trained, discard)": (GREEN, "learned drafters (tiny-scale)"),
        "diffusion (trained, reconcile)": (GREEN, "learned drafters (tiny-scale)"),
    }
    fig, ax = _fig(9.6, 5.2, bottom=0.12, left=0.28)
    seen = {}
    for i, (nm, acc) in enumerate(rows):
        color, label = family.get(nm, (DART, None))
        ax.barh(i, acc, height=0.6, color=color,
                label=label if label not in seen else None)
        seen[label] = True
        ax.text(acc + 0.013, i, f"{acc:.3f}", va="center", fontsize=10,
                color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.2)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("draft acceptance rate (accepted / proposed)")
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(color=GRID, linewidth=0.9)
    ax.legend(loc="lower right", framealpha=0)
    run = data["run"]
    _header(fig, ax, "Draft acceptance by lane",
            f"Seeded tiny target · {run['prompts']} prompts × "
            f"{run['max_new']} tokens · every lane byte-exact - acceptance "
            "is purely a speed lever")
    _save(fig, "08_draft_acceptance_by_lane.png")


FIGURES = {
    "01": fig_01_cost_vs_fidelity,
    "02": fig_02_exactness_under_kv_compression,
    "03": fig_03_kv_cache_memory_at_scale,
    "04": fig_04_acceleration_landscape,
    "05": fig_05_speed_vs_fidelity,
    "06": fig_06_technique_coverage,
    "07": fig_07_canvas_reconciliation,
    "08": fig_08_draft_acceptance_by_lane,
}


def main() -> None:
    wanted = sys.argv[1:] or sorted(FIGURES)
    for key in wanted:
        FIGURES[key.zfill(2)]()


if __name__ == "__main__":
    main()
