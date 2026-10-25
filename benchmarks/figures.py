from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from benchmarks.timing import summarize_samples

METHODS = ("hf_greedy", "hf_prompt_lookup", "target_greedy", "target_only", "dart", "dart_model_norm")
LABELS = {
    "hf_greedy": "HF greedy", "hf_prompt_lookup": "HF prompt lookup",
    "target_greedy": "Target greedy", "target_only": "Target-only engine",
    "dart": "DART · native RMSNorm", "dart_model_norm": "DART · original RMSNorm",
}
SHORT = {**LABELS, "dart_model_norm": "Original DART"}
BG, INK, MUTED, GRID = "#FAF9F5", "#141413", "#5E5D59", "#E5E4DD"
BLUE, SAGE, PINK, LILAC = "#C5D3E0", "#CED6BF", "#EBCECE", "#CBCADB"


def summarize(report: dict) -> dict:
    samples, outputs = report["timing"]["samples"], report["correctness"]["outputs"]
    repeats, seed = report["timing"]["repeats"], report["seed"]
    if not outputs or type(repeats) is not int or repeats < 2:
        raise ValueError("figures require outputs and at least two repeats")
    if len({row["id"] for row in outputs}) != len(outputs):
        raise ValueError("prompt IDs must be unique")
    expected = {(name, repeat, index) for name in METHODS for repeat in range(repeats) for index in range(len(outputs))}
    counts = Counter((row["method"], row["repeat"], row["prompt_index"]) for row in samples)
    if set(counts) != expected or set(counts.values()) != {1}:
        raise ValueError("timing matrix must contain every method, repeat and prompt exactly once")
    for row in samples:
        output = outputs[row["prompt_index"]]
        if row["output_hash"] != output["output_hash"] or row["tokens"] != output["generated_tokens"]:
            raise ValueError("timed outputs must match the correctness report")

    def group(rows):
        timing = summarize_samples(rows, "hf_prompt_lookup", seed)
        fastest = min((name for name in METHODS if name != "dart"), key=lambda name: timing[name]["mean_ms"])
        return {
            "timing": timing, "fastest_control": fastest,
            "vs_fastest": summarize_samples(rows, fastest, seed)["dart"],
            "norm_ablation": summarize_samples(rows, "dart_model_norm", seed)["dart"],
        }

    return {
        "prompts": len(outputs), "repeats": repeats,
        "budgets": sorted({row["max_new_tokens"] for row in outputs}),
        "prompts_by_workload": dict(Counter(row["workload"] for row in outputs)),
        "aggregate": group(samples),
        "workloads": {
            name: group([row for row in samples if outputs[row["prompt_index"]]["workload"] == name])
            for name in sorted({row["workload"] for row in outputs})
        },
        "cases": [{
            "id": output["id"], "workload": output["workload"], "budget": output["max_new_tokens"],
            **group([row for row in samples if row["prompt_index"] == index]),
        } for index, output in enumerate(outputs)],
    }


def _font(path: Path) -> str:
    import fontTools
    import matplotlib
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    from matplotlib import font_manager

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    family = "DART Figure Sans " + digest[:12]
    directory = Path(matplotlib.get_cachedir()) / "dart-fonts" / (digest + "-" + fontTools.__version__ + "-opsz18-v1")
    directory.mkdir(parents=True, exist_ok=True)  # FreeType keeps font files open on Windows.
    for weight, style in ((400, "Regular"), (500, "Medium"), (600, "SemiBold")):
        target = directory / f"figure-{weight}.ttf"
        if target.exists():
            font_manager.fontManager.addfont(str(target))
            continue
        with TTFont(path) as font:
            if "fvar" not in font:
                raise ValueError("font-file must be a variable font with a weight axis")
            axes = {axis.axisTag for axis in font["fvar"].axes}
            if "wght" not in axes:
                raise ValueError("font-file must have a weight axis")
            settings = {"wght": weight, **({"opsz": 18} if "opsz" in axes else {})}
            static = instantiateVariableFont(font, settings, inplace=False)
            names = {1: family, 2: style, 4: family + " " + style,
                     6: "DARTFigureSans" + digest[:12] + "-" + style, 16: family, 17: style}
            for name in static["name"].names:
                if name.nameID in names:
                    name.string = names[name.nameID].encode(name.getEncoding())
            static.save(target)
            static.close()
        font_manager.fontManager.addfont(str(target))
    return family


def render(data: dict, directory: Path, *, font_file: Path | None = None, previews: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    directory.mkdir(parents=True, exist_ok=True)
    names = ("methods", "workloads", "normalization", "cases")
    destinations = [directory / (name + ".svg") for name in names]
    if previews is not None:
        previews.mkdir(parents=True, exist_ok=True)
        destinations.extend(previews / (name + ".png") for name in names)
    if any(path.exists() for path in destinations):
        raise FileExistsError("figure paths must be new")
    exports = {}

    def frame(title, subtitle, metric, height):
        fig = plt.figure(figsize=(9.82, height / 100), dpi=100, facecolor=BG)
        fig.text(44 / 982, 1 - 48 / height, title, fontsize=18, weight="semibold", va="center")
        fig.text(44 / 982, 1 - 86 / height, subtitle, fontsize=10.5, color=MUTED, va="center")
        fig.text(938 / 982, 1 - 48 / height, metric, fontsize=10.5, color=MUTED, ha="right", va="center")
        return fig

    def axes(fig, box, grid="x"):
        ax = fig.add_axes(box, facecolor=BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis=grid, color=GRID, linewidth=.7)
        ax.tick_params(length=0, pad=10)
        return ax

    def save(fig, name):
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for item in fig.findobj(match=matplotlib.text.Text):
                if item.get_visible() and item.get_text().strip():
                    box = item.get_window_extent(renderer)
                    if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                        raise ValueError("figure text leaves the canvas: " + item.get_text())
            path = directory / (name + ".svg")
            fig.savefig(path, metadata={"Date": None, "Creator": None, "Title": name})
            raw = b"\n".join(line.rstrip() for line in path.read_bytes().splitlines()) + b"\n"
            size = f'width="982" height="{round(fig.get_figheight() * 100)}"'.encode()
            raw = re.sub(rb'width="[^"]+" height="[^"]+"', lambda match: size, raw, count=1)
            ET.fromstring(raw)
            path.write_bytes(raw)
            if previews is not None:
                fig.savefig(previews / (name + ".png"), dpi=100)
            exports[name] = hashlib.sha256(raw).hexdigest()
        finally:
            plt.close(fig)

    def whisker(ax, value, interval, y, color=SAGE):
        low, high = interval
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", markersize=8,
                    markerfacecolor=color, markeredgecolor=INK, markeredgewidth=.6,
                    ecolor=INK, elinewidth=.8, capsize=3, capthick=.8)

    with ExitStack() as stack:
        family = "DejaVu Sans"
        if font_file is not None:
            family = _font(font_file)
        stack.enter_context(plt.rc_context({
            "font.family": family, "font.size": 11.5, "text.color": INK,
            "axes.labelcolor": INK, "axes.edgecolor": "#7D7C77", "axes.linewidth": .75,
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
            "svg.fonttype": "path", "svg.hashsalt": "dart-v03", "savefig.facecolor": BG,
        }))

        timing = data["aggregate"]["timing"]
        order = sorted(METHODS, key=lambda name: timing[name]["mean_ms"])
        fig = frame("Six decoding methods", "Mean request latency · All baselines use native RMSNorm unless marked", "Lower is better", 610)
        ax = axes(fig, [302 / 982, 120 / 610, 502 / 982, 350 / 610])
        maximum = max(row["ci95_ms"][1] for row in timing.values()) / 1000 * 1.08
        for index, name in enumerate(order):
            row, y = timing[name], len(order) - 1 - index
            value = row["mean_ms"] / 1000
            low, high = np.array(row["ci95_ms"]) / 1000
            color = SAGE if name == "dart" else LILAC if name == "dart_model_norm" else BLUE
            ax.barh(y, value, height=.38, color=color, edgecolor=INK, linewidth=.6)
            ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="none", ecolor=INK, capsize=3, elinewidth=.7)
            ax.text((44 - 302) / 502, y, LABELS[name], transform=ax.get_yaxis_transform(), va="center",
                    fontsize=11, weight="semibold" if name == "dart" else "normal")
            ax.text((938 - 302) / 502, y, f"{value:.2f} s", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=14, weight="semibold" if name == "dart" else "medium")
        ax.set(xlim=(0, maximum), ylim=(-.6, len(order) - .4), yticks=[], xlabel="Mean latency (seconds)")
        caption = f"{data['prompts']} cases × {data['repeats']} repeats · Whiskers show 95% intervals for these fixed cases"
        fig.text(44 / 982, 34 / 610, caption, fontsize=9.5, color=MUTED)
        save(fig, "methods")

        groups = list(data["workloads"])
        fig = frame("Across workloads", "HF prompt lookup vs DART · Same weights, precision and normalization", "Lower is better", 590)
        ax = axes(fig, [88 / 982, 95 / 590, 850 / 982, 340 / 590], grid="y")
        maximum = max(data["workloads"][name]["timing"][method]["ci95_ms"][1]
                      for name in groups for method in ("hf_prompt_lookup", "dart")) / 1000
        for method, offset, color in (("hf_prompt_lookup", -.19, BLUE), ("dart", .19, SAGE)):
            for index, name in enumerate(groups):
                row = data["workloads"][name]["timing"][method]
                value = row["mean_ms"] / 1000
                low, high = np.array(row["ci95_ms"]) / 1000
                x = index + offset
                ax.bar(x, value, width=.32, color=color, edgecolor=INK, linewidth=.6)
                ax.errorbar(x, value, yerr=[[value - low], [high - value]], fmt="none", ecolor=INK, capsize=3, elinewidth=.7)
                ax.text(x, high + maximum * .025, f"{value:.2f}", ha="center", fontsize=10.5)
        ax.set(ylim=(0, maximum * 1.22), xticks=range(len(groups)), xticklabels=[name.title() for name in groups], ylabel="Mean latency (seconds)")
        fig.legend(handles=[Patch(facecolor=BLUE, edgecolor=INK, linewidth=.6, label="HF prompt lookup"),
                            Patch(facecolor=SAGE, edgecolor=INK, linewidth=.6, label="DART")],
                   loc="center right", bbox_to_anchor=(938 / 982, 1 - 125 / 590), ncol=2, frameon=False, fontsize=11)
        counts = set(data["prompts_by_workload"].values())
        case_scope = f"{next(iter(counts))} fixed cases per workload" if len(counts) == 1 else "Fixed cases by workload"
        budgets = " / ".join(str(value) for value in data["budgets"])
        fig.text(44 / 982, 30 / 590, case_scope + " · " + budgets + "-token output budgets", fontsize=9.5, color=MUTED)
        save(fig, "workloads")

        rows = [("All cases", data["aggregate"]["norm_ablation"])]
        rows += [(name.title(), data["workloads"][name]["norm_ablation"]) for name in groups]
        fig = frame("One normalization change", "DART with native vs original RMSNorm · Identical weights and drafting policy", "Higher is better", 540)
        ax = axes(fig, [230 / 982, 115 / 540, 574 / 982, 280 / 540])
        limits = [v for _, row in rows for v in row["ci95_speedup"]]
        ax.axvline(1, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)))
        for index, (label, row) in enumerate(rows):
            y = len(rows) - 1 - index
            whisker(ax, row["speedup"], row["ci95_speedup"], y, LILAC if index == 0 else SAGE)
            ax.text((44 - 230) / 574, y, label, transform=ax.get_yaxis_transform(), va="center", weight="semibold" if index == 0 else "normal")
            ax.text((938 - 230) / 574, y, f"{row['speedup']:.2f}×", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=14)
        ax.set(xlim=(min(.95, min(limits) - .05), max(limits) + .05), ylim=(-.5, len(rows) - .5), yticks=[], xlabel="Original latency / native latency")
        fig.text(44 / 982, 30 / 540, "1× = equal latency · Paired 95% repeat intervals · No extra learned parameters", fontsize=9.5, color=MUTED)
        save(fig, "normalization")

        rows = data["cases"]
        height = 240 + len(rows) * 43
        fig = frame("Every timed case", "DART vs each case’s fastest measured control · Output budgets shown beside each case", "Higher is better", height)
        ax = axes(fig, [250 / 982, 115 / height, 500 / 982, (height - 250) / height])
        limits = [value for row in rows for value in row["vs_fastest"]["ci95_speedup"]]
        ax.axvline(1, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)))
        for index, row in enumerate(rows):
            result, y = row["vs_fastest"], len(rows) - 1 - index
            whisker(ax, result["speedup"], result["ci95_speedup"], y, SAGE if result["speedup"] >= 1 else PINK)
            ax.text((44 - 250) / 500, y, f"{row['workload'].title()} · {row['budget']} tokens", transform=ax.get_yaxis_transform(), va="center", fontsize=11)
            ax.text((938 - 250) / 500, y + .15, f"{result['speedup']:.2f}×", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12, weight="medium")
            ax.text((938 - 250) / 500, y - .22, "vs " + SHORT[row["fastest_control"]], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, color=MUTED)
        ax.set(xlim=(min(.9, min(limits) - .05), max(limits) + .05), ylim=(-.55, len(rows) - .45), yticks=[], xlabel="Control latency / DART latency")
        fig.text(44 / 982, 30 / height, "1× = equal latency · Fixed-case repeat intervals, conditional on the observed fastest control", fontsize=9, color=MUTED)
        save(fig, "cases")
    return exports


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate v0.3 figures from complete six-method raw timings.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True, help="directory for new SVG files and figure-data.json")
    parser.add_argument("--previews", help="optional directory for PNG previews")
    parser.add_argument("--font-file", help="optional variable font; release figures use Inter at optical size 18")
    args = parser.parse_args(argv)
    source, destination = Path(args.report), Path(args.out)
    audit_path = destination / "figure-data.json"
    if audit_path.exists():
        parser.error("figure-data.json already exists")
    raw = source.read_bytes()
    data = summarize(json.loads(raw))
    font = Path(args.font_file) if args.font_file else None
    hashes = render(data, destination, font_file=font, previews=Path(args.previews) if args.previews else None)
    audit = {"input_sha256": hashlib.sha256(raw).hexdigest(), "figures_sha256": hashes, "data": data,
             "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest() if font else None,
             "font_scope": "variable font at weights 400/500/600 and optical size 18 when available" if font else "Matplotlib DejaVu Sans"}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(hashes)} figures and {audit_path}")


if __name__ == "__main__":
    main()
