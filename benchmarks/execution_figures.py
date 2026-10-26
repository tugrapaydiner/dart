"""Render the execution study from complete raw session records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.execution_analysis import analyze_directory
from benchmarks.execution_protocol import BASELINES, write_protocol
from benchmarks.figures import BG, BLUE, GRID, INK, LILAC, MUTED, PINK, SAGE, _font

NAMES = ("methods", "ablation", "workloads", "cases")
LABELS = {
    "current_dart": "Eager DART", "static_dart": "Static buffers",
    "replay_dart": "DART", "replay_greedy": "Direct greedy",
    "replay_hf_greedy": "HF greedy", "replay_hf_lookup": "HF prompt lookup",
}


def render(data: dict, directory: Path, *, font_file: Path | None = None,
           previews: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if data["split"] != "test" or len(data["sessions"]) != 3:
        raise ValueError("execution figures require the complete three-session study")
    destinations = [directory / (name + ".svg") for name in NAMES]
    if previews is not None:
        destinations += [previews / (name + ".png") for name in NAMES]
    if any(path.exists() for path in destinations):
        raise FileExistsError("figure paths must be new")
    directory.mkdir(parents=True, exist_ok=True)
    if previews is not None:
        previews.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def frame(title, subtitle, height, direction):
        fig = plt.figure(figsize=(9.82, height / 100), dpi=100, facecolor=BG)
        fig.text(44 / 982, 1 - 48 / height, title, fontsize=18, weight="semibold", va="center")
        fig.text(44 / 982, 1 - 86 / height, subtitle, fontsize=10.5, color=MUTED, va="center")
        fig.text(938 / 982, 1 - 48 / height, direction, fontsize=10.5, color=MUTED, ha="right", va="center")
        return fig

    def axes(fig, height, left, bottom=125):
        width = 804 - left
        ax = fig.add_axes([left / 982, bottom / height, width / 982, (height - bottom - 132) / height], facecolor=BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.tick_params(length=0, pad=10)
        return ax, (44 - left) / width, (938 - left) / width

    def footer(fig, lines):
        height = fig.get_figheight() * 100
        for y, line in zip((48, 25), lines):
            fig.text(44 / 982, y / height, line, fontsize=9.5, color=MUTED)

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
            hashes[name] = hashlib.sha256(raw).hexdigest()
        finally:
            plt.close(fig)

    def latencies(name, order, title, subtitle, height, note):
        group = data["aggregate"]
        fig = frame(title, subtitle, height, "Lower is better")
        ax, left, right = axes(fig, height, 270)
        maximum = 0
        for index, method in enumerate(order):
            y = len(order) - 1 - index
            mean = group["mean_latency_ms"][method]
            values = [row[method] for row in group["session_mean_latency_ms"]]
            maximum = max(maximum, *values)
            color = SAGE if method == "replay_dart" else LILAC if method == "static_dart" else BLUE
            ax.barh(y, mean, height=.38, color=color, edgecolor=INK, linewidth=.6)
            ax.errorbar(mean, y, xerr=[[mean - min(values)], [max(values) - mean]],
                        fmt="none", ecolor=INK, elinewidth=.8, capsize=3, capthick=.8)
            label = "CUDA graph replay" if name == "ablation" and method == "replay_dart" else LABELS[method]
            weight = "semibold" if method == "replay_dart" else "normal"
            ax.text(left, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=11, weight=weight)
            ax.text(right, y, f"{mean:,.0f} ms", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=14, weight=weight)
        ax.set(xlim=(0, maximum * 1.08), ylim=(-.6, len(order) - .4), yticks=[], xlabel="Mean request latency (ms)")
        footer(fig, [note, "Whiskers: range of three session means, not confidence intervals."])
        save(fig, name)

    def ratios(name, rows, title, subtitle, height, note):
        fig = frame(title, subtitle, height, "Higher is better")
        ax, left, right = axes(fig, height, 270)
        intervals = [bound for _, row in rows for bound in row["session_ratio_ci95"]]
        low, high = min(.95, *intervals), max(1.05, *intervals)
        padding = (high - low) * .1
        ax.axvline(1, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)))
        for index, (label, row) in enumerate(rows):
            y = len(rows) - 1 - index
            value = row["geometric_session_ratio"]
            lower, upper = row["session_ratio_ci95"]
            color = LILAC if label == "All cases" else SAGE if value >= 1 else PINK
            ax.errorbar(value, y, xerr=[[value - lower], [upper - value]], fmt="o", markersize=8,
                        markerfacecolor=color, markeredgecolor=INK, markeredgewidth=.6,
                        ecolor=INK, elinewidth=.8, capsize=3, capthick=.8)
            ax.text(left, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=11)
            ax.text(right, y, f"{value:.2f}×", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=14)
        ax.set(xlim=(max(0, low - padding), high + padding), ylim=(-.6, len(rows) - .4), yticks=[],
               xlabel="Direct greedy latency / DART latency")
        footer(fig, [note, "Points: geometric mean · 95% Student-t intervals over three session log ratios (df = 2)."])
        save(fig, name)

    family = _font(font_file) if font_file is not None else "DejaVu Sans"
    with plt.rc_context({
        "font.family": family, "font.size": 11.5, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#7D7C77", "axes.linewidth": .75,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10.5,
        "svg.fonttype": "path", "svg.hashsalt": "dart-execution-v1", "savefig.facecolor": BG,
    }):
        order = sorted(("replay_dart", *BASELINES), key=data["aggregate"]["mean_latency_ms"].__getitem__)
        latencies("methods", order, "Matched decoding methods", "All four paths use the same CUDA graph executor", 530,
                  "12 fixed cases · Three fresh processes · Three timed repeats per method and case")
        latencies("ablation", ["current_dart", "static_dart", "replay_dart"], "The effect of replay",
                  "Same weights, normalization, inputs and four-token drafting policy", 460,
                  "Static and replay share KV buffers and copy work; replay captures execution.")
        rows = [("All cases", data["aggregate"]["candidate_comparisons"]["replay_greedy"])]
        rows += [(name.title(), group["candidate_comparisons"]["replay_greedy"]) for name, group in data["workloads"].items()]
        ratios("workloads", rows, "Where drafting helps", "DART vs direct greedy · Both use CUDA graph replay", 580,
               "1× = equal latency · Four fixed workloads; subgroup intervals are descriptive.")
        rows = [(f"{group['workload'].title()} · {group['max_new_tokens']} tokens", group["candidate_comparisons"]["replay_greedy"])
                for group in data["cases"].values()]
        ratios("cases", rows, "Every timed case", "All case means and intervals, including slower requests", 250 + 44 * len(rows),
               "1× = equal latency · Labels show output budgets; natural EOS is retained.")
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed raw execution run")
    parser.add_argument("--out", type=Path, required=True, help="new figure directory")
    parser.add_argument("--previews", type=Path, help="optional PNG directory")
    parser.add_argument("--font-file", type=Path, help="optional variable font; release figures use Inter")
    args = parser.parse_args(argv)
    audit_path = args.out / "figure-data.json"
    if audit_path.exists():
        parser.error("figure-data.json already exists")
    try:
        data = analyze_directory(args.run)
        hashes = render(data, args.out, font_file=args.font_file, previews=args.previews)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    write_protocol(audit_path, {
        "data": data, "figures_sha256": hashes,
        "font_sha256": hashlib.sha256(args.font_file.read_bytes()).hexdigest() if args.font_file else None,
        "latency_whiskers": "Range of three process means, not a confidence interval.",
        "ratio_whiskers": data["uncertainty"],
    })
    print(json.dumps({"figures": list(hashes), "protocol_hash": data["protocol_hash"]}))


if __name__ == "__main__":
    main()
