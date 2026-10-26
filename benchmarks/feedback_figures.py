"""Render the feedback study from complete raw session records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.execution_protocol import file_hash, write_protocol
from benchmarks.feedback_analysis import analyze
from benchmarks.figures import BG, BLUE, GRID, INK, LILAC, MUTED, PINK, SAGE, _font

NAMES = ("methods", "feedback", "workloads", "cases")
LABELS = {"entropy_off4": "DART", "replay_greedy": "Direct greedy",
          "replay_hf_greedy": "HF greedy", "replay_hf_lookup": "HF prompt lookup"}


def read_data(directory: Path) -> dict:
    paths = [directory / "protocol.json", directory / "status.json"]
    plan, status = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if plan["split"] != "test" or plan["sessions"] != 3:
        raise ValueError("figures require the complete three-session final study")
    if {path.name for path in directory.glob("session-*.json")} != {f"session-{i}.json" for i in range(3)}:
        raise ValueError("session files do not match the protocol")
    paths += [directory / f"session-{i}.json" for i in range(3)]
    data = analyze(plan, status, [json.loads(path.read_text(encoding="utf-8")) for path in paths[2:]])
    data["artifact_hashes"] = {path.name: file_hash(path) for path in paths}
    return data


def render(data: dict, directory: Path, *, font_file: Path | None = None,
           previews: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    def axes(fig, height, left=270, bottom=125):
        ax = fig.add_axes([left / 982, bottom / height, (804 - left) / 982, (height - bottom - 132) / height], facecolor=BG)
        style(ax)
        return ax, (44 - left) / (804 - left), (938 - left) / (804 - left)

    def style(ax):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.tick_params(length=0, pad=10)

    def footer(fig, first, second):
        height = fig.get_figheight() * 100
        for y, line in ((48, first), (25, second)):
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

    family = _font(font_file) if font_file else "DejaVu Sans"
    with plt.rc_context({
        "font.family": family, "font.size": 11.5, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#7D7C77", "axes.linewidth": .75,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10.5,
        "svg.fonttype": "path", "svg.hashsalt": "dart-feedback-v1", "savefig.facecolor": BG,
    }):
        group = data["nonrepetitive"]
        fig = frame("Beyond synthetic repetition", "36 code, instruction and structured cases · Same CUDA graph executor", 530, "Lower is better")
        ax, left, right = axes(fig, 530)
        order = sorted(LABELS, key=group["mean_latency_ms"].__getitem__)
        maximum = max(row[method] for row in group["session_mean_latency_ms"] for method in order)
        for index, method in enumerate(order):
            y, value = len(order) - 1 - index, group["mean_latency_ms"][method]
            values = [row[method] for row in group["session_mean_latency_ms"]]
            weight = "semibold" if method == "entropy_off4" else "normal"
            ax.barh(y, value, height=.38, color=SAGE if method == "entropy_off4" else BLUE, edgecolor=INK, linewidth=.6)
            ax.errorbar(value, y, xerr=[[value - min(values)], [max(values) - value]], fmt="none", ecolor=INK, elinewidth=.8, capsize=3)
            ax.text(left, y, LABELS[method], transform=ax.get_yaxis_transform(), va="center", fontsize=11, weight=weight)
            ax.text(right, y, f"{value:.1f} ms", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=14, weight=weight)
        ax.set(xlim=(0, maximum * 1.05), ylim=(-.6, 3.6), yticks=[], xlabel="Mean request latency (ms)")
        footer(fig, "Three fresh processes · Three timed repeats per method and case",
               "Whiskers: range of three session means, not confidence intervals.")
        save(fig, "methods")

        fig = frame("Unused feedback costs time", "Fixed4 entropy on vs off · Same 48 cases in each process", 530, "Lower is better")
        ax, left, right = axes(fig, 530, bottom=185)
        values = []
        for index, row in enumerate(data["aggregate"]["session_mean_latency_ms"]):
            y, on, off = 2 - index, row["entropy_on4"], row["entropy_off4"]
            values += [on, off]
            ax.plot([off, on], [y, y], color=MUTED, linewidth=1.2)
            for value, color, label in ((on, PINK, "Entropy on"), (off, SAGE, "Entropy off")):
                ax.plot(value, y, "o", color=color, markersize=10, markeredgecolor=INK, markeredgewidth=.6,
                        label=label if index == 0 else None)
                ax.annotate(f"{value:.1f}", (value, y), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=10)
            ax.text(left, y, f"Process {index + 1}", transform=ax.get_yaxis_transform(), va="center", fontsize=11)
            ax.text(right, y, f"{100 * (1 - off / on):.1f}% less", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=13)
        padding = (max(values) - min(values)) * .2
        ax.set(xlim=(min(values) - padding, max(values) + padding), ylim=(-.5, 2.6), yticks=[], xlabel="Mean request latency (ms)")
        ax.legend(loc="upper left", bbox_to_anchor=(-.02, -.27), frameon=False, ncol=2, fontsize=10, handletextpad=.4)
        footer(fig, "Paired ablation: same engine, draft policy and inputs; only unused entropy is toggled.",
               "Absolute timings vary across processes. All recorded timings remain included.")
        save(fig, "feedback")

        rows = [("All 48 cases", data["aggregate"]), ("Without repetition", data["nonrepetitive"])]
        rows += [(name.title(), data["workloads"][name]) for name in ("code", "instruction", "structured", "repetition")]
        fig = frame("Where drafting helps", "DART vs direct greedy · Both use CUDA graph replay", 620, "Higher is better")
        ax, left, right = axes(fig, 620)
        ax.axvline(1, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)))
        bounds = [1.0]
        for index, (label, group) in enumerate(rows):
            row = group["candidate_comparisons"]["replay_greedy"]
            y, value = len(rows) - 1 - index, row["geometric_session_ratio"]
            low, high = row["session_ratio_ci95"]
            bounds += [low, high]
            color = LILAC if index < 2 else SAGE if value >= 1 else PINK
            ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", markersize=8, markerfacecolor=color,
                        markeredgecolor=INK, markeredgewidth=.6, ecolor=INK, elinewidth=.8, capsize=3)
            ax.text(left, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=11)
            ax.text(right, y, f"{value:.2f}×", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=14)
        padding = (max(bounds) - min(bounds)) * .12
        ax.set(xlim=(min(bounds) - padding, max(bounds) + padding), ylim=(-.6, 5.6), yticks=[], xlabel="Direct greedy latency / DART latency")
        footer(fig, "12 cases per workload · Without repetition: 36 cases · 1× = equal latency",
               "Points: geometric mean · 95% Student-t intervals over three session log ratios (df = 2).")
        save(fig, "workloads")

        fig = frame("Every case stays in", "All 48 case means and intervals · Pink marks a mean slower than direct greedy", 670, "Higher is better")
        bounds = [bound for group in data["cases"].values() for bound in group["candidate_comparisons"]["replay_greedy"]["session_ratio_ci95"]]
        low, high = min(.95, *bounds), max(1.05, *bounds)
        padding = (high - low) * .07
        for column, workload in enumerate(("code", "instruction", "structured", "repetition")):
            start = 44 + column * 231
            ax = fig.add_axes([(start + 47) / 982, 133 / 670, 148 / 982, 385 / 670], facecolor=BG)
            style(ax)
            ax.axvline(1, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)))
            groups = [(name, group) for name, group in data["cases"].items() if group["workload"] == workload]
            for index, (name, group) in enumerate(groups):
                row = group["candidate_comparisons"]["replay_greedy"]
                value = row["geometric_session_ratio"]
                lower, upper = row["session_ratio_ci95"]
                color = SAGE if row["mean_latency_reduction"] >= 0 else PINK
                ax.errorbar(value, 11 - index, xerr=[[value - lower], [upper - value]], fmt="o", markersize=5.5,
                            markerfacecolor=color, markeredgecolor=INK, markeredgewidth=.5, ecolor=MUTED, elinewidth=.7, capsize=2)
                ax.text(-.09, 11 - index, f"{name[-2:]} / {group['max_new_tokens']}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9)
            ax.set(xlim=(max(0, low - padding), high + padding), ylim=(-.65, 11.65), yticks=[], xlabel="Speed ratio")
            ax.tick_params(axis="x", labelsize=9)
            fig.text((start + 47) / 982, 555 / 670, workload.title(), fontsize=12, weight="semibold")
        footer(fig, "Row labels: case ID / output budget · 1× = equal latency · Natural EOS retained",
               "95% intervals over three session log ratios; case intervals are descriptive, without multiplicity correction.")
        save(fig, "cases")
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previews", type=Path)
    parser.add_argument("--font-file", type=Path, help="optional variable font; published figures use Inter")
    args = parser.parse_args(argv)
    audit = args.out / "figure-data.json"
    if audit.exists():
        parser.error("figure-data.json already exists")
    try:
        data = read_data(args.run)
        hashes = render(data, args.out, font_file=args.font_file, previews=args.previews)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    write_protocol(audit, {"data": data, "figures_sha256": hashes,
                           "font_sha256": file_hash(args.font_file) if args.font_file else None,
                           "latency_whiskers": "Range of three process means, not a confidence interval.",
                           "ratio_whiskers": data["uncertainty"]})
    print(json.dumps({"figures": list(hashes), "protocol_hash": data["protocol_hash"]}))


if __name__ == "__main__":
    main()
