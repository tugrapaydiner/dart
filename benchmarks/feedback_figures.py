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

NAMES = ("methods", "feedback", "workloads", "budgets", "cases")
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

    def frame(title, subtitle, height, direction="Lower is better"):
        fig = plt.figure(figsize=(9.82, height / 100), dpi=100, facecolor=BG)
        fig.patch.set_edgecolor(INK)
        fig.patch.set_linewidth(.7)
        fig.text(25 / 982, 1 - 38 / height, title, fontsize=13, weight="semibold", va="center")
        fig.text(25 / 982, 1 - 71 / height, subtitle, fontsize=10, color=MUTED, va="center")
        fig.text(957 / 982, 1 - 38 / height, direction, fontsize=10, color=MUTED, ha="right", va="center")
        return fig

    def axes(fig, left=85, bottom=94, right=38, top=124, grid="y"):
        from matplotlib.ticker import MaxNLocator

        height = fig.get_figheight() * 100
        ax = fig.add_axes([left / 982, bottom / height, (982 - left - right) / 982,
                           (height - bottom - top) / height], facecolor=BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis=grid, color=GRID, linewidth=.8)
        ax.tick_params(length=0, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(5, prune="upper"))
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="upper"))
        return ax

    def legend(ax, labels, colors, *, columns=2):
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=color, edgecolor=INK, linewidth=.6, label=label)
                   for label, color in zip(labels, colors)]
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 1.06), ncol=columns,
                  frameon=False, fontsize=10, handlelength=1, handleheight=1, columnspacing=1.8)

    def footer(fig, line):
        height = fig.get_figheight() * 100
        fig.text(25 / 982, 23 / height, line, fontsize=9, color=MUTED)

    def range_error(group, method):
        value = group["mean_latency_ms"][method]
        values = [row[method] for row in group["session_mean_latency_ms"]]
        return [[value - min(values)], [max(values) - value]]

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
        "font.family": family, "font.size": 10.5, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#7D7C77", "axes.linewidth": .75,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10,
        "svg.fonttype": "path", "svg.hashsalt": "dart-feedback-v2", "savefig.facecolor": BG,
    }):
        group = data["nonrepetitive"]
        fig = frame("Latency against matched baselines", "36 code, instruction and structured cases · Synthetic repetition excluded", 520)
        ax = axes(fig)
        order = sorted(LABELS, key=group["mean_latency_ms"].__getitem__)
        maximum = max(row[method] for row in group["session_mean_latency_ms"] for method in order)
        colors = {"entropy_off4": SAGE, "replay_greedy": BLUE,
                  "replay_hf_greedy": LILAC, "replay_hf_lookup": PINK}
        for index, method in enumerate(order):
            value = group["mean_latency_ms"][method]
            error = range_error(group, method)
            ax.bar(index, value, width=.53, color=colors[method], edgecolor=INK, linewidth=.65)
            ax.errorbar(index, value, yerr=error, fmt="none", ecolor=INK, elinewidth=.8, capsize=4)
            ax.text(index, value + error[1][0] + maximum * .04, f"{value:.1f}", ha="center", fontsize=11)
        ax.set(xlim=(-.65, 3.65), ylim=(0, maximum * 1.19), xticks=range(4),
               xticklabels=[LABELS[name] for name in order], ylabel="Mean request latency (ms)")
        ax.tick_params(axis="x", colors=INK)
        for tick, method in zip(ax.get_xticklabels(), order):
            tick.set_weight("semibold" if method == "entropy_off4" else "normal")
        footer(fig, "Same replay executor · Three processes · Whiskers show the range of process means")
        save(fig, "methods")

        fig = frame("The cost of unused entropy", "Same engine, draft policy and 48 inputs · Only unused feedback is toggled", 490)
        ax = axes(fig, left=142, right=60, top=136, bottom=88, grid="x")
        maximum = max(row[method] for row in data["aggregate"]["session_mean_latency_ms"]
                      for method in ("entropy_on4", "entropy_off4"))
        for index, row in enumerate(data["aggregate"]["session_mean_latency_ms"]):
            y = 2 - index
            for offset, method, color in ((.19, "entropy_on4", PINK), (-.19, "entropy_off4", LILAC)):
                value = row[method]
                ax.barh(y + offset, value, height=.30, color=color, edgecolor=INK, linewidth=.65)
                ax.text(value + maximum * .013, y + offset, f"{value:.1f}", va="center", fontsize=10)
        ax.set(xlim=(0, maximum * 1.10), ylim=(-.55, 2.55), yticks=[2, 1, 0],
               yticklabels=["Process 1", "Process 2", "Process 3"], xlabel="Mean request latency (ms)")
        ax.tick_params(axis="y", colors=INK)
        legend(ax, ["Entropy on", "Entropy off"], [PINK, LILAC])
        footer(fig, "Paired within each process · Absolute latency varies across processes")
        save(fig, "feedback")

        workloads = ("code", "instruction", "structured", "repetition")
        fig = frame("Latency across workloads", "12 fresh cases per workload · Both methods use CUDA graph replay", 520)
        ax = axes(fig, top=137)
        maximum = max(row[method] for group in data["workloads"].values()
                      for row in group["session_mean_latency_ms"] for method in ("entropy_off4", "replay_greedy"))
        for index, workload in enumerate(workloads):
            group = data["workloads"][workload]
            for offset, method, color in ((-.18, "entropy_off4", SAGE), (.18, "replay_greedy", BLUE)):
                value, error = group["mean_latency_ms"][method], range_error(group, method)
                ax.bar(index + offset, value, width=.29, color=color, edgecolor=INK, linewidth=.65)
                ax.errorbar(index + offset, value, yerr=error, fmt="none", ecolor=INK, elinewidth=.7, capsize=3)
                ax.text(index + offset, value + error[1][0] + maximum * .035,
                        f"{value:.1f}", ha="center", fontsize=9.5)
        ax.set(xlim=(-.6, 3.6), ylim=(0, maximum * 1.17), xticks=range(4),
               xticklabels=[name.title() for name in workloads], ylabel="Mean request latency (ms)")
        ax.tick_params(axis="x", colors=INK)
        legend(ax, ["DART", "Direct greedy"], [SAGE, BLUE])
        footer(fig, "Three processes · Whiskers show the range of process means")
        save(fig, "workloads")

        fig = frame("Latency by output budget", "All 48 cases, including repetition · 16 cases at each budget", 520)
        ax = axes(fig, right=207, bottom=95)
        budgets = sorted(map(int, data["budgets"]))
        markers = {"entropy_off4": "o", "replay_greedy": "s", "replay_hf_greedy": "D", "replay_hf_lookup": "^"}
        values = []
        for method in LABELS:
            means = [data["budgets"][str(budget)]["mean_latency_ms"][method] for budget in budgets]
            values += means
            ax.plot(budgets, means, color=colors[method], linewidth=2, marker=markers[method],
                    markersize=8, markeredgecolor=INK, markeredgewidth=.6)
            ax.annotate(LABELS[method], (budgets[-1], means[-1]), xytext=(15, 0), textcoords="offset points",
                        va="center", fontsize=10, weight="semibold" if method == "entropy_off4" else "normal")
        ax.set(xlim=(min(budgets) - 3, max(budgets) + 2), ylim=(0, max(values) * 1.12), xticks=budgets,
               xlabel="Maximum new tokens", ylabel="Mean request latency (ms)")
        footer(fig, "Budgets contain different cases · Lines connect observed means, not a controlled length sweep")
        save(fig, "budgets")

        from matplotlib.patches import Rectangle

        fig = frame("Every case, including the losses", "Mean latency reduction vs direct greedy (%) · Negative values are slower", 630, "Higher is better")
        ax = fig.add_axes([25 / 982, 77 / 630, 932 / 982, 432 / 630])
        ax.set(xlim=(-.86, 4), ylim=(-.65, 12.4))
        ax.axis("off")
        ax.text(-.82, 12.05, "Case", fontsize=10, color=MUTED)
        for column, workload in enumerate(workloads):
            ax.text(column + .47, 12.05, workload.title(), ha="center", fontsize=10.5, weight="semibold")
            groups = sorted((name, group) for name, group in data["cases"].items() if group["workload"] == workload)
            for index, (name, group) in enumerate(groups):
                row = group["candidate_comparisons"]["replay_greedy"]
                value, y = 100 * row["mean_latency_reduction"], 11 - index
                color = SAGE if value >= 0 else PINK
                ax.add_patch(Rectangle((column + .03, y - .39), .88, .78, facecolor=color,
                                       edgecolor=GRID, linewidth=.5))
                ax.text(column + .47, y, f"{value:+.1f}", ha="center", va="center", fontsize=11,
                        weight="semibold" if value < 0 else "normal")
                if column == 0:
                    ax.text(-.82, y, name[-2:], va="center", fontsize=10, color=MUTED)
        footer(fig, "Each cell is one case mean · All case intervals, token budgets and IDs remain in the raw analysis")
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
                           "case_cells": "Percent mean latency reduction vs direct greedy; negative means slower.",
                           "budget_lines": "Observed means across different cases, not a controlled length sweep.",
                           "analysis_uncertainty": data["uncertainty"]})
    print(json.dumps({"figures": list(hashes), "protocol_hash": data["protocol_hash"]}))


if __name__ == "__main__":
    main()
