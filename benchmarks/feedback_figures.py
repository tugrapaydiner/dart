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
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    destinations = [directory / (name + ".svg") for name in NAMES]
    if previews is not None:
        destinations += [previews / (name + ".png") for name in NAMES]
    if any(path.exists() for path in destinations):
        raise FileExistsError("figure paths must be new")
    directory.mkdir(parents=True, exist_ok=True)
    if previews is not None:
        previews.mkdir(parents=True, exist_ok=True)
    hashes = {}
    width = 982
    colors = {"entropy_off4": SAGE, "replay_greedy": BLUE,
              "replay_hf_greedy": LILAC, "replay_hf_lookup": PINK}
    workloads = ("code", "instruction", "structured", "repetition")

    def frame(title, height, context):
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)
        fig.text(32 / width, 1 - 32 / height, title, fontsize=14, weight="semibold", va="top")
        fig.text(950 / width, 1 - 35 / height, context, fontsize=10.5, color=MUTED, ha="right", va="top")
        return fig

    def label(fig, x, y, text, **kwargs):
        height = fig.get_figheight() * 100
        return fig.text(x / width, 1 - y / height, text, va="center", **kwargs)

    def axes(fig, left, top, plot_width, plot_height, grid="x"):
        height = fig.get_figheight() * 100
        ax = fig.add_axes([left / width, 1 - (top + plot_height) / height,
                           plot_width / width, plot_height / height], facecolor=BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis=grid, color=GRID, linewidth=.8)
        ax.tick_params(length=0, pad=9)
        return ax

    def footer(fig, text):
        label(fig, 32, fig.get_figheight() * 100 - 28, text, fontsize=10, color=MUTED)

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
        "font.family": family, "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#9A9991", "axes.linewidth": .7,
        "xtick.color": MUTED, "ytick.color": INK, "xtick.labelsize": 10.5,
        "svg.fonttype": "path", "svg.hashsalt": "dart-feedback-editorial-v3", "savefig.facecolor": BG,
    }):
        group = data["nonrepetitive"]
        reduction = group["candidate_comparisons"]["replay_greedy"]["mean_latency_reduction"]
        fig = frame("Latency against matched baselines", 530, "Supra-50M · FP32 · batch one")
        label(fig, 32, 111, f"{abs(reduction) * 100:.1f}%", fontsize=36, weight="semibold")
        label(fig, 245, 97, "less latency" if reduction >= 0 else "more latency", fontsize=17, weight="semibold")
        label(fig, 245, 128, "than direct greedy · 36 cases excluding synthetic repetition", fontsize=11, color=MUTED)
        label(fig, 32, 178, "METHOD", fontsize=9.5, color=MUTED)
        label(fig, 950, 178, "MEAN REQUEST TIME", fontsize=9.5, color=MUTED, ha="right")
        ax = axes(fig, 208, 202, 626, 232)
        order = sorted(LABELS, key=group["mean_latency_ms"].__getitem__)
        maximum = max(row[method] for row in group["session_mean_latency_ms"] for method in order)
        for index, method in enumerate(order):
            y, value = 3 - index, group["mean_latency_ms"][method]
            ax.barh(y, value, height=.47, color=colors[method], edgecolor=INK, linewidth=.6)
            ax.errorbar(value, y, xerr=range_error(group, method), fmt="none", ecolor=INK, elinewidth=.8, capsize=4)
            ax.text((950 - 208) / 626, y, f"{value:.1f} ms", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=12, weight="semibold" if method == "entropy_off4" else "normal")
        ax.set(xlim=(0, math.ceil(maximum * 1.02 / 25) * 25), ylim=(-.6, 3.6), yticks=[3, 2, 1, 0],
               yticklabels=[LABELS[name] for name in order], xlabel="Latency (ms) →")
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.label.set_size(10.5)
        for tick, method in zip(ax.get_yticklabels(), order):
            tick.set_weight("semibold" if method == "entropy_off4" else "normal")
        footer(fig, "Same weights and replay executor · Whiskers span three process means · Lower is better")
        save(fig, "methods")

        fig = frame("Latency by workload", 480, "12 cases per workload")
        legend = [Line2D([], [], marker="o", linestyle="none", color=INK, markerfacecolor=colors[method],
                         markeredgewidth=.7, markersize=8, label=LABELS[method])
                  for method in ("entropy_off4", "replay_greedy")]
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(24 / width, 1 - 72 / 480),
                   ncol=2, frameon=False, handletextpad=.3, columnspacing=1.6)
        label(fig, 950, 97, "DART VS GREEDY", fontsize=9.5, color=MUTED, ha="right")
        ax = axes(fig, 180, 138, 610, 240)
        maximum = max(data["workloads"][name]["mean_latency_ms"][method]
                      for name in workloads for method in ("entropy_off4", "replay_greedy"))
        for index, name in enumerate(workloads):
            group, y = data["workloads"][name], 3 - index
            candidate, control = [group["mean_latency_ms"][method] for method in ("entropy_off4", "replay_greedy")]
            ax.plot([candidate, control], [y, y], color="#B4B7AE", linewidth=2, zorder=2)
            for value, color in ((candidate, SAGE), (control, BLUE)):
                ax.scatter(value, y, s=100, facecolor=color, edgecolor=INK, linewidth=.65, zorder=3)
                ax.annotate(f"{value:.1f}", (value, y), xytext=(0, 13), textcoords="offset points",
                            ha="center", fontsize=10.5)
            reduction = 1 - candidate / control
            ax.text((950 - 180) / 610, y, f"{abs(reduction) * 100:.1f}% " + ("less" if reduction >= 0 else "more"),
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=14, weight="semibold")
        ax.set(xlim=(0, maximum * 1.12), ylim=(-.65, 3.65), yticks=[3, 2, 1, 0],
               yticklabels=[name.title() for name in workloads], xlabel="Mean request latency (ms) →")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        footer(fig, "Means over three processes · Repetition is synthetic · Lower is better")
        save(fig, "workloads")

        comparison = data["aggregate"]["candidate_comparisons"]["entropy_on4"]
        improvement = comparison["mean_latency_reduction"] * 100
        fig = frame("Remove unused feedback", 420, f"{abs(improvement):.1f}% " + ("less latency" if improvement >= 0 else "more latency"))
        label(fig, 32, 83, "Same engine, fixed draft length and 48 inputs. Only entropy computation changes.", fontsize=11, color=MUTED)
        rows = data["aggregate"]["session_mean_latency_ms"]
        values = [row[method] for row in rows for method in ("entropy_on4", "entropy_off4")]
        span = max(max(values) - min(values), 1)
        low, high = max(0, min(values) - span * .3), max(values) + span * .3
        for index, row in enumerate(rows):
            ax = axes(fig, 78 + index * 303, 147, 220, 166, grid="y")
            before, after = row["entropy_on4"], row["entropy_off4"]
            ax.plot([0, 1], [before, after], color="#A7A8A2", linewidth=1.6)
            for x, value, color in ((0, before, PINK), (1, after, LILAC)):
                ax.scatter(x, value, s=110, facecolor=color, edgecolor=INK, linewidth=.7, zorder=3)
                ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 13), textcoords="offset points",
                            ha="center", fontsize=11.5, weight="semibold")
            ax.set(xlim=(-.23, 1.23), ylim=(low, high), xticks=[0, 1], xticklabels=["Entropy on", "Entropy off"])
            ax.set_title(f"Process {index + 1}", fontsize=11.5, pad=18)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(axis="x", colors=INK, labelsize=10.5)
            if index == 0:
                ax.set_ylabel("Latency (ms)", fontsize=10.5)
            else:
                ax.tick_params(axis="y", labelleft=False)
        footer(fig, "Paired within each process · Shared vertical scale · Absolute latency varies between processes")
        save(fig, "feedback")

        fig = frame("Latency at each output budget", 470, "16 cases per budget")
        handles = [Patch(facecolor=colors[method], edgecolor=INK, linewidth=.6, label=LABELS[method]) for method in LABELS]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(22 / width, 1 - 72 / 470),
                   ncol=4, frameon=False, handlelength=1, columnspacing=1.8)
        maximum = max(group["mean_latency_ms"][method] for group in data["budgets"].values() for method in LABELS)
        for index, budget in enumerate(sorted(data["budgets"], key=int)):
            ax = axes(fig, 78 + index * 303, 160, 240, 215, grid="y")
            group = data["budgets"][budget]
            for position, method in enumerate(LABELS):
                value = group["mean_latency_ms"][method]
                ax.bar(position, value, width=.68, color=colors[method], edgecolor=INK, linewidth=.6)
                ax.text(position, value + maximum * .035, f"{value:.0f}", ha="center", fontsize=10.5)
            ax.set(xlim=(-.65, 3.65), ylim=(0, maximum * 1.17), xticks=[])
            ax.set_title(f"{budget} tokens", fontsize=13, weight="semibold", pad=22)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if index == 0:
                ax.set_ylabel("Mean latency (ms)", fontsize=10.5)
            else:
                ax.tick_params(axis="y", labelleft=False)
        footer(fig, "Different cases at each budget · Shared scale · EOS may stop generation earlier")
        save(fig, "budgets")

        cases = data["cases"]
        values = [row["candidate_comparisons"]["replay_greedy"]["mean_latency_reduction"] * 100 for row in cases.values()]
        wins = sum(value > 0 for value in values)
        fig = frame(f"{wins} of {len(cases)} case means are faster", 570, "DART vs direct greedy")
        label(fig, 32, 82, "Latency reduction (%) · Left of zero is slower · Every case is shown", fontsize=11, color=MUTED)
        low, high = min(-20, math.floor(min(values) / 20) * 20 - 5), max(75, math.ceil(max(values) / 20) * 20)
        for column, workload in enumerate(workloads):
            ax = axes(fig, 60 + column * 235, 159, 190, 310)
            ax.set_title(workload.title(), fontsize=12, weight="semibold", pad=23)
            ax.axvline(0, color="#7F8078", linewidth=.9)
            groups = sorted((name, group) for name, group in cases.items() if group["workload"] == workload)
            for index, (_name, group) in enumerate(groups):
                value = group["candidate_comparisons"]["replay_greedy"]["mean_latency_reduction"] * 100
                color, y = (SAGE if value >= 0 else PINK), 11 - index
                ax.plot([0, value], [y, y], color=color, linewidth=3, solid_capstyle="round")
                ax.scatter(value, y, s=38, facecolor=color, edgecolor=INK, linewidth=.5, zorder=3)
                if value < 0:
                    ax.annotate(f"{value:.1f}", (0, y), xytext=(8, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=10, weight="semibold")
            ax.set(xlim=(low, high), ylim=(-.65, 11.65), yticks=list(reversed(range(12))),
                   yticklabels=[name[-2:] for name, _ in groups])
            ticks = MaxNLocator(4).tick_values(low, high)
            ax.set_xticks([tick for tick in ticks if low <= tick <= high])
            ax.tick_params(axis="y", colors=MUTED, labelsize=9.5, pad=6)
            ax.tick_params(axis="x", labelsize=10)
        label(fig, 491, 514, "← Slower      Mean latency reduction (%)      Faster →", ha="center", fontsize=10.5, color=MUTED)
        footer(fig, "Case IDs within each workload · All intervals and token budgets are available in the raw records")
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
                           "latency_whiskers": "Methods chart: range of three process means, not a confidence interval.",
                           "workload_points": "Paired method means with arithmetic latency reduction; no interval is drawn.",
                           "feedback_slopes": "Paired entropy on/off means within each process; shared vertical scale.",
                           "case_points": "One mean reduction per case vs direct greedy; negative values are slower.",
                           "budget_bars": "Different cases per budget; shared zero-based scale, not a controlled length sweep.",
                           "analysis_uncertainty": data["uncertainty"]})
    print(json.dumps({"figures": list(hashes), "protocol_hash": data["protocol_hash"]}))


if __name__ == "__main__":
    main()
