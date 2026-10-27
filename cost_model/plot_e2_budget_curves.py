"""Export standalone SVG research plots and CSV from a completed E2 report.

Requires ReportLab only for rendering; the replay runner does not need it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors


STYLES = (
    ("random", "Random (4 seeds)", "#6b7280"),
    ("uncertainty_only", "Uncertainty", "#0072b2"),
    ("impact_only", "Impact", "#009e73"),
    ("srop_proxy", "SROP-Proxy", "#cc79a7"),
    ("srop_refit", "SROP-Refit", "#d55e00"),
)


def dense_plot_summary(report, checkpoint_dir):
    """Use every saved prefix so sparse exported budgets cannot hide harm."""
    folder = checkpoint_dir / report["experiment_id"].split(":")[1][:16]
    raw = {}
    for task in report["tasks"]:
        with gzip.open(folder / (task["task_id"].split(":")[1] + ".json.gz"), "rt", encoding="utf-8") as handle:
            raw[task["task_id"]] = json.load(handle)["raw"]
        if "baseline_experiment_id" in report["configuration"]:
            baseline_folder = checkpoint_dir / report["configuration"]["baseline_experiment_id"].split(":")[1][:16]
            with gzip.open(baseline_folder / (task["task_id"].split(":")[1] + ".json.gz"), "rt", encoding="utf-8") as handle:
                previous = json.load(handle)["raw"]
            previous["policies"].update(raw[task["task_id"]]["policies"])
            raw[task["task_id"]] = previous
    for group in ("all_configs", "positive_initial_regret", "zero_initial_regret"):
        tasks = [task for task in report["tasks"] if group == "all_configs" or
                 ((task["initial_regret_ms"] > 1e-9) == (group == "positive_initial_regret"))]
        for mode, modes in report["summary"][group]["modes"].items():
            for policy, values in modes["policies"].items():
                curve = []
                for budget in range(17):
                    rows = []
                    for task in tasks:
                        trajectories = raw[task["task_id"]]["policies"][policy]["full_trajectories"]
                        points = []
                        for trajectory in trajectories:
                            limit = len(trajectory) - 1
                            if mode == "stopping":
                                limit = next((index for index, point in enumerate(trajectory) if point["stop_recommended"]), limit)
                            points.append(trajectory[min(budget, limit)])
                        keys = ("budget", "top1_regret_ms", "profiling_wall_time_s", "algorithm_wall_time_s")
                        rows.append({key: statistics.fmean(point[key] for point in points) for key in keys})
                    curve.append({"requested_budget": budget,
                        "measurements_mean": statistics.fmean(row["budget"] for row in rows),
                        **{key: statistics.fmean(row[key] for row in rows) for key in keys if key != "budget"},
                        "harmed_configs": sum(row["top1_regret_ms"] > task["initial_regret_ms"] + 1e-9
                                              for task, row in zip(tasks, rows))})
                values["curve"] = curve
    return report


def panel(drawing, summary, mode, metric, x, y, title, ylabel):
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = x + 53, y + 35, 343, 175
    policies = summary["modes"][mode]["policies"]
    styles = [style for style in STYLES if style[0] in policies]
    plot.data = [[(point["requested_budget"], point[metric]) for point in policies[name]["curve"]]
                 for name, _, _ in styles]
    plot.joinedLines = 1
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, 16
    plot.xValueAxis.valueSteps = [0, 1, 2, 4, 8, 16]
    plot.yValueAxis.valueMin = 0
    maximum = max(value for series in plot.data for _, value in series)
    plot.yValueAxis.valueMax = max(maximum * 1.1, 1e-5)
    plot.yValueAxis.labelTextFormat = "%.3g"
    for axis in (plot.xValueAxis, plot.yValueAxis):
        axis.labels.fontName = "Helvetica"
        axis.labels.fontSize = 8
        axis.strokeColor = colors.HexColor("#9ca3af")
    for index, (_, _, color) in enumerate(styles):
        plot.lines[index].strokeColor = colors.HexColor(color)
        plot.lines[index].strokeWidth = 2.0 if index == 3 else 1.3
        if index == 0:
            plot.lines[index].strokeDashArray = [4, 2]
    drawing.add(plot)
    drawing.add(String(x + 53, y + 229, title, fontName="Helvetica-Bold", fontSize=11))
    drawing.add(String(x + 53, y + 215, ylabel, fontSize=8, fillColor=colors.HexColor("#4b5563")))
    drawing.add(String(x + 170, y + 10, "Requested measurements", fontSize=9))


def figure(report, *, costs):
    drawing = Drawing(870, 590)
    drawing.add(Rect(0, 0, 870, 590, fillColor=colors.white, strokeColor=None))
    drawing.add(String(28, 565, "E2 recorded-data replay: " + ("measurement and CPU costs" if costs else "regret vs profiling budget"),
                       fontName="Helvetica-Bold", fontSize=16))
    styles = [style for style in STYLES if style[0] in report["summary"]["all_configs"]["modes"]["forced"]["policies"]]
    for index, (_, label, color) in enumerate(styles):
        x = 35 + index * (810 / len(styles))
        drawing.add(Line(x, 539, x + 23, 539, strokeColor=colors.HexColor(color), strokeWidth=2))
        drawing.add(String(x + 29, 536, label, fontSize=10))
    all_configs = report["summary"]["all_configs"]
    if costs:
        entries = [(all_configs, "forced", "profiling_wall_time_s", "Forced", "Recorded profiling wall time (s)"),
                   (all_configs, "stopping", "profiling_wall_time_s", "With score-based stopping", "Recorded profiling wall time (s)"),
                   (all_configs, "forced", "algorithm_wall_time_s", "Forced", "CPU replay algorithm wall time (s)"),
                   (all_configs, "stopping", "algorithm_wall_time_s", "With score-based stopping", "CPU replay algorithm wall time (s)")]
    else:
        positive = report["summary"]["positive_initial_regret"]
        zero = report["summary"]["zero_initial_regret"]
        entries = [(all_configs, "forced", "top1_regret_ms", f"All {all_configs['configs']} configs: forced", "Mean Top-1 regret (ms)"),
                   (all_configs, "stopping", "top1_regret_ms", f"All {all_configs['configs']} configs: stopping", "Mean Top-1 regret (ms)"),
                   (positive, "forced", "top1_regret_ms", f"Initially nonzero: {positive['configs']} configs", "Mean Top-1 regret (ms), forced"),
                   (zero, "stopping", "top1_regret_ms", f"Initially zero: {zero['configs']} configs", "Mean Top-1 regret (ms), stopping")]
    for index, (summary, mode, metric, title, ylabel) in enumerate(entries):
        panel(drawing, summary, mode, metric, 13 + (index % 2) * 425, 275 if index < 2 else 30, title, ylabel)
    drawing.add(String(28, 12, "Fixed pools include Oracle-discovered candidates. Repeated configurations are correlated; development diagnostics only.",
                       fontSize=8, fillColor=colors.HexColor("#4b5563")))
    return drawing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs":
        raise ValueError("only a completed full frozen report can be exported")
    report = dense_plot_summary(report, args.checkpoint_dir or args.report.parent / "diagnostics_checkpoints")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for costs, name in ((False, "regret_budget"), (True, "profiling_costs")):
        renderSVG.drawToFile(figure(report, costs=costs), str(args.out_dir / (name + ".svg")))
    with (args.out_dir / "budget_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "mode", "policy", "requested_budget", "measurements_mean", "regret_ms",
                         "profiling_wall_s", "algorithm_wall_s", "harmed_configs"])
        for group in ("all_configs", "positive_initial_regret", "zero_initial_regret"):
            for mode, values in report["summary"][group]["modes"].items():
                for policy, result in values["policies"].items():
                    for point in result["curve"]:
                        writer.writerow([group, mode, policy, point["requested_budget"], point["measurements_mean"],
                            point["top1_regret_ms"], point["profiling_wall_time_s"], point["algorithm_wall_time_s"], point["harmed_configs"]])
    print(args.out_dir)


if __name__ == "__main__":
    main()
