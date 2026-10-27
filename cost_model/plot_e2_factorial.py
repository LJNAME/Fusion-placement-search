"""Standalone ReportLab plots and CSV for the frozen factorial experiment."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors


STYLES = (("legacy_proxy", "Old + Proxy", "#6b7280"), ("legacy_refit", "Old + Refit", "#cc79a7"),
          ("distance_proxy", "Distance + Proxy", "#0072b2"), ("distance_refit", "Distance + Refit", "#d55e00"))


def dense_curves(report, directory):
    parents = [json.loads((directory / name).read_text(encoding="utf-8")) for name in
               ("e2c_graph_wide_budget_curves.json", "e2c_refit_evaluation.json", "e2c_uncertainty_stopping.json")]
    folders = [directory / "diagnostics_checkpoints" / source["experiment_id"].split(":")[1][:16] for source in [*parents, report]]
    raw = {name: [] for name, _, _ in STYLES}
    for task in report["tasks"]:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        stored = []
        for folder in folders:
            with gzip.open(folder / filename, "rt", encoding="utf-8") as handle:
                stored.append(json.load(handle))
        trajectories = (stored[0]["raw"]["policies"]["srop_proxy"]["full_trajectories"][0],
                        stored[1]["raw"]["policies"]["srop_refit"]["full_trajectories"][0],
                        stored[2]["trajectories"]["distance_floor"], stored[3]["trajectory"])
        for (name, _, _), trajectory in zip(STYLES, trajectories):
            stop = next((index for index, point in enumerate(trajectory) if point["stop_recommended"]), len(trajectory)-1)
            raw[name].append([trajectory[min(budget, stop)] for budget in range(17)])
    return {name: [{"budget": budget,
        "mean_measurements": statistics.fmean(trajectory[budget]["budget"] for trajectory in trajectories),
        **{key: statistics.fmean(trajectory[budget][key] for trajectory in trajectories) for key in
           ("top1_regret_ms", "profiling_wall_time_s", "algorithm_wall_time_s")}}
        for budget in range(17)] for name, trajectories in raw.items()}


def bar_panel(drawing, values, x, y, title, ylabel):
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = x+48, y+40, 345, 165
    chart.data = [[values[name] for name, _, _ in STYLES]]
    chart.categoryAxis.categoryNames = ["Old/P", "Old/R", "Dist/P", "Dist/R"]
    chart.categoryAxis.labels.fontSize = chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(max(values.values()) * 1.15, 1.0 if "stop" in title.lower() else 1e-5)
    chart.valueAxis.labelTextFormat = "%.3g"
    for index, (_, _, color) in enumerate(STYLES):
        chart.bars[(0, index)].fillColor = colors.HexColor(color)
        chart.bars[(0, index)].strokeColor = None
    drawing.add(chart)
    drawing.add(String(x+48, y+229, title, fontName="Helvetica-Bold", fontSize=11))
    drawing.add(String(x+48, y+213, ylabel, fontSize=8, fillColor=colors.HexColor("#4b5563")))


def line_panel(drawing, curves, key, x, y, title, ylabel):
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = x+48, y+35, 345, 165
    plot.data = [[(row["budget"], row[key]) for row in curves[name]] for name, _, _ in STYLES]
    plot.joinedLines = 1
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, 16
    plot.xValueAxis.valueSteps = [0, 1, 2, 4, 8, 16]
    plot.yValueAxis.valueMin = 0
    plot.yValueAxis.valueMax = max(value for series in plot.data for _, value in series) * 1.1 or 1.0
    plot.yValueAxis.labelTextFormat = "%.3g"
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 8
    for index, (_, _, color) in enumerate(STYLES):
        plot.lines[index].strokeColor = colors.HexColor(color)
        plot.lines[index].strokeWidth = 2 if index == 3 else 1.4
    drawing.add(plot)
    drawing.add(String(x+48, y+229, title, fontName="Helvetica-Bold", fontSize=11))
    drawing.add(String(x+48, y+213, ylabel, fontSize=8, fillColor=colors.HexColor("#4b5563")))
    drawing.add(String(x+160, y+10, "Requested budget", fontSize=9))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs":
        raise ValueError("complete frozen factorial experiment required")
    curves = dense_curves(report, args.report.parent)
    summary = report["summary"]["all"]["cells"]
    drawing = Drawing(870, 590)
    drawing.add(Rect(0, 0, 870, 590, fillColor=colors.white, strokeColor=None))
    drawing.add(String(28, 565, "E2 stopping: interval x hypothetical update", fontName="Helvetica-Bold", fontSize=16))
    for index, (_, label, color) in enumerate(STYLES):
        x = 35+index*205
        drawing.add(Line(x, 539, x+23, 539, strokeColor=colors.HexColor(color), strokeWidth=2))
        drawing.add(String(x+29, 536, label, fontSize=9))
    bar_panel(drawing, {name: value["mean_regret_budget_auc_ms"] for name, value in summary.items()},
              13, 275, "Mean regret-budget AUC", "Common 0..16 horizon (ms), lower is better")
    bar_panel(drawing, {name: value["wrong_stop_configs"] for name, value in summary.items()},
              438, 275, "Wrong stops", "Number of configurations; P=Proxy, R=Refit")
    line_panel(drawing, curves, "top1_regret_ms", 13, 30, "All frozen configurations", "Mean Top-1 regret after stopping (ms)")
    line_panel(drawing, curves, "mean_measurements", 438, 30, "Measurements actually used", "Mean revealed objects; decision held after stopping")
    drawing.add(String(28, 12, "Same frozen Oracle-containing pools. Correlated development configurations; interaction is descriptive, not a significance test.", fontSize=8))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(drawing, str(args.out_dir / "factorial_stopping.svg"))
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "auc_ms", "wrong_stops", "mean_measurements", "final_regret_ms", "profiling_wall_s", "algorithm_wall_s"])
        for name, _, _ in STYLES:
            value = summary[name]
            end = value["curve"][-1]
            writer.writerow([name, value["mean_regret_budget_auc_ms"], value["wrong_stop_configs"], value["mean_stop_budget"],
                             end["top1_regret_ms"], end["profiling_wall_time_s"], end["algorithm_wall_time_s"]])
    with (args.out_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "budget", "mean_measurements", "regret_ms", "profiling_wall_s", "algorithm_wall_s"])
        for name, _, _ in STYLES:
            for point in curves[name]:
                writer.writerow([name, point["budget"], point["mean_measurements"], point["top1_regret_ms"],
                                 point["profiling_wall_time_s"], point["algorithm_wall_time_s"]])
    print(args.out_dir)


if __name__ == "__main__":
    main()
