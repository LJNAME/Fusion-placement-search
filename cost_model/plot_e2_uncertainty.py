"""Export uncertainty/stop diagnostics as standalone SVG and CSV (ReportLab)."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors


STYLES = (("legacy", "Legacy family", "#6b7280"),
          ("strict_family", "Strict-fold family", "#0072b2"),
          ("distance_floor", "Distance + floor", "#d55e00"))


def dense_stop_curves(report, directory):
    baseline = json.loads((directory / "e2c_graph_wide_budget_curves.json").read_text(encoding="utf-8"))
    folder = directory / "diagnostics_checkpoints" / baseline["experiment_id"].split(":")[1][:16]
    legacy = {}
    for task in report["tasks"]:
        with gzip.open(folder / (task["task_id"].split(":")[1] + ".json.gz"), "rt", encoding="utf-8") as handle:
            trajectory = json.load(handle)["raw"]["policies"]["srop_proxy"]["full_trajectories"][0]
        stop = next((index for index, point in enumerate(trajectory) if point["stop_recommended"]), len(trajectory)-1)
        legacy[task["task_id"]] = [trajectory[min(budget, stop)] for budget in range(17)]
    curves = {}
    for variant, _, _ in STYLES:
        points = []
        for budget in range(17):
            rows = [legacy[task["task_id"]][budget] if variant == "legacy" else task["variants"][variant]["curve"][budget]
                    for task in report["tasks"]]
            points.append({"budget": budget,
                "mean_measurements": statistics.fmean(row["budget"] if variant == "legacy" else row["measurements_mean"] for row in rows),
                **{key: statistics.fmean(row[key] for row in rows) for key in
                   ("top1_regret_ms", "profiling_wall_time_s", "algorithm_wall_time_s")}})
        curves[variant] = points
    return curves


def bar_panel(drawing, groups, metric, x, y, title, ylabel, percent=False):
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = x + 48, y + 40, 345, 160
    chart.data = [[group[variant][metric] * (100 if percent else 1) for group in groups] for variant, _, _ in STYLES]
    chart.categoryAxis.categoryNames = ["Model holdout", "Sequence holdout", "Active initial"]
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100 if percent else math.ceil(max(value for row in chart.data for value in row))
    chart.valueAxis.valueStep = 25 if percent else 1
    for index, (_, _, color) in enumerate(STYLES):
        chart.bars[index].fillColor = colors.HexColor(color)
        chart.bars[index].strokeColor = None
    drawing.add(chart)
    if percent:
        drawing.add(Line(chart.x, chart.y + chart.height * 0.9, chart.x + chart.width, chart.y + chart.height * 0.9,
                         strokeColor=colors.HexColor("#9ca3af"), strokeDashArray=[3, 3]))
    drawing.add(String(x + 48, y + 225, title, fontName="Helvetica-Bold", fontSize=11))
    drawing.add(String(x + 48, y + 210, ylabel, fontSize=8, fillColor=colors.HexColor("#4b5563")))


def line_panel(drawing, curves, metric, x, y, title, ylabel):
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = x + 48, y + 35, 345, 165
    chart.data = [[(point["budget"], point[metric]) for point in curves[variant]] for variant, _, _ in STYLES]
    chart.joinedLines = 1
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, 16
    chart.xValueAxis.valueSteps = [0, 1, 2, 4, 8, 16]
    chart.yValueAxis.valueMin = 0
    chart.yValueAxis.valueMax = max(max(y for _, y in row) for row in chart.data) * 1.1 or 1.0
    chart.yValueAxis.labelTextFormat = "%.3g"
    chart.xValueAxis.labels.fontSize = chart.yValueAxis.labels.fontSize = 8
    for index, (_, _, color) in enumerate(STYLES):
        chart.lines[index].strokeColor = colors.HexColor(color)
        chart.lines[index].strokeWidth = 2 if index == 2 else 1.4
    drawing.add(chart)
    drawing.add(String(x + 48, y + 225, title, fontName="Helvetica-Bold", fontSize=11))
    drawing.add(String(x + 48, y + 210, ylabel, fontSize=8, fillColor=colors.HexColor("#4b5563")))
    drawing.add(String(x + 160, y + 10, "Requested budget", fontSize=9))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--stopping", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    stopping = json.loads(args.stopping.read_text(encoding="utf-8"))
    if stopping["execution_status"] != "complete" or stopping["scope"] != "all_frozen_configs":
        raise ValueError("complete frozen stop study required")
    if coverage["configuration"]["dataset_id"] != stopping["configuration"]["dataset_id"]:
        raise ValueError("coverage and stopping datasets differ")
    groups = [split["summary"]["all"] for split in coverage["zero_shot"]] + [coverage["active_initial"]["summary"]["all"]]
    curves = dense_stop_curves(stopping, args.stopping.parent)
    drawing = Drawing(870, 590)
    drawing.add(Rect(0, 0, 870, 590, fillColor=colors.white, strokeColor=None))
    drawing.add(String(28, 565, "Training-only uncertainty: coverage and stopping", fontName="Helvetica-Bold", fontSize=16))
    for index, (_, label, color) in enumerate(STYLES):
        x = 35 + index * 270
        drawing.add(Line(x, 539, x + 23, 539, strokeColor=colors.HexColor(color), strokeWidth=2))
        drawing.add(String(x+29, 536, label, fontSize=10))
    bar_panel(drawing, groups, "coverage", 13, 275, "Coverage", "Percent inside q90; dotted line = 90%", True)
    bar_panel(drawing, groups, "mean_interval_score_log", 438, 275, "Width and miss penalty", "Central-90% interval score (log units), lower is better")
    line_panel(drawing, curves, "top1_regret_ms", 13, 30, "All frozen configurations", "Mean Top-1 regret after score-based stopping (ms)")
    line_panel(drawing, curves, "mean_measurements", 438, 30, "Measurements actually used", "Mean revealed objects; decisions held after stop")
    counts = [group["legacy"]["objects"] for group in groups]
    drawing.add(String(28, 12, f"{counts[0]}/{counts[1]} held-out objects; {counts[2]} deduplicated active object-state pairs. Development diagnostics; correlated configurations.", fontSize=8))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(drawing, str(args.out_dir / "uncertainty_stopping.svg"))
    with (args.out_dir / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "variant", "objects", "coverage", "mean_q90_log", "max_q90_log", "interval_score_log"])
        for group, values in zip(("model_holdout", "sequence_holdout", "active_initial"), groups):
            for variant, _, _ in STYLES:
                row = values[variant]
                writer.writerow([group, variant, row["objects"], row["coverage"], row["mean_q90_log"], row["max_q90_log"], row["mean_interval_score_log"]])
    with (args.out_dir / "stopping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "budget", "mean_measurements", "regret_ms", "profiling_wall_s", "algorithm_wall_s"])
        for variant, _, _ in STYLES:
            for row in curves[variant]:
                writer.writerow([variant, row["budget"], row["mean_measurements"], row["top1_regret_ms"], row["profiling_wall_time_s"], row["algorithm_wall_time_s"]])
    print(args.out_dir)


if __name__ == "__main__":
    main()
