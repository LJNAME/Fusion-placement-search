"""Standalone ReportLab CPU benchmark figure and CSV."""

import argparse
import csv
import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs":
        raise ValueError("complete frozen experiment required")
    summary = report["summary"]
    groups = [summary["all"], summary["by_scale"]["layer"], summary["by_scale"]["model"]]
    drawing = Drawing(850, 400)
    drawing.add(Rect(0, 0, 850, 400, fillColor=colors.white, strokeColor=None))
    drawing.add(String(25, 372, "SROP hypothetical observations: exact scenario batching", fontName="Helvetica-Bold", fontSize=16))
    drawing.add(String(25, 351, "84 frozen configurations; all candidate ranks, acquisition scores, selections and stops unchanged.", fontSize=10))
    for index, (metric, title) in enumerate((("mean_algorithm_wall_time_s", "Fit + ranking + acquisition CPU time"),
                                           ("mean_selection_wall_time_s", "Acquisition CPU time"))):
        x = 30+index*415
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = x+45, 105, 310, 175
        chart.data = [[group["variants"][variant][metric] for group in groups]
                      for variant in ("scalar_scenarios", "scenario_batch")]
        chart.categoryAxis.categoryNames = ["All", "Layer", "Full graph"]
        chart.categoryAxis.labels.fontSize = chart.valueAxis.labels.fontSize = 9
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(value for series in chart.data for value in series)*1.15
        chart.valueAxis.labelTextFormat = "%.3g"
        for series, color in enumerate(("#0072b2", "#d55e00")):
            chart.bars[series].fillColor = colors.HexColor(color)
            chart.bars[series].strokeColor = None
        drawing.add(chart)
        drawing.add(String(x+45, 309, title, fontName="Helvetica-Bold", fontSize=12))
        drawing.add(String(x+45, 291, "Seconds; mean of per-config paired medians", fontSize=9))
    for x, color, label in ((250, "#0072b2", "Scalar scenarios"), (445, "#d55e00", "Scenario batch")):
        drawing.add(Rect(x, 66, 14, 10, fillColor=colors.HexColor(color), strokeColor=None))
        drawing.add(String(x+20, 66, label, fontSize=10))
    config = report["configuration"]
    drawing.add(String(25, 35, f"Alternating AB/BA pairs, {config['repeats_per_variant']} repeats per variant; {config['workers']} concurrent CPU workers. Audit excluded from timed runs.", fontSize=9))
    drawing.add(String(25, 18, "Fixed Oracle-containing development pools. This is CPU replay acceleration, not end-to-end GPU training speedup.", fontSize=9))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    svg = args.out_dir / "scoring_benchmark.svg"
    renderSVG.drawToFile(drawing, str(svg))
    svg.write_text(svg.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["group", "configs", "old_algorithm_s", "batch_algorithm_s", "ratio_of_means", "median_paired_speedup", "old_selection_s", "batch_selection_s"])
        rows = [("all", summary["all"]), *summary["by_scale"].items(), *summary["by_model"].items()]
        for name, group in rows:
            old, new = (group["variants"][variant] for variant in ("scalar_scenarios", "scenario_batch"))
            writer.writerow([name, group["configs"], old["mean_algorithm_wall_time_s"], new["mean_algorithm_wall_time_s"],
                group["mean_time_ratio"], group["median_paired_speedup"], old["mean_selection_wall_time_s"], new["mean_selection_wall_time_s"]])
    print(args.out_dir)


if __name__ == "__main__":
    main()
