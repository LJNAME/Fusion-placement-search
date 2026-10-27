"""Standalone ReportLab figure and CSV for the wall-cost ablation."""

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
    names = ("analytic_repetitions", "family_wall_median")
    summary = report["summary"]["all"]["cells"]
    drawing = Drawing(820, 555)
    drawing.add(Rect(0, 0, 820, 555, fillColor=colors.white, strokeColor=None))
    drawing.add(String(25, 525, "E2 wall-cost normalization: fewer measurements, worse cost-budget AUC",
                       fontName="Helvetica-Bold", fontSize=13))
    drawing.add(String(25, 504, "Distance + Refit held fixed. 84 correlated development configurations; zero wrong stops in both variants.", fontSize=9))
    panels = (("mean_step_actual_cost_auc_ms", "Regret versus actual profiling cost", "Normalized step AUC (ms), lower is better"),
              ("mean_count_auc_ms", "Regret versus measurement count", "Normalized 0..16 AUC (ms), lower is better"),
              ("mean_measurements", "Measurements used at stopping", "Mean count"),
              ("mean_profiling_wall_time_s", "Recorded profiling cost at stopping", "Mean seconds; CPU acquisition cost excluded"))
    for index, (metric, title, unit) in enumerate(panels):
        x, y = 20+(index%2)*400, 270-(index//2)*230
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = x+52, y+39, 300, 135
        values = [summary[name][metric] for name in names]
        chart.data = [values]
        chart.categoryAxis.categoryNames = ["Analytic x 360", "Family wall median"]
        chart.categoryAxis.labels.fontSize = chart.valueAxis.labels.fontSize = 8
        chart.valueAxis.valueMin, chart.valueAxis.valueMax = 0, max(values)*1.25
        chart.valueAxis.labelTextFormat = "%.3g"
        for column, color in enumerate(("#0072b2", "#d55e00")):
            chart.bars[(0, column)].fillColor = colors.HexColor(color)
            chart.bars[(0, column)].strokeColor = None
        drawing.add(chart)
        drawing.add(String(x+30, y+212, title, fontName="Helvetica-Bold", fontSize=11))
        drawing.add(String(x+30, y+195, unit, fontSize=8))
    drawing.add(String(25, 24, "Common per-task full-pool wall-cost horizon; observations become available after completion. Oracle-containing fixed pools.", fontSize=8))
    drawing.add(String(25, 10, "Cost predictor is predeclared and trained on visible wall labels only. This result does not establish an end-to-end speedup.", fontSize=8))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    svg = args.out_dir / "cost_normalization.svg"
    renderSVG.drawToFile(drawing, str(svg))
    svg.write_text(svg.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", *summary[names[0]].keys()], lineterminator="\n")
        writer.writeheader()
        for name in names:
            writer.writerow({"variant": name, **summary[name]})
    print(args.out_dir)


if __name__ == "__main__":
    main()
