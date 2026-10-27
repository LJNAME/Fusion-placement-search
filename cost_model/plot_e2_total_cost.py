"""Standalone ReportLab total-cost curves and CSV; no inference dependencies."""

import argparse
import csv
import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.lib import colors


STYLES = (("random", "Random", "#777777"), ("uncertainty_only", "Uncertainty", "#e69f00"),
          ("impact_only", "Impact", "#009e73"), ("srop_proxy", "Proxy", "#56b4e9"),
          ("srop_refit", "Refit", "#d55e00"), ("initial_only", "Initial only", "#cc79a7"),
          ("reference_only", "Reference only", "#222222"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs":
        raise ValueError("complete frozen report required")
    drawing = Drawing(920, 650)
    drawing.add(Rect(0, 0, 920, 650, fillColor=colors.white, strokeColor=None))
    drawing.add(String(27, 624, "E2: measured CPU + recorded BW1000 profiling cost", fontName="Helvetica-Bold", fontSize=16))
    for index, (_, label, color) in enumerate(STYLES):
        x, y = 32+(index%4)*225, 599-(index//4)*20
        drawing.add(Line(x, y, x+20, y, strokeColor=colors.HexColor(color), strokeWidth=2))
        drawing.add(String(x+25, y-3, label, fontSize=10))
    for index, (group, mode, title) in enumerate((("all", "forced", "All 84: fixed budget"),
        ("all", "stopping", "All 84: score-stop prefix replay"),
        ("positive_initial", "forced", "Positive initial regret: fixed budget"),
        ("positive_initial", "stopping", "Positive initial regret: score-stop prefix"))):
        x, y = 25+(index%2)*455, 302-(index//2)*240
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = x+55, y+35, 345, 160
        policies = report["summary"][group][mode]
        plot.data = [[(row["budget_s"], row["step_auc_ms"]) for row in policies[name]["curve"]]
                     for name, _, _ in STYLES]
        plot.joinedLines = 1
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 1, 16
        plot.xValueAxis.valueSteps = [1, 2, 4, 8, 16]
        plot.yValueAxis.valueMin = 0
        plot.yValueAxis.valueMax = max(v for series in plot.data for _, v in series)*1.1 or 1
        plot.yValueAxis.labelTextFormat = "%.3g"
        plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 8
        for line_index, (_, _, color) in enumerate(STYLES):
            plot.lines[line_index].strokeColor = colors.HexColor(color)
            plot.lines[line_index].strokeWidth = 1.6
        drawing.add(plot)
        drawing.add(String(x+55, y+222, title, fontName="Helvetica-Bold", fontSize=11))
        drawing.add(String(x+55, y+204, "Normalized regret-time AUC (ms), lower is better", fontSize=8))
        drawing.add(String(x+150, y+12, "Serial cost budget (seconds)", fontSize=9))
    drawing.add(String(27, 30, "Shared preparation and initial fit/rank fee; observations become usable after refit. Primary horizon = 4 seconds.", fontSize=9))
    drawing.add(String(27, 14, "Oracle-containing fixed development pools; correlated tasks; mixed-clock offline replay, not end-to-end GPU wall time.", fontSize=9))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    svg = args.out_dir/"total_cost.svg"
    renderSVG.drawToFile(drawing, str(svg))
    svg.write_text(svg.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    with (args.out_dir/"curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["group", "mode", "policy", "budget_s", "step_auc_ms", "regret_ms", "published_measurements", "completed_measurements"])
        groups = [(name, report["summary"][name]) for name in ("all", "positive_initial", "zero_initial")]
        groups += list(report["summary"]["by_model"].items())
        groups += list(report["summary"]["by_scale"].items())
        for name, group in groups:
            for mode in ("forced", "stopping"):
                for policy, _, _ in STYLES:
                    for row in group[mode][policy]["curve"]:
                        writer.writerow([name, mode, policy, *[row[key] for key in
                            ("budget_s", "step_auc_ms", "regret_ms", "published_measurements", "completed_measurements")]])
    print(args.out_dir)


if __name__ == "__main__":
    main()
