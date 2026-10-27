"""Separate forced one-object diagnostics from score-based stopping outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .e2_diagnostics import REGRET_TOLERANCE_MS, _write_json


POLICIES = ("uncertainty_only", "impact_only", "srop_proxy")


def audit_first_step(report):
    rows = []
    for task in report["tasks"]:
        if "one_step" not in task:
            continue
        zero, step = task["zero"], task["one_step"]
        policies = {}
        for name in POLICIES:
            selection = step["policies"][name]
            forced = selection["selected_gain_ms"]
            stopped = bool(selection["stop_recommended"])
            policies[name] = {
                "selected_object_id": selection["selected_object_id"],
                "selection_score": selection["selection_score"],
                "stops_before_measurement": stopped,
                "forced_gain_ms": forced,
                "gain_with_stopping_ms": 0.0 if stopped else forced,
                "regret_with_stopping_ms": zero["top1_regret_ms"] if stopped else selection["selected_regret_ms"],
            }
        rows.append({"task_id": task["task_id"], "split_id": zero["split_id"],
                     "model_id": zero["model_id"], "scale": zero["scale"],
                     "initial_train_count": zero["initial_train_count"], "subset_seed": zero["subset_seed"],
                     "initial_top1_regret_ms": zero["top1_regret_ms"], "policies": policies})

    def gain_summary(values):
        return {
            "improved_configs": sum(value > REGRET_TOLERANCE_MS for value in values),
            "harmed_configs": sum(value < -REGRET_TOLERANCE_MS for value in values),
            "unchanged_configs": sum(abs(value) <= REGRET_TOLERANCE_MS for value in values),
            "mean_gain_ms": statistics.fmean(values) if values else None,
        }

    summaries = {}
    comparisons = {}
    for mode, field in (("forced_measurement", "forced_gain_ms"), ("with_stopping", "gain_with_stopping_ms")):
        summaries[mode] = {policy: gain_summary([row["policies"][policy][field] for row in rows]) for policy in POLICIES}
        comparisons[mode] = {}
        for baseline in ("uncertainty_only", "impact_only"):
            difference = [row["policies"]["srop_proxy"][field] - row["policies"][baseline][field] for row in rows]
            counts = gain_summary(difference)
            comparisons[mode]["srop_proxy_vs_" + baseline] = {
                "better_configs": counts["improved_configs"], "worse_configs": counts["harmed_configs"],
                "tied_configs": counts["unchanged_configs"], "mean_additional_gain_ms": counts["mean_gain_ms"],
            }
    return {
        "format": "fusion-placement-search.e2-first-step-audit", "version": "1.0",
        "experiment_id": report["experiment_id"],
        "scope": "development fixed-candidate replay; first measurement only; repeated configurations are not independent workloads",
        "stop_rule": "maximum acquisition score <= 0: retain initial decision without revealing a label",
        "completed_configs": len(rows),
        "policy_outcomes": summaries, "pairwise_comparisons": comparisons,
        "nonpositive_score_forced_improvements": {
            policy: sum(row["policies"][policy]["stops_before_measurement"] and
                        row["policies"][policy]["forced_gain_ms"] > REGRET_TOLERANCE_MS for row in rows)
            for policy in POLICIES},
        "tasks": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = audit_first_step(json.loads(args.report.read_text(encoding="utf-8")))
    _write_json(args.output, audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
