"""Descriptive pilot reports; no rate estimates, causal verdicts, or cost labels."""
import json
from pathlib import Path
import statistics

SCOPE = "pilot_only_not_profile_admission"
REPORT_RULES = {
    "version": "ffn-anchor-report.v2", "scope": SCOPE, "primary_readout": "Down0",
    "independent_runs_per_workload_strategy": 1,
    "crossing_rate_estimate": None,
    "forbidden_claims": ["R3 reduces crossing rate", "R3 is more stable", "Down-specific cause proved",
                         "run05 shared-window cause proved", "single-sample bias proved"],
    "noise_exceedance_criterion": None,
    "protocol_adoption": "manual review required; no automatic R3 adoption",
    "pilot_pass": "four pipelines complete, identity/contract/path verified, anchors/timestamps and all endpoint readout fields present",
    "quality_gate": "separate result; preserve the frozen R3 thresholds and all R1 failures",
}


def endpoint_readouts(rounds, repeats):
    rows = []
    for row in rounds:
        for phase in ("pre", "post"):
            samples = row[phase]
            if len(samples) != repeats or repeats not in (1, 3):
                raise ValueError("endpoint readout count differs from protocol")
            values = [s["batch_ms"] * 1000 / 50 for s in samples]
            quartiles = statistics.quantiles(values, n=4, method="inclusive") if repeats == 3 else None
            rows.append({"attempt": row["attempt"], "phase": phase,
                "R1_single_us": values[0] if repeats == 1 else None,
                "R3_median_us": statistics.median(values) if repeats == 3 else None,
                "R3_IQR_us": quartiles[2] - quartiles[0] if quartiles else None,
                "IQR_status": "estimated_within_endpoint" if quartiles else "not_estimable_single_replay",
                "missing_strategy_fields": "not_applicable", "raw_us": values})
    return rows


def build_report(directory, records):
    directory = Path(directory)
    endpoints, details = {}, {}
    for cell_id, record in records.items():
        if not record.get("pipeline_complete"):
            continue
        result = json.loads((directory / cell_id / "result.json").read_text(encoding="utf-8"))
        rounds = json.loads((directory / cell_id / "samples.json").read_text(encoding="utf-8"))
        endpoints[cell_id] = endpoint_readouts(rounds, result["cell"]["endpoint_replays"])
        details[cell_id] = result["quality"]
    down = {name: records.get(name, {}).get("status", "not_run") for name in ("down_r1", "down_r3")}
    if all(s == "quality_passed" for s in down.values()):
        interpretation = "Failure not reproduced under this single-device pilot; run05 cause remains unresolved."
    elif any(s == "quality_failed" for s in down.values()):
        interpretation = "Quality failure reproduced under pilot conditions; no Down-specific cause established."
    else:
        interpretation = "Down0 evidence incomplete; no quality or causal conclusion."
    comparisons = {}
    for workload in ("down", "up"):
        one, three = workload + "_r1", workload + "_r3"
        if one in details and three in details:
            r1, r3 = details[one], details[three]
            comparisons[workload] = {"target_R1_median_us": r1["target_median_us"],
                "target_R3_median_us": r3["target_median_us"],
                "target_R3_minus_R1_us": r3["target_median_us"] - r1["target_median_us"],
                "target_R1_iqr_over_median": r1["target_iqr_over_median"],
                "target_R3_iqr_over_median": r3["target_iqr_over_median"],
                "anchor_endpoints": {phase: {
                    "R1_median_of_seven_single_readings_us": statistics.median(r["R1_single_us"] for r in endpoints[one] if r["phase"] == phase),
                    "R3_median_of_seven_endpoint_medians_us": statistics.median(r["R3_median_us"] for r in endpoints[three] if r["phase"] == phase),
                } for phase in ("pre", "post")},
                "noise_exceedance": "not_assessed_no_preregistered_criterion",
                "interpretation": "descriptive comparison of two ordered runs; not a bias or rate estimate"}
    return {"primary_readout": {"workload": "Down0", "cells": down, "interpretation": interpretation},
            "scope": SCOPE, "rules": REPORT_RULES, "cells": records,
            "endpoint_readouts": endpoints, "cell_quality": details, "comparisons": comparisons,
            "protocol_review_required": True, "r3_adopted": False,
            "admission_eligible": False, "followup_matrix_open": False}


def render_report(report):
    down = report["primary_readout"]
    lines = ["PRIMARY READOUT: Down0 | " + str(down["cells"]), down["interpretation"],
             "", "Scope: " + SCOPE + "; these numbers MUST NOT enter Profile admission.",
             "One independent run per workload/strategy. Crossing rates and R3 stability improvements are not estimable.",
             "Noise exceedance is not assessed; protocol review required, R3 not adopted, seven-item matrix closed.", "",
             "| Cell | Status | Quality reasons | Scope |", "|---|---|---|---|"]
    for name, row in report["cells"].items():
        reasons = report["cell_quality"].get(name, {}).get("cell_reasons", [])
        lines.append(f"| {name} | {row['status']} | {reasons} | {SCOPE} |")
    lines += ["", "| Workload | Target R1 median (us) | Target R3 median (us) | R3 minus R1 (us) | Scope |",
              "|---|---|---|---|---|"]
    for name, row in report["comparisons"].items():
        lines.append(f"| {name} | {row['target_R1_median_us']} | {row['target_R3_median_us']} | {row['target_R3_minus_R1_us']} | {SCOPE} |")
    lines += ["", "| Cell / attempt / endpoint | R1 single (us) | R3 median (us) | R3 IQR (us) | IQR status | Round reasons | Scope |",
              "|---|---|---|---|---|---|---|"]
    for name, readings in report["endpoint_readouts"].items():
        for row in readings:
            fields = [row[k] if row[k] is not None else "not_applicable" for k in ("R1_single_us", "R3_median_us", "R3_IQR_us")]
            reasons = report["cell_quality"][name]["rounds"][row["attempt"]]["invalid_reasons"]
            lines.append(f"| {name}/{row['attempt']}/{row['phase']} | {fields[0]} | {fields[1]} | {fields[2]} | {row['IQR_status']} | {reasons} | {SCOPE} |")
    return "\n".join(lines) + "\n"
