"""Second step after holdout validation: rescore the fixed 48 FFN plans.

GPU0 SiLU/Mul take same-batch S=256 measurements; quality-passed Copies
replace the four link proxies. GPU1 activation/Fused, GEMM and the GPU0
Fused Profile are unchanged. An unchanged best plan is not accuracy evidence.
"""
import argparse
import copy
import json
from pathlib import Path

from calibration.stage_size_calibration import read
from cost_model.ffn_decision_sensitivity import FFNSensitivity
from cost_model.hardware_profile import GroupCost

SILU, MUL = ("L000.mlp.silu",), ("L000.mlp.gated_mul",)
OLD_CRITICAL_US = {(0, 1, 524288): 13.99139, (1, 0, 1572864): 15.41639}


def summary(audit):
    rows = audit.evaluate("all_communication", 1.)
    best = rows[0]
    local = next(r for r in rows if not r["cross_device"])
    cross = next(r for r in rows if r["cross_device"])
    return {"best": {k: best[k] for k in ("plan_id", "fused", "placement", "cross_device", "cost_us")},
            "best_zero_communication_us": local["cost_us"], "best_cross_device_us": cross["cost_us"],
            "cross_minus_local_us": cross["cost_us"] - local["cost_us"],
            "ranking": [{k: r[k] for k in ("plan_id", "cost_us", "cross_device")} for r in rows]}


def apply(audit, stage_us, copy_us):
    sources = {}
    for (nodes, device), cost in list(audit.costs.items()):
        if device == 0 and nodes in (SILU, MUL) and stage_us.get(nodes) is not None:
            value = stage_us[nodes] / 1e6
            audit.costs[nodes, device] = GroupCost(value, value, 0., 0., .95, "measured_stage_size_batch",
                                                   ("measurement_context:isolated_stage",))
        sources["+".join(nodes) + f"@gpu{device}"] = audit.costs[nodes, device].source
    for key, cost in list(audit.transfers.items()):
        src, dst, size = key
        value = copy_us.get((src, dst, int(size)))
        if value is not None:
            audit.transfers[key] = type(cost)(value / 1e6, .95, "measured_graph_copy_batch",
                                              ("route:peer_capability_not_certified",))
        sources[f"copy/{int(size)}/{src}->{dst}"] = audit.transfers[key].source
    audit.cache.clear()
    return sources


def build_report(validation):
    if validation["linear_fit"]["metrics"]["coverage"].startswith("0/"):
        raise ValueError("Holdout prediction validation must exist before the search check")
    baseline = FFNSensitivity()
    before = summary(baseline)
    before_threshold = baseline.break_even("all_communication")
    medians = validation["stage_medians_us"]
    # Same-batch exact S=256 medians; the fitted line is not used where exact exists.
    stage_us = {SILU: medians.get("silu/S256"), MUL: medians.get("mul/S256")}
    copy_us = {}
    for case, info in validation["copies"].items():
        _, size, direction = case.split("/")
        src, dst = map(int, direction.split("->"))
        if info["status"] == "quality_passed":
            copy_us[src, dst, int(size)] = info["median_us"]
    calibrated = FFNSensitivity()
    sources = apply(calibrated, stage_us, copy_us)
    after = summary(calibrated)
    threshold = calibrated.break_even("all_communication")
    return {"scope": "search check after prediction validation; not a measured FFN reference",
        "order": ["prediction accuracy (validation.json)", "this 48-plan rescoring"],
        "inputs": {"gpu0_stage_us": {"silu": stage_us[SILU], "mul": stage_us[MUL]},
                   "measured_copy_us": {f"copy/{k[2]}/{k[0]}->{k[1]}": v for k, v in sorted(copy_us.items())},
                   "copies_left_as_proxy": sorted(c for c, i in validation["copies"].items()
                                                  if i["status"] != "quality_passed")},
        "cost_sources_after": sources,
        "before": {k: v for k, v in before.items() if k != "ranking"},
        "after": {k: v for k, v in after.items() if k != "ranking"},
        "best_plan_changed": before["best"]["plan_id"] != after["best"]["plan_id"],
        "ranking_after": after["ranking"],
        "communication_threshold": {
            "before_all_communication_scale": before_threshold.get("switch_bracket"),
            "after_all_communication_scale_relative_to_measured": threshold.get("switch_bracket"),
            "after_endpoints": threshold["cross_minus_best_zero_communication_us_at_endpoints"],
            "old_critical_transfer_us": {f"copy/{k[2]}/{k[0]}->{k[1]}": v for k, v in OLD_CRITICAL_US.items()},
            "measured_vs_old_critical": {f"copy/{k[2]}/{k[0]}->{k[1]}": (copy_us.get(k), (copy_us[k] > v) if k in copy_us else None)
                                         for k, v in OLD_CRITICAL_US.items()}},
        "unchanged_proxies": ["GPU0/GPU1 GEMM", "GPU1 SiLU/Mul/Fused", "GPU0 Fused exact Profile kept"],
        "best_unchanged_is_not_accuracy_evidence": True, "measured_ffn_reference_plans": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(read(args.validation))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("before", "after", "best_plan_changed")}, indent=2))


if __name__ == "__main__":
    main()
