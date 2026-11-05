"""P1: default-SDMA Copy marked unstable; effect of its range on the 48 FFN plans.

Formal communication costs stay proxies (no failed median becomes a Profile).
Observed default-SDMA samples, kept per batch and never pooled with the
SDMA-off control, drive counterfactual scenarios only. For each scenario the
GPU0 Projection scale at which a cross-GPU plan would first win is also
searched, because GPU0-optimal under current compute assumptions does not
prove GPU0 is the true optimum.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics

from cost_model.ffn_decision_sensitivity import FFNSensitivity
from cost_model.profile_prediction_validation import read

BATCHES = {
    "batch01_default_sdma": "data/stage_size_calibration_20261106_01/review/evaluation.json",
    "copy_sdma_default": "data/copy_sdma_default_20261106_01/review/evaluation.json",
}
CONTROL = "data/copy_sdma_disabled_20261106_01/review/evaluation.json"
STATISTICS = ("min", "p25", "median", "mean_including_outliers", "p75", "max")
GPU0_SCALE_SEARCH = [1., 8.]


def describe(samples):
    q1, q2, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    median = statistics.median(samples)
    return {"n": len(samples), "min": min(samples), "p25": q1, "median": median, "p75": q3,
            "max": max(samples), "mean_including_outliers": statistics.fmean(samples),
            "iqr_over_median": (q3 - q1) / median,
            "over_3x_median": sum(v > 3 * median for v in samples),
            "over_1ms": sum(v > 1000 for v in samples)}


def transfer_key(case):
    _, size, direction = case.split("/")
    src, dst = map(int, direction.split("->"))
    return src, dst, float(size)


def communication_status():
    status = {}
    for name, path in BATCHES.items():
        cases = read(path)["cases"]
        status[name] = {c: {"status": "unstable" if q["status"] != "quality_passed" else "quality_passed_single_batch",
                            "quality": q["status"], "samples_us": q["samples_us"], **describe(q["samples_us"])}
                        for c, q in cases.items() if c.startswith("copy/")}
    control = {c: {"quality": q["status"], **describe(q["samples_us"])}
               for c, q in read(CONTROL)["cases"].items()}
    return status, control


class ScenarioProfile:
    def __init__(self, owner, transfers_us, gpu0_projection_scale):
        self.owner, self.transfers_us, self.scale = owner, transfers_us, gpu0_projection_scale

    def __getattr__(self, name):
        return getattr(self.owner.profile, name)

    def estimate_group(self, graph, group):
        cost = self.owner.costs[tuple(group.node_ids), group.device]
        if group.device == 0 and graph.nodes[group.node_ids[0]].op_type == "matmul" and self.scale != 1.:
            s = self.scale
            return replace(cost, duration=cost.duration * s, compute_time=cost.compute_time * s,
                           memory_time=cost.memory_time * s, launch_time=cost.launch_time * s)
        return cost

    def estimate_transfer(self, source, target, nbytes):
        cost = self.owner.transfers[source, target, nbytes]
        value = self.transfers_us.get((source, target, nbytes))
        return cost if value is None else replace(cost, duration=value / 1e6, source="observed_unstable_range_scenario")


def rank(audit, transfers_us, scale=1.):
    profile = ScenarioProfile(audit, transfers_us, scale)
    rows = []
    for plan in audit.saved["plans"]:
        estimate = audit.model.predict_prepared(audit.prepared[plan["plan_id"]], profile)
        rows.append({"plan_id": plan["plan_id"], "fused": plan["fused"], "placement": plan["placement"],
                     "cross_device": estimate.cross_device_bytes > 0, "cost_us": estimate.makespan * 1e6})
    rows.sort(key=lambda r: (r["cost_us"], r["plan_id"]))
    local = next(r for r in rows if not r["cross_device"])
    cross = next(r for r in rows if r["cross_device"])
    return rows, local, cross


def gpu0_projection_break_even(audit, transfers_us):
    """Smallest GPU0 Projection scale in the search interval where a cross plan wins."""
    low, high = GPU0_SCALE_SEARCH
    margin = lambda s: (lambda r: r[2]["cost_us"] - r[1]["cost_us"])(rank(audit, transfers_us, s))
    grid = [low + i * (high - low) / 28 for i in range(29)]
    values = [margin(s) for s in grid]
    first = next((i for i, v in enumerate(values) if v < 0), None)
    if first is None:
        return {"interval": GPU0_SCALE_SEARCH, "switch_bracket": None, "margin_at_ends_us": [values[0], values[-1]]}
    if first == 0:
        return {"interval": GPU0_SCALE_SEARCH, "switch_bracket": [low, low], "margin_at_ends_us": [values[0], values[-1]]}
    a, b = grid[first - 1], grid[first]
    for _ in range(40):
        m = (a + b) / 2
        a, b = (m, b) if margin(m) >= 0 else (a, m)
    _, _, cross = rank(audit, transfers_us, b)
    return {"interval": GPU0_SCALE_SEARCH, "grid_points": len(grid), "switch_bracket": [a, b],
            "winning_cross_plan_at_switch": {k: cross[k] for k in ("plan_id", "fused", "placement", "cost_us")},
            "note": "first sign change on a 29-point grid then bisection; later re-crossings not excluded"}


def build_report():
    audit = FFNSensitivity()
    status, control = communication_status()
    baseline_rows, base_local, base_cross = rank(audit, {})
    scenarios = [{"name": "proxy_baseline", "batch": None, "statistic": None, "transfers_us": {}}]
    for batch, cases in status.items():
        for stat in STATISTICS:
            scenarios.append({"name": f"{batch}:{stat}", "batch": batch, "statistic": stat,
                              "transfers_us": {transfer_key(c): v[stat] for c, v in cases.items()}})
    results = []
    for scenario in scenarios:
        rows, local, cross = rank(audit, scenario["transfers_us"])
        results.append({"name": scenario["name"], "batch": scenario["batch"], "statistic": scenario["statistic"],
            "transfers_us": {f"copy/{int(k[2])}/{k[0]}->{k[1]}": v for k, v in sorted(scenario["transfers_us"].items())},
            "best": {k: rows[0][k] for k in ("plan_id", "fused", "placement", "cross_device", "cost_us")},
            "best_changed_from_proxy": rows[0]["plan_id"] != baseline_rows[0]["plan_id"],
            "best_zero_communication_us": local["cost_us"], "best_cross_device_us": cross["cost_us"],
            "cross_minus_local_us": cross["cost_us"] - local["cost_us"],
            "best_cross_plan": {k: cross[k] for k in ("plan_id", "fused", "placement")},
            "gpu0_projection_break_even": gpu0_projection_break_even(audit, scenario["transfers_us"])})
    return {"scope": "P1 communication instability analysis; counterfactual, not Profile admission",
        "communication_status": {"execution_contract": "default SDMA (FFN default)", "status": "unstable",
            "per_batch": status, "batches_pooled": False,
            "formal_profile_from_failed_medians": False, "formal_cost_source": "analytic_ffn_link_proxy (unchanged)"},
        "sdma_disabled_control": {"cases": control, "used_for_ffn_cost": False},
        "proxy_transfers_us": {f"copy/{int(k[2])}/{k[0]}->{k[1]}": v.duration * 1e6 for k, v in sorted(audit.transfers.items())},
        "scenarios": results,
        "any_scenario_changes_best": any(r["best_changed_from_proxy"] for r in results),
        "compute_held_fixed_except": "GPU0 Projection scale in break-even search",
        "interpretation_limits": ["GPU0-optimal under current compute proxies is not proof of the true optimum",
                                  "GEMM and GPU1 costs remain proxies", "FFN measured reference plans: 0/48"],
        "model_or_mcts_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"any_scenario_changes_best": report["any_scenario_changes_best"]}))


if __name__ == "__main__":
    main()
