"""CPU break-even analysis over the fixed 48-plan FFN and scheduling policy."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cost_model import ffn_profile_search as ffn
from cost_model.fusion_kernel_profile import key_id
from cost_model.profile_prediction_validation import read
from integration.lightop.silu_mul import adapter_registry
from simulator import FusionGroup, Plan


MODES = ("all_communication", "communication_latency", "communication_byte_service",
         "gpu1_projection", "gpu1_activation")


class PerturbedProfile:
    def __init__(self, owner, mode, scale):
        self.owner, self.mode, self.scale = owner, mode, scale

    def __getattr__(self, name):
        return getattr(self.owner.profile, name)

    def estimate_group(self, graph, group):
        cost = self.owner.costs[tuple(group.node_ids), group.device]
        changed = group.device == 1 and (
            self.mode == "gpu1_projection" and graph.nodes[group.node_ids[0]].op_type == "matmul"
            or self.mode == "gpu1_activation" and graph.nodes[group.node_ids[0]].op_type != "matmul")
        if not changed:
            return cost
        return replace(cost, duration=cost.duration * self.scale, compute_time=cost.compute_time * self.scale,
                       memory_time=cost.memory_time * self.scale, launch_time=cost.launch_time * self.scale)

    def estimate_transfer(self, source, target, nbytes):
        cost = self.owner.transfers[source, target, nbytes]
        latency = self.owner.hardware.comm_latency
        if self.mode == "all_communication":
            duration = cost.duration * self.scale
        elif self.mode == "communication_latency":
            duration = latency * self.scale + cost.duration - latency
        elif self.mode == "communication_byte_service":
            duration = latency + (cost.duration - latency) * self.scale
        else:
            duration = cost.duration
        return replace(cost, duration=duration)


class FFNSensitivity:
    def __init__(self):
        config = read(ffn.CONFIG)
        _, self.graph, self.hardware, measured, self.identity = ffn.build_case(config)
        base = ffn.FFNProxyBase(self.hardware, measured, config, config["scenarios"][0])
        self.profile = ffn.NativeMeasuredFirstProfile(self.graph, base, measured.store, measured.bindings)
        if self.profile.interference != 0:
            raise ValueError("monotonic one-axis audit requires zero compute/send interference")
        self.model = ffn.StructuralCostModel(registry=adapter_registry(), scheduling_policy=ffn.SCHEDULING_POLICY)
        self.saved = read("data/ffn_schedule_audit_v1/report.json")["scenario_results"][0]
        self.prepared, self.costs, self.transfers = {}, {}, {}
        for row in self.saved["plans"]:
            plan = Plan(groups=[FusionGroup(g["nodes"], g["device"], {"pattern_id": g["pattern"]})
                                for g in row["scoring_groups"]])
            prepared = self.model.prepare(self.graph, plan, self.profile.n_devs)
            self.prepared[row["plan_id"]] = prepared
            for group in prepared.structure.groups:
                self.costs[tuple(group.node_ids), group.device] = self.profile.estimate_group(self.graph, group)
            for dep in prepared.structure.dependencies:
                if dep.cross_device:
                    src = prepared.structure.groups[dep.src_group].device
                    dst = prepared.structure.groups[dep.dst_group].device
                    self.transfers[src, dst, dep.bytes] = self.profile.estimate_transfer(src, dst, dep.bytes)
        self.cache = {}

    def evaluate(self, mode, scale):
        if mode not in MODES or not 0 <= scale <= 1:
            raise ValueError("audit modes use the explicit exploration interval [0,1]")
        cache_key = mode, scale
        if cache_key not in self.cache:
            profile = PerturbedProfile(self, mode, scale)
            rows = []
            for original in self.saved["plans"]:
                estimate = self.model.predict_prepared(self.prepared[original["plan_id"]], profile)
                rows.append({"plan_id": original["plan_id"], "fused": original["fused"],
                    "placement": original["placement"], "cross_device": estimate.cross_device_bytes > 0,
                    "cost_us": estimate.makespan * 1e6,
                    "computation_sum_us_not_additive_makespan": estimate.computation_time * 1e6,
                    "communication_sum_us_not_additive_makespan": estimate.communication_time * 1e6,
                    "estimate": estimate.to_dict()})
            self.cache[cache_key] = sorted(rows, key=lambda row: (row["cost_us"], row["plan_id"]))
        return self.cache[cache_key]

    def margin(self, mode, scale):
        rows = self.evaluate(mode, scale)
        cross = next(r for r in rows if r["cross_device"])
        local = next(r for r in rows if not r["cross_device"])
        return cross["cost_us"] - local["cost_us"]

    def break_even(self, mode):
        low, high = 0., 1.
        endpoints = {str(s): self.margin(mode, s) for s in (low, high)}
        if endpoints["0.0"] >= 0 or endpoints["1.0"] <= 0:
            return {"mode": mode, "exploration_interval_not_confidence_bounds": [0, 1],
                "cross_minus_best_zero_communication_us_at_endpoints": endpoints,
                "switch_bracket": None, "reason": "no strict cross/local switch inside this one-axis interval"}
        # All changed costs are nonnegative and nondecreasing in scale. With
        # fixed priority and zero interference, max-plus resource recurrences
        # make every plan nondecreasing. Local GPU0 plans stay constant.
        for _ in range(42):
            middle = (low + high) / 2
            if self.margin(mode, middle) < 0:
                low = middle
            else:
                high = middle
        midpoint = (low + high) / 2
        witnesses = {"below": max(0., midpoint - 1e-4), "above": min(1., midpoint + 1e-4)}
        return {"mode": mode, "exploration_interval_not_confidence_bounds": [0, 1],
            "cross_minus_best_zero_communication_us_at_endpoints": endpoints,
            "switch_bracket": [low, high], "bisection_iterations": 42,
            "signed_margin_us_at_bracket": [self.margin(mode, low), self.margin(mode, high)],
            "witnesses": {name: {"scale": s, "best": self.evaluate(mode, s)[0],
                                  "margin_us": self.margin(mode, s)} for name, s in witnesses.items()},
            "transfer_costs_us_at_midpoint": [
                {"source": src, "destination": dst, "bytes": size,
                 "duration_us": PerturbedProfile(self, mode, midpoint).estimate_transfer(src, dst, size).duration * 1e6}
                for src, dst, size in sorted(self.transfers)]}


def build_report():
    audit = FFNSensitivity()
    baseline = audit.evaluate("all_communication", 1.)
    saved = {p["plan_id"]: p for p in audit.saved["plans"]}
    if len(baseline) != 48 or any(abs(p["cost_us"] - saved[p["plan_id"]]["cost_us"]) > 1e-9 for p in baseline):
        raise ValueError("P2 baseline differs from the frozen scheduling audit")
    thresholds = [audit.break_even(mode) for mode in MODES]
    old_links = read("data/bw1000_2card/raw_measurements.json")
    review = read("data/ffn_b4_run05_review/report.json")
    local_reference = read("data/lightop_local_pair_20261104_01/review/evaluation.json")
    copy_rows = [{k: row.get(k) for k in ("row_id", "grid_status", "admitted", "median_event_us_per_call")}
                 for row in review["formal_rows"] if row["row_id"].startswith("copy/")]
    return {"scope": "P2 fixed FFN CPU counterfactuals; not measured speedup or reference regret",
        "scheduling_policy": ffn.SCHEDULING_POLICY, "baseline_plans": baseline,
        "sensitivity_axes": thresholds,
        "communication_probe_grid": [{"scale": s, "best": audit.evaluate("all_communication", s)[0]}
                                     for s in (0., .1, .2, .3, .5, 1.)],
        "comparison_semantics": "recompute resource-constrained makespan; component sums are diagnostics, not an additive formula",
        "other_costs_held_fixed": True, "source_queries_and_profiles_unchanged": True,
        "not_swept": ["GPU0 projection cost increases", "simultaneous parameter changes",
                      "chain composition residuals and runtime scheduling mismatch"],
        "existing_local_activation_reference": {
            "source": "data/lightop_local_pair_20261104_01/review/evaluation.json",
            "reference_cost_us": local_reference["reference_cost_us"],
            "role": "historical GPU0 local development evidence; not complete FFN or fresh holdout",
            "reference_supports_native_local_choice": local_reference["reference_cost_us"]["native"]
                < local_reference["reference_cost_us"]["fused"]},
        "communication_evidence": {
            "supported_current_graph_scale_interval": None, "ranking_robustness_under_credible_bounds": "unresolved",
            "historical_host_wall": {"source": "data/bw1000_2card/raw_measurements.json",
                "HIP": old_links["environment"]["torch_hip"], "directions": len(old_links["links"]),
                "measured_sizes_bytes": sorted({p["size_bytes"] for link in old_links["links"] for p in link["points"]}),
                "reason_not_bounds": "different timing boundary/runtime; parameters also generate this baseline, not independent bounds"},
            "historical_graph_copy_review": {"source": "data/ffn_b4_run05_review/report.json", "rows": copy_rows,
                "reason_not_bounds": "different frozen recipe/window contract; two failed quality, none admitted as current search costs"}},
        "minimum_conditional_reference": {
            "purpose": "resolve zero-communication GPU0 Native versus the Up-on-GPU1 alternative only if needed",
            "communication": [{"source": 0, "destination": 1, "bytes": 524288},
                              {"source": 1, "destination": 0, "bytes": 1572864}],
            "remaining_compute_uncertainty": "GPU0/GPU1 Up Projection and schedule composition remain proxies",
            "automatic_gpu_collection": False},
        "profile_only_plans": 0, "independent_reference_regret": None,
        "srop_enabled": False, "new_gpu_execution": False,
        "baseline_report_sha256": hashlib.sha256((ffn.ROOT / "data/ffn_schedule_audit_v1/report.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in a.items() if k not in ("witnesses", "transfer_costs_us_at_midpoint")}
                      for a in report["sensitivity_axes"]], indent=2))


if __name__ == "__main__":
    main()
