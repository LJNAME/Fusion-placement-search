"""Evaluator-only diagnosis of prior wrong stops; never used for acquisition."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
from pathlib import Path

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from simulator.dataset import plan_from_dict
from simulator.hardware import load_hardware
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_diagnostics import _read_checkpoint, _write_json, workload_for_task
from .e2_refit_acquisition import NearestRefitResponse
from .e2_residual import NearestShapePredictor
from .e2_srop_proxy import _active_profile, _oracle_profile, build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract


def feature_only_profile(public, protocol, query_id, baseline):
    """Refit scaling on a query's public features without adding its label/NN anchor."""
    visible = [row for row in public if "target" in row]
    synthetic = copy.deepcopy(next(row for row in public if row["object_id"] == query_id))
    synthetic["target"] = {"log_residual": 0.0}  # fit_mean uses feature rows only.
    predictor = NearestShapePredictor().fit_mean([*visible, synthetic], protocol)
    predictor.records = visible
    predictor.matrix = predictor.encoder.transform(visible)
    updated = copy.copy(baseline)
    updated.durations_s = dict(baseline.durations_s)
    updated._group_costs = {}
    for row in public:
        if "target" not in row:
            updated.durations_s[row["object_id"]] = row["analytic"]["estimate_s"] * math.exp(predictor._nearest(row))
    return updated


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs":
        raise ValueError("completed full frozen refit experiment required")
    parent = json.loads((directory / "e2c_graph_wide_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    records_by_id = {row["object_id"]: row for row in records}
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    task_by_id = {task["task_id"]: task for task in suite["tasks"]}
    parent_by_id = {task["task_id"]: task for task in parent["tasks"]}
    candidate_root = directory / "diagnostics_checkpoints" / parent["experiment_id"].split(":")[1][:16]
    refit_root = directory / "diagnostics_checkpoints" / report["experiment_id"].split(":")[1][:16]
    refit_by_id = {task["task_id"]: task for task in report["tasks"]}
    rows = []
    for failed in report["previous_wrong_stops"]:
        task = task_by_id[failed["task_id"]]
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, task["scale"])
        runtime = active_task_contract(task, contract, manifest)
        mapping = object_map_for_workload(records, workload)
        profile, uncertainty = _active_profile(records, runtime, hardware, mapping,
            split_id=task["split_id"], revealed_object_ids=())
        public = make_dataset_view(records, runtime, split_id=task["split_id"], mode="active_calibration", role="sampler")
        stored = _read_checkpoint(candidate_root / (task["task_id"].split(":")[1] + ".json.gz"))
        ranker = BatchedPlacementRanker(graph, [plan_from_dict(plan) for plan in stored["candidate_plans"]], 2, ((0, 1), (1, 0)))
        current_top1 = ranker.rank(profile)[0].plan_id
        oracle_ranked = ranker.rank(_oracle_profile(records, hardware, mapping))
        oracle_costs = {item.plan_id: item.makespan for item in oracle_ranked}
        helpful = []
        for observation in parent_by_id[task["task_id"]]["one_step"]["observations"]:
            if observation["retrained_gain_ms"] <= 1e-9:
                continue
            key = observation["object_id"]
            response = NearestRefitResponse(public, runtime["prediction_protocol"], key)
            samples = []
            for quantile in runtime["active_profiling_protocol"]["srop_proxy_standard_normal_quantiles"]:
                value = profile.durations_s[key] * math.exp(quantile * uncertainty[key] /
                    runtime["active_profiling_protocol"]["uncertainty_q90_normalizer"])
                ranked = ranker.rank(response.profile(profile, value))
                current = next(item.makespan for item in ranked if item.plan_id == current_top1)
                samples.append({"hypothetical_duration_s": value,
                                "decision_loss_ms": max(0.0, current - ranked[0].makespan) * 1e3})
            true_s = observation["measured_duration_s"]
            feature_top1 = ranker.rank(feature_only_profile(public, runtime["prediction_protocol"], key, profile))[0].plan_id
            feature_regret = max(0.0, oracle_costs[feature_top1] - oracle_ranked[0].makespan) * 1e3
            helpful.append({key: observation[key] for key in ("object_id", "operator", "shape", "predicted_duration_s",
                "measured_duration_s", "actual_abs_log_error", "uncertainty_q90_log", "inside_q90_interval",
                "target_only_gain_ms", "retrained_gain_ms")} | {
                "public_sample_envelope_contains_actual": min(row["hypothetical_duration_s"] for row in samples) <= true_s <=
                                                          max(row["hypothetical_duration_s"] for row in samples),
                "hypothetical_refit_samples": samples,
                "feature_only_control_regret_ms": feature_regret,
                "indirectly_affected_object_ids": list(response.affected_object_ids)})
        policy = refit_by_id[task["task_id"]]["policies"]["srop_refit"]
        stop = policy["traces"][0]["stop_budget"]
        selected = policy["traces"][0]["selected_object_ids"]
        revealed = selected[:stop]
        stop_profile, stop_uncertainty = _active_profile(records, runtime, hardware, mapping,
            split_id=task["split_id"], revealed_object_ids=revealed)
        stop_public = make_dataset_view(records, runtime, split_id=task["split_id"],
            mode="active_calibration", role="sampler", revealed_object_ids=revealed)
        stop_top1 = ranker.rank(stop_profile)[0].plan_id
        remaining = [key for key in selected if key not in revealed]
        helpful_at_stop = []
        for key in remaining:
            updated, _ = _active_profile(records, runtime, hardware, mapping,
                split_id=task["split_id"], revealed_object_ids=[*revealed, key])
            decision = ranker.rank(updated)[0].plan_id
            regret = max(0.0, oracle_costs[decision] - oracle_ranked[0].makespan) * 1e3
            if failed["new_stop_regret_ms"] - regret > 1e-9:
                true_s = records_by_id[key]["measurement"]["duration_s"]
                error = abs(math.log(true_s / stop_profile.durations_s[key]))
                response = NearestRefitResponse(stop_public, runtime["prediction_protocol"], key)
                sample_losses = []
                for quantile in runtime["active_profiling_protocol"]["srop_proxy_standard_normal_quantiles"]:
                    value = stop_profile.durations_s[key] * math.exp(quantile * stop_uncertainty[key] /
                        runtime["active_profiling_protocol"]["uncertainty_q90_normalizer"])
                    ranked = ranker.rank(response.profile(stop_profile, value))
                    current = next(item.makespan for item in ranked if item.plan_id == stop_top1)
                    sample_losses.append(max(0.0, current - ranked[0].makespan) * 1e3)
                helpful_at_stop.append({"object_id": key, "operator": records_by_id[key]["kernel_key"]["operator"],
                    "regret_after_measurement_ms": regret, "predicted_duration_s": stop_profile.durations_s[key],
                    "measured_duration_s": true_s, "actual_abs_log_error": error,
                    "uncertainty_q90_log": stop_uncertainty[key], "inside_q90_interval": error <= stop_uncertainty[key],
                    "hypothetical_refit_losses_ms": sample_losses})
        with gzip.open(refit_root / (task["task_id"].split(":")[1] + ".json.gz"), "rt", encoding="utf-8") as handle:
            trajectory = json.load(handle)["raw"]["policies"]["srop_refit"]["full_trajectories"][0]
        rows.append({**failed, "helpful_measurement_analysis": helpful,
            "revealed_before_new_stop": revealed,
            "original_helpful_key_state_at_new_stop": [{"object_id": item["object_id"],
                "predicted_duration_s": stop_profile.durations_s[item["object_id"]],
                "uncertainty_q90_log": stop_uncertainty.get(item["object_id"])} for item in helpful],
            "improving_single_measurements_at_new_stop": helpful_at_stop,
            "next_two_forced_object_ids": selected[stop:stop+2],
            "forced_regret_after_one_more_ms": trajectory[min(stop+1, len(trajectory)-1)]["top1_regret_ms"],
            "forced_regret_after_two_more_ms": trajectory[min(stop+2, len(trajectory)-1)]["top1_regret_ms"]})
    # Full-budget checks include all tasks; the failure analysis above is a
    # post-hoc diagnostic of the four previous failures, not a selected test set.
    baseline = json.loads((directory / "e2c_graph_wide_budget_curves.json").read_text(encoding="utf-8"))
    baseline_by_id = {task["task_id"]: task for task in baseline["tasks"]}
    for task in report["tasks"]:
        for name, policy in baseline_by_id[task["task_id"]]["policies"].items():
            if task["policies"][name] != policy:
                raise AssertionError("baseline policy changed")
        trace = task["policies"]["srop_refit"]["traces"][0]
        if trace["full_budget_regret_ms"] > 1e-9 or trace["full_budget_strict_top_k_recall"] != 1.0:
            raise AssertionError("full budget failed to recover Oracle")
    wrong = [task for task in report["tasks"] if task["policies"]["srop_refit"]["traces"][0]["stop_regret_ms"] > 1e-9]
    output = {"format": "fusion-placement-search.e2-refit-stop-audit", "version": "1.0",
        "experiment_id": report["experiment_id"], "scope": "evaluator-only post-hoc failure analysis; hidden outcomes do not enter acquisition",
        "configs": len(report["tasks"]), "unchanged_baseline_policy_checks": len(report["tasks"]) * 4,
        "full_budget_recovery_trajectories": len(report["tasks"]), "new_wrong_stop_configs": len(wrong),
        "previous_wrong_stop_analysis": rows}
    _write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "previous_wrong_stop_analysis"}, indent=2))
    for row in rows:
        print(row["model_id"], row["scale"], "new_stop", row["new_stop_budget"], row["new_stop_regret_ms"],
              [(item["operator"], item["inside_q90_interval"], [round(sample["decision_loss_ms"], 6) for sample in item["hypothetical_refit_samples"]])
               for item in row["helpful_measurement_analysis"]])
        print("helpful singles at stop:", len(row["improving_single_measurements_at_new_stop"]),
              "forced next-two regret:", row["forced_regret_after_one_more_ms"], row["forced_regret_after_two_more_ms"])


if __name__ == "__main__":
    main()
