"""Verify all stop runs and diagnose failed distance intervals in the evaluator."""

import argparse
import json
import math
from pathlib import Path

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator.dataset import plan_from_dict
from simulator.hardware import load_hardware
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_diagnostics import _read_checkpoint, _write_json, workload_for_task
from .e2_srop_proxy import _active_profile, _oracle_profile, ObjectCounterfactualProfile, build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    baseline = json.loads((directory / "e2c_graph_wide_budget_curves.json").read_text(encoding="utf-8"))
    parent = json.loads((directory / "e2c_graph_wide_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    by_key = {row["object_id"]: row for row in records}
    task_by_id = {task["task_id"]: task for task in suite["tasks"]}
    baseline_by_id = {task["task_id"]: task for task in baseline["tasks"]}
    parent_by_id = {task["task_id"]: task for task in parent["tasks"]}
    if (report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs"
            or report["configuration"]["baseline_report_id"] != content_id("uncertainty-stop-baseline", baseline)
            or {task["task_id"] for task in report["tasks"]} != set(task_by_id)):
        raise ValueError("complete unchanged frozen experiment required")
    checkpoint_root = directory / "diagnostics_checkpoints" / report["experiment_id"].split(":")[1][:16]
    candidate_root = directory / "diagnostics_checkpoints" / parent["experiment_id"].split(":")[1][:16]
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    recovery = stopped = 0
    wrong = {name: set() for name in ("legacy", "strict_family", "distance_floor")}
    for row in report["tasks"]:
        expected = baseline_by_id[row["task_id"]]["policies"]["srop_proxy"]
        if row["variants"]["legacy"]["regret_budget_auc_ms"] != expected["modes"]["stopping"]["regret_budget_auc_ms"]:
            raise AssertionError("legacy baseline changed")
        stored = _read_checkpoint(checkpoint_root / (row["task_id"].split(":")[1] + ".json.gz"))
        pool = set(parent_by_id[row["task_id"]]["zero"]["measurement_pool_object_ids"])
        for variant in wrong:
            if row["variants"][variant]["stop_regret_ms"] > 1e-9:
                wrong[variant].add(row["task_id"])
            if variant == "legacy":
                continue
            trajectory = stored["trajectories"][variant]
            if any(point["stop_recommended"] for point in trajectory[:-1]):
                raise AssertionError("revealed labels after the stopping recommendation")
            final = trajectory[-1]
            if not final["stop_recommended"] and set(final["selected_object_ids"]) != pool:
                raise AssertionError("trajectory ended before score stop or pool exhaustion")
            if len(final["selected_object_ids"]) != len(set(final["selected_object_ids"])) or not set(final["selected_object_ids"]) <= pool:
                raise AssertionError("invalid measurement trajectory")
            if not row["variants"][variant]["full_budget_recovers_all_candidates"]:
                raise AssertionError("full-reveal check failed")
            stopped += 1
            recovery += 1

    failures = []
    for row in report["tasks"]:
        if row["task_id"] not in wrong["distance_floor"]:
            continue
        task = task_by_id[row["task_id"]]
        current = row["variants"]["distance_floor"]
        runtime = active_task_contract(task, contract, manifest)
        runtime["active_profiling_protocol"]["uncertainty_variant"] = "distance_floor"
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, task["scale"])
        mapping = object_map_for_workload(records, workload)
        candidate = _read_checkpoint(candidate_root / (row["task_id"].split(":")[1] + ".json.gz"))
        plans = [plan_from_dict(plan) for plan in candidate["candidate_plans"]]
        ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
        oracle = ranker.rank(_oracle_profile(records, hardware, mapping))
        truth = {item.plan_id: item.makespan for item in oracle}
        revealed = current["selected_object_ids"]
        profile, uncertainty = _active_profile(records, runtime, hardware, mapping,
            split_id=task["split_id"], revealed_object_ids=revealed)
        remaining = set(parent_by_id[row["task_id"]]["zero"]["measurement_pool_object_ids"]) - set(revealed)
        observations = []
        for key in sorted(remaining):
            actual = by_key[key]["measurement"]["duration_s"]
            direct = ObjectCounterfactualProfile(profile, key, 0.0, multiplier=actual/profile.durations_s[key])
            direct_regret = max(0.0, truth[ranker.rank(direct)[0].plan_id] - oracle[0].makespan) * 1e3
            updated, _ = _active_profile(records, runtime, hardware, mapping,
                split_id=task["split_id"], revealed_object_ids=[*revealed, key])
            updated_regret = max(0.0, truth[ranker.rank(updated)[0].plan_id] - oracle[0].makespan) * 1e3
            error = abs(math.log(actual/profile.durations_s[key]))
            max_z = max(abs(value) for value in runtime["active_profiling_protocol"]["srop_proxy_standard_normal_quantiles"])
            sample_log_radius = max_z * uncertainty[key] / runtime["active_profiling_protocol"]["uncertainty_q90_normalizer"]
            observations.append({"object_id": key, "operator": by_key[key]["kernel_key"]["operator"],
                "abs_log_error": error, "q90_log": uncertainty[key], "inside_q90": error <= uncertainty[key],
                "inside_sample_envelope": error <= sample_log_radius,
                "target_only_gain_ms": current["stop_regret_ms"] - direct_regret,
                "full_refit_gain_ms": current["stop_regret_ms"] - updated_regret})
        failures.append({key: row[key] for key in ("task_id", "model_id", "scale", "initial_train_count", "subset_seed")} |
            {"old_wrong_stop": row["task_id"] in wrong["legacy"], "stop_budget": current["stop_budget"],
             "stop_regret_ms": current["stop_regret_ms"], "unmeasured_object_analysis": observations})
    output = {"format": "fusion-placement-search.e2-uncertainty-stop-audit", "version": "1.0",
        "experiment_id": report["experiment_id"], "verified_configs": len(report["tasks"]),
        "first_score_stop_checks": stopped, "separate_full_reveal_profile_checks": recovery,
        "legacy_results_preserved": True, "wrong_stop_counts": {key: len(value) for key, value in wrong.items()},
        "distance_fixed_legacy_failures": len(wrong["legacy"] - wrong["distance_floor"]),
        "distance_introduced_failures": len(wrong["distance_floor"] - wrong["legacy"]),
        "scope": "hidden labels used only in evaluator failure diagnosis, never calibration or acquisition",
        "distance_failed_configurations": failures}
    _write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "distance_failed_configurations"}, indent=2))


if __name__ == "__main__":
    main()
