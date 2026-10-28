"""Audit received summaries; post-hoc rescoring is diagnostic, never zero-shot."""

import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.evaluate_gpu_holdout import prediction_metrics
from calibration.gpu_holdout import load_freeze
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_residual import _spearman
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def main():
    base = Path(__file__).resolve().parent
    freeze, manifest = load_freeze(base.parent)
    zero = json.loads((base / "zero_shot_report.json").read_text(encoding="utf-8"))
    execution = json.loads((base / "execution_report.json").read_text(encoding="utf-8"))
    audit = json.loads((base.parent / "feature_overlap_audit.json").read_text(encoding="utf-8"))
    assert zero["freeze_id"] == execution["freeze_id"] == freeze["freeze_id"]
    objects = {row["object_id"]: row for row in zero["objects"]}
    assert len(objects) == len(zero["objects"]) == 56
    assert set(objects) == set(freeze["zero_shot_predictions"])
    assert not set(objects).intersection(freeze["train_object_ids"])
    for key, row in objects.items():
        for field, value in freeze["zero_shot_predictions"][key].items():
            assert row[field] == value
        assert row["inside_q90"] == (abs(math.log(row["measured_s"] / row["nearest_s"])) <= row["q90_log"])
    recomputed = prediction_metrics(list(objects.values()))
    for method in recomputed:
        for key, value in recomputed[method].items():
            assert math.isclose(value, zero["all"][method][key], rel_tol=1e-12, abs_tol=1e-12)
    alias_ids = {row["object_id"] for row in audit["feature_identical_to_training"]}
    features = {row["object_id"]: row for row in freeze["query_features"]}
    by_id = {row["model_id"]: row for row in execution["workloads"]}
    assert set(by_id) == {row["model_id"] for row in freeze["workloads"]}
    hardware = hardware_from_dict(freeze["hardware"])
    durations = {key: row["measured_s"] for key, row in objects.items()}
    workloads = []
    for sealed in freeze["workloads"]:
        result = by_id[sealed["model_id"]]
        candidates = {row["plan_id"]: row for row in result["candidates"]}
        assert len(candidates) == len(result["candidates"])
        assert set(candidates) == set(sealed["execution_plan_ids"])
        predicted = {row["plan_id"]: row["makespan_s"] for row in sealed["predicted_ranking"]}
        assert result["predicted_top1_id"] == sealed["predicted_top1_id"]
        assert all(row["predicted_s"] == predicted[key] for key, row in candidates.items())
        selected = candidates[result["predicted_top1_id"]]
        reference = candidates[sealed["reference_plan_id"]]
        best = min(candidates, key=lambda key: candidates[key]["median_s"])
        assert result["measured_subset_best_id"] == best
        regret = selected["median_s"] / candidates[best]["median_s"] - 1
        assert math.isclose(regret, result["top1_relative_regret"], abs_tol=1e-12)
        assert math.isclose(reference["median_s"] / selected["median_s"], result["selected_speedup_over_single_device"], abs_tol=1e-12)
        spearman = _spearman([row["predicted_s"] for row in candidates.values()], [row["median_s"] for row in candidates.values()])
        assert math.isclose(spearman, result["ranking_spearman"], abs_tol=1e-12)
        # New labels are used ONLY below for a fixed-subset, post-hoc diagnosis.
        workload = E2Workload(sealed["model_id"], "post-hoc-diagnostic", sealed["model_id"], sealed["model_family"], sealed["sequence_length"])
        mapping = object_map_for_workload(freeze["query_features"], workload)
        graph = graph_from_dict(sealed["graph"])
        plans = [plan_from_dict(sealed["candidate_plans"][key]) for key in candidates]
        profile = DatasetLatencyProfile(hardware, mapping, durations, {key: "oracle_exact" for key in durations})
        reranked = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0))).rank(profile)
        measured_micro_cost = {row.plan_id: row.makespan for row in reranked}
        post_selected = candidates[reranked[0].plan_id]
        plan_rows = []
        for key, row in candidates.items():
            plan = plan_from_dict(sealed["candidate_plans"][key])
            placement = {node: group.device for group in plan.groups for node in group.node_ids}
            cross_edges = [edge for edge in sealed["graph"]["edges"] if placement[edge["src"]] != placement[edge["dst"]]]
            plan_rows.append({**row, "is_reference": key == sealed["reference_plan_id"],
                "post_hoc_measured_micro_s": measured_micro_cost[key],
                "cross_device_edges": len(cross_edges),
                "nodes_by_device": {str(device): sum(value == device for value in placement.values()) for device in (0, 1)},
                "max_to_median_sample_ratio": row["max_s"] / row["median_s"]})
        workloads.append({"model_id": sealed["model_id"], "reference_is_measured_subset_best": best == sealed["reference_plan_id"],
            "predicted_top1_ms": selected["predicted_s"] * 1000,
            "selected_actual_ms": selected["median_s"] * 1000,
            "single_gpu_actual_ms": reference["median_s"] * 1000,
            "selected_regret_pct": regret * 100,
            "selected_prediction_underestimate_pct_of_actual": (1 - selected["predicted_s"] / selected["median_s"]) * 100,
            "ranking_spearman": spearman,
            "post_hoc_measured_micro_selected_id": reranked[0].plan_id,
            "post_hoc_measured_micro_selected_regret_pct": (post_selected["median_s"] / candidates[best]["median_s"] - 1) * 100,
            "post_hoc_measured_micro_ranking_spearman": _spearman([measured_micro_cost[key] for key in candidates], [row["median_s"] for row in candidates.values()]),
            "candidates": plan_rows})
    worst = sorted(objects.values(), key=lambda row: abs(row["measured_s"] - row["nearest_s"]) / row["measured_s"], reverse=True)[:8]
    output = {"freeze_id": freeze["freeze_id"], "summary_consistency_checks_pass": True,
        "provenance_limit": "Only the two supplied summary reports were received; raw journals and environments for kernel collection were not independently audited.",
        "input_sha256": {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ("zero_shot_report.json", "execution_report.json")},
        "all": recomputed,
        "feature_distinct_52": prediction_metrics([row for key, row in objects.items() if key not in alias_ids]),
        "feature_alias_4": prediction_metrics([row for key, row in objects.items() if key in alias_ids]),
        "coverage_count": sum(row["inside_q90"] for row in objects.values()),
        "replica_ratio_max_min": {"median": statistics.median(row["replica_ratio_max_min"] for row in objects.values()), "max": max(row["replica_ratio_max_min"] for row in objects.values())},
        "worst_nearest_objects": [{**row, "ape": abs(row["measured_s"]-row["nearest_s"])/row["measured_s"], "shape": features[row["object_id"]]["kernel_key"]["shape"]} for row in worst],
        "post_hoc_scope": "Only existing measured candidates rescored with newly observed microkernel medians. No candidate discovery, refitting, threshold tuning, or changes to the frozen predictions. Exact microkernel labels are not a whole-plan runtime oracle.",
        "workloads": workloads}
    (base/"analysis.json").write_bytes((json.dumps(output, indent=2, ensure_ascii=True)+"\n").encode("utf-8"))
    print(json.dumps({key: value for key, value in output.items() if key not in ("worst_nearest_objects", "workloads")}, indent=2))
    for row in workloads:
        print(json.dumps({key: value for key, value in row.items() if key != "candidates"}))


if __name__ == "__main__":
    main()
