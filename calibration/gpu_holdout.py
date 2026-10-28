"""Freeze unseen sequence-shape predictions and GPU reference candidates before labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from calibration.e2_dataset import _feature_record, analytic_kernel_estimate
from calibration.e2_manifest import build_shape_manifest, content_id, default_model_specs, validate_shape_manifest
from calibration.e2_records import load_jsonl_records
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_diagnostics import _write_json
from cost_model.e2_residual import NearestShapePredictor
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, build_workload_graph, object_map_for_workload
from cost_model.e2_uncertainty import TrainingDistanceUncertainty
from cost_model.placement_probes import graph_wide_candidates
from cost_model.search_sensitivity import generate_candidate_pool, plan_id
from simulator import make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.hardware import hardware_to_dict, load_hardware


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT/"data/bw1000_gpu_holdout_v1"
INPUTS = ("data/bw1000_e2/profiling_dataset.jsonl", "data/bw1000_e2/shape_manifest.json",
          "data/bw1000_e2/split_contract.json", "data/bw1000_2card/bw1000_2card.calibrated.json")
SOURCES = ("calibration/gpu_holdout.py", "calibration/evaluate_gpu_holdout.py", "calibration/profile_e2_bw1000.py",
    "calibration/e2_manifest.py", "calibration/e2_dataset.py", "calibration/e2_records.py",
    "calibration/verify_e2_measurements.py", "cost_model/e2_residual.py", "cost_model/e2_uncertainty.py",
    "cost_model/e2_srop_proxy.py", "cost_model/search_sensitivity.py", "cost_model/legal_search.py",
    "cost_model/legal_policy.py", "cost_model/structure.py", "cost_model/structural.py",
    "cost_model/e2_audit.py", "cost_model/hardware_profile.py", "calibration/e2_contract.py",
    "cost_model/batched_placement_ranker.py", "cost_model/placement_probes.py",
    "execution/gpu_holdout.py", "execution/reference.py", "simulator/model_graphs/dense_layer.py",
    "simulator/model_graphs/transformer.py", "simulator/model_graphs/spec.py", "simulator/fusion_rules.py",
    "simulator/shape_model.py", "simulator/hardware.py", "simulator/schema.py", "simulator/dataset.py",
    "simulator/placement_artifacts.py", "simulator/validate.py", "tools/run_gpu_holdout.sh")


def file_hashes(paths):
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


def public_queries(manifest, hardware):
    result = []
    for item in manifest["objects"]:
        if not item["measurement"]["enabled"]:
            continue
        analytic = analytic_kernel_estimate(item, hardware)
        result.append({"object_id": item["object_id"], "kernel_key": item["kernel_key"],
            "features": _feature_record(item, analytic, hardware), "analytic": analytic,
            "usages": item["usages"], "label_access": {"visible": False}})
    return result


def select_execution_candidates(ranked, graph, exchange):
    reference = make_reference_plan(graph, device=0)
    reference_id = plan_id(reference, exchange)
    split = copy.deepcopy(reference)
    for group in split.groups[len(split.groups)//2:]:
        group.device = 1
    split_id = plan_id(split, exchange)
    selected = list(dict.fromkeys([*(row.plan_id for row in ranked[:4]), reference_id, split_id]))
    return selected, reference_id, split


def build_freeze(*, iterations=64, seeds=(0, 1, 7, 23)):
    train = load_jsonl_records(ROOT/INPUTS[0])
    old_manifest = json.loads((ROOT/INPUTS[1]).read_text(encoding="utf-8"))
    contract = json.loads((ROOT/INPUTS[2]).read_text(encoding="utf-8"))
    hardware = load_hardware(ROOT/INPUTS[3])
    manifest = build_shape_manifest(default_model_specs((1536, 2048)), **{
        key: old_manifest["environment_contract"][key]
        for key in ("hardware_class", "framework_version", "accelerator_version")})
    queries = public_queries(manifest, hardware)
    known = {row["object_id"] for row in train}
    fresh = {row["object_id"] for row in queries}
    if known & fresh:
        raise ValueError("held-out full Kernel Keys overlap the existing measured training labels")
    if len(train) != 112 or len(known) != len(train):
        raise ValueError("the frozen development baseline must have 112 distinct known labels")
    visible = [{**row, "label_access": {"visible": True}} for row in train]
    prediction_protocol = contract["prediction_protocol"]
    predictor = NearestShapePredictor().fit_mean(visible, prediction_protocol)
    uncertainty = TrainingDistanceUncertainty().fit(visible, prediction_protocol)
    predictions = {}
    for row in queries:
        mean = predictor._nearest(row)
        predictions[row["object_id"]] = {"analytic_s": row["analytic"]["estimate_s"],
            "nearest_s": row["analytic"]["estimate_s"]*math.exp(mean),
            "log_residual": mean, "q90_log": uncertainty.q90(row, "distance_floor")}
    workloads = []
    exchange = ((0, 1), (1, 0))
    for model in manifest["models"]:
        spec = model["spec"]
        workload = E2Workload(model["model_id"], "new-sequence-holdout", model["model_id"],
                              model["model_family"], int(spec["seq_length"]))
        graph = build_workload_graph(workload, "layer")
        mapping = object_map_for_workload(queries, workload)
        profile = DatasetLatencyProfile(hardware, mapping,
            {key: value["nearest_s"] for key, value in predictions.items()},
            {key: "frozen_zero_shot" for key in predictions})
        searched, search_meta = generate_candidate_pool(graph, profile, candidate_count=12,
            seeds=seeds, iterations=iterations, rollouts=3, include_planned=False)
        proposals, probe_meta = graph_wide_candidates(graph, searched, profile.device_classes)
        # Include public uniform and half-graph baselines regardless of prediction rank.
        reference = make_reference_plan(graph, device=0)
        split = copy.deepcopy(reference)
        for group in split.groups[len(split.groups)//2:]:
            group.device = 1
        by_id = {plan_id(plan, exchange): plan for plan in [*proposals, reference, split]}
        ranker = BatchedPlacementRanker(graph, list(by_id.values()), 2, exchange)
        ranked = ranker.rank(profile)
        selected, reference_id, _ = select_execution_candidates(ranked, graph, exchange)
        if any(len(group.node_ids) != 1 for key in selected for group in by_id[key].groups):
            raise ValueError("this GPU reference validation supports executable singleton groups only")
        if not any(len({group.device for group in by_id[key].groups}) == 2 for key in selected):
            raise AssertionError("GPU validation requires a cross-device candidate")
        workloads.append({"model_id": model["model_id"], "model_family": model["model_family"],
            "sequence_length": spec["seq_length"], "scope": "single-layer-synthetic-forward-eager",
            "graph": graph_to_dict(graph), "candidate_plans": {key: plan_to_dict(value) for key, value in by_id.items()},
            "predicted_ranking": [{"plan_id": row.plan_id, "makespan_s": row.makespan} for row in ranked],
            "execution_plan_ids": selected, "reference_plan_id": reference_id,
            "predicted_top1_id": ranked[0].plan_id, "candidate_search": search_meta,
            "structural_probes": probe_meta})
    payload = {"format": "fusion-placement-search.gpu-holdout-freeze", "version": "1.0",
        "method_baseline": "9173c57", "manifest_id": manifest["manifest_id"],
        "input_sha256": file_hashes(INPUTS), "source_sha256": file_hashes(SOURCES),
        "train_object_ids": sorted(known), "training_labels": len(train), "query_features": queries,
        "zero_shot_predictions": predictions, "hardware": hardware_to_dict(hardware), "workloads": workloads,
        "protocol": {"sequences": [1536, 2048], "micro_batch_size": 1, "dtype": "bf16",
            "novelty": "full Kernel Keys disjoint from the 112 repository training labels; external exposure is unverified",
            "predictor": "nearest shape mean plus training-only distance-floor q90; no refit on new labels",
            "candidate_generation": "frozen predictor-only MCTS plus label-independent structural probes; no Oracle search candidates",
            "search": {"iterations": iterations, "seeds": list(seeds), "rollouts": 3},
            "kernel_profiling": {"assignment": "replicate", "devices": [0, 1], "expected_replicas": 2,
                "warmup": 10, "iterations_per_sample": 50, "samples": 7, "max_robust_cv_target": .1},
            "execution": {"warmup": 5, "iterations": 20, "repeats": 2, "seed": 1234,
                "rtol": .01, "atol": .001, "top1_relative_regret_target": .05,
                "inputs": "canonical CPU-generated tensors copied to each runtime device; same values for every plan",
                "order": "rotate candidate order by workload and reverse in repeat 1",
                "selection": "frozen predicted Top-4 plus uniform device-zero and half-graph split; deduplicated",
                "timing": "uninstrumented eager forward; synchronize all participating GPUs before/after CPU wall timer"},
            "links": "reuse the existing calibrated link profile; record current GPU peer access; do not refit on holdout execution times",
            "acceptance": "complete formal two-device labels, finite and close outputs; ranking/speed targets reported even on failure",
            "scope": "new-shape zero-shot prediction and executable-layer validation; not active-calibration benefit, training throughput or real fusion validation"}}
    return manifest, {**payload, "freeze_id": content_id("gpu-holdout-freeze", payload)}


def load_freeze(bundle=DEFAULT_BUNDLE):
    bundle = Path(bundle)
    freeze = json.loads((bundle/"freeze.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle/"manifest.json").read_text(encoding="utf-8"))
    validate_shape_manifest(manifest)
    payload = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != content_id("gpu-holdout-freeze", payload) or freeze["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("frozen predictions/candidates or manifest changed")
    if freeze["source_sha256"] != file_hashes(freeze["source_sha256"]) or freeze["input_sha256"] != file_hashes(freeze["input_sha256"]):
        raise ValueError("frozen source or training/hardware inputs changed; use the frozen revision")
    fresh = {row["object_id"] for row in freeze["query_features"]}
    if fresh & set(freeze["train_object_ids"]) or fresh != set(freeze["zero_shot_predictions"]):
        raise ValueError("training/holdout keys overlap or prediction coverage changed")
    if any("target" in row or "measurement" in row for row in freeze["query_features"]):
        raise ValueError("held-out feature rows must not contain labels")
    return freeze, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.bundle.exists():
            raise FileExistsError("do not overwrite a frozen bundle; choose a new directory")
        manifest, freeze = build_freeze()
        args.bundle.mkdir(parents=True)
        _write_json(args.bundle/"manifest.json", manifest)
        _write_json(args.bundle/"freeze.json", freeze)
    freeze, manifest = load_freeze(args.bundle)
    print(f"freeze_id: {freeze['freeze_id']}")
    print(f"training_labels: {freeze['training_labels']}; unseen_kernel_keys: {len(freeze['query_features'])}")
    print(f"workloads: {len(freeze['workloads'])}; execution_candidates: {sum(len(w['execution_plan_ids']) for w in freeze['workloads'])}")
    print("GPU measurements and execution: PENDING (bundle check is CPU-only)")


if __name__ == "__main__":
    main()
