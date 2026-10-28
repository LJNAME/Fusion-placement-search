"""Seal a new single-layer epsilon validation before seq768/1280 labels."""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import zipfile

from calibration.e2_manifest import build_shape_manifest, content_id, default_model_specs, validate_shape_manifest
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import ROOT, INPUTS, SOURCES, file_hashes, load_freeze, public_queries
from cost_model.e2_residual import NearestShapePredictor
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.e2_uncertainty import TrainingDistanceUncertainty
from cost_model.placement_probes import graph_wide_candidates
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.search_sensitivity import generate_candidate_pool, permute_plan_devices, plan_id
from cost_model.structural import StructuralCostModel
from execution.audit_gpu_holdout import write_json
from simulator import make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.hardware import load_hardware, hardware_to_dict
from simulator.model_graphs import build_decoder_layer


DEFAULT_BUNDLE = ROOT / "data/reference_validation_v1"
HOST = "data/received_reference_host_cpu_v1/bw1000_reference_host_cpu_v1/host_submission_profile.json"
EXPOSED = "data/bw1000_gpu_holdout_v1/freeze.json"
EXTRA_SOURCES = ("calibration/reference_validation.py", "execution/evaluate_reference_validation.py",
    "execution/audit_gpu_holdout.py", "cost_model/reference_semantics.py", "tools/run_reference_validation.sh")


def layer_manifest():
    specs = default_model_specs((768, 1280))
    manifest = build_shape_manifest(specs)
    graphs = {}
    for model in manifest["models"]:
        spec = next(s for s in specs if asdict(s) == model["spec"])
        graphs[model["model_id"]] = build_decoder_layer(spec, layer_id=0)
    selected = []
    for raw in manifest["objects"]:
        item = copy.deepcopy(raw)
        usages = []
        for usage in item["usages"]:
            graph = graphs[usage["model_id"]]
            matches = [n.id for n in graph.nodes.values() if n.extra["local_name"] == usage["local_name"]]
            if matches:
                usages.append({**usage, "occurrence_count": len(matches), "example_node_id": matches[0]})
        if usages:
            item["usages"] = usages
            selected.append(item)
    manifest["objects"] = selected
    manifest["scope"] = "single decoder layer only; no embedding, final model norm or vocabulary head"
    manifest["manifest_id"] = content_id("manifest", {k: v for k, v in manifest.items() if k != "manifest_id"})
    validate_shape_manifest(manifest)
    return manifest, graphs


def feature_alias_audit(queries, known, protocol):
    def key(row):
        f = row["features"]
        return content_id("predictor-feature-vector", {"numeric": {k: f["numeric"][k] for k in protocol["numeric_features"]},
            "categorical": {k: f["categorical"][k] for k in protocol["categorical_features"]}})
    index = {}
    for row in known:
        index.setdefault(key(row), []).append(row["object_id"])
    aliases = [{"object_id": row["object_id"], "known_feature_alias_ids": sorted(index[key(row)]),
                "new_usages": row["usages"],
                "feature_fingerprint": key(row)} for row in queries if key(row) in index]
    return {"fresh_objects": len(queries), "feature_novel_objects": len(queries) - len(aliases), "aliases": aliases,
        "rule": "Compare exactly the nearest predictor numeric/categorical input columns, not JSON record IDs or unused fields.",
        "interpretation": "Full-key novelty and predictor-feature novelty are separate; retain and report alias objects."}


def build_freeze(*, iterations=64, seeds=(0, 1, 7, 23)):
    old, _ = load_freeze()
    manifest, graphs = layer_manifest()
    train = load_jsonl_records(ROOT / INPUTS[0])
    contract = json.loads((ROOT / INPUTS[2]).read_text())
    hardware = load_hardware(ROOT / INPUTS[3])
    queries = public_queries(manifest, hardware)
    exposed = set(old["train_object_ids"]) | set(old["zero_shot_predictions"])
    fresh = {r["object_id"] for r in queries}
    if fresh & exposed:
        raise ValueError("new layer keys overlap previously exposed full keys")
    audit = feature_alias_audit(queries, [*train, *old["query_features"]], contract["prediction_protocol"])
    public_train = [{**row, "label_access": {"visible": True}} for row in train]
    predictor = NearestShapePredictor().fit_mean(public_train, contract["prediction_protocol"])
    uncertainty = TrainingDistanceUncertainty().fit(public_train, contract["prediction_protocol"])
    predictions = {}
    for row in queries:
        mean = predictor._nearest(row)
        predictions[row["object_id"]] = {"analytic_s": row["analytic"]["estimate_s"],
            "nearest_s": row["analytic"]["estimate_s"] * math.exp(mean), "log_residual": mean,
            "q90_log": uncertainty.q90(row, "distance_floor")}
    host = HostSubmissionProfile(**json.loads((ROOT / HOST).read_text()))
    if host.calibrated:
        raise ValueError("this preregistration expects the unchanged, explicitly unaccepted experimental host profile")
    exchange, workloads = ((0, 1), (1, 0)), []
    for model in manifest["models"]:
        graph, model_id = graphs[model["model_id"]], model["model_id"]
        mapping = object_map_for_workload(queries, E2Workload(model_id, "unseen-sequence-layer", model_id,
            model["model_family"], model["spec"]["seq_length"]))
        profile = DatasetLatencyProfile(hardware, mapping, {k: r["nearest_s"] for k, r in predictions.items()},
            {k: "frozen_zero_shot" for k in predictions})
        searched, search_meta = generate_candidate_pool(graph, profile, candidate_count=12, seeds=seeds,
            iterations=iterations, rollouts=3, include_planned=False)
        proposals, probe_meta = graph_wide_candidates(graph, searched, profile.device_classes)
        reference = make_reference_plan(graph, device=0)
        split = copy.deepcopy(reference)
        for group in split.groups[len(split.groups) // 2:]:
            group.device = 1
        classes = {plan_id(p, exchange): p for p in [*proposals, reference, split]}
        plans, physical_to_class = {}, {}
        for class_id, plan in classes.items():
            for direction in exchange:
                physical = permute_plan_devices(plan, direction)
                key = plan_id(physical)
                plans[key], physical_to_class[key] = physical, class_id
        methods = {"reference_host": ReferenceEagerCostModel(host), "structural_dag": StructuralCostModel(),
                   "reference_zero_host": ReferenceEagerCostModel()}
        rankings = {name: sorted([{"plan_id": key, "makespan_s": method.predict(graph, plan, profile).makespan}
                                 for key, plan in plans.items()], key=lambda r: (r["makespan_s"], r["plan_id"]))
                    for name, method in methods.items()}
        selected_classes = {plan_id(reference, exchange), plan_id(split, exchange)}
        top_sets = {}
        for name, ranking in rankings.items():
            ordered_classes = list(dict.fromkeys(physical_to_class[r["plan_id"]] for r in ranking))
            selected_classes.update(ordered_classes[:4])
            top_sets[name] = sorted(r["plan_id"] for r in ranking if r["makespan_s"] <= ranking[0]["makespan_s"] + 1e-12)
        selected = sorted(k for k, c in physical_to_class.items() if c in selected_classes)
        workloads.append({"model_id": model_id, "model_family": model["model_family"], "sequence_length": model["spec"]["seq_length"],
            "scope": "single-layer-synthetic-forward-eager", "graph": graph_to_dict(graph),
            "candidate_plans": {k: plan_to_dict(p) for k, p in plans.items()}, "physical_to_class": physical_to_class,
            "predicted_ranking": rankings["reference_host"], "method_rankings": rankings, "method_top1_sets": top_sets,
            "method_selected_ids": {name: min(ids) for name, ids in top_sets.items()},
            "execution_plan_ids": selected, "reference_plan_id": plan_id(reference),
            "predicted_top1_id": min(top_sets["reference_host"]),
            "candidate_search": search_meta, "structural_probes": probe_meta})
    inputs = tuple(INPUTS) + (HOST, EXPOSED)
    sources = tuple(dict.fromkeys((*SOURCES, *EXTRA_SOURCES)))
    protocol = copy.deepcopy(old["protocol"])
    protocol.update(sequences=[768, 1280], scope="independent sequence-layer validation of experimental cost predictions; no SROP benefit claim",
        predictor="Old112 nearest mean/distance-floor q90 plus unchanged unaccepted host profile; no target-label refit",
        candidate_generation="Shared structural-MCTS proposal pool plus label-free probes, then sealed three-method rescoring; not ReferenceEager-guided search",
        search={"iterations": iterations, "seeds": list(seeds), "rollouts": 3},
        epsilon_evaluation={"primary_relative_regret": .01, "sensitivity": [.001, .005, .01],
            "reference": "best measured candidate subset, never global Oracle optimum",
            "physical_tie_policy": "Choose lexicographically smallest physical ID before labels; also report full tie best/worst regret and exchange-class diagnostics",
            "stop_rule_enabled": False, "no_candidate_reranking_after_labels": True},
        label_access={"zero_shot": "Target labels may only be consumed by measurement and evaluation, never predictors or candidate generation",
            "active_calibration": "Disabled in this bundle; a separate preregistration is required",
            "prior_exposure": "112 train labels and 56 development full keys excluded; external/private label exposure unverified"})
    protocol["execution"]["selection"] = "Union of Top4 exchange classes from each of three methods plus single-device/half-graph controls; execute both physical directions of every selected class"
    protocol["novelty"] = "Single-layer complete Keys disjoint from 168 repository-exposed objects; predictor feature aliases reported separately"
    payload = {"format": "fusion-placement-search.gpu-holdout-freeze", "version": "1.0", "experiment": "reference-epsilon-validation-v1",
        "manifest_id": manifest["manifest_id"], "source_sha256": file_hashes(sources), "input_sha256": file_hashes(inputs),
        "train_object_ids": old["train_object_ids"], "training_labels": len(train), "exposed_object_ids": sorted(exposed),
        "query_features": queries, "zero_shot_predictions": predictions, "hardware": hardware_to_dict(hardware),
        "host_profile": asdict(host), "feature_alias_audit": audit, "workloads": workloads, "protocol": protocol}
    return manifest, {**payload, "freeze_id": content_id("gpu-holdout-freeze", payload)}


def check(bundle):
    freeze, manifest = load_freeze(bundle)
    fresh = {r["object_id"] for r in freeze["query_features"]}
    if fresh & set(freeze["exposed_object_ids"]) or freeze["host_profile"]["calibrated"]:
        raise ValueError("independent-key boundary or host acceptance changed")
    actual_usages = {(r["model_id"], r["local_name"]) for row in freeze["query_features"] for r in row["usages"]}
    expected_usages = {(w["model_id"], n["extra"]["local_name"]) for w in freeze["workloads"]
                       for n in w["graph"]["nodes"] if n["op_type"] != "input"}
    if actual_usages != expected_usages:
        raise ValueError("measurement pool differs from executable single-layer nodes")
    for workload in freeze["workloads"]:
        mapping = object_map_for_workload(freeze["query_features"], E2Workload(workload["model_id"], "check", workload["model_id"],
            workload["model_family"], workload["sequence_length"]))
        for node in workload["graph"]["nodes"]:
            if node["op_type"] != "input" and node["extra"]["local_name"] not in mapping:
                raise ValueError("single-layer query coverage incomplete")
        if not set(k for ids in workload["method_top1_sets"].values() for k in ids) <= set(workload["execution_plan_ids"]):
            raise ValueError("a predicted tie member is not scheduled for execution")
    return freeze, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "package"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/reference_validation_v1.deployment.zip")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.bundle.exists():
            raise FileExistsError("never overwrite a frozen validation bundle")
        manifest, frozen = build_freeze()
        args.bundle.mkdir(parents=True)
        write_json(args.bundle / "manifest.json", manifest)
        write_json(args.bundle / "freeze.json", frozen)
        write_json(args.bundle / "feature_alias_audit.json", frozen["feature_alias_audit"])
    freeze, manifest = check(args.bundle)
    if args.action == "package":
        paths = set(freeze["source_sha256"]) | set(freeze["input_sha256"])
        for package in ("calibration", "cost_model", "execution", "simulator", "integration"):
            paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / package).rglob("*.py") if "tests" not in path.parts)
        paths.update(path.relative_to(ROOT).as_posix() for path in args.bundle.glob("*.json"))
        paths.add("calibration/REFERENCE_VALIDATION.md")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(paths):
                archive.writestr(name, (ROOT / name).read_bytes())
        print("deployment archive:", args.archive, "files:", len(paths))
    print("freeze_id:", freeze["freeze_id"])
    print("layer objects:", len(manifest["objects"]), "enabled:", len(freeze["query_features"]),
        "feature aliases:", len(freeze["feature_alias_audit"]["aliases"]))
    print("tasks:", len(freeze["workloads"]), "physical execution candidates:", sum(len(w["execution_plan_ids"]) for w in freeze["workloads"]))
    print("No target labels collected by prepare/check; primary epsilon=1%; experimental host remains unaccepted.")


if __name__ == "__main__":
    main()
