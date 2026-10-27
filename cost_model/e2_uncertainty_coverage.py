"""Measure held-out interval coverage with training-only calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from calibration.e2_dataset import make_dataset_view
from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from .e2_diagnostics import _write_json, measurement_pool, workload_for_task
from .e2_residual import NearestShapePredictor, _operator, _target
from .e2_srop_proxy import build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract, build_task_suite
from .e2_uncertainty import PROTOCOL, VARIANTS, TrainingDistanceUncertainty, interval_metrics


def summarize_rows(rows):
    return {"all": {variant: interval_metrics(rows, variant) for variant in VARIANTS},
        "by_family": {family: {variant: interval_metrics([row for row in rows if row["family"] == family], variant)
                              for variant in VARIANTS} for family in sorted({row["family"] for row in rows})},
        "by_distance_bin": {name: {variant: interval_metrics([row for row in rows if row["distance_bin"] == name], variant)
                                   for variant in VARIANTS} for name in ("near", "mid", "far")},
        "by_family_seen": {str(seen): {variant: interval_metrics([row for row in rows if row["family_seen"] == seen], variant)
                                      for variant in VARIANTS} for seen in (False, True)}}


def evaluate_objects(public, predictor, calibrated, oracle_by_id, object_ids, setting_id):
    public_by_id = {row["object_id"]: row for row in public}
    rows = []
    for key in sorted(object_ids):
        row = public_by_id[key]
        if "target" in row or "measurement" in row:
            raise AssertionError("evaluation object must be hidden from predictor and calibrator")
        prediction = predictor.predict(row)
        description = calibrated.describe(row)
        rows.append({"training_setting_id": setting_id, "object_id": key, "family": _operator(row),
            "predicted_duration_s": row["analytic"]["estimate_s"] * math.exp(prediction.mean),
            "abs_log_error": abs(_target(oracle_by_id[key]) - prediction.mean),
            "q90": {"legacy": prediction.q90, "strict_family": description["strict_family"],
                    "distance_floor": description["distance_floor"]},
            **{name: description[name] for name in ("nearest_distance", "distance_ratio", "distance_bin", "neighbor_object_id", "family_seen", "family_calibration")}})
    return rows


def run_coverage(records, manifest, contract, suite):
    oracle = {row["object_id"]: row for row in records}
    cache = {}
    settings = {}
    def public_state(runtime, split_id, mode):
        public = make_dataset_view(records, runtime, split_id=split_id, mode=mode, role="predictor")
        visible = [row for row in public if "target" in row]
        setting_id = content_id("uncertainty-training-state", sorted(row["object_id"] for row in visible))
        if setting_id not in cache:
            predictor = NearestShapePredictor().fit(visible, runtime["prediction_protocol"])
            calibrated = TrainingDistanceUncertainty().fit(visible, runtime["prediction_protocol"])
            cache[setting_id] = predictor, calibrated
            settings[setting_id] = {"train_object_ids": sorted(row["object_id"] for row in visible),
                "global_parameters": calibrated.global_parameters, "family_parameters": calibrated.by_family,
                "fold_rule": "each fold trains on train_object_ids minus its object_id",
                "folds": [{key: fold[key] for key in ("object_id", "family", "neighbor_object_id", "distance", "abs_log_error", "family_seen_in_fold")}
                          for fold in calibrated.folds]}
        return public, *cache[setting_id], setting_id

    zero_shot = []
    for split in contract["splits"]:
        public, predictor, calibrated, state = public_state(contract, split["split_id"], "zero_shot")
        rows = evaluate_objects(public, predictor, calibrated, oracle, split["hidden_object_ids"], state)
        zero_shot.append({"split_id": split["split_id"], "hidden_objects": len(rows),
                          "summary": summarize_rows(rows), "objects": rows})
    configurations = []
    unique = {}
    for task in suite["tasks"]:
        runtime = active_task_contract(task, contract, manifest)
        public, predictor, calibrated, state = public_state(runtime, task["split_id"], "active_calibration")
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, task["scale"])
        pool = measurement_pool(graph, object_map_for_workload(records, workload), task["train_object_ids"])
        rows = evaluate_objects(public, predictor, calibrated, oracle, pool, state)
        for row in rows:
            identity = (state, row["object_id"])
            if identity in unique and unique[identity] != row:
                raise AssertionError("same public training state and object produced different calibration")
            unique[identity] = row
        configurations.append({"task_id": task["task_id"], "model_id": task["model_id"], "scale": task["scale"],
            "initial_train_count": task["initial_train_count"], "subset_seed": task["subset_seed"],
            "training_setting_id": state, "measurement_objects": len(rows), "summary": summarize_rows(rows),
            "object_ids": list(pool)})
    unique_rows = list(unique.values())
    return {"zero_shot": zero_shot, "calibration_training_states": settings,
        "active_initial": {"configs": len(configurations), "unique_training_state_objects": len(unique_rows),
            "summary": summarize_rows(unique_rows),
            "by_initial_train_count": {str(count): summarize_rows([row for row in unique_rows
                if len(settings[row["training_setting_id"]]["train_object_ids"]) == count])
                for count in sorted({len(state["train_object_ids"]) for state in settings.values()})},
            "objects": unique_rows, "configurations": configurations}}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    if suite != build_task_suite(manifest, contract):
        raise ValueError("frozen suite changed")
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    result = run_coverage(records, manifest, contract, suite)
    config = {"suite_id": suite["suite_id"], "dataset_id": content_id("diagnostics-data", records),
        "protocol": PROTOCOL, "prediction_protocol": contract["prediction_protocol"],
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / file).read_bytes()
            for file in ("e2_uncertainty_coverage.py", "e2_uncertainty.py", "e2_residual.py"))).hexdigest()}
    report = {"format": "fusion-placement-search.e2-uncertainty-coverage", "version": "1.0",
        "configuration": config, "experiment_id": content_id("uncertainty-coverage", config),
        "scope": "development diagnostics; hidden labels used only for evaluation; repeated object-state pairs deduplicated",
        **result}
    _write_json(args.output, report)
    for split in report["zero_shot"]:
        print(split["split_id"], json.dumps(split["summary"]["all"]))
    print("active_initial", json.dumps(report["active_initial"]["summary"]["all"]))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
