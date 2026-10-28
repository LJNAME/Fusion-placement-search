"""Score fresh GPU kernel measurements against predictions sealed before collection."""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

from calibration.e2_manifest import content_id
from calibration.e2_records import load_environment_registry, load_jsonl_records
from calibration.gpu_holdout import DEFAULT_BUNDLE, load_freeze
from calibration.verify_e2_measurements import validate_measurement_set
from cost_model.e2_diagnostics import _write_json
from cost_model.e2_residual import _spearman


def prediction_metrics(rows):
    result = {}
    for method in ("analytic", "nearest"):
        actual = [row["measured_s"] for row in rows]
        predicted = [row[f"{method}_s"] for row in rows]
        errors = [abs(a-p)/a for a, p in zip(actual, predicted)]
        result[method] = {"objects": len(rows), "mape": statistics.fmean(errors),
            "median_ape": statistics.median(errors), "p90_ape": float(np.quantile(errors, .9)),
            "log_mae": statistics.fmean(abs(math.log(a/p)) for a, p in zip(actual, predicted)),
            "spearman": _spearman(actual, predicted)}
    result["nearest"]["q90_coverage"] = statistics.fmean(row["inside_q90"] for row in rows)
    return result


def evaluate_measurements(freeze, manifest, records, environments):
    quality = validate_measurement_set(manifest, environments, records, expected_replicas=2)
    if not quality["complete"] or quality["over_replicated_object_ids"]:
        raise ValueError("formal collection requires exactly two successful records per enabled object")
    expected = freeze["protocol"]["kernel_profiling"]
    by_id = {}
    for row in records:
        if row["status"] != "ok":
            continue
        timing = row["timing"]
        if (timing["warmup"] != expected["warmup"] or timing["iterations_per_sample"] != expected["iterations_per_sample"] or
                len(timing["samples"]) != expected["samples"]):
            raise ValueError("quick/changed timing recipes cannot be used for the formal holdout")
        by_id.setdefault(row["object_id"], []).append(row)
    features = {row["object_id"]: row for row in freeze["query_features"]}
    rows = []
    for key, prediction in freeze["zero_shot_predictions"].items():
        source = by_id[key]
        if {row["physical_device_index"] for row in source} != {0, 1}:
            raise ValueError("each object must be independently replicated on physical devices 0 and 1")
        measured = statistics.median(row["timing"]["summary"]["median_s"] for row in source)
        absolute_log_error = abs(math.log(measured/prediction["nearest_s"]))
        rows.append({"object_id": key, "family": features[key]["kernel_key"]["operator"],
            "model_ids": sorted({usage["model_id"] for usage in features[key]["usages"]}),
            "measured_s": measured, **prediction, "inside_q90": absolute_log_error <= prediction["q90_log"],
            "raw_record_ids": [row["record_id"] for row in source],
            "replica_ratio_max_min": max(row["timing"]["summary"]["median_s"] for row in source)/
                                     min(row["timing"]["summary"]["median_s"] for row in source)})
    return {"freeze_id": freeze["freeze_id"], "format": "fusion-placement-search.gpu-holdout-zero-shot",
        "quality": quality, "quality_target_pass": quality["max_robust_sample_cv"] <= expected["max_robust_cv_target"],
        "data_id": content_id("gpu-holdout-measurements", records), "all": prediction_metrics(rows),
        "by_family": {family: prediction_metrics([row for row in rows if row["family"] == family])
                      for family in sorted({row["family"] for row in rows})},
        "by_model": {model: prediction_metrics([row for row in rows if model in row["model_ids"]])
                     for model in sorted({model for row in rows for model in row["model_ids"]})},
        "objects": rows, "scope": "sealed zero-shot only; no new labels used for fitting, normalization, q90 or candidate generation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze, manifest = load_freeze(args.bundle)
    report = evaluate_measurements(freeze, manifest, load_jsonl_records(args.raw), load_environment_registry(args.environments))
    _write_json(args.output, report)
    print(f"quality_target_pass: {report['quality_target_pass']}")
    print(json.dumps(report["all"], indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
