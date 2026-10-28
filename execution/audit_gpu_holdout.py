"""Reproduce a sealed execution report and audit its raw timing blocks."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import DEFAULT_BUNDLE, load_freeze
from execution.gpu_holdout import summarize_execution
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))


def timing_statistics(samples):
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {"samples": len(samples), "median_ms": median * 1000,
        "mean_ms": statistics.fmean(samples) * 1000,
        "min_ms": min(samples) * 1000, "max_ms": max(samples) * 1000,
        "p90_ms": float(np.quantile(samples, .9)) * 1000,
        "p95_ms": float(np.quantile(samples, .95)) * 1000,
        "standard_cv": statistics.pstdev(samples) / statistics.fmean(samples),
        "robust_cv": 1.4826 * mad / median,
        "samples_above_twice_median": sum(value > 2 * median for value in samples)}


def audit_execution(freeze, records, environment, original):
    environment_id = content_id("gpu-holdout-execution-environment", environment)
    if original["freeze_id"] != freeze["freeze_id"] or original["environment"] != environment or original["environment_id"] != environment_id:
        raise ValueError("report, freeze and execution environment do not match")
    slots = {(workload["model_id"], plan_id, repeat)
        for workload in freeze["workloads"] for plan_id in workload["execution_plan_ids"]
        for repeat in range(freeze["protocol"]["execution"]["repeats"])}
    counts = Counter((row["model_id"], row["plan_id"], row["repeat"]) for row in records)
    if set(counts) != slots or any(value != 1 for value in counts.values()):
        raise ValueError("missing, extra or duplicate execution slots")
    record_ids = set()
    for row in records:
        payload = {key: value for key, value in row.items() if key != "record_id"}
        if row["record_id"] != content_id("gpu-holdout-execution-sample", payload) or row["record_id"] in record_ids:
            raise ValueError("raw record hash mismatch or duplicate record ID")
        record_ids.add(row["record_id"])
        if row["freeze_id"] != freeze["freeze_id"] or row["environment_id"] != environment_id or row["status"] != "ok":
            raise ValueError("unexpected freeze, environment or failed record")
        if not row["samples_s"] or any(not math.isfinite(value) or value <= 0 for value in row["samples_s"]):
            raise ValueError("execution samples must be finite and positive")
        if row["median_s"] != statistics.median(row["samples_s"]):
            raise ValueError("stored block median differs from raw samples")
    reproduced = summarize_execution(freeze, records)
    if any(original.get(key) != value for key, value in reproduced.items()):
        raise ValueError("raw journal does not reproduce the original execution report")
    hardware = hardware_from_dict(freeze["hardware"])
    workloads = []
    for workload in freeze["workloads"]:
        graph = graph_from_dict(workload["graph"])
        by_plan = {}
        outliers = []
        for plan_id in workload["execution_plan_ids"]:
            entries = sorted((row for row in records if row["model_id"] == workload["model_id"] and row["plan_id"] == plan_id), key=lambda row: row["repeat"])
            placed = PlacedOpGraph(graph, hardware, plan_from_dict(workload["candidate_plans"][plan_id]))
            if any(row["artifact_candidate_id"] != placed.candidate_id for row in entries):
                raise ValueError("artifact candidate ID does not match frozen graph/hardware/plan")
            samples = [value for entry in entries for value in entry["samples_s"]]
            median = statistics.median(samples)
            for entry in entries:
                for index, value in enumerate(entry["samples_s"]):
                    if value > 2 * median:
                        outliers.append({"plan_id": plan_id, "repeat": entry["repeat"], "sample_index": index,
                            "duration_ms": value * 1000, "ratio_to_pooled_median": value / median})
            blocks = [{"repeat": entry["repeat"], **timing_statistics(entry["samples_s"])} for entry in entries]
            by_plan[plan_id] = {"plan_id": plan_id, "is_reference": plan_id == workload["reference_plan_id"],
                "is_predicted_top1": plan_id == workload["predicted_top1_id"],
                "pooled": timing_statistics(samples), "rounds": blocks,
                "round_1_over_0_change_pct": (entries[1]["median_s"] / entries[0]["median_s"] - 1) * 100}
        rounds = []
        for repeat in range(freeze["protocol"]["execution"]["repeats"]):
            medians = {key: value["rounds"][repeat]["median_ms"] for key, value in by_plan.items()}
            ranking = sorted(medians, key=lambda key: (medians[key], key))
            rounds.append({"repeat": repeat, "ranking": ranking,
                "reference_is_best": ranking[0] == workload["reference_plan_id"],
                "predicted_top1_regret_pct": (medians[workload["predicted_top1_id"]] / medians[ranking[0]] - 1) * 100})
        workloads.append({"model_id": workload["model_id"], "reference_is_best_in_both_rounds": all(row["reference_is_best"] for row in rounds),
            "complete_ranking_same_in_both_rounds": rounds[0]["ranking"] == rounds[1]["ranking"],
            "max_absolute_round_change_pct": max(abs(value["round_1_over_0_change_pct"]) for value in by_plan.values()),
            "rounds": rounds, "candidates": list(by_plan.values()), "long_tail_samples": outliers})
    return {"format": "fusion-placement-search.gpu-holdout-log-audit", "freeze_id": freeze["freeze_id"],
        "checks": {"record_hashes": True, "environment_hash": True, "artifact_ids": True,
            "exact_expected_slots": True, "original_report_reproduced": True},
        "records": len(records), "timing_samples": sum(len(row["samples_s"]) for row in records),
        "correctness_pass": reproduced["correctness_pass"],
        "samples_above_twice_candidate_median": sum(len(row["long_tail_samples"]) for row in workloads),
        "long_tail_rule": "Descriptive >2x pooled candidate median; no samples removed from results.",
        "inference_limit": "Only two sequential blocks per candidate, not paired iterations or independent workload replicates; no significance or causal attribution claim.",
        "workloads": workloads}


def representative_plans(freeze, original, model_id):
    workload = next(row for row in freeze["workloads"] if row["model_id"] == model_id)
    result = next(row for row in original["workloads"] if row["model_id"] == model_id)
    measured = {row["plan_id"]: row["median_s"] for row in result["candidates"]}
    reference, selected = workload["reference_plan_id"], workload["predicted_top1_id"]
    if reference == selected:
        raise ValueError("diagnosis requires different reference and predicted plans")
    lower, upper = sorted((measured[reference], measured[selected]))
    middle = [key for key in workload["execution_plan_ids"] if key not in (reference, selected) and lower < measured[key] < upper]
    if not middle:
        raise ValueError("no measured intermediate plan in this workload")
    intermediate = min(middle, key=lambda key: (abs(measured[key] - (lower + upper) / 2), key))
    return {"model_id": model_id,
        "scope": "Post-hoc diagnostic selection from existing measured candidates; not an independent performance test.",
        "selection_rule": "single-device reference, frozen predicted Top1, and measured candidate nearest their latency midpoint",
        "plans": [{"role": role, "plan_id": key, "old_measured_ms": measured[key] * 1000}
            for role, key in (("single_gpu_reference", reference), ("predicted_top1", selected), ("intermediate", intermediate))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze, _ = load_freeze(args.bundle)
    original = json.loads(args.report.read_text(encoding="utf-8"))
    result = audit_execution(freeze, load_jsonl_records(args.samples), json.loads(args.environment.read_text(encoding="utf-8")), original)
    result["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("samples", args.samples), ("environment", args.environment), ("report", args.report))}
    result["representatives"] = representative_plans(freeze, original, "qwen2.5-3b-s2048-b1")
    write_json(args.output, result)
    print(f"records={result['records']} samples={result['timing_samples']} original_report_reproduced=True")
    for row in result["workloads"]:
        print(f"{row['model_id']} single_gpu_best_both_rounds={row['reference_is_best_in_both_rounds']} complete_ranking_stable={row['complete_ranking_same_in_both_rounds']}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
