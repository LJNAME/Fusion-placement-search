"""Execute sealed layer candidates on two GPUs with identical canonical tensors."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import socket
import statistics

import torch

from calibration.e2_manifest import content_id
from calibration.e2_records import append_jsonl_record, load_jsonl_records
from calibration.gpu_holdout import DEFAULT_BUNDLE, load_freeze
from calibration.profile_e2_bw1000 import _validate_runtime, parse_int_csv
from cost_model.e2_diagnostics import _write_json
from cost_model.e2_residual import _spearman
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict
from .reference import _ReferenceExecutor, _sync


class CanonicalReferenceExecutor(_ReferenceExecutor):
    """Generate tensors once on CPU so device placement cannot change their values."""
    def __init__(self, placed, devices, seed, templates):
        self.templates = templates
        super().__init__(placed, devices, seed)

    def _random(self, shape, dtype, device, salt):
        key = (tuple(shape), str(dtype), self.seed, salt)
        if key not in self.templates:
            generator = torch.Generator(device="cpu").manual_seed(self.seed+salt)
            self.templates[key] = torch.randn(shape, dtype=dtype, device="cpu", generator=generator)*.02
        return self.templates[key].to(device)


def compare_outputs(actual, reference, *, rtol, atol):
    actual, reference = actual.float().cpu(), reference.float().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if actual.shape != reference.shape:
        return {"pass": False, "finite": finite, "shape_match": False}
    if not finite:
        return {"pass": False, "finite": False, "shape_match": True}
    difference = (actual-reference).abs()
    return {"pass": finite and bool(torch.isclose(actual, reference, rtol=rtol, atol=atol).all()),
        "finite": finite, "shape_match": True, "output_shape": list(actual.shape),
        "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()),
        "rtol": rtol, "atol": atol}


def measure_candidate(placed, devices, parameters, templates, reference):
    executor = CanonicalReferenceExecutor(placed, devices, parameters["seed"], templates)
    with torch.no_grad():
        output, _, _, _ = executor.run_once(measure_groups=False)
        accuracy = compare_outputs(output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del output
        for _ in range(parameters["warmup"]):
            output, _, _, _ = executor.run_once(measure_groups=False)
            del output
        for device in executor.devices:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(parameters["iterations"]):
            output, _, _, elapsed = executor.run_once(measure_groups=False)
            samples.append(elapsed)
            del output
        _sync(executor.devices)
        peak = [int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                for device in executor.devices]
    return {"samples_s": samples, "median_s": statistics.median(samples),
        "correctness": accuracy, "peak_memory_bytes": peak}


def summarize_execution(freeze, records):
    workloads = []
    parameters = freeze["protocol"]["execution"]
    for workload in freeze["workloads"]:
        predicted = {row["plan_id"]: row["makespan_s"] for row in workload["predicted_ranking"]}
        candidates = []
        for key in workload["execution_plan_ids"]:
            entries = [row for row in records if row["status"] == "ok" and row["model_id"] == workload["model_id"] and row["plan_id"] == key]
            if len(entries) != parameters["repeats"] or {row["repeat"] for row in entries} != set(range(parameters["repeats"])):
                raise ValueError("execution coverage is incomplete or duplicated")
            if any(len(row["samples_s"]) != parameters["iterations"] for row in entries):
                raise ValueError("execution sample count differs from the frozen recipe")
            samples = [value for row in entries for value in row["samples_s"]]
            candidates.append({"plan_id": key, "predicted_s": predicted[key], "median_s": statistics.median(samples),
                "min_s": min(samples), "max_s": max(samples),
                "correctness_pass": all(row["correctness"]["pass"] for row in entries),
                "max_abs_error": max((row["correctness"]["max_abs_error"] for row in entries
                                      if "max_abs_error" in row["correctness"]), default=None),
                "peak_memory_bytes": [max(row["peak_memory_bytes"][index] for row in entries) for index in range(2)]})
        measured = {row["plan_id"]: row["median_s"] for row in candidates}
        best_id = min(measured, key=lambda key: (measured[key], key))
        selected_id = workload["predicted_top1_id"]
        relative_regret = measured[selected_id]/measured[best_id]-1
        workloads.append({"model_id": workload["model_id"], "candidates": candidates,
            "predicted_top1_id": selected_id, "measured_subset_best_id": best_id,
            "top1_relative_regret": relative_regret,
            "top1_within_5pct": relative_regret <= parameters["top1_relative_regret_target"],
            "selected_speedup_over_single_device": measured[workload["reference_plan_id"]]/measured[selected_id],
            "selected_prediction_ratio_measured_over_predicted": measured[selected_id]/predicted[selected_id],
            "ranking_spearman": _spearman([row["predicted_s"] for row in candidates], [row["median_s"] for row in candidates]),
            "correctness_pass": all(row["correctness_pass"] for row in candidates)})
    return {"format": "fusion-placement-search.gpu-holdout-execution", "freeze_id": freeze["freeze_id"],
        "scope": "actual two-GPU eager synthetic-layer forward; best is limited to sealed measured candidates",
        "correctness_pass": all(row["correctness_pass"] for row in workloads),
        "all_top1_within_5pct": all(row["top1_within_5pct"] for row in workloads),
        "runtime_failure_attempts": sum(row["status"] != "ok" for row in records), "workloads": workloads,
        "claims": "not full-model training, active-profiling benefit, true fused kernels or global placement optimality"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--devices", type=parse_int_csv, default=(0, 1))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze, manifest = load_freeze(args.bundle)
    if tuple(args.devices) != (0, 1):
        raise ValueError("the frozen two-GPU protocol uses physical devices 0,1")
    # There is deliberately no CPU fallback in this entry point.
    _validate_runtime(manifest, args.devices)
    devices = tuple(f"cuda:{index}" for index in args.devices)
    environment = {"host": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "hip": torch.version.hip,
        "peer_access": {f"{source}->{target}": bool(torch.cuda.can_device_access_peer(source, target))
                        for source, target in ((0, 1), (1, 0))},
        "devices": [{"index": index, "name": torch.cuda.get_device_properties(index).name,
                     "total_memory": torch.cuda.get_device_properties(index).total_memory} for index in args.devices]}
    environment_id = content_id("gpu-holdout-execution-environment", environment)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    journal = args.out_dir/"execution_samples.jsonl"
    records = load_jsonl_records(journal)
    if any(row["freeze_id"] != freeze["freeze_id"] or row["environment_id"] != environment_id for row in records):
        raise ValueError("execution journal belongs to a different frozen set or host environment; use a new output directory")
    _write_json(args.out_dir/"execution_environment.json", environment)
    hardware = hardware_from_dict(freeze["hardware"])
    parameters = freeze["protocol"]["execution"]
    completed = {(row["model_id"], row["plan_id"], row["repeat"]) for row in records if row["status"] == "ok"}
    templates = {}
    for workload_index, workload in enumerate(freeze["workloads"]):
        graph = graph_from_dict(workload["graph"])
        plans = workload["candidate_plans"]
        reference_placed = PlacedOpGraph(graph, hardware, plan_from_dict(plans[workload["reference_plan_id"]]))
        reference_executor = CanonicalReferenceExecutor(reference_placed, devices, parameters["seed"], templates)
        with torch.no_grad():
            output, _, _, _ = reference_executor.run_once(measure_groups=False)
            reference = output.float().cpu()
            del output
        del reference_executor
        gc.collect()
        predicted = {row["plan_id"]: row["makespan_s"] for row in workload["predicted_ranking"]}
        for repeat in range(parameters["repeats"]):
            candidates = workload["execution_plan_ids"]
            offset = workload_index % len(candidates)
            order = candidates[offset:]+candidates[:offset]
            if repeat % 2:
                order = list(reversed(order))
            for key in order:
                if (workload["model_id"], key, repeat) in completed:
                    continue
                placed = PlacedOpGraph(graph, hardware, plan_from_dict(plans[key]), predicted_makespan=predicted[key])
                row = {"freeze_id": freeze["freeze_id"], "environment_id": environment_id,
                    "model_id": workload["model_id"], "plan_id": key, "repeat": repeat,
                    "artifact_candidate_id": placed.candidate_id}
                try:
                    row.update(measure_candidate(placed, devices, parameters, templates, reference), status="ok")
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    row["record_id"] = content_id("gpu-holdout-execution-sample", row)
                    append_jsonl_record(journal, row)
                    _write_json(args.out_dir/"execution_report.json", {"freeze_id": freeze["freeze_id"],
                        "execution_status": "incomplete_runtime_failure", "failed_candidate": row})
                    raise
                row["record_id"] = content_id("gpu-holdout-execution-sample", row)
                append_jsonl_record(journal, row)
                records.append(row)
                completed.add((workload["model_id"], key, repeat))
                print(f"{workload['model_id']} repeat={repeat} plan={key[-10:]} median_ms={row['median_s']*1e3:.4f} correctness={row['correctness']['pass']}", flush=True)
        del reference
    report = summarize_execution(freeze, records)
    report.update(environment=environment, environment_id=environment_id, execution_status="complete")
    _write_json(args.out_dir/"execution_report.json", report)
    print(f"correctness_pass: {report['correctness_pass']}; all_top1_within_5pct: {report['all_top1_within_5pct']}")
    print(f"report: {args.out_dir/'execution_report.json'}")
    if not report["correctness_pass"]:
        raise SystemExit("GPU output correctness failed; keep all records for diagnosis")


if __name__ == "__main__":
    main()
