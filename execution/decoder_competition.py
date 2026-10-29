"""Collect and evaluate the sealed A3 development candidates on two BW GPUs."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import time
import uuid

import torch

from calibration.decoder_competition import check, DEFAULT_BUNDLE
from calibration.e2_manifest import content_id
from calibration.e2_records import append_jsonl_record, load_jsonl_records
from execution.audit_gpu_holdout import write_json, timing_statistics
from execution.decoder_reference import ContinuousDecoderExecutor
from execution.gpu_holdout import compare_outputs
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def robust_sigma(values):
    median = statistics.median(values)
    return 1.4826 * statistics.median(abs(x - median) for x in values)


def validate_records(frozen, records, environment):
    env_id = content_id("a3-environment", environment)
    slots = {s["slot_id"]: s for s in frozen["schedule"]}
    seen = set()
    for row in records:
        key = row["slot_id"]
        if key in seen or key not in slots:
            raise ValueError("duplicate or unexpected A3 slot")
        seen.add(key)
        if any(row[k] != v for k, v in slots[key].items()):
            raise ValueError("record does not match frozen schedule")
        if row["freeze_id"] != frozen["freeze_id"] or row["environment_id"] != env_id:
            raise ValueError("record freeze/environment mismatch")
        payload = {k: v for k, v in row.items() if k != "record_id"}
        if row["record_id"] != content_id("a3-execution-block", payload):
            raise ValueError("A3 block hash mismatch")
        if row["status"] == "ok":
            values = row["samples_s"]
            if len(values) != frozen["protocol"]["iterations"] or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("A3 timing block differs from formal recipe")
            if row["median_s"] != statistics.median(values):
                raise ValueError("A3 median differs from raw samples")
            if not row["correctness"]["pass"] or not row["correctness"]["finite"]:
                raise ValueError("successful A3 block contains failed correctness")
        elif row["status"] != "error":
            raise ValueError("unexpected A3 status")
    return len(seen) == len(slots) and all(r["status"] == "ok" for r in records)


def summarize(frozen, records, environment):
    complete = validate_records(frozen, records, environment)
    report = {"format": "fusion-placement-search.a3-competition-report", "freeze_id": frozen["freeze_id"],
        "complete": complete, "record_count": len(records), "expected_blocks": len(frozen["schedule"]),
        "failed_slots": [r["slot_id"] for r in records if r["status"] != "ok"],
        "environment": environment, "workloads": [],
        "limits": ["Development fixed-subset comparison, not independent SROP or global optimality.",
            "No samples/windows removed. Three windows and MAD thresholds are descriptive, not confidence intervals.",
            "New kernel measurements/refit/re-search are not performed; profiling value remains untested.",
            "Only selected actual-size candidates get runtime correctness checks; not all A2 candidates or pretrained training."]}
    if not complete:
        return report
    protocol = frozen["protocol"]
    for task in frozen["workloads"]:
        rows = [r for r in records if r["task_id"] == task["task_id"]]
        windows = []
        for repeat in range(protocol["repeats"]):
            block = [r for r in rows if r["repeat"] == repeat]
            times, noises, drift = {}, {}, {}
            for key in task["plans"]:
                entries = [r for r in block if r["plan_id"] == key]
                times[key] = statistics.median(r["median_s"] for r in entries)
                noises[key] = max(robust_sigma(r["samples_s"]) for r in entries)
                if key in task["uniform_plan_ids"]:
                    before = next(r for r in entries if r["phase"] == "before")
                    after = next(r for r in entries if r["phase"] == "after")
                    drift[key] = after["median_s"] / before["median_s"] - 1
            resumed = len({r["collection_run_id"] for r in block}) != 1
            windows.append({"repeat": repeat, "median_s_by_plan": times, "robust_sigma_s_by_plan": noises,
                "anchor_relative_drift": drift, "spans_collection_processes": resumed,
                "quality_pass": not resumed and all(abs(v) <= protocol["anchor_drift_limit"] for v in drift.values())})
        actual = {key: statistics.median(w["median_s_by_plan"][key] for w in windows) for key in task["plans"]}
        best = min(actual.values())
        candidates = []
        for key in task["plans"]:
            relative_to_single, differences, separated = [], [], []
            for window in windows:
                times, sigmas = window["median_s_by_plan"], window["robust_sigma_s_by_plan"]
                baseline = min(task["uniform_plan_ids"], key=lambda k: (times[k], k))
                delta = times[key] - times[baseline]
                threshold = max(protocol["noise_relative_floor"] * min(times[key], times[baseline]),
                                protocol["noise_mad_multiplier"] * (sigmas[key] + sigmas[baseline]))
                relative_to_single.append(times[key] / times[baseline] - 1)
                differences.append(delta)
                separated.append(abs(delta) > threshold)
            consistent = (all(v > 0 for v in differences) or all(v < 0 for v in differences)) and all(separated)
            stable = consistent and all(w["quality_pass"] for w in windows)
            candidates.append({"plan_id": key, "median_s": actual[key], "relative_regret": actual[key] / best - 1,
                "raw_sample_distribution": timing_statistics([v for r in rows if r["plan_id"] == key for v in r["samples_s"]]),
                "relative_to_window_best_single": relative_to_single,
                "repeat_consistent_difference_beyond_noise": stable,
                "repeat_consistent_faster_than_single": stable and all(v < 0 for v in differences),
                "window_noise_tests_pass": separated,
                "relative_window_regrets": [w["median_s_by_plan"][key] / min(w["median_s_by_plan"].values()) - 1 for w in windows]})
        selected = task["initial_selected_id"]
        chosen = next(c for c in candidates if c["plan_id"] == selected)
        uniform_regrets = {str(d): actual[k] / best - 1 for d, k in enumerate(task["uniform_plan_ids"])}
        selected_regret = actual[selected] / best - 1
        ties = [actual[k] / best - 1 for k in task["initial_predicted_ties"]]
        report["workloads"].append({"task_id": task["task_id"], "candidates": candidates, "windows": windows,
            "measured_subset_best_s": best, "initial_preselected_id": selected,
            "initial_relative_regret": selected_regret, "initial_relative_regret_by_window": chosen["relative_window_regrets"],
            "initial_epsilon_success": {str(e): selected_regret <= e for e in protocol["epsilon_values"]},
            "initial_tie_regret_range_diagnostic": [min(ties), max(ties)], "fixed_single_gpu_regrets": uniform_regrets,
            "has_initial_loss_over_primary_epsilon": selected_regret > protocol["primary_epsilon"],
            "quality_windows": sum(w["quality_pass"] for w in windows),
            "repeat_consistent_cross_gpu_advantages": sum(c["repeat_consistent_faster_than_single"] for c in candidates
                                                         if c["plan_id"] not in task["uniform_plan_ids"]),
            "kernel_profiling_value": "not assessed"})
    report["evidence_summary"] = {
        "tasks": len(report["workloads"]),
        "tasks_with_initial_loss_over_primary_epsilon": sum(t["has_initial_loss_over_primary_epsilon"] for t in report["workloads"]),
        "tasks_with_all_windows_quality_pass": sum(t["quality_windows"] == protocol["repeats"] for t in report["workloads"]),
        "tasks_with_repeat_consistent_cross_gpu_advantage": sum(t["repeat_consistent_cross_gpu_advantages"] > 0 for t in report["workloads"])}
    return report


def collect_environment(protocol):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("A3 requires two GPUs; no CPU fallback")
    expected = protocol["runtime"]
    if str(torch.__version__).split("+")[0] != expected["torch"] or torch.version.hip != expected["hip"]:
        raise RuntimeError("A3 Torch/HIP differs from sealed runtime")
    devices = []
    for index in protocol["devices"]:
        properties = torch.cuda.get_device_properties(index)
        if expected["device_name_contains"] not in properties.name.upper():
            raise RuntimeError("A3 requires the sealed BW hardware class")
        devices.append({"index": index, "name": properties.name, "memory": properties.total_memory,
                        "uuid": str(getattr(properties, "uuid", "unavailable"))})
    return {"host": socket.gethostname(), "platform": platform.platform(), "torch": str(torch.__version__),
        "hip": torch.version.hip, "devices": devices,
        "visibility": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        "peer_access": {f"{a}->{b}": torch.cuda.can_device_access_peer(a, b) for a, b in ((0, 1), (1, 0))}}


def measure_block(placed, protocol, templates, gold):
    start = time.perf_counter()
    executor = ContinuousDecoderExecutor(placed, ("cuda:0", "cuda:1"), protocol["seed"], templates)
    with torch.no_grad():
        output, *_ = executor.run_once(False)
        accuracy = compare_outputs(output, gold, rtol=protocol["rtol"], atol=protocol["atol"])
        del output
        if not accuracy["pass"]:
            raise RuntimeError(f"actual-size candidate correctness failed: {accuracy}")
        for _ in range(protocol["warmup"]):
            output, *_ = executor.run_once(False)
            del output
        for index in protocol["devices"]:
            torch.cuda.reset_peak_memory_stats(index)
        samples = []
        for _ in range(protocol["iterations"]):
            output, _, _, elapsed = executor.run_once(False)
            samples.append(elapsed)
            del output
    return {"samples_s": samples, "median_s": statistics.median(samples), "correctness": accuracy,
        "peak_memory_bytes": [torch.cuda.max_memory_allocated(d) for d in protocol["devices"]],
        "block_wall_s": time.perf_counter() - start}


def run(frozen, results):
    protocol = frozen["protocol"]
    environment = collect_environment(protocol)
    results.mkdir(parents=True, exist_ok=True)
    env_path = results / "environment.json"
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError("resume environment changed; preserve this directory and use a separate attempt")
    write_json(env_path, environment)
    journal = results / "execution_samples.jsonl"
    records = load_jsonl_records(journal) if journal.exists() else []
    validate_records(frozen, records, environment)
    if any(r["status"] == "error" for r in records):
        raise ValueError("recorded failure retained; refusing to replace or skip failed slot")
    seen = {r["slot_id"] for r in records}
    env_id = content_id("a3-environment", environment)
    run_id = uuid.uuid4().hex
    hardware = hardware_from_dict(frozen["hardware"])
    for task in frozen["workloads"]:
        pending = [s for s in frozen["schedule"] if s["task_id"] == task["task_id"] and s["slot_id"] not in seen]
        if not pending:
            continue
        graph = graph_from_dict(task["graph"])
        templates = {}
        try:
            with torch.no_grad():
                baseline = PlacedOpGraph(graph, hardware, plan_from_dict(task["plans"][task["uniform_plan_ids"][0]]))
                executor = ContinuousDecoderExecutor(baseline, ("cuda:0", "cuda:1"), protocol["seed"], templates)
                output, *_ = executor.run_once(False)
                gold = output.cpu().clone()
                del output, executor
        except Exception as exc:
            write_json(results / f"setup_failure_{task['task_id']}_{run_id}.json", {
                "freeze_id": frozen["freeze_id"], "task_id": task["task_id"], "environment_id": env_id,
                "collection_run_id": run_id, "error_type": type(exc).__name__, "message": str(exc)})
            raise
        for slot in pending:
            row = {**slot, "freeze_id": frozen["freeze_id"], "environment_id": env_id, "collection_run_id": run_id}
            try:
                placed = PlacedOpGraph(graph, hardware, plan_from_dict(task["plans"][slot["plan_id"]]))
                row.update(measure_block(placed, protocol, templates, gold), status="ok")
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, message=str(exc))
                row["record_id"] = content_id("a3-execution-block", row)
                append_jsonl_record(journal, row)
                raise
            row["record_id"] = content_id("a3-execution-block", row)
            append_jsonl_record(journal, row)
            records.append(row)
            gc.collect()
            print(f"{slot['slot_id']} {slot['task_id']} {slot['phase']} ms={row['median_s']*1000:.6f}", flush=True)
        del templates, gold
        gc.collect()
    report = summarize(frozen, records, environment)
    write_json(results / "competition_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "evaluate"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", type=Path, default=Path("data/a3_decoder_competition_v1_results"))
    args = parser.parse_args()
    frozen = check(args.bundle)
    if args.action == "run":
        report = run(frozen, args.out_dir)
    else:
        report = summarize(frozen, load_jsonl_records(args.out_dir / "execution_samples.jsonl"),
                           json.loads((args.out_dir / "environment.json").read_text()))
        write_json(args.out_dir / "competition_report.json", report)
    print(f"complete={report['complete']} blocks={report['record_count']}/{report['expected_blocks']}")
    for task in report["workloads"]:
        print(f"{task['task_id']} initial_regret_pct={task['initial_relative_regret']*100:.6f} "
              f"quality_windows={task['quality_windows']} repeat_consistent_cross_gpu_advantages={task['repeat_consistent_cross_gpu_advantages']}")
    print("report:", args.out_dir / "competition_report.json")


if __name__ == "__main__":
    main()
