"""Bounded training-side executor-lifetime diagnostic, not a calibration rerun."""

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time

import torch

from calibration.e2_manifest import content_id
from calibration.e2_records import append_jsonl_record, load_jsonl_records
from calibration.gpu_holdout import DEFAULT_BUNDLE, ROOT, load_freeze
from calibration.profile_e2_bw1000 import _validate_runtime
from execution.audit_gpu_holdout import timing_statistics, write_json
from execution.diagnose_gpu_holdout import expected_transfers
from execution.gpu_holdout import CanonicalReferenceExecutor, compare_outputs
from execution.host_submission import SubmissionProbeExecutor, training_cases
from simulator import PlacedOpGraph, make_reference_plan
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


PHASES = ("before", "middle", "after")
SOURCES = ("execution/diagnose_reference_state.py", "execution/host_submission.py",
    "execution/reference.py", "execution/gpu_holdout.py", "simulator/model_graphs/dense_layer.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_protocol(freeze, manifest):
    cases = [x for x in training_cases(freeze, manifest, (1024,))
             if x["model_id"] == "qwen3-0.6b-s1024-b1" and x["plan_name"] in ("uniform0", "uniform1", "mlp_up")]
    cells = [{"case_id": x["case_id"], "lifecycle": life, "middle_kind": middle}
             for x in cases for life in ("rebuild_block", "reuse_triplet") for middle in ("cpu_probe", "baseline_only")]
    schedule = []
    for repeat in range(3):
        offset = repeat * 5 % len(cells)
        order = cells[offset:] + cells[:offset]
        if repeat % 2:
            order = list(reversed(order))
        for cell in order:
            schedule.extend({**cell, "repeat": repeat, "phase": phase} for phase in PHASES)
    payload = {"format": "fusion-placement-search.reference-state-diagnostic", "version": "1.0",
        "freeze_id": freeze["freeze_id"], "cases": cases, "schedule": schedule,
        "parameters": {k: freeze["protocol"]["execution"][k] for k in ("seed", "rtol", "atol", "warmup", "iterations")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        "reference_targets": {"probe_deviation": .15, "baseline_drift": .10},
        "scope": "Three known training placements, two executor lifetimes, real/sham probe, three rounds. No Events or new labels.",
        "interpretation": "Descriptive factorial diagnostic; executor rebuilding also changes allocation/cache history. No host-profile export or acceptance promotion."}
    return {"protocol_id": content_id("reference-state-protocol", payload), **payload}


def cache_state(executor):
    tensors, counts = [], {}
    for name in ("_weights", "_inputs", "_norm_weights", "_rope_tables"):
        cache = getattr(executor, name)
        counts[name] = len(cache)
        for key, value in sorted(cache.items()):
            for index, tensor in enumerate(value if isinstance(value, tuple) else (value,)):
                tensors.append((name, str(key), index, str(tensor.device), list(tensor.shape), tensor.data_ptr()))
    return {"identity": content_id("reference-executor-cache", tensors), "entries": counts}


def memory_state(devices):
    return [{"device": str(device), "allocated": torch.cuda.memory_allocated(device),
             "reserved": torch.cuda.memory_reserved(device),
             "allocation_retries": torch.cuda.memory_stats(device).get("num_alloc_retries")}
            for device in devices if torch.device(device).type == "cuda"]


def measure_block(executor, parameters, reference, *, probe):
    def run():
        if probe:
            output, sample = executor.run_probe()
            return output, sample["total_s"], sample
        output, _, _, elapsed = executor.run_once(False)
        return output, elapsed, None

    with torch.no_grad():
        output, _, _ = run()
        first = compare_outputs(output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del output
        for _ in range(parameters["warmup"]):
            output, _, _ = run()
            del output
        ready = cache_state(executor)
        samples, probe_samples = [], []
        for index in range(parameters["iterations"]):
            output, elapsed, sample = run()
            samples.append(elapsed)
            if sample is not None:
                probe_samples.append(sample)
            if index + 1 < parameters["iterations"]:
                del output
        last = compare_outputs(output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del output
        finished = cache_state(executor)
    return {"samples_s": samples, "probe_samples": probe_samples,
        "catalog": executor.catalog if probe else {}, "cache_before_samples": ready, "cache_after_samples": finished,
        "correctness": {"pass": first["pass"] and last["pass"], "first": first, "last": last}}


def summarize(protocol, records):
    keys = ("case_id", "lifecycle", "middle_kind", "repeat", "phase")
    require([{k: row[k] for k in keys} for row in records] == protocol["schedule"], "missing, reordered or duplicate blocks")
    cases = {x["case_id"]: x for x in protocol["cases"]}
    triplets, groups = [], defaultdict(list)
    for index in range(0, len(records), 3):
        block = records[index:index + 3]
        first = block[0]
        stats = {row["phase"]: timing_statistics(row["samples_s"]) for row in block}
        before, middle, after = (stats[phase]["median_ms"] for phase in PHASES)
        instances = [row["executor_instance"] for row in block]
        require(len(set(instances)) == (1 if first["lifecycle"] == "reuse_triplet" else 3), "executor lifecycle mismatch")
        if first["lifecycle"] == "reuse_triplet":
            require(len({row["cache_before_samples"]["identity"] for row in block}) == 1, "reused executor changed warmed cache identity")
        entry = {**{k: first[k] for k in keys if k != "phase"}, "plan_name": cases[first["case_id"]]["plan_name"],
            "statistics": stats, "after_over_before_change_pct": (after / before - 1) * 100,
            "middle_over_before_change_pct": (middle / before - 1) * 100,
            "middle_over_after_change_pct": (middle / after - 1) * 100,
            "middle_within_reference_15pct": abs(middle / before - 1) <= .15 and abs(middle / after - 1) <= .15,
            "baseline_within_reference_10pct": abs(after / before - 1) <= .10,
            "executor_instances": instances}
        triplets.append(entry)
        groups[first["case_id"], first["lifecycle"], first["middle_kind"]].append(entry)
    return {"triplets": triplets, "cells": [{"case_id": key[0], "plan_name": cases[key[0]]["plan_name"],
            "lifecycle": key[1], "middle_kind": key[2], "rounds": len(rows),
            "baseline_change_pct_by_round": [r["after_over_before_change_pct"] for r in rows],
            "rounds_within_reference_10pct": sum(r["baseline_within_reference_10pct"] for r in rows)}
            for key, rows in groups.items()],
        "scope": "No calibrated flag or host parameters: diagnostic only. Three rounds are not independent tasks or a significance test."}


def audit(directory, bundle=DEFAULT_BUNDLE):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol, environment = read("protocol.json"), read("environment.json")
    freeze, manifest = load_freeze(bundle)
    require(content_id("compare", protocol) == content_id("compare", build_protocol(freeze, manifest)), "protocol/source mismatch")
    env_id = content_id("reference-state-environment", {k: v for k, v in environment.items() if k != "environment_id"})
    require(env_id == environment["environment_id"], "environment hash mismatch")
    records = load_jsonl_records(directory / "blocks.jsonl")
    cases = {x["case_id"]: x for x in protocol["cases"]}
    hardware = hardware_from_dict(freeze["hardware"])
    for row in records:
        require(row["record_id"] == content_id("reference-state-block", {k: v for k, v in row.items() if k != "record_id"}), "record hash mismatch")
        require(row["protocol_id"] == protocol["protocol_id"] and row["environment_id"] == env_id
                and row["freeze_id"] == freeze["freeze_id"], "lineage mismatch")
        require(row["correctness"]["pass"] and len(row["samples_s"]) == protocol["parameters"]["iterations"], "incorrect or incomplete block")
        require(all(math.isfinite(value) and value > 0 for value in row["samples_s"]), "invalid forward timing")
        require(row["cache_before_samples"] == row["cache_after_samples"], "cache changed during measured samples")
        probe = row["phase"] == "middle" and row["middle_kind"] == "cpu_probe"
        require(len(row["probe_samples"]) == (len(row["samples_s"]) if probe else 0), "unexpected probe coverage")
        require(not probe or row["samples_s"] == [s["total_s"] for s in row["probe_samples"]], "probe sample mismatch")
        require(all(o["gpu_stream_window_s"] is None for s in row["probe_samples"] for o in s["operations"]), "unexpected device windows")
        if probe:
            case = cases[row["case_id"]]
            placed = PlacedOpGraph(graph_from_dict(case["graph"]), hardware, plan_from_dict(case["plan"]))
            require(Counter((x["source"], x["destination"], x["payload_bytes"]) for x in expected_transfers(placed)) ==
                    Counter((x["source"], x["destination"], x["payload_bytes"]) for x in row["catalog"].values() if x["kind"] == "copy"),
                    "copy catalog mismatch")
            for sample in row["probe_samples"]:
                require(Counter(o["key"] for o in sample["operations"]) == Counter(row["catalog"].keys()), "operation catalog mismatch")
                require(math.isclose(sample["total_s"], sample["start_s"] + sum(o["dispatch_step_s"] for o in sample["operations"])
                                     + sample["busy_final_sync_s_not_a_host_parameter"], abs_tol=1e-12), "forward decomposition mismatch")
    summary = summarize(protocol, records)
    require(read("diagnostic_report.json") == summary, "summary mismatch")
    require(read("collection_status.json") == {"status": "complete_diagnostic", "records": len(records)}, "status mismatch")
    return {"sources_and_training_scope": True, "record_hashes_and_order": True, "cache_lifetime_and_no_events": True,
        "summary_reproduced": True, "records": len(records), "measured_forwards": sum(len(row["samples_s"]) for row in records),
        "calibration_promoted": False}


def collect(args):
    freeze, manifest = load_freeze(args.bundle)
    _validate_runtime(manifest, (0, 1))
    protocol = build_protocol(freeze, manifest)
    require(not args.out_dir.exists() or not any(args.out_dir.iterdir()), "use a new empty directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    devices, hardware, templates = ("cuda:0", "cuda:1"), hardware_from_dict(freeze["hardware"]), {}
    environment = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "hip": torch.version.hip, "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "devices": [{"device": d, "name": torch.cuda.get_device_properties(d).name} for d in devices],
        "peer_access": {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a, b in ((0, 1), (1, 0))}}
    env_id = content_id("reference-state-environment", environment)
    write_json(args.out_dir / "protocol.json", protocol)
    write_json(args.out_dir / "environment.json", {"environment_id": env_id, **environment})
    write_json(args.out_dir / "collection_status.json", {"status": "collecting", "records": 0})
    records, serial = [], 0
    cases = {x["case_id"]: x for x in protocol["cases"]}
    parameters = protocol["parameters"]
    try:
        graph = graph_from_dict(protocol["cases"][0]["graph"])
        ref_executor = CanonicalReferenceExecutor(PlacedOpGraph(graph, hardware, make_reference_plan(graph, device=0)),
            devices, parameters["seed"], templates)
        with torch.no_grad():
            output, _, _, _ = ref_executor.run_once(False)
            reference = output.float().cpu()
            del output, ref_executor
        gc.collect()
        for start in range(0, len(protocol["schedule"]), 3):
            slots = protocol["schedule"][start:start + 3]
            case = cases[slots[0]["case_id"]]
            placed = PlacedOpGraph(graph_from_dict(case["graph"]), hardware, plan_from_dict(case["plan"]))
            shared, shared_serial = None, None
            if slots[0]["lifecycle"] == "reuse_triplet":
                shared = SubmissionProbeExecutor(placed, devices, parameters["seed"], templates, environment_id=env_id)
                serial += 1
                shared_serial = serial
            for slot in slots:
                executor, instance = shared, shared_serial
                if executor is None:
                    executor = SubmissionProbeExecutor(placed, devices, parameters["seed"], templates, environment_id=env_id)
                    serial += 1
                    instance = serial
                probe = slot["phase"] == "middle" and slot["middle_kind"] == "cpu_probe"
                before_memory = memory_state(devices)
                wall_start, process_start = time.perf_counter(), time.process_time()
                result = measure_block(executor, parameters, reference, probe=probe)
                context = {"block_wall_s": time.perf_counter() - wall_start, "process_cpu_s": time.process_time() - process_start,
                    "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                    "memory_before": before_memory, "memory_after": memory_state(devices)}
                row = {**slot, "protocol_id": protocol["protocol_id"], "environment_id": env_id, "freeze_id": freeze["freeze_id"],
                    "executor_instance": instance, "context": context, **result}
                row["record_id"] = content_id("reference-state-block", row)
                append_jsonl_record(args.out_dir / "blocks.jsonl", row)
                records.append(row)
                print(f"block={len(records)}/108 round={slot['repeat']} {case['plan_name']} {slot['lifecycle']} {slot['middle_kind']} {slot['phase']} correct={result['correctness']['pass']}", flush=True)
                require(result["correctness"]["pass"], "output mismatch; keep the raw block")
                del executor
                gc.collect()
            del shared
            gc.collect()
        write_json(args.out_dir / "diagnostic_report.json", summarize(protocol, records))
        write_json(args.out_dir / "collection_status.json", {"status": "complete_diagnostic", "records": len(records)})
        write_json(args.out_dir / "audit_report.json", audit(args.out_dir, args.bundle))
        print(f"report: {args.out_dir / 'diagnostic_report.json'}; diagnostic only")
    except Exception as exc:
        write_json(args.out_dir / "collection_status.json", {"status": "incomplete", "records": len(records), "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out-dir", type=Path)
    group.add_argument("--audit-dir", type=Path)
    args = parser.parse_args(argv)
    if args.audit_dir:
        result = audit(args.audit_dir, args.bundle)
        write_json(args.audit_dir / "audit_report.json", result)
        print(json.dumps(result, indent=2))
    else:
        collect(args)


if __name__ == "__main__":
    main()
