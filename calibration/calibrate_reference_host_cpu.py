"""Fixed three-round, CPU-only submission probes on 16 old training cases."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
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

import torch

from calibration.calibrate_reference_host import measure_probe
from calibration.e2_manifest import content_id
from calibration.e2_records import append_jsonl_record, load_jsonl_records
from calibration.gpu_holdout import DEFAULT_BUNDLE, ROOT, load_freeze
from calibration.profile_e2_bw1000 import _validate_runtime
from cost_model.reference_semantics import HostSubmissionProfile
from execution.audit_gpu_holdout import timing_statistics, write_json
from execution.diagnose_gpu_holdout import expected_transfers
from execution.gpu_holdout import CanonicalReferenceExecutor, measure_candidate
from execution.host_submission import training_cases
from simulator import PlacedOpGraph, make_reference_plan
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


ROUNDS = 3
MODES = ("baseline_before", "cpu_probe", "baseline_after")
DEVIATION_TARGET = .15
DRIFT_TARGET = .10
SOURCES = ("calibration/calibrate_reference_host_cpu.py", "calibration/calibrate_reference_host.py",
    "execution/host_submission.py", "execution/reference.py", "execution/gpu_holdout.py",
    "cost_model/reference_semantics.py", "simulator/model_graphs/dense_layer.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_protocol(freeze, manifest):
    cases = training_cases(freeze, manifest)
    schedule = []
    for repeat in range(ROUNDS):
        offset = (5 * repeat) % len(cases)
        order = cases[offset:] + cases[:offset]
        if repeat % 2:
            order = list(reversed(order))
        schedule.extend({"repeat": repeat, "case_id": case["case_id"], "mode": mode}
                        for case in order for mode in MODES)
    payload = {"format": "fusion-placement-search.reference-host-cpu-calibration", "version": "1.0",
        "freeze_id": freeze["freeze_id"], "cases": cases, "rounds": ROUNDS, "schedule": schedule,
        "parameters": {key: freeze["protocol"]["execution"][key] for key in ("seed", "rtol", "atol", "warmup", "iterations")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        "acceptance": {"probe_deviation_target": DEVIATION_TARGET, "baseline_shift_target": DRIFT_TARGET,
            "rule": "Every case in every round must pass both observer comparisons and baseline drift; no favorable-round selection.",
            "across_round_baseline_range": "Reported descriptively, not a new acceptance gate."},
        "scope": "Old training cases only; no device Events/profiler, service-window fitting, new sequence labels or independent validation.",
        "aggregation": "Median within each case-round, median across its three rounds, then median across all 16 cases; all rounds retained."}
    return {"protocol_id": content_id("reference-host-cpu-protocol", payload), **payload}


def summarize_cpu_blocks(protocol, records):
    require([{k: row[k] for k in ("repeat", "case_id", "mode")} for row in records] == protocol["schedule"],
            "missing, extra, reordered or duplicate CPU blocks")
    by_slot = {(row["case_id"], row["repeat"], row["mode"]): row for row in records}
    parameters, quality, cases = defaultdict(lambda: defaultdict(list)), [], []
    for case in protocol["cases"]:
        baselines, round_quality = [], []
        for repeat in range(protocol["rounds"]):
            blocks = {mode: by_slot[case["case_id"], repeat, mode] for mode in MODES}
            stats = {mode: timing_statistics(row["samples_s"]) for mode, row in blocks.items()}
            before, probe, after = (stats[mode]["median_ms"] / 1000 for mode in MODES)
            ratios, shift = [probe / before, probe / after], after / before - 1
            passed = all(abs(value - 1) <= DEVIATION_TARGET for value in ratios) and abs(shift) <= DRIFT_TARGET
            entry = {"case_id": case["case_id"], "model_id": case["model_id"], "plan_name": case["plan_name"],
                "repeat": repeat, "probe_over_before_ratio": ratios[0], "probe_over_after_ratio": ratios[1],
                "baseline_shift_fraction": shift, "target_pass": passed, "block_statistics": stats}
            quality.append(entry)
            round_quality.append(entry)
            baselines.append(statistics.median([*blocks["baseline_before"]["samples_s"], *blocks["baseline_after"]["samples_s"]]))
            per_key = defaultdict(list)
            for sample in blocks["cpu_probe"]["probe_samples"]:
                per_key["start_s"].append(sample["start_s"])
                per_key["finish_s"].append(sample["idle_sync_s"])
                for operation in sample["operations"]:
                    item = blocks["cpu_probe"]["catalog"][operation["key"]]
                    if item["kind"] == "copy":
                        per_key["copy_submit_s"].append(operation["dispatch_step_s"])
                    else:
                        per_key["op:" + item["operator"]].append(operation["dispatch_step_s"])
                        per_key["local:" + item["local_name"]].append(operation["dispatch_step_s"])
            for key, values in per_key.items():
                parameters[case["case_id"]][key].append(statistics.median(values))
        cases.append({"case_id": case["case_id"], "model_id": case["model_id"], "plan_name": case["plan_name"],
            "all_rounds_pass": all(x["target_pass"] for x in round_quality),
            "baseline_round_medians_ms": [value * 1000 for value in baselines],
            "baseline_across_round_range_fraction": max(baselines) / min(baselines) - 1})
    combined = defaultdict(list)
    for values in parameters.values():
        for key, rounds in values.items():
            combined[key].append(statistics.median(rounds))
    operators = {key[3:]: statistics.median(values) for key, values in combined.items() if key.startswith("op:")}
    local = {key: statistics.median(values) for key, values in combined.items() if key.startswith("local:")}
    passed = bool(quality) and bool(combined["copy_submit_s"]) and all(x["target_pass"] for x in quality)
    return {"host_parameters": {"start_s": statistics.median(combined["start_s"]),
            "finish_s": statistics.median(combined["finish_s"]), "node_submit_s": statistics.median(operators.values()),
            "copy_submit_s": statistics.median(combined["copy_submit_s"]), "operator_submit_s": {**operators, **local},
            "calibrated": passed}, "case_rounds": quality, "cases": cases,
        "case_round_targets_pass": sum(x["target_pass"] for x in quality),
        "case_targets_pass": sum(x["all_rounds_pass"] for x in cases),
        "all_cpu_probe_targets_pass": passed, "aggregation": protocol["aggregation"]}


def audit_collection(directory, bundle=DEFAULT_BUNDLE):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol, environment = read("protocol.json"), read("environment.json")
    report, host, status = read("calibration_report.json"), read("host_submission_profile.json"), read("collection_status.json")
    freeze, manifest = load_freeze(bundle)
    require(content_id("cpu-protocol-comparison", protocol) == content_id("cpu-protocol-comparison", build_protocol(freeze, manifest)),
            "protocol, training cases or collection sources differ")
    env_id = content_id("reference-host-cpu-environment", {k: v for k, v in environment.items() if k != "environment_id"})
    require(env_id == environment["environment_id"], "environment hash mismatch")
    records = load_jsonl_records(directory / "blocks.jsonl")
    cases = {case["case_id"]: case for case in protocol["cases"]}
    hardware = hardware_from_dict(freeze["hardware"])
    for row in records:
        require(row["record_id"] == content_id("reference-host-cpu-block", {k: v for k, v in row.items() if k != "record_id"}),
                "record hash mismatch")
        require(row["freeze_id"] == freeze["freeze_id"] and row["environment_id"] == env_id
                and row["protocol_id"] == protocol["protocol_id"], "block lineage mismatch")
        case = cases[row["case_id"]]
        require(all(row[key] == case[key] for key in ("model_id", "plan_name")), "case identity mismatch")
        require(row["correctness"]["pass"], "output mismatch")
        require(len(row["samples_s"]) == protocol["parameters"]["iterations"]
                and all(math.isfinite(x) and x > 0 for x in row["samples_s"]), "invalid sample count or timing")
        if row["mode"] != "cpu_probe":
            require(row["median_s"] == statistics.median(row["samples_s"]), "baseline median mismatch")
            continue
        require([x["total_s"] for x in row["probe_samples"]] == row["samples_s"], "probe timing mismatch")
        placed = PlacedOpGraph(graph_from_dict(case["graph"]), hardware, plan_from_dict(case["plan"]))
        expected = Counter((x["source"], x["destination"], x["payload_bytes"]) for x in expected_transfers(placed))
        observed = Counter((x["source"], x["destination"], x["payload_bytes"])
                           for x in row["catalog"].values() if x["kind"] == "copy")
        require(expected == observed, "copy catalog mismatch")
        for sample in row["probe_samples"]:
            require(Counter(x["key"] for x in sample["operations"]) == Counter(row["catalog"].keys()), "operation catalog mismatch")
            for operation in sample["operations"]:
                require(operation["gpu_stream_window_s"] is None, "CPU protocol contains device windows")
                require(all(math.isfinite(operation[k]) and operation[k] >= 0 for k in
                            ("api_call_s", "start_s", "dispatch_step_s", "bookkeeping_after_s")), "invalid CPU operation")
                require(math.isclose(operation["dispatch_step_s"], operation["api_call_s"] + operation["bookkeeping_after_s"],
                                     abs_tol=1e-12), "dispatch decomposition mismatch")
            require(math.isclose(sample["total_s"], sample["start_s"] + sum(x["dispatch_step_s"] for x in sample["operations"])
                                 + sample["busy_final_sync_s_not_a_host_parameter"], abs_tol=1e-12), "forward decomposition mismatch")
    summary = summarize_cpu_blocks(protocol, records)
    require(all(report[k] == v for k, v in summary.items()), "summary does not reproduce")
    data_id = content_id("reference-host-cpu-data", records)
    reproduced = asdict(HostSubmissionProfile(**summary["host_parameters"], source=f"reference-host-cpu-calibration:{data_id}"))
    require(host == report["host_profile"] == reproduced, "host export mismatch")
    expected_status = "complete" if host["calibrated"] else "complete_cpu_probe_perturbed"
    forwards = sum(len(x["samples_s"]) for x in records)
    require(report["data_id"] == data_id and report["protocol_id"] == protocol["protocol_id"]
            and report["environment_id"] == env_id, "report lineage mismatch")
    require(report["records"] == status["records"] == len(records) and report["measured_forwards"] == forwards
            and report["status"] == status["status"] == expected_status, "status mismatch")
    return {"checks": {"sources_training_protocol_and_environment": True, "journal_hashes_and_slots": True,
            "no_device_windows": True, "summary_and_host_exports": True}, "records": len(records),
        "measured_forwards": forwards, "case_round_targets_pass": summary["case_round_targets_pass"],
        "case_targets_pass": summary["case_targets_pass"], "profile_accepted": host["calibrated"],
        "scope": "Collection integrity and observer checks only; no independent performance or SROP validation."}


def collect(args):
    freeze, manifest = load_freeze(args.bundle)
    _validate_runtime(manifest, (0, 1))  # Deliberately no CPU fallback.
    protocol = build_protocol(freeze, manifest)
    require(not args.out_dir.exists() or not any(args.out_dir.iterdir()), "use an empty output directory; do not mix runs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    environment = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "logical_cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "torch": torch.__version__, "hip": torch.version.hip,
        "devices": [{"index": i, "name": torch.cuda.get_device_properties(i).name,
            "total_memory": torch.cuda.get_device_properties(i).total_memory} for i in (0, 1)],
        "peer_access": {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a, b in ((0, 1), (1, 0))}}
    env_id = content_id("reference-host-cpu-environment", environment)
    write_json(args.out_dir / "protocol.json", protocol)
    write_json(args.out_dir / "environment.json", {"environment_id": env_id, **environment})
    write_json(args.out_dir / "collection_status.json", {"status": "collecting", "records": 0})
    devices, hardware, templates = ("cuda:0", "cuda:1"), hardware_from_dict(freeze["hardware"]), {}
    references, records = {}, []
    cases = {case["case_id"]: case for case in protocol["cases"]}
    parameters = protocol["parameters"]
    try:
        for slot in protocol["schedule"]:
            case = cases[slot["case_id"]]
            graph = graph_from_dict(case["graph"])
            if case["model_id"] not in references:
                executor = CanonicalReferenceExecutor(PlacedOpGraph(graph, hardware, make_reference_plan(graph, device=0)),
                    devices, parameters["seed"], templates)
                with torch.no_grad():
                    output, _, _, _ = executor.run_once(False)
                    references[case["model_id"]] = output.float().cpu()
                    del output, executor
                gc.collect()
            placed = PlacedOpGraph(graph, hardware, plan_from_dict(case["plan"]))
            result = (measure_probe(placed, devices, parameters, templates, references[case["model_id"]],
                        mode="cpu_probe", environment_id=env_id) if slot["mode"] == "cpu_probe" else
                      measure_candidate(placed, devices, parameters, templates, references[case["model_id"]]))
            row = {**slot, "freeze_id": freeze["freeze_id"], "protocol_id": protocol["protocol_id"],
                "environment_id": env_id, "model_id": case["model_id"], "plan_name": case["plan_name"], **result}
            row["record_id"] = content_id("reference-host-cpu-block", row)
            append_jsonl_record(args.out_dir / "blocks.jsonl", row)
            records.append(row)
            print(f"block={len(records)}/{len(protocol['schedule'])} round={slot['repeat']} {case['model_id']} {case['plan_name']} {slot['mode']} correctness={result['correctness']['pass']}", flush=True)
            require(result["correctness"]["pass"], "output mismatch; retain the failed block")
            gc.collect()
        summary = summarize_cpu_blocks(protocol, records)
        data_id = content_id("reference-host-cpu-data", records)
        host = asdict(HostSubmissionProfile(**summary["host_parameters"], source=f"reference-host-cpu-calibration:{data_id}"))
        status = "complete" if host["calibrated"] else "complete_cpu_probe_perturbed"
        write_json(args.out_dir / "host_submission_profile.json", host)
        write_json(args.out_dir / "calibration_report.json", {"status": status, "protocol_id": protocol["protocol_id"],
            "environment_id": env_id, "data_id": data_id, "records": len(records),
            "measured_forwards": sum(len(x["samples_s"]) for x in records), **summary, "host_profile": host,
            "limitations": ["Host/API calibration is local to this host/runtime and may include driver blocking.",
                "Busy final GPU wait is excluded from finish_s; only already-idle sync is used.",
                "All rounds, cases and samples are retained even if acceptance fails; old runs are not merged.",
                "No body windows, service-cost refit, default-search change or independent validation."]})
        write_json(args.out_dir / "collection_status.json", {"status": status, "records": len(records)})
        result = audit_collection(args.out_dir, args.bundle)
        write_json(args.out_dir / "audit_report.json", result)
        print(f"case-rounds={summary['case_round_targets_pass']}/48 cases={summary['case_targets_pass']}/16 calibrated={host['calibrated']}")
    except Exception as exc:
        write_json(args.out_dir / "collection_status.json", {"status": "incomplete", "records": len(records),
            "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out-dir", type=Path)
    action.add_argument("--audit-dir", type=Path)
    args = parser.parse_args(argv)
    if args.audit_dir is not None:
        result = audit_collection(args.audit_dir, args.bundle)
        write_json(args.audit_dir / "audit_report.json", result)
        print(json.dumps(result, indent=2))
    else:
        collect(args)


if __name__ == "__main__":
    main()
