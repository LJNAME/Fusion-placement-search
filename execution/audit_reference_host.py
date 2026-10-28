"""Audit received training-side submission probes without accepting their profile."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import ROOT, load_freeze
from cost_model.e2_srop_proxy import DatasetLatencyProfile
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from execution.audit_gpu_holdout import timing_statistics, write_json
from execution.diagnose_gpu_holdout import expected_transfers
from execution.host_submission import summarize_blocks, training_cases
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol, environment = read("protocol.json"), read("environment.json")
    report, host = read("calibration_report.json"), read("host_submission_profile.json")
    windows, status = read("body_window_candidates.json"), read("collection_status.json")
    records = load_jsonl_records(directory / "blocks.jsonl")
    freeze, manifest = load_freeze()
    hardware = hardware_from_dict(freeze["hardware"])
    for value, field, prefix in ((protocol, "protocol_id", "reference-host-calibration-protocol"),
                                (environment, "environment_id", "reference-host-environment")):
        require(value[field] == content_id(prefix, {k: v for k, v in value.items() if k != field}), field + " mismatch")
    require(protocol["freeze_id"] == freeze["freeze_id"], "freeze mismatch")
    require(content_id("training-cases", protocol["cases"]) == content_id("training-cases", training_cases(freeze, manifest)),
            "training cases differ from the declared recipe")
    require(all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                for name, digest in protocol["source_sha256"].items()), "collection source mismatch")
    cases = {row["case_id"]: row for row in protocol["cases"]}
    modes = {"baseline_before", "baseline_after", "cpu_probe", "window_probe"}
    counts = Counter((row["case_id"], row["mode"]) for row in records)
    require(set(counts) == {(key, mode) for key in cases for mode in modes}
            and all(n == 1 for n in counts.values()), "missing, extra or duplicate blocks")
    probe_operations = 0
    for row in records:
        require(row["record_id"] == content_id("reference-host-calibration-block",
                {k: v for k, v in row.items() if k != "record_id"}), "record hash mismatch")
        require(all(row[key] == source[key] for key, source in
                    (("freeze_id", freeze), ("protocol_id", protocol), ("environment_id", environment))), "block lineage mismatch")
        case = cases[row["case_id"]]
        require(all(row[key] == case[key] for key in ("model_id", "plan_name")), "block case mismatch")
        require(row["correctness"]["pass"], "incorrect output")
        require(len(row["samples_s"]) == protocol["parameters"]["iterations"]
                and all(math.isfinite(x) and x > 0 for x in row["samples_s"]), "invalid forward samples")
        if row["mode"].startswith("baseline"):
            require(row["median_s"] == statistics.median(row["samples_s"]), "baseline median mismatch")
            continue
        require([s["total_s"] for s in row["probe_samples"]] == row["samples_s"], "probe samples mismatch")
        graph, plan = graph_from_dict(case["graph"]), plan_from_dict(case["plan"])
        placed = PlacedOpGraph(graph, hardware, plan)
        expected = Counter((x["source"], x["destination"], x["payload_bytes"]) for x in expected_transfers(placed))
        observed = Counter((x["source"], x["destination"], x["payload_bytes"])
                           for x in row["catalog"].values() if x["kind"] == "copy")
        require(observed == expected, "copy catalog mismatch")
        for item in row["catalog"].values():
            if item["kind"] == "node":
                require(item["body_id"] == content_id("reference-body-window", item["body_key"]), "body hash mismatch")
                require(item["body_key"]["environment_id"] == environment["environment_id"], "body environment mismatch")
        for sample in row["probe_samples"]:
            operations = sample["operations"]
            require(Counter(x["key"] for x in operations) == Counter(row["catalog"].keys()), "operation catalog coverage mismatch")
            probe_operations += len(operations)
            for operation in operations:
                require(all(math.isfinite(operation[key]) and operation[key] >= 0 for key in
                            ("start_s", "api_call_s", "dispatch_step_s", "bookkeeping_after_s")), "invalid CPU timing")
                require(math.isclose(operation["dispatch_step_s"], operation["api_call_s"] + operation["bookkeeping_after_s"],
                                     abs_tol=1e-12), "dispatch decomposition mismatch")
                value = operation["gpu_stream_window_s"]
                require(value is None or (math.isfinite(value) and value >= 0), "invalid stream window")
                require(row["mode"] != "cpu_probe" or value is None, "CPU probe contains device windows")
            body_end = sample["start_s"] + sum(x["dispatch_step_s"] for x in operations)
            require(math.isclose(sample["total_s"], body_end + sample["busy_final_sync_s_not_a_host_parameter"],
                                 abs_tol=1e-12), "forward decomposition mismatch")
    summary = summarize_blocks(records)
    require(all(report[key] == value for key, value in summary.items()), "stored summary differs from raw records")
    data_id = content_id("reference-host-calibration-data", records)
    reproduced_host = asdict(HostSubmissionProfile(**summary["host_parameters"], source=f"reference-host-calibration:{data_id}"))
    require(host == reproduced_host == report["host_profile"], "host export mismatch")
    require(report["data_id"] == data_id and report["environment_id"] == environment["environment_id"]
            and report["protocol_id"] == protocol["protocol_id"], "report lineage mismatch")
    expected_status = "complete" if host["calibrated"] else "complete_cpu_probe_perturbed"
    forwards = sum(len(row["samples_s"]) for row in records)
    require(report["status"] == status["status"] == expected_status and report["records"] == status["records"] == len(records)
            and report["measured_forwards"] == forwards, "completion status mismatch")
    require(windows["windows"] == summary["body_window_candidates"]
            and windows["status"] == "diagnostic_candidates_only", "window export mismatch")
    catalog = {x["body_id"]: x["body_key"] for row in records if row["mode"] == "window_probe"
               for x in row["catalog"].values() if x["kind"] == "node"}
    require(windows["body_catalog"] == catalog and windows["environment_id"] == environment["environment_id"], "body catalog export mismatch")

    details = []
    for case_id, case in cases.items():
        blocks = {r["mode"]: r for r in records if r["case_id"] == case_id}
        quality = next(x for x in summary["cases"] if x["case_id"] == case_id)
        cpu = blocks["cpu_probe"]["probe_samples"]
        details.append({**quality, "block_statistics": {mode: timing_statistics(row["samples_s"]) for mode, row in blocks.items()},
            "cpu_probe_dispatch_median_ms": statistics.median(s["start_s"] + sum(o["dispatch_step_s"] for o in s["operations"]) for s in cpu) * 1000,
            "cpu_probe_busy_tail_median_ms": statistics.median(s["busy_final_sync_s_not_a_host_parameter"] for s in cpu) * 1000,
            "cpu_probe_host_fraction_median": statistics.median((s["total_s"] - s["busy_final_sync_s_not_a_host_parameter"]) / s["total_s"] for s in cpu),
            "cpu_probe_idle_sync_median_us": statistics.median(s["idle_sync_s"] for s in cpu) * 1e6})

    # Development-only ablation on this same host. Rejected candidate host
    # parameters remain rejected; original training microbench labels are used
    # unchanged, with operator-body/environment mismatch explicitly retained.
    train = load_jsonl_records(ROOT / "data/bw1000_e2/profiling_dataset.jsonl")
    durations = {r["object_id"]: r["measurement"]["duration_s"] for r in train}
    predictions = defaultdict(list)
    for case in cases.values():
        graph, plan = graph_from_dict(case["graph"]), plan_from_dict(case["plan"])
        mapping = {graph.nodes[node].extra["local_name"]: key for node, key in case["node_training_object_ids"].items() if key}
        profile = DatasetLatencyProfile(hardware, mapping, durations, {key: "old_training_measurement" for key in durations})
        measured = statistics.median([s for r in records if r["case_id"] == case["case_id"]
                                      and r["mode"].startswith("baseline") for s in r["samples_s"]])
        predictions[case["model_id"]].append({"plan_name": case["plan_name"], "baseline_pooled_ms": measured * 1000,
            "zero_host_ms": ReferenceEagerCostModel().predict(graph, plan, profile).makespan * 1000,
            "rejected_host_candidate_ms": ReferenceEagerCostModel(HostSubmissionProfile(**host)).predict(graph, plan, profile).makespan * 1000})
    workloads = []
    for model_id, plans in predictions.items():
        best = min(x["baseline_pooled_ms"] for x in plans)
        ablations = {}
        for method in ("zero_host_ms", "rejected_host_candidate_ms"):
            selected = min(plans, key=lambda x: (x[method], x["plan_name"]))
            ablations[method] = {"selected": selected["plan_name"], "regret_pct": (selected["baseline_pooled_ms"] / best - 1) * 100,
                "mape_pct": statistics.fmean(abs(x[method] / x["baseline_pooled_ms"] - 1) for x in plans) * 100}
        workloads.append({"model_id": model_id, "plans": plans, "ablations": ablations})
    return {"format": "fusion-placement-search.reference-host-audit", "checks": {"hashes_sources_and_lineage": True,
            "training_cases_and_keys": True, "samples_operations_and_copy_catalogs": True, "all_exports_reproduced": True},
        "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in
            ("protocol.json", "environment.json", "blocks.jsonl", "calibration_report.json", "host_submission_profile.json",
             "body_window_candidates.json", "collection_status.json")},
        "records": len(records), "measured_forwards": forwards, "probe_operations": probe_operations,
        "correctness_blocks_pass": len(records), "cpu_targets_pass": sum(x["cpu_probe"]["target_pass"] for x in details),
        "cpu_deviation_only_targets_pass": sum(all(abs(x["cpu_probe"][key] - 1) <= .15 for key in
            ("over_before_ratio", "over_after_ratio")) for x in details),
        "window_targets_pass": sum(x["window_probe"]["target_pass"] for x in details),
        "all_window_data_complete": summary["all_window_data_complete"],
        "baseline_drift_targets_pass": sum(abs(x["baseline_shift_fraction"]) <= .1 for x in details),
        "failed_cpu_cases": [{"model_id": x["model_id"], "plan_name": x["plan_name"],
            "baseline_before_ms": x["baseline_before_s"] * 1000, "baseline_after_ms": x["baseline_after_s"] * 1000,
            "baseline_shift_pct": x["baseline_shift_fraction"] * 100} for x in details if not x["cpu_probe"]["target_pass"]],
        "cpu_ratio_range": [min(x["cpu_probe"][key] for x in details for key in ("over_before_ratio", "over_after_ratio")),
                            max(x["cpu_probe"][key] for x in details for key in ("over_before_ratio", "over_after_ratio"))],
        "window_ratio_range": [min(x["window_probe"][key] for x in details for key in ("over_before_ratio", "over_after_ratio")),
                               max(x["window_probe"][key] for x in details for key in ("over_before_ratio", "over_after_ratio"))],
        "profile_accepted": host["calibrated"], "cases": details, "training_reconstruction": workloads,
        "limitations": ["Training-side reconstruction with exposed labels, not independent validation or SROP benefit.",
            "Rejected host parameters are a diagnostic only; no failed case or outlier is removed and no target is relaxed.",
            "Window probes fail observer targets; their body windows are not substituted into service or link costs.",
            "CPU API time may include driver blocking; dispatch duration is not pure kernel-launch overhead.",
            "Old operator microbench implementation and environment compatibility remain unverified."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    write_json(args.directory / "audit_report.json", result)
    print(f"records={result['records']} forwards={result['measured_forwards']} CPU={result['cpu_targets_pass']}/16 window={result['window_targets_pass']}/16 accepted={result['profile_accepted']}")
    for row in result["training_reconstruction"]:
        print(row["model_id"], row["ablations"])


if __name__ == "__main__":
    main()
