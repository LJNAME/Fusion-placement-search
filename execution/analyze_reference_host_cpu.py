"""Describe received CPU probes and training-only reconstruction; no acceptance changes."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from calibration.calibrate_reference_host_cpu import audit_collection
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import ROOT, load_freeze
from cost_model.e2_srop_proxy import DatasetLatencyProfile
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from execution.audit_gpu_holdout import write_json
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def analyze(directory):
    directory = Path(directory)
    audit = audit_collection(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol, report = read("protocol.json"), read("calibration_report.json")
    records = load_jsonl_records(directory / "blocks.jsonl")
    host = HostSubmissionProfile(**read("host_submission_profile.json"))
    failed = [row for row in report["case_rounds"] if not row["target_pass"]]
    failed_ids = {row["case_id"] for row in failed}
    details = []
    for row in report["case_rounds"]:
        if row["case_id"] not in failed_ids:
            continue
        block = next(x for x in records if x["case_id"] == row["case_id"] and x["repeat"] == row["repeat"] and x["mode"] == "cpu_probe")
        operations = defaultdict(lambda: defaultdict(list))
        for sample in block["probe_samples"]:
            for item in sample["operations"]:
                for key in ("api_call_s", "dispatch_step_s", "bookkeeping_after_s"):
                    operations[item["key"]][key].append(item[key])
        samples = block["probe_samples"]
        details.append({**row,
            "cpu_dispatch_median_ms": statistics.median(x["start_s"] + sum(o["dispatch_step_s"] for o in x["operations"]) for x in samples) * 1000,
            "busy_final_sync_median_ms": statistics.median(x["busy_final_sync_s_not_a_host_parameter"] for x in samples) * 1000,
            "operations": [{"key": key, "catalog": block["catalog"][key],
                **{field.removesuffix("_s") + "_us": statistics.median(values) * 1e6 for field, values in fields.items()}}
                for key, fields in operations.items()]})

    # Round-specific host parameter variation is descriptive. Do not discard
    # the failed round or use cross-round variation as a new acceptance gate.
    round_parameters = []
    for repeat in range(protocol["rounds"]):
        grouped = defaultdict(list)
        for block in records:
            if block["repeat"] != repeat or block["mode"] != "cpu_probe":
                continue
            values = defaultdict(list)
            for sample in block["probe_samples"]:
                values["start_s"].append(sample["start_s"])
                values["finish_s"].append(sample["idle_sync_s"])
                for operation in sample["operations"]:
                    item = block["catalog"][operation["key"]]
                    key = "copy_submit_s" if item["kind"] == "copy" else "local:" + item["local_name"]
                    values[key].append(operation["dispatch_step_s"])
            for key, items in values.items():
                grouped[key].append(statistics.median(items))
        round_parameters.append({key: statistics.median(values) for key, values in grouped.items()})

    freeze, _ = load_freeze()
    hardware = hardware_from_dict(freeze["hardware"])
    train = load_jsonl_records(ROOT / "data/bw1000_e2/profiling_dataset.jsonl")
    durations = {row["object_id"]: row["measurement"]["duration_s"] for row in train}
    by_model = defaultdict(list)
    for case in protocol["cases"]:
        graph, plan = graph_from_dict(case["graph"]), plan_from_dict(case["plan"])
        mapping = {graph.nodes[node].extra["local_name"]: key for node, key in case["node_training_object_ids"].items() if key}
        profile = DatasetLatencyProfile(hardware, mapping, durations, {key: "old_training_measurement" for key in durations})
        estimate = {"zero_host_ms": ReferenceEagerCostModel().predict(graph, plan, profile).makespan * 1000,
            "rejected_host_candidate_ms": ReferenceEagerCostModel(host).predict(graph, plan, profile).makespan * 1000}
        for repeat in range(protocol["rounds"]):
            times = [value for row in records if row["case_id"] == case["case_id"] and row["repeat"] == repeat
                     and row["mode"].startswith("baseline") for value in row["samples_s"]]
            by_model[case["model_id"], repeat].append({"plan_name": case["plan_name"],
                "baseline_pooled_ms": statistics.median(times) * 1000, **estimate})
    reconstruction = []
    for (model_id, repeat), plans in by_model.items():
        best = min(row["baseline_pooled_ms"] for row in plans)
        metrics = {}
        for method in ("zero_host_ms", "rejected_host_candidate_ms"):
            selected = min(plans, key=lambda row: (row[method], row["plan_name"]))
            metrics[method] = {"selected": selected["plan_name"],
                "regret_pct": (selected["baseline_pooled_ms"] / best - 1) * 100,
                "mape_pct": statistics.fmean(abs(row[method] / row["baseline_pooled_ms"] - 1) for row in plans) * 100}
        reconstruction.append({"model_id": model_id, "repeat": repeat, "metrics": metrics, "plans": plans})
    q = report["case_rounds"]
    ratios = [row[key] for row in q for key in ("probe_over_before_ratio", "probe_over_after_ratio")]
    return {"audit": audit, "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in
            ("protocol.json", "environment.json", "blocks.jsonl", "host_submission_profile.json", "calibration_report.json")},
        "all_correctness_blocks_pass": all(row["correctness"]["pass"] for row in records),
        "cpu_deviation_only_pass": sum(all(abs(row[key] - 1) <= .15 for key in
            ("probe_over_before_ratio", "probe_over_after_ratio")) for row in q),
        "baseline_drift_pass": sum(abs(row["baseline_shift_fraction"]) <= .1 for row in q),
        "cpu_probe_ratio_range": [min(ratios), max(ratios)],
        "max_across_round_baseline_range_pct": max(row["baseline_across_round_range_fraction"] for row in report["cases"]) * 100,
        "samples_above_twice_block_median": sum(stats["samples_above_twice_median"]
            for row in q for stats in row["block_statistics"].values()),
        "training_reconstruction_summary": {"training_workloads": len({row["model_id"] for row in reconstruction}),
            "rounds": protocol["rounds"], "independent_validation": False,
            **{method: {key + "_range": [min(row["metrics"][method][key] for row in reconstruction),
                                         max(row["metrics"][method][key] for row in reconstruction)]
                for key in ("regret_pct", "mape_pct")}
               for method in ("zero_host_ms", "rejected_host_candidate_ms")}},
        "failed_case_rounds": failed, "failed_case_all_rounds": details,
        "round_parameter_ranges_us": {key: {"by_round": [row[key] * 1e6 for row in round_parameters],
            "max_over_min_change_pct": (max(row[key] for row in round_parameters) / min(row[key] for row in round_parameters) - 1) * 100}
            for key in round_parameters[0]},
        "training_reconstruction": reconstruction,
        "limitations": ["Parameters use all training cases and rounds including the failure; acceptance remains false.",
            "Reconstruction reuses these calibration tasks and is not independent validation or SROP evidence.",
            "Uninstrumented baseline has no per-operation times; current records cannot locate its drift in a specific API or GPU kernel.",
            "CPU call times can include driver waits and GPU overlap; event-free does not establish zero observer effect.",
            "Parameter ranges and three-round comparisons are descriptive, not significance or new acceptance tests."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    write_json(args.directory / "analysis_report.json", result)
    print("audit:", result["audit"])
    print("CPU ratio range:", result["cpu_probe_ratio_range"])
    for row in result["failed_case_all_rounds"]:
        print("failed-case repeat", row["repeat"], "baseline", row["block_statistics"]["baseline_before"]["median_ms"],
            row["block_statistics"]["baseline_after"]["median_ms"], "dispatch", row["cpu_dispatch_median_ms"],
            "busy sync", row["busy_final_sync_median_ms"])
    for row in result["training_reconstruction"]:
        print(row["model_id"], row["repeat"], row["metrics"])


if __name__ == "__main__":
    main()
