"""Post-hoc profiling opportunity on exposed, fixed GPU candidate subsets.

This is not active calibration. Exact single-key replacements do not refit a
predictor, and the all-key endpoint is not an upper bound over partial updates.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import load_freeze
from calibration.reference_validation import check, DEFAULT_BUNDLE
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from execution.audit_gpu_holdout import audit_execution, write_json
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def decision_metrics(scores, measured, epsilon=.01):
    """Select from predictions first, then evaluate against exposed GPU data."""
    if set(scores) != set(measured) or not scores:
        raise ValueError("prediction and measurement candidates must match")
    if any(not math.isfinite(value) or value <= 0 for value in [*scores.values(), *measured.values()]):
        raise ValueError("candidate times must be finite and positive")
    minimum = min(scores.values())
    ties = sorted(key for key, value in scores.items() if value <= minimum + 1e-12)
    selected, best = ties[0], min(measured.values())
    regret = measured[selected] / best - 1
    return {"selected_id": selected, "predicted_tie_ids": ties,
        "measured_ms": measured[selected] * 1000, "relative_regret": regret,
        "within_epsilon": regret <= epsilon,
        "epsilon_excess_ms": max(0., measured[selected] - (1 + epsilon) * best) * 1000,
        "tie_regret_range": [min(measured[key] / best - 1 for key in ties),
                             max(measured[key] / best - 1 for key in ties)]}


def assess_replacements(score, initial_durations, exact_durations, measured, epsilon=.01):
    """Enumerate one-key corrections and the all-key endpoint, never subsets."""
    if set(initial_durations) != set(exact_durations):
        raise ValueError("initial and exact kernel keys must match")
    initial = decision_metrics(score(initial_durations), measured, epsilon)
    full = decision_metrics(score(exact_durations), measured, epsilon)
    replacements = []
    for key, value in sorted(exact_durations.items()):
        decision = decision_metrics(score({**initial_durations, key: value}), measured, epsilon)
        replacements.append({"object_id": key, "decision": decision,
            "selection_changed": decision["selected_id"] != initial["selected_id"],
            "realized_time_saving_ms": initial["measured_ms"] - decision["measured_ms"],
            "epsilon_excess_reduction_ms": initial["epsilon_excess_ms"] - decision["epsilon_excess_ms"]})
    return {"initial": initial, "all_kernel_labels": full,
        "full_reveal_realized_saving_ms": initial["measured_ms"] - full["measured_ms"],
        "single_key_corrections": replacements,
        "single_key_selection_changes": sum(row["selection_changed"] for row in replacements),
        "single_key_positive_savings": sum(row["realized_time_saving_ms"] > 1e-9 for row in replacements),
        "single_key_harmful_changes": sum(row["realized_time_saving_ms"] < -1e-9 for row in replacements),
        "max_single_key_saving_ms": max((row["realized_time_saving_ms"] for row in replacements), default=0.)}


def analyze_case(freeze, workload, kernel_report, execution_report, host, epsilon):
    graph = graph_from_dict(workload["graph"])
    hardware = hardware_from_dict(freeze["hardware"])
    mapping = object_map_for_workload(freeze["query_features"], E2Workload(workload["model_id"],
        "exposed-development", workload["model_id"], workload["model_family"], workload["sequence_length"]))
    keys = {mapping[str(node.extra.get("local_name", node.id))] for node in graph.nodes.values()
            if node.op_type != "input"}
    exact = {row["object_id"]: row["measured_s"] for row in kernel_report["objects"]}
    initial = {key: freeze["zero_shot_predictions"][key]["nearest_s"] for key in sorted(keys)}
    exact = {key: exact[key] for key in sorted(keys)}
    source = next(row for row in execution_report["workloads"] if row["model_id"] == workload["model_id"])
    measured = {row["plan_id"]: row["median_s"] for row in source["candidates"]}
    plans = {key: plan_from_dict(workload["candidate_plans"][key]) for key in workload["execution_plan_ids"]}
    if set(measured) != set(plans):
        raise ValueError("GPU candidate subset differs from freeze")
    model = ReferenceEagerCostModel(host)

    def score(durations):
        profile = DatasetLatencyProfile(hardware, mapping, durations,
                                       {key: "exposed_diagnostic" for key in durations})
        return {key: model.predict(graph, plan, profile).makespan for key, plan in plans.items()}

    analysis = assess_replacements(score, initial, exact, measured, epsilon)
    reference = workload["reference_plan_id"]
    best = min(measured.values())
    return {"model_id": workload["model_id"], "freeze_id": freeze["freeze_id"],
        "measured_candidates": len(plans), "correctable_kernel_keys": len(keys),
        "execution_host": execution_report["environment"]["host"],
        "fixed_reference_regret": measured[reference] / best - 1,
        "fixed_reference_within_epsilon": measured[reference] / best - 1 <= epsilon,
        **analysis}


def analyze(root, epsilon=.01):
    current, _ = check(root / "data/reference_validation_v1")
    host = HostSubmissionProfile(**current["host_profile"])
    cases = [(root / "data/bw1000_gpu_holdout_v1", root / "data/bw1000_gpu_holdout_v1/results_received_20261028"),
             (root / "data/reference_validation_v1", root / "data/received_reference_validation_v1/reference_validation_v1_results")]
    workloads, inputs = [], []
    for bundle, results in cases:
        freeze, _ = load_freeze(bundle)
        read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
        kernel, execution = read("zero_shot_report.json"), read("execution_report.json")
        if kernel["freeze_id"] != freeze["freeze_id"]:
            raise ValueError("kernel report belongs to another freeze")
        audit_execution(freeze, load_jsonl_records(results / "execution_samples.jsonl"),
                        read("execution_environment.json"), execution)
        paths = [bundle / "freeze.json", results / "zero_shot_report.json", results / "execution_report.json",
                 results / "execution_samples.jsonl", results / "execution_environment.json"]
        inputs.extend(paths)
        for workload in freeze["workloads"]:
            row = analyze_case(freeze, workload, kernel, execution, host, epsilon)
            workloads.append(row)
            print(f"{row['model_id']}: initial_regret={row['initial']['relative_regret']:.6f} "
                  f"full_regret={row['all_kernel_labels']['relative_regret']:.6f} "
                  f"single_changes={row['single_key_selection_changes']}", flush=True)
    return {"format": "fusion-placement-search.reference-profiling-opportunity", "version": "1.0",
        "scope": "Exposed development diagnosis on fixed measured candidate subsets, not independent SROP evidence.",
        "epsilon": epsilon, "host_profile": current["host_profile"],
        "input_sha256": {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": {"tasks": len(workloads),
            "initial_within_epsilon": sum(row["initial"]["within_epsilon"] for row in workloads),
            "all_labels_within_epsilon": sum(row["all_kernel_labels"]["within_epsilon"] for row in workloads),
            "fixed_reference_within_epsilon": sum(row["fixed_reference_within_epsilon"] for row in workloads),
            "single_key_corrections": sum(len(row["single_key_corrections"]) for row in workloads),
            "single_key_selection_changes": sum(row["single_key_selection_changes"] for row in workloads),
            "single_key_positive_savings": sum(row["single_key_positive_savings"] for row in workloads)},
        "workloads": workloads,
        "limitations": [
            "The old four tasks use kernel report labels; their original kernel raw journal is not re-audited here.",
            "Both batches are exposed; historical hosts are kept separate per task and host compatibility is unverified.",
            "Exact times are pooled across physical GPUs; no GPU-specific prediction or post-label physical-card selection.",
            "One-key replacement does not simulate posterior/refit updates or their influence on other kernel keys.",
            "Full revelation and one-key replacements do not bound all partial subsets or unseen candidates/tasks.",
            "No active strategy, stopping certificate, total-cost advantage or future-workload performance is established.",
            "Host and link parameters stay fixed, host remains unaccepted and operator-body/layout coverage unverified."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reference_profiling_opportunity_v1.json"))
    args = parser.parse_args()
    report = analyze(DEFAULT_BUNDLE.parents[1])
    write_json(args.output, report)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
