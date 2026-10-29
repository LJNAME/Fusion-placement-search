"""Post-hoc A3 latency-model ablations with unchanged kernels and GPU labels."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from calibration.decoder_competition import check, DEFAULT_BUNDLE
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import ROOT, INPUTS
from cost_model.e2_residual import _spearman
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel, _payload_bytes
from cost_model.structure import analyze_plan_structure
from cost_model.structural import StructuralCostModel
from execution.analyze_reference_validation import compare_report
from execution.audit_gpu_holdout import write_json
from execution.decoder_competition import summarize
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


PRIMARY_METHODS = ("structural_dag", "serial_with_host", "reference_zero_host", "reference_host")
METHODS = (*PRIMARY_METHODS, "serial_zero_host")


def serial_duration(graph, plan, profile, host=None):
    """Strictly add operator service, full copies and optional host work.

    Input nodes are cached roots or tensor aliases, with zero GPU service.
    Host work cannot overlap GPU work in this deliberately simple baseline.
    """
    if any(len(group.node_ids) != 1 for group in plan.groups):
        raise ValueError("serial reference baseline requires singleton groups")
    structure = analyze_plan_structure(graph, plan, profile.n_devs)
    service = sum(profile.estimate_group(graph, group).duration for group in structure.groups
                  if graph.nodes[group.node_ids[0]].op_type != "input")
    copies = [dep for dep in structure.dependencies if dep.cross_device]
    communication = sum(profile.estimate_transfer(structure.groups[dep.src_group].device, structure.groups[dep.dst_group].device,
                        _payload_bytes(graph.nodes[dep.src_node])).duration for dep in copies)
    submission = 0.
    if host is not None:
        submission = host.start_s + host.finish_s + len(copies) * host.copy_submit_s + sum(
            host.node_cost(node.op_type, str(node.extra.get("local_name", node.id))) for node in graph.nodes.values())
    return service + communication + submission


def collapse_predicted_ties(values, tolerance=1e-12):
    collapsed, anchor = {}, None
    for value in sorted(values):
        if anchor is None or value > anchor + tolerance:
            anchor = value
        collapsed[value] = anchor
    return [collapsed[value] for value in values]


def metrics(scores, actual, epsilons=(.001, .005, .01)):
    if not scores or set(scores) != set(actual):
        raise ValueError("score and GPU candidate sets differ")
    if any(not math.isfinite(v) or v <= 0 for v in [*scores.values(), *actual.values()]):
        raise ValueError("latencies must be finite and positive")
    best = min(actual.values())
    ties = sorted(key for key, value in scores.items() if value <= min(scores.values()) + 1e-12)
    selected = ties[0]
    relative = actual[selected] / best - 1
    predicted = [scores[key] for key in actual]
    truth = list(actual.values())
    return {"selected_id": selected, "predicted_tie_ids": ties, "selected_measured_ms": actual[selected] * 1000,
        "relative_regret": relative, "epsilon_success": {str(e): relative <= e for e in epsilons},
        "predicted_tie_regret_range": [min(actual[k] / best - 1 for k in ties), max(actual[k] / best - 1 for k in ties)],
        "mape": statistics.fmean(abs(p / a - 1) for p, a in zip(predicted, truth)),
        "signed_relative_bias": statistics.fmean(p / a - 1 for p, a in zip(predicted, truth)),
        "spearman": _spearman(collapse_predicted_ties(predicted), truth)}


def evaluate(results, *, bundle=DEFAULT_BUNDLE, repeats=5):
    if repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    frozen = check(bundle)
    read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    original = read("competition_report.json")
    reproduced = summarize(frozen, load_jsonl_records(results / "execution_samples.jsonl"), read("environment.json"))
    compare_report(reproduced, original)
    if not reproduced["complete"]:
        raise ValueError("ablation requires complete original A3 data")
    hardware = hardware_from_dict(frozen["hardware"])
    host = HostSubmissionProfile(**frozen["host_profile"])
    train = load_jsonl_records(ROOT / INPUTS[0])
    measured = {t["task_id"]: {r["plan_id"]: r["median_s"] for r in t["candidates"]} for t in original["workloads"]}
    tasks = []
    for task in frozen["workloads"]:
        graph = graph_from_dict(task["graph"])
        plans = {key: plan_from_dict(value) for key, value in sorted(task["plans"].items())}
        mapping = object_map_for_workload(train, E2Workload("ablation", "exposed-development",
            f"{task['family']}-s256-b1", task["family"], 256))
        profile = DatasetLatencyProfile(hardware, mapping, task["kernel_durations_s"],
                                       {key: "sealed_original112_nearest" for key in task["kernel_durations_s"]})
        dag, zero, reference = StructuralCostModel(), ReferenceEagerCostModel(), ReferenceEagerCostModel(host)
        methods = {"structural_dag": lambda p: dag.predict(graph, p, profile).makespan,
            "serial_with_host": lambda p: serial_duration(graph, p, profile, host),
            "reference_zero_host": lambda p: zero.predict(graph, p, profile).makespan,
            "reference_host": lambda p: reference.predict(graph, p, profile).makespan,
            "serial_zero_host": lambda p: serial_duration(graph, p, profile)}
        # One complete unmeasured pass warms shared latency caches uniformly.
        scores = {name: {key: method(plan) for key, plan in plans.items()} for name, method in methods.items()}
        if any(not math.isclose(scores["reference_host"][key], value, rel_tol=1e-12, abs_tol=1e-15)
               for key, value in task["initial_scores_s"].items()):
            raise ValueError("Reference host scoring no longer reproduces sealed initial predictions")
        cpu_ns = {name: [] for name in METHODS}
        for repeat in range(repeats):
            offset = repeat % len(METHODS)
            order = METHODS[offset:] + METHODS[:offset]
            if repeat % 2:
                order = tuple(reversed(order))
            for name in order:
                start = time.perf_counter_ns()
                current = {key: methods[name](p) for key, p in plans.items()}
                elapsed = time.perf_counter_ns() - start
                if current != scores[name]:
                    raise ValueError("timed scoring changed predictions")
                cpu_ns[name].append(elapsed)
        rows = {}
        actual = measured[task["task_id"]]
        for name in METHODS:
            timed = cpu_ns[name]
            rows[name] = {**metrics(scores[name], actual, frozen["protocol"]["epsilon_values"]),
                "scores_s": scores[name], "cpu_pool_time_ms": [value / 1e6 for value in timed],
                "median_cpu_pool_ms": statistics.median(timed) / 1e6,
                "median_cpu_us_per_candidate": statistics.median(timed) / len(plans) / 1000}
        tasks.append({"task_id": task["task_id"], "nodes": len(graph.nodes), "edges": len(graph.edges),
            "candidates": len(plans), "methods": rows,
            "serial_host_and_reference_same_selection": rows["serial_with_host"]["selected_id"] == rows["reference_host"]["selected_id"]})
        print(task["task_id"], {name: {"mape_pct": round(r["mape"] * 100, 3),
              "regret_pct": round(r["relative_regret"] * 100, 3),
              "cpu_us": round(r["median_cpu_us_per_candidate"], 1)} for name, r in rows.items()}, flush=True)
    summary = {name: {"mean_task_mape": statistics.fmean(t["methods"][name]["mape"] for t in tasks),
        "mean_task_spearman": statistics.fmean(t["methods"][name]["spearman"] for t in tasks),
        "mean_task_relative_regret": statistics.fmean(t["methods"][name]["relative_regret"] for t in tasks),
        "primary_epsilon_success_tasks": sum(t["methods"][name]["relative_regret"] <= frozen["protocol"]["primary_epsilon"] for t in tasks),
        "mean_task_cpu_us_per_candidate": statistics.fmean(t["methods"][name]["median_cpu_us_per_candidate"] for t in tasks)} for name in METHODS}
    inputs = ("competition_report.json", "execution_samples.jsonl", "environment.json")
    return {"format": "fusion-placement-search.a3-cost-model-ablation", "freeze_id": frozen["freeze_id"],
        "scope": "Post-hoc development ablation, no label fitting or new GPU collection.",
        "primary_methods": list(PRIMARY_METHODS), "additional_control": "serial_zero_host",
        "definitions": {"structural_dag": "Unmodified original DAG scalar predict API, including its original input service fallback.",
            "serial_with_host": "Sum(non-input operator service) + sum(full-copy cost) + host start/finish + per-node/per-copy submission; no overlap.",
            "reference_zero_host": "Consumer-copy, two-sided streams, zero GPU input service, zero host times.",
            "reference_host": "Same Reference semantics with unchanged unaccepted host profile; scores checked against A3 freeze.",
            "serial_zero_host": "Paired diagnostic removing host terms from serial_with_host; not a fifth trained model."},
        "timing": {"repeats": repeats, "warmup_pool_passes": 1,
            "scope": "Current scalar APIs: DAG/Reference include validation, structures, memory/trace work; serial returns latency only. Shared graph/profile initialization excluded, lookups warmed. Not abstract algorithm complexity or optimized/prepared/batched throughput.",
            "method_order": "Rotate per repeat, reverse on odd repeats; fixed candidate order; all raw pool timings retained.",
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count()}},
        "input_sha256": {name: hashlib.sha256((results / name).read_bytes()).hexdigest() for name in inputs},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": summary, "tasks": tasks,
        "limitations": ["Four correlated development tasks, known seq256 kernel keys, 120 measured candidates; no independent novelty or universal optimality proof.",
            "Same frozen kernel durations and communication costs; no target-label calibration, score scaling or baseline tuning.",
            "Serial main baseline receives identical frozen host parameters; no-host serial is reported separately to avoid confounding host availability.",
            "DAG retains original input service; cached/alias inputs are zero GPU work in serial and Reference. This offset affects absolute error as well as scheduling assumptions.",
            "Equal regret cannot establish that the richer model is necessary; error/ranking and scoring cost must be considered separately.",
            "Forward reference implementation is not formally fully serialized: host enqueue order alone does not prove absence of all cross-device overlap.",
            "Host calibration remains unaccepted; CPU timings on this local host are not BW server overhead or GPU execution time."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "data/received_a3_decoder_competition_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/a3_cost_model_ablation.v1.json")
    parser.add_argument("--timing-repeats", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(args.results, repeats=args.timing_repeats)
    write_json(args.output, report)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
