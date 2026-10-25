"""Verify exact cached/reference E2 rankings and benchmark complete graphs."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path

from calibration.e2_records import load_jsonl_records
from simulator import default_fusion_registry, make_reference_plan
from simulator.hardware import load_hardware

from .e2_audit import PreparedCandidateRanker
from .e2_srop_proxy import WORKLOADS, _active_profile, build_workload_graph, object_map_for_workload, ObjectCounterfactualProfile
from .search_sensitivity import _rank_plans, device_exchange_permutations


def benchmark(records, contract, hardware, *, scale="model", repeats=3):
    workload = WORKLOADS["sequence_qwen3b_s1024"]
    graph = build_workload_graph(workload, scale)
    object_map = object_map_for_workload(records, workload)
    profile, _ = _active_profile(records, contract, hardware, object_map,
                                split_id=workload.split_id, revealed_object_ids=())
    reference = make_reference_plan(graph, device=0)
    plans = [reference]
    for fraction in (0.25, 0.50, 0.75):
        plan = copy.deepcopy(reference)
        for group in plan.groups[int(len(plan.groups) * fraction):]:
            group.device = 1
        plans.append(plan)
    exchange = device_exchange_permutations(profile.device_classes)
    registry = default_fusion_registry()
    prepared = PreparedCandidateRanker(graph, plans, hardware.n_devs, exchange, registry)
    object_id = object_map["self_attention.q_rope"]
    profiles = [profile] + [ObjectCounterfactualProfile(profile, object_id, delta)
                            for delta in (-0.50, 0.50)]
    baseline_times, prepared_times = [], []
    maximum_error = 0.0
    for _ in range(repeats):
        for current_profile in profiles:
            started = time.perf_counter()
            expected = _rank_plans(graph, plans, current_profile, registry, exchange)
            baseline_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            actual = prepared.rank(current_profile)
            prepared_times.append(time.perf_counter() - started)
            assert [item.plan_id for item in expected] == [item.plan_id for item in actual]
            for left, right in zip(expected, actual):
                maximum_error = max(maximum_error, abs(left.makespan - right.makespan))
                assert left.critical_path == right.critical_path
                assert left.confidence == right.confidence
                assert left.makespan == right.makespan
    return {
        "workload_id": workload.workload_id, "scale": scale,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "contract_id": contract["contract_id"], "hardware_profile": hardware.name,
        "nodes": len(graph.nodes), "candidates": len(plans),
        "profiles": len(profiles), "repeats": repeats,
        "max_makespan_error_s": maximum_error,
        "reference_median_s": statistics.median(baseline_times),
        "prepared_median_s": statistics.median(prepared_times),
        "speedup": statistics.median(baseline_times) / statistics.median(prepared_times),
        "preparation_wall_time_s": prepared.preparation_wall_time_s,
        "reference_total_s": sum(baseline_times),
        "prepared_total_including_preparation_s": sum(prepared_times) + prepared.preparation_wall_time_s,
        "ranking_and_critical_path_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("layer", "model"), default="model")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
    contract = json.loads((root / "data/bw1000_e2/split_contract.json").read_text(encoding="utf-8"))
    result = benchmark(records, contract,
                       load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json"),
                       scale=args.scale, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    print("Fusion E2c prepared scoring verification: OK")


if __name__ == "__main__":
    main()
