"""CPU-only audit of the existing forward memory estimate; no search or GPU work."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.kerv_add_rmsnorm_probe import sha
from cost_model.kerv_add_rmsnorm_profile import load_profile
from cost_model.native_add_rmsnorm_profile import load_record
from cost_model.qwen25_add_rmsnorm import (
    PROFILE_DIRECTORY, SCHEDULING_POLICY, NativeChainCostModel, reference_plan, search_registry,
)
from cost_model.qwen25_static_contract import StaticContractProfile
from cost_model.search_evaluation import qwen25_3b_spec
from integration.flagscale.kerv_add_rmsnorm import KervAddRmsnormAdapter
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import graph_fingerprint
from tools.audit_qwen25_placement import three_plans

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
GIB = 1024 ** 3


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def metadata_check(rows, shape_field, size_field):
    missing, mismatches = [], []
    for name, extra, stored in rows:
        shape, dtype = extra.get(shape_field), extra.get("dtype")
        if not shape or dtype not in DTYPE_BYTES:
            missing.append(name)
            continue
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if stored != expected:
            mismatches.append({"id": name, size_field: stored, "from_shape_dtype": expected})
    return {"count": len(rows), "missing_shape_or_dtype": missing, "byte_mismatches": mismatches}


def run(output):
    graph = build_model_graph(qwen25_3b_spec())
    hardware = load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    measured, _ = load_profile(PROFILE_DIRECTORY)
    profile = StaticContractProfile(graph, hardware, KervAddRmsnormAdapter(), measured,
                                    load_record(ROOT / "data/native_add_rmsnorm_timing_01"))
    registry = search_registry(profile)
    model = NativeChainCostModel(registry=registry, scheduling_policy=SCHEDULING_POLICY)
    node_check = metadata_check([(n, v.extra, v.out_bytes) for n, v in graph.nodes.items()],
                                "output_shape", "out_bytes")
    edge_check = metadata_check([(f"{i}:{e.src}->{e.dst}", e.extra, e.bytes)
                                for i, e in enumerate(graph.edges)], "shape", "bytes")
    assert not any(c[k] for c in (node_check, edge_check)
                   for k in ("missing_shape_or_dtype", "byte_mismatches"))
    previous_path = ROOT / "data/qwen25_placement_audit_v1/report.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    assert graph_fingerprint(graph) == previous["graph_fingerprint"]
    order = {n: i for i, n in enumerate(graph.topological_order())}
    consumers = {n: [] for n in graph.nodes}
    for edge in graph.edges:
        consumers[edge.src].append(edge.dst)
    lifetimes = {n: {"shape": list(v.extra["output_shape"]), "dtype": v.extra["dtype"],
                    "bytes": int(v.out_bytes), "producer_topological_index": order[n],
                    "consumer_nodes": consumers[n],
                    "last_consumer_topological_index": max((order[c] for c in consumers[n]), default=None),
                    "modeled_intervals": {}}
                 for n, v in graph.nodes.items()}
    plans = three_plans(graph, registry)
    scores = {}
    for name, plan in plans.items():
        estimate = model.predict(graph, plan, profile)
        groups = {n: i for i, g in enumerate(plan.groups) for n in g.node_ids}
        timing = {g["group_id"]: g for g in estimate.details["groups"]}
        transfers = {(t["src_group"], t["dst_group"]): t for t in estimate.details["transfers"]}
        old = previous["plans"][name]
        assert abs(estimate.makespan * 1e3 - old["predicted_ms"]) < 1e-10
        assert list(estimate.peak_memory) == [old["per_device"][str(d)]["peak_memory_bytes"]
                                              for d in range(profile.n_devs)]
        eliminated = []
        for n, v in graph.nodes.items():
            src = f"fg{groups[n]:04d}"
            external = [c for c in consumers[n] if groups[c] != groups[n]]
            if consumers[n] and not external:
                eliminated.append(n)
                lifetimes[n]["modeled_intervals"][name] = {"internal_only_not_materialized": True}
                continue
            release = []
            for c in external:
                dst = f"fg{groups[c]:04d}"
                release.append(transfers[(src, dst)]["end"] if plan.groups[groups[n]].device != plan.groups[groups[c]].device
                               else timing[dst]["end"])
            lifetimes[n]["modeled_intervals"][name] = {
                "device": plan.groups[groups[n]].device,
                "publish_ms": timing[src]["end"] * 1e3,
                "release_ms": max(release, default=estimate.makespan) * 1e3,
            }
        parameters = [sum(graph.nodes[n].param_bytes for g in plan.groups if g.device == d for n in g.node_ids)
                      for d in range(profile.n_devs)]
        shares = [{"consumer": n, "owner": v.extra["shared_param_with"],
                   "owner_device": plan.groups[groups[v.extra["shared_param_with"]]].device,
                   "consumer_device": plan.groups[groups[n]].device,
                   "shared_weight_bytes": int(graph.nodes[v.extra["shared_param_with"]].param_bytes)}
                  for n, v in graph.nodes.items() if v.extra.get("shared_param_with")
                  and groups[n] != groups[v.extra["shared_param_with"]]
                  and plan.groups[groups[n]].device != plan.groups[groups[v.extra["shared_param_with"]]].device]
        scores[name] = {
            "predicted_ms": estimate.makespan * 1e3,
            "peak_memory_bytes": list(estimate.peak_memory),
            "peak_memory_gib": [p / GIB for p in estimate.peak_memory],
            "resident_parameter_bytes": parameters,
            "workspace_explicitly_declared_groups": sum("workspace_bytes" in g.extra for g in plan.groups),
            "workspace_bytes_used_by_estimator": sum(g.extra.get("workspace_bytes", 0) for g in plan.groups),
            "internal_only_outputs_eliminated": eliminated,
            "cross_device_transfers": estimate.details["transfers"],
            "unmodeled_cross_device_shared_parameters": shares,
            "within_current_forward_estimate_and_total_capacity": all(
                p <= profile.max_memory(d) for d, p in enumerate(estimate.peak_memory)),
            "actual_available_budget_feasible": None,
            "training_feasible": None,
        }
    native = model.predict(graph, reference_plan(graph, registry, fused=False, device=0), profile)
    source_paths = [Path(__file__), previous_path, ROOT / "cost_model/structural.py",
                    ROOT / "cost_model/structure.py", ROOT / "cost_model/search_evaluation.py",
                    ROOT / "simulator/model_graphs/transformer.py", ROOT / "simulator/model_graphs/dense_layer.py",
                    ROOT / "integration/flagscale/kerv_add_rmsnorm.py",
                    ROOT / "cost_model/qwen25_static_contract.py",
                    ROOT / "cost_model/qwen25_add_rmsnorm.py",
                    ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json",
                    PROFILE_DIRECTORY / "review/profile.json",
                    ROOT / "data/native_add_rmsnorm_timing_01/review/profile.json"]
    report = {
        "scope": "existing static forward memory estimator; not a training or runtime feasibility certificate",
        "graph": {"fingerprint": graph_fingerprint(graph), "nodes": node_check, "edges": edge_check,
                  "phases": dict(Counter(v.extra.get("phase", "unknown") for v in graph.nodes.values())),
                  "dtype_counts": dict(Counter(v.extra["dtype"] for v in graph.nodes.values())),
                  "parameter_bytes": sum(v.param_bytes for v in graph.nodes.values()),
                  "saved_activation_metadata_sum_bytes_not_training_peak": sum(
                      v.extra.get("saved_activation_bytes", 0) for v in graph.nodes.values())},
        "devices": [{"device": d, "total_capacity_bytes": profile.max_memory(d),
                     "total_capacity_gib": profile.max_memory(d) / GIB,
                     "default_search_budget_equals_total_capacity": True,
                     "actual_available_budget_bytes": None, "runtime_reserve_bytes": None}
                    for d in range(profile.n_devs)],
        "plans": scores,
        "native_gpu0_comparison": {"predicted_ms": native.makespan * 1e3,
                                   "peak_memory_bytes": list(native.peak_memory),
                                   "native_pairs_contracted_for_scheduling": len(native.details["native_chain_execution_units"]),
                                   "equal_peak_does_not_validate_inplace_accounting": True},
        "lifetime_semantics": "logical outputs published at group end; local last consumer completion or remote copy completion; destination buffer starts at copy END; allocate before free at ties",
        "not_accounted": ["backward retained activations", "gradients", "optimizer/master-weight states",
                          "allocator/context/graph-pool reserves", "undeclared kernel workspace",
                          "physical storage alias deduplication", "cross-device tied weight residency",
                          "destination buffer during in-flight copy", "outputs allocated during kernel execution"],
        "search_admission": "existing final candidates require estimated peak <= capacity; infeasible exploration gets finite penalty; default capacity is device total, not observed free memory",
        "static_forward_estimate_is_proven_upper_bound": False,
        "new_search_runs": 0, "new_gpu_measurements": 0,
        "production_memory_or_search_code_changed": False,
        "verification": {"shape_dtype_byte_checks": "passed", "prior_three_plan_scores_and_peaks": "identical"},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
    }
    output.mkdir(parents=True, exist_ok=False)
    write(output / "report.json", report)
    write(output / "tensor_lifetimes.json", {"scope": "modeled logical intervals, not observed tensor allocations", "nodes": lifetimes})
    print(json.dumps({"graph": report["graph"], "plans": scores, "native_gpu0": report["native_gpu0_comparison"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
