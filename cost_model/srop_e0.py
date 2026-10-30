"""CPU E0 checks for fusion/placement freedom, Kernel lookup and output graphs.

The coupling witness is deliberately synthetic. Real graph inventories report
legacy shape coverage separately from complete group-Kernel identity coverage.
No GPU, downstream parallel configuration, or whole-graph execution is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from simulator import (
    DeviceSpec, FusionGroup, FusionPatternRegistry, HardwareSpec, OpGraph, OpNode,
    Plan, TensorEdge, default_fusion_registry, make_reference_plan, validate_fusion_plan,
)
from simulator.dataset import graph_from_dict, plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import (
    PlacedOpGraph, graph_fingerprint, hardware_fingerprint,
    placed_op_graph_artifact, plan_fingerprint, validate_placed_op_graph,
)

from .exact_search import exact_legal_search
from .fusion_kernel_profile import (
    GroupKernelKeys, KernelLatency, KeyedKernelProfile, candidate_structure,
    inventory, key_id,
)
from .hardware_profile import AnalyticHardwareProfile
from .legal_search import LegalFusionSearchSpace, legal_mcts_search
from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import build_evaluation_graph
from .structural import StructuralCostModel


ROOT = Path(__file__).resolve().parents[1]
MODES = ("placement_only", "fusion_only", "sequential", "joint")


def singleton_registry(registry):
    return FusionPatternRegistry(p for p in registry.patterns() if p.max_nodes == 1)


def partition_signature(plan):
    return tuple((group.extra["pattern_id"], tuple(group.node_ids)) for group in plan.groups)


def contract_partition(graph, template, registry):
    """Expose only device choices after the fusion partition has been fixed."""
    structure = StructuralCostModel(registry=registry).prepare(graph, template,
                                                              max(g.device for g in template.groups) + 1).structure
    macro = OpGraph(name=graph.name + ":fixed-partition")
    for group in structure.groups:
        macro.add_node(OpNode(str(group.index), "fixed_partition", group.flops,
                              group.output_bytes, group.parameter_bytes))
    for dep in structure.dependencies:
        macro.add_edge(TensorEdge(str(dep.src_group), str(dep.dst_group), dep.bytes))

    def expand(placement):
        devices = {int(group.node_ids[0]): group.device for group in placement.groups}
        result = copy.deepcopy(template)
        result.name = "sequential-fixed-partition"
        for index, group in enumerate(result.groups):
            group.device = devices[index]
        return result

    return macro, expand


class _Scorer:
    def __init__(self, graph, profile, registry):
        self.graph, self.profile = graph, profile
        self.model = StructuralCostModel(registry=registry)
        self.calls = 0
        self.unique = set()

    def __call__(self, plan):
        self.calls += 1
        self.unique.add(plan_fingerprint(plan))
        return self.model.predict(self.graph, plan, self.profile)


def run_mode(graph, profile, registry, mode, *, method, iterations=128, seed=0,
             rollouts=3, fixed_device=0, require_executable=False):
    """Four modes sharing one scorer; sequential freezes fusion before placement.

    Sequential MCTS splits the same total iteration budget across its two stages.
    Exact mode enumerates each restricted space and reports both stage counts.
    These timings are E0 diagnostics, not a fixed-wall-clock SROP comparison.
    """
    if mode not in MODES or method not in {"exact", "mcts"}:
        raise ValueError("unsupported E0 search mode or method")
    if method == "mcts" and iterations < (2 if mode == "sequential" else 1):
        raise ValueError("insufficient MCTS iterations for search stages")
    scorer = _Scorer(graph, profile, registry)
    stages = []

    def search(search_graph, rules, devices, budget, expand=lambda plan: plan):
        before = scorer.calls
        def evaluate(plan):
            return scorer(expand(plan))
        started = time.perf_counter()
        if method == "exact":
            result = exact_legal_search(search_graph, evaluate, profile.n_devs,
                top_k=1, registry=rules, require_executable=require_executable,
                allowed_devices=devices)
            winner = result.best_plan
            stage = {"space_size": result.search_space_size,
                     "feasible_plans": result.feasible_plans}
        else:
            def value(plan):
                estimate = evaluate(plan)
                return math.inf if estimate.details.get("overflow_devices") else estimate.makespan
            winner = legal_mcts_search(search_graph, value, profile.n_devs,
                iterations=budget, rollouts=rollouts, seed=seed, n_best=1,
                registry=rules, require_executable=require_executable,
                allowed_devices=devices)[0]
            stage = {"iterations": budget, "rollouts": rollouts, "seed": seed}
        stage.update(elapsed_s=time.perf_counter() - started,
                     cost_model_evaluations=scorer.calls - before)
        stages.append(stage)
        return expand(winner)

    started = time.perf_counter()
    all_devices = tuple(range(profile.n_devs))
    if mode == "sequential":
        first = iterations // 2
        template = search(graph, registry, (fixed_device,), first)
        macro, expand = contract_partition(graph, template, registry)
        winner = search(macro, singleton_registry(registry), all_devices,
                        iterations - first, expand)
        if partition_signature(winner) != partition_signature(template):
            raise AssertionError("sequential placement changed the frozen fusion partition")
    else:
        rules = singleton_registry(registry) if mode == "placement_only" else registry
        devices = (fixed_device,) if mode == "fusion_only" else all_devices
        winner = search(graph, rules, devices, iterations)
    elapsed = time.perf_counter() - started
    # Validation/scoring below is separately outside the reported search clock.
    estimate = StructuralCostModel(registry=registry).predict(graph, winner, profile)
    if estimate.details.get("overflow_devices"):
        raise ValueError("search returned a memory-infeasible output graph")
    validate_fusion_plan(graph, winner, profile.n_devs, registry=registry,
                         require_executable=require_executable)
    placed = PlacedOpGraph(graph, profile.keys.hardware, winner,
        predicted_makespan=estimate.makespan, predicted_peak_memory=estimate.peak_memory,
        extra={"scope": "E0_search_validation", "deployable": False,
               "latency_scope": "synthetic_coupling_fixture"})
    validate_placed_op_graph(placed, registry=registry)
    artifact = placed_op_graph_artifact(placed)
    return {"mode": mode, "method": method, "seed": seed,
            "makespan_s": estimate.makespan, "elapsed_s": elapsed,
            "cost_model_evaluations": scorer.calls, "unique_plans": len(scorer.unique),
            "stages": stages, "plan": plan_to_dict(winner),
            "partition": [[p, list(nodes)] for p, nodes in partition_signature(winner)],
            "used_devices": sorted({group.device for group in winner.groups}),
            "peak_memory": list(estimate.peak_memory),
            "output_constraints_pass": True, "artifact": artifact}


def coupling_fixture():
    """A declared counterexample to fusion-first under device-dependent costs.

    The five-node fork/join has 32 singleton placements and 16 fused placements.
    On device 0, fusion is slower (12 ms vs 4+4 ms); on device 1 it is faster
    (1 ms vs 8+8 ms). GEMMs prefer device 0. This is not measured GPU evidence.
    """
    graph = OpGraph(name="synthetic-device-dependent-fusion")
    for node_id, operator in (("gate", "matmul"), ("up", "matmul"),
                              ("silu", "silu"), ("mul", "mul"), ("out", "matmul")):
        attrs = {"output_shape": [4, 8], "dtype": "bf16", "compute_dtype": "fp32",
                 "layout": {"contiguity": "contiguous", "order": "row_major"},
                 "phase": "forward", "layer": 0}
        if operator == "matmul":
            attrs["shape"] = {"m": 4, "n": 8, "k": 8}
        graph.add_node(OpNode(node_id, operator, 64., 64., 0., attrs))
    for source, target in (("gate", "silu"), ("up", "mul"),
                            ("silu", "mul"), ("mul", "out")):
        graph.add_edge(TensorEdge(source, target, 64.))
    hardware = HardwareSpec(
        devices=[DeviceSpec(1e12, 1e12, 1e9, name="synthetic-" + str(device),
                            extra={"class": "synthetic", "physical_index": device,
                                   "source": "controlled-E0-fixture-v1"})
                 for device in range(2)],
        comm_bw=1e12, comm_latency=.0001, launch_overhead=0., overlap_send=True,
        name="synthetic-E0-not-GPU")
    registry = default_fusion_registry()
    bindings = {p.pattern_id: {"id": "synthetic." + p.pattern_id, "version": "e0-v1"}
                for p in registry.patterns()}
    keys = GroupKernelKeys(graph, hardware, bindings,
        {"framework": "synthetic", "framework_version": "synthetic-v1",
         "accelerator": "none", "accelerator_version": "no-accelerator"})
    space = LegalFusionSearchSpace(graph, 2, registry, require_executable=False)
    latencies = {}
    for candidate in space.candidates:
        for device in range(2):
            group = candidate_structure(graph, candidate, device)
            if len(group.node_ids) > 1:
                duration = (.012, .001)[device]
            else:
                duration = ((.001, .010) if graph.nodes[group.node_ids[0]].op_type == "matmul"
                            else (.004, .008))[device]
            key, gaps = keys.describe(group)
            if gaps:
                raise AssertionError("synthetic fixture is missing Kernel metadata")
            latencies[key_id(key)] = KernelLatency(duration, "synthetic", "declared-E0-cost-table-v1")
    profile = KeyedKernelProfile(keys, latencies, AnalyticHardwareProfile(hardware))
    return graph, hardware, registry, profile


def run_coupling_checks(*, seeds=(0, 1, 7, 23), iterations=128):
    graph, hardware, registry, profile = coupling_fixture()
    before = graph_fingerprint(graph)
    exact = {mode: run_mode(graph, profile, registry, mode, method="exact") for mode in MODES}
    joint_cost = exact["joint"]["makespan_s"]
    joint = exact["joint"]["artifact"]
    if not joint_cost < min(exact[mode]["makespan_s"] for mode in MODES if mode != "joint"):
        raise AssertionError("controlled fixture did not expose fusion/placement coupling")
    mcts = []
    for seed in seeds:
        for mode in MODES:
            row = run_mode(graph, profile, registry, mode, method="mcts",
                           seed=seed, iterations=iterations)
            gap = row["makespan_s"] - exact[mode]["makespan_s"]
            row["exact_gap_s"] = gap
            row["exact_gap_relative"] = gap / exact[mode]["makespan_s"]
            if abs(gap) > 1e-12:
                raise AssertionError(f"E0 MCTS missed the {mode} exact optimum for seed {seed}: {gap}")
            mcts.append(row)
    if graph_fingerprint(graph) != before:
        raise AssertionError("E0 searches modified the input graph")
    return {
        "scope": "synthetic_mechanism_test_not_empirical_advantage",
        "graph_hash": before, "hardware_hash": hardware_fingerprint(hardware),
        "default_registry_modified": False, "deployable_fusion_claim": False,
        "exact": exact, "mcts": mcts,
        "kernel_inventory": inventory(profile.keys, registry, latencies=profile.latencies),
        "joint_improvement_over_sequential": 1. - joint_cost / exact["sequential"]["makespan_s"],
        "representative_output": joint,
    }


def run_e0(*, graph_paths=(), hardware_path=None, raw_path=None, iterations=128,
           kerv_capability=None, flagscale_root=None):
    hardware_path = Path(hardware_path or ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    raw_path = Path(raw_path or ROOT / "data/bw1000_2card/raw_measurements.json")
    hardware = load_hardware(hardware_path)
    legacy = MeasuredHardwareProfile.from_files(hardware_path, raw_path)
    graphs = [(str(path), graph_from_dict(json.loads(Path(path).read_text(encoding="utf-8"))))
              for path in graph_paths]
    if not graphs:
        graphs = [(scale, build_evaluation_graph(scale)) for scale in ("layer", "model", "model_3b")]
    rows = []
    registry = default_fusion_registry()
    bindings = None
    capability = None
    if kerv_capability is not None:
        from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE, check_capability, development_registry
        capability = check_capability(kerv_capability, flagscale_root or DEFAULT_FLAGSCALE)
        registry = development_registry(capability)
        bindings = {"silu_mul": capability["binding"]}
    for label, graph in graphs:
        keys = GroupKernelKeys(graph, hardware, bindings)
        explorative = inventory(keys, registry, legacy_profile=legacy)
        executable = inventory(keys, registry, legacy_profile=legacy, require_executable=True)
        reference = make_reference_plan(graph, device=0)
        estimate = StructuralCostModel().predict(graph, reference, legacy)
        placed = PlacedOpGraph(graph, hardware, reference,
            predicted_makespan=estimate.makespan, predicted_peak_memory=estimate.peak_memory,
            extra={"scope": "reference_output_constraint_check_not_search_benefit"})
        validate_placed_op_graph(placed)
        artifact = placed_op_graph_artifact(placed)
        rows.append({"label": label, "graph_hash": graph_fingerprint(graph),
                     "nodes": len(graph.nodes), "edges": len(graph.edges),
                     "reference_output_constraints": {
                         "pass": True, "graph_hash": artifact["graph_hash"],
                         "candidate_id": artifact["candidate_id"],
                         "groups": len(artifact["fusion_groups"]),
                         "peak_memory": list(estimate.peak_memory),
                         "scope": "all_device_zero_singleton_reference_only"},
                     "exploration": explorative, "executable_only": executable})
    sources = ("cost_model/fusion_kernel_profile.py", "cost_model/srop_e0.py",
               "cost_model/tests/test_srop_e0.py", "cost_model/search_evaluation.py",
               "cost_model/legal_search.py", "cost_model/exact_search.py",
               "cost_model/structural.py", "cost_model/structure.py",
               "cost_model/hardware_profile.py", "cost_model/estimate.py",
               "cost_model/measured_profile.py", "simulator/fusion_rules.py",
               "simulator/placement_artifacts.py", "simulator/schema.py",
               "simulator/hardware.py", "simulator/dataset.py", "simulator/validate.py",
               "simulator/model_graphs/spec.py", "simulator/model_graphs/transformer.py",
               "simulator/model_graphs/dense_layer.py")
    if capability is not None:
        sources += ("calibration/kerv_silu_mul.py", "cost_model/tests/test_kerv_capability.py")
    inputs = (hardware_path, raw_path, *graph_paths)
    if kerv_capability is not None:
        inputs += (Path(kerv_capability),)
    return {
        "format": "fusion-placement-search.srop-e0", "version": "1.0",
        "scope": "development_CPU_mechanism_and_profile_coverage_audit",
        "mechanism": run_coupling_checks(iterations=iterations), "graph_inventories": rows,
        "development_capability": capability,
        "readiness": {
            "controlled_joint_search_and_output_checks_pass": True,
            "real_complete_group_kernel_profile_available": all(
                row["exploration"]["complete_measured_kernel_profile"] for row in rows),
            "project_team_capability_contract_available": False,
            "development_capability_source_available": capability is not None,
            "gpu_collection_performed": False, "SROP_benefit_assessed": False,
        },
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "input_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for path in inputs},
        "limitations": [
            "Coupling latencies are intentionally synthetic; no measured speedup or SROP gain is claimed.",
            "Default multi-node patterns remain planned; exploratory artifacts are not deployable.",
            "Legacy measured_exact_shape hits do not bind implementation version, layout or a group Kernel Key.",
            "This legacy BW1000 audit does not invalidate E2 complete-key records; reuse needs an explicit audited mapping.",
            "Inventories enumerate local candidates x devices; they do not exhaust full-model placed graphs.",
            "Complete Kernel measurements would still leave link, scheduling and aggregation assumptions.",
            "The downstream capability contract is unavailable; no parallel config or whole-graph execution is validated.",
            "A supplied KERV capability is development-only and may impose unresolved contiguous-layout requirements.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, action="append", default=[],
                        help="Optional upstream-adapted OpGraph JSON; repeat for multiple graphs")
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--raw-profile", type=Path)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--kerv-capability", type=Path,
                        help="Source-pinned development substitute for the later downstream capability")
    parser.add_argument("--flagscale-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/srop_e0_v1")
    args = parser.parse_args(argv)
    report = run_e0(graph_paths=args.graph, hardware_path=args.hardware,
                    raw_path=args.raw_profile, iterations=args.iterations,
                    kerv_capability=args.kerv_capability, flagscale_root=args.flagscale_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output = report["mechanism"]["representative_output"]
    (args.out_dir / "synthetic_joint.placed-op-graph.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("E0 controlled search checks: PASS (synthetic costs; not SROP benefit)")
    for mode, row in report["mechanism"]["exact"].items():
        print(f"  {mode}: {row['makespan_s'] * 1000:.6f} ms; stages={row['stages']}")
    for row in report["graph_inventories"]:
        inv = row["exploration"]
        print(f"  {row['label']}: {inv['kernel_key_count']} group keys; "
              f"{inv['measured_key_count']} complete measured entries; legacy={inv['legacy_lookup_sources_by_usage']}")
    if report["development_capability"] is not None:
        print("KERV development capability: SOURCE PINNED; GPU validation/latency: NOT RUN")
        print("The final downstream capability file is not required for development checks.")
    else:
        print("Project capability contract and complete measured fusion Profile: PENDING")
    print("Report:", args.out_dir / "report.json")


if __name__ == "__main__":
    main()
