"""Two local GPU0 choices through the existing scheduler and search APIs.

This is a CPU development check using previously observed implementation
variants. It is not an independent performance evaluation or placement study.
"""
import argparse
import copy
import json
from pathlib import Path

from calibration.lightop_profile_calibration import load_reviewed_profile, ROOT
from cost_model.exact_search import exact_legal_search
from cost_model.fusion_kernel_profile import GroupKernelKeys, key_id
from cost_model.hardware_profile import GroupCost
from cost_model.legal_search import legal_mcts_search
from cost_model.native_stage_profile import NativeMeasuredFirstProfile, ReviewedStageStore
from cost_model.structural import StructuralCostModel
from integration.lightop.silu_mul import adapter_registry, group_inputs
from simulator import FusionGroup, OpGraph
from simulator.placement_artifacts import graph_fingerprint


FUSED_RAW = ROOT / "data/lightop_profile_calibration_20261104_01/raw"
STAGE_RAW = ROOT / "data/native_stage_calibration_20261105_01/raw"


def signature(plan):
    return tuple((tuple(g.node_ids), g.device, g.extra["pattern_id"]) for g in plan.groups)


def build_case():
    fused = load_reviewed_profile(FUSED_RAW)
    node_ids, device = fused.keys._selected
    source_graph = fused.keys.graph
    gate, up = group_inputs(source_graph, node_ids)
    graph = OpGraph(name="observed_layer0_activation_gpu0", extra={"output_nodes": [node_ids[-1]]})
    for name in (gate, up, *node_ids):
        node = copy.deepcopy(source_graph.nodes[name])
        if name in (gate, up):
            node.op_type, node.flops, node.param_bytes = "input", 0., 0.
        graph.add_node(node)
    graph.edges = [copy.deepcopy(e) for e in source_graph.edges if e.src in graph.nodes and e.dst in graph.nodes]
    store = ReviewedStageStore(STAGE_RAW)
    records = {r["stage"]: r for r in store.records.values()}
    nr, fr = records["silu"]["identity"]["runtime"], fused.keys.runtime
    mapping = {"torch_distribution": "torch_distribution", "torch_git_revision": "torch_git_revision",
        "torch_extension_sha256": "torch_extension_sha256", "torch_build_sha256": "torch_build_sha256",
        "hip": "accelerator_version", "driver": "driver_version", "device": "execution_device"}
    matching = {n: nr[n] == fr[f] for n, f in mapping.items()}
    matching.update({"architecture": records["silu"]["identity"]["device"]["architecture"] == fused.keys.environment["device_architecture"],
        "flags": all(nr[n] == fr["flags"][f] for n, f in (("grad_enabled", "grad_enabled"),
            ("autocast_enabled", "cuda_autocast_enabled"), ("deterministic_algorithms", "deterministic_algorithms")))})
    if not all(matching.values()):
        raise ValueError("Existing stage and Fused runtime conditions do not match")

    # Keep an immutable Fused variant before assigning materialized Native
    # intermediates/aten-out implementations to the search graph.
    fused_view = copy.deepcopy(graph)
    for name, stage in zip(node_ids, ("silu", "mul")):
        impl = records[stage]["identity"]["implementation"]
        observed = impl["tensors"]["output"]
        graph.nodes[name].extra.update({k: copy.deepcopy(observed[k]) for k in
            ("dtype", "layout", "strides", "layout_provenance")})
        graph.nodes[name].extra["implementation"] = {"id": impl["recipe"]["id"], "version": impl["source_sha256"]}
    bindings = {node_ids[0]: {"identity": records["silu"]["identity"], "operands": {"gate": gate}},
                node_ids[1]: {"identity": records["mul"]["identity"], "operands": {"left": node_ids[0], "up": up}}}

    class LocalBase:
        def __getattr__(self, name):
            return getattr(fused, name)

        def estimate_group(self, current_graph, group):
            if group.device != device:
                raise ValueError("This local check only has GPU0 costs")
            if len(group.node_ids) == 1 and current_graph.nodes[group.node_ids[0]].op_type == "input":
                return GroupCost(0., 0., 0., 0., 1., "logical_input")
            if tuple(group.node_ids) != node_ids or group.pattern_id != "silu_mul":
                raise ValueError("Uncovered cost in the local experiment; no hidden analytic fallback")
            # Establish identical canonical identity on the boundary slice,
            # then query the original strict Profile on its own snapshot.
            key, gaps = GroupKernelKeys(fused_view, fused.keys.hardware, fused.keys.bindings, fused.keys.runtime).describe(group)
            original, original_gaps = fused.keys.describe(group)
            if gaps or original_gaps or key_id(key) != key_id(original):
                raise ValueError("Local Fused variant does not match the accepted Kernel Key")
            return fused.estimate_group(source_graph, group)

    key, gaps = fused.keys.describe(FusionGroup(list(node_ids), device, {"pattern_id": "silu_mul"}))
    if gaps:
        raise ValueError("Accepted Fused identity is incomplete")
    profile = NativeMeasuredFirstProfile(graph, LocalBase(), store, bindings)
    return graph, profile, {"matching_runtime_fields": matching,
        "native_stage_identities": {s: key_id(r["identity"]) for s, r in records.items()},
        "native_stage_provenance": {s: r["provenance"] for s, r in records.items()},
        "native_stage_source": STAGE_RAW.relative_to(ROOT).as_posix(),
        "fused_kernel_key_id": key_id(key), "fused_source": FUSED_RAW.relative_to(ROOT).as_posix(),
        "fused_measurement_contract_id": fused.measurement_contract_id,
        "case": "layer0 activation slice with observed gate/up boundary tensors, not complete FFN"}


def build_report(seeds=(0, 1, 7)):
    graph, profile, identity = build_case()
    registry = adapter_registry()
    model = StructuralCostModel(registry=registry)
    fingerprint = graph_fingerprint(graph)
    estimates = {}

    def score(plan):
        if graph_fingerprint(graph) != fingerprint:
            raise ValueError("Search graph mutated")
        estimate = model.predict(graph, plan, profile)
        if estimate.cross_device_bytes or any(g["source"] not in (
                "logical_input", "measured_native_isolated_stage", "measured_group_kernel") for g in estimate.details["groups"]):
            raise ValueError("Unexpected cost source or communication")
        estimates[signature(plan)] = estimate
        return estimate.makespan

    exact = exact_legal_search(graph, score, profile.n_devs, registry=registry,
                               allowed_devices=[0], top_k=2, max_evaluations=2)
    if exact.search_space_size != 2:
        raise ValueError("Expected exactly two same-device implementation choices")
    candidates = []
    for plan in exact.plans:
        estimate = estimates[signature(plan)]
        fused_choice = any(len(g.node_ids) > 1 for g in plan.groups)
        candidates.append({"choice": "fused" if fused_choice else "native",
            "profile_identity_ids": [identity["fused_kernel_key_id"]] if fused_choice else list(identity["native_stage_identities"].values()),
            "plan": [{"nodes": list(g.node_ids), "device": g.device, "pattern": g.extra["pattern_id"]} for g in plan.groups],
            "cost_us": estimate.makespan * 1e6, "estimate": estimate.to_dict()})
    mcts = []
    for seed in seeds:
        plans = legal_mcts_search(graph, score, profile.n_devs, registry=registry, allowed_devices=[0],
                                  iterations=24, rollouts=1, n_best=2, seed=seed)
        cost = score(plans[0])
        mcts.append({"seed": seed, "best_cost_us": cost * 1e6,
                     "best_matches_exact": signature(plans[0]) == signature(exact.best_plan),
                     "cost_model_gap_us": (cost - exact.optimal_score) * 1e6})
    return {"scope": "CPU development check of local measured-cost search, not independent validation",
        "identity": identity, "graph_fingerprint": fingerprint, "allowed_devices": [0],
        "search_space_size": exact.search_space_size, "candidates": candidates,
        "exact_choice": candidates[0]["choice"], "mcts": mcts,
        "all_candidates_have_recorded_cost_sources": True, "analytic_fallback_used": False,
        "new_gpu_execution": False, "full_ffn_coverage": False, "placement_benefit_tested": False,
        "mcts_efficiency_or_superiority_tested": False, "independent_reference_regret_us": None,
        "limitations": ["Native is a scheduled sum of isolated-stage costs; Fused is a measured complete call",
            "Only two candidates; the singleton baseline is available to MCTS",
            "Existing historical chain data participated in development; no new holdout claims"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"exact_choice": result["exact_choice"], "costs_us": {r["choice"]: r["cost_us"] for r in result["candidates"]},
                      "mcts_matches_exact": all(r["best_matches_exact"] for r in result["mcts"])}))


if __name__ == "__main__":
    main()
