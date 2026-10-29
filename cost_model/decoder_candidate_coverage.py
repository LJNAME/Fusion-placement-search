"""A2 structural coverage audit and label-free decoder placement probes.

Coverage is finite and descriptive. No duration, label, search score or GPU
measurement is used to add probes or to claim performance competition.
"""

import argparse
import copy
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path

from calibration.reference_validation import check, DEFAULT_BUNDLE
from execution.audit_gpu_holdout import write_json
from execution.decoder_reference import ContinuousDecoderExecutor
from simulator import PlacedOpGraph, make_reference_plan, validate_fusion_plan
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict
from simulator.model_graphs.decoder_stack import build_decoder_stack
from .placement_probes import graph_wide_candidates
from .reference_semantics import _payload_bytes
from .search_evaluation import qwen3_spec, qwen25_3b_spec
from .search_sensitivity import plan_id, permute_plan_devices


EXCHANGE = ((0, 1), (1, 0))
SUPPORTED = {"input", "matmul", "rmsnorm", "rope", "softmax", "silu", "mul", "add"}


def branch_groups(graph):
    """Use graph local names, keeping each branch's pre-join chain together."""
    by_layer = {}
    for node in graph.nodes.values():
        by_layer.setdefault(node.extra["layer"], {})[node.extra["local_name"]] = node.id
    result = {}
    for layer, names in sorted(by_layer.items()):
        chains = [[names[name] for name in sequence if name in names] for sequence in (
            ("self_attention.q_proj", "self_attention.q_norm", "self_attention.q_rope"),
            ("self_attention.k_proj", "self_attention.k_norm", "self_attention.k_rope"),
            ("self_attention.v_proj",), ("mlp.gate_proj", "mlp.silu"), ("mlp.up_proj",))]
        if any(not chain for chain in chains):
            raise ValueError("A2 audit requires dense decoder branches")
        result[layer] = {"qkv": chains[:3], "gate_up": chains[3:]}
    return result


def expand_directions(plans):
    return list({plan_id(p): p for original in plans for direction in EXCHANGE
                 for p in [permute_plan_devices(original, direction)]}.values())


def structured_candidates(graph, base_plans):
    """Preserve bases and add local branch masks and all topology-prefix cuts."""
    reference = make_reference_plan(graph)
    plans = list(copy.deepcopy(base_plans))
    order = reference.nodes()

    def add(nodes, label):
        proposal = copy.deepcopy(reference)
        proposal.name = f"a2:{label}"
        for group in proposal.groups:
            group.device = int(group.node_ids[0] in nodes)
        plans.append(proposal)

    add(set(), "uniform")
    for layer, families in branch_groups(graph).items():
        for name, chains in families.items():
            for mask in product((0, 1), repeat=len(chains)):
                add({key for bit, chain in zip(mask, chains) if bit for key in chain},
                    f"layer{layer}:{name}:{''.join(map(str, mask))}")
    for cut in range(1, len(order)):
        add(set(order[cut:]), f"prefix-cut:{cut}")
    layers = sorted(branch_groups(graph))
    for boundary in layers[1:]:
        add({key for key, node in graph.nodes.items() if node.extra["layer"] >= boundary},
            f"layer-cut:{boundary}")
    result = expand_directions(plans)
    for plan in result:
        validate_fusion_plan(graph, plan, 2, require_executable=True)
    return result


def audit_candidates(graph, plans, hardware):
    if hardware.n_devs != 2:
        raise ValueError("A2 audit requires two logical devices")
    plans = list(plans)
    eligible, failures, rows = {}, [], []
    legal_count, registry_count = 0, 0
    for index, plan in enumerate(plans):
        try:
            validate_fusion_plan(graph, plan, 2)
        except ValueError as exc:
            failures.append({"index": index, "stage": "search_legality", "reason": str(exc)})
            continue
        legal_count += 1
        try:
            validate_fusion_plan(graph, plan, 2, require_executable=True)
        except ValueError as exc:
            failures.append({"index": index, "stage": "registry_executability", "reason": str(exc)})
            continue
        registry_count += 1
        try:
            if any(node.op_type not in SUPPORTED for node in graph.nodes.values()):
                raise ValueError("unsupported Reference operator")
            ContinuousDecoderExecutor(PlacedOpGraph(graph, hardware, plan), ("cpu", "cpu"))
        except (ValueError, RuntimeError) as exc:
            failures.append({"index": index, "stage": "reference_eligibility", "reason": str(exc)})
            continue
        eligible[plan_id(plan)] = plan
    assignments = []
    ids = set(eligible)
    order, keys = graph.topological_order(), sorted(graph.nodes)
    for key, plan in sorted(eligible.items()):
        devices = {node: group.device for group in plan.groups for node in group.node_ids}
        assignments.append(devices)
        crosses = [edge for edge in graph.edges if devices[edge.src] != devices[edge.dst]]
        rows.append({"plan_id": key, "exchange_class_id": plan_id(plan, EXCHANGE),
            "devices_used": sorted(set(devices.values())), "copy_count": len(crosses),
            "full_tensor_copy_bytes": sum(_payload_bytes(graph.nodes[edge.src]) for edge in crosses),
            "two_sided_copy_barriers_in_reference_model": len(crosses),
            "host_node_dispatches": len(plan.groups), "host_copy_dispatches": len(crosses),
            "flops_by_device": [sum(n.flops for n in graph.nodes.values() if devices[n.id] == d) for d in (0, 1)],
            "parameter_bytes_by_device": [sum(n.param_bytes for n in graph.nodes.values() if devices[n.id] == d) for d in (0, 1)],
            "missing_exchange_partner": plan_id(permute_plan_devices(plan, (1, 0))) not in ids})

    branches = []
    for layer, families in branch_groups(graph).items():
        for name, chains in families.items():
            required = {''.join(map(str, bits)) for bits in product((0, 1), repeat=len(chains))}
            projection, colocated, isolated = set(), set(), set()
            branch_nodes = {node for chain in chains for node in chain}
            for devices in assignments:
                bits = [devices[chain[0]] for chain in chains]
                pattern = ''.join(map(str, bits))
                projection.add(pattern)
                if any(len({devices[key] for key in chain}) != 1 for chain in chains):
                    continue
                colocated.add(pattern)
                outside = {devices[key] for key in keys if key not in branch_nodes}
                if len(outside) == 1:
                    background = next(iter(outside))
                    isolated.add(''.join(str(bit ^ background) for bit in bits))
            branches.append({"layer": layer, "family": name, "required_patterns": sorted(required),
                "projection_patterns": sorted(projection), "colocated_chain_patterns": sorted(colocated),
                "isolated_branch_masks": sorted(isolated), "missing_projection_patterns": sorted(required - projection),
                "missing_colocated_chain_patterns": sorted(required - colocated),
                "missing_isolated_branch_masks": sorted(required - isolated)})

    cuts = [cut for cut in range(1, len(order)) if any(
        len({devices[key] for key in order[:cut]}) == 1 and
        len({devices[key] for key in order[cut:]}) == 1 and
        devices[order[0]] != devices[order[-1]] for devices in assignments)]
    layer_ids = sorted(branch_groups(graph))
    layer_cuts = [layer for layer in layer_ids[1:] if any(
        len({devices[key] for key in keys if graph.nodes[key].extra["layer"] < layer}) == 1 and
        len({devices[key] for key in keys if graph.nodes[key].extra["layer"] >= layer}) == 1 and
        devices[order[0]] != devices[order[-1]] for devices in assignments)]
    return {"input_candidates": len(plans), "search_legal_candidates": legal_count,
        "registry_executable_candidates": registry_count, "reference_eligible_unique_candidates": len(eligible),
        "exchange_classes": len({row["exchange_class_id"] for row in rows}),
        "missing_exchange_partners": sum(row["missing_exchange_partner"] for row in rows),
        "single_device_directions": sorted({row["devices_used"][0] for row in rows if len(row["devices_used"]) == 1}),
        "topology_prefix_cuts": cuts, "missing_topology_prefix_cuts": sorted(set(range(1, len(order))) - set(cuts)),
        "whole_layer_cuts": layer_cuts, "missing_whole_layer_cuts": sorted(set(layer_ids[1:]) - set(layer_cuts)),
        "branches": branches, "failures": failures, "candidates": rows,
        "limits": "Reference eligibility is a static constructor check, not GPU execution, memory feasibility, numerical or performance evidence. Copy barriers are model assumptions, not observed synchronization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/a2_decoder_candidate_coverage.v1.json"))
    args = parser.parse_args()
    frozen, _ = check(DEFAULT_BUNDLE)
    hardware = hardware_from_dict(frozen["hardware"])
    old, fresh = [], []
    for workload in frozen["workloads"]:
        graph = graph_from_dict(workload["graph"])
        plans = {key: plan_from_dict(value) for key, value in workload["candidate_plans"].items()}
        old.append({"model_id": workload["model_id"],
            "sealed_proposals": audit_candidates(graph, list(plans.values()), hardware),
            "sealed_measured_subset": audit_candidates(graph, [plans[key] for key in workload["execution_plan_ids"]], hardware)})
    for family, spec in (("qwen3-0.6b", qwen3_spec(2)), ("qwen2.5-3b", qwen25_3b_spec())):
        for layers in (2, 4):
            graph = build_decoder_stack(replace(spec, num_layers=layers))
            base, metadata = graph_wide_candidates(graph, [make_reference_plan(graph)], ("bw", "bw"))
            base = expand_directions(base)
            expanded = structured_candidates(graph, base)
            baseline, covered = audit_candidates(graph, base, hardware), audit_candidates(graph, expanded, hardware)
            fresh.append({"model_id": f"{family}-decoder{layers}-s256-b1", "nodes": len(graph.nodes), "edges": len(graph.edges),
                "base_source": "graph-wide-v1 probes plus singleton reference; no new search or labels",
                "base_probe_metadata": metadata, "base": baseline, "expanded": covered,
                "base_preserved": {plan_id(p) for p in base} <= {plan_id(p) for p in expanded}})
            print(f"{fresh[-1]['model_id']}: physical={len(base)}->{len(expanded)} "
                  f"isolated_missing={sum(len(b['missing_isolated_branch_masks']) for b in baseline['branches'])}"
                  f"->{sum(len(b['missing_isolated_branch_masks']) for b in covered['branches'])}", flush=True)
    summary = {"sealed_tasks": len(old),
        "sealed_proposal_physical_candidates": sum(w["sealed_proposals"]["input_candidates"] for w in old),
        "sealed_measured_physical_candidates": sum(w["sealed_measured_subset"]["input_candidates"] for w in old),
        "sealed_measured_tasks_missing_qkv_partitions": sum(any(b["family"] == "qkv" and b["missing_projection_patterns"]
             for b in w["sealed_measured_subset"]["branches"]) for w in old),
        "decoder_tasks": len(fresh), "base_physical_candidates": sum(w["base"]["input_candidates"] for w in fresh),
        "expanded_physical_candidates": sum(w["expanded"]["input_candidates"] for w in fresh),
        "base_missing_isolated_branch_masks": sum(len(b["missing_isolated_branch_masks"]) for w in fresh for b in w["base"]["branches"]),
        "expanded_missing_isolated_branch_masks": sum(len(b["missing_isolated_branch_masks"]) for w in fresh for b in w["expanded"]["branches"]),
        "base_missing_topology_prefix_cuts": sum(len(w["base"]["missing_topology_prefix_cuts"]) for w in fresh),
        "expanded_missing_topology_prefix_cuts": sum(len(w["expanded"]["missing_topology_prefix_cuts"]) for w in fresh),
        "expanded_missing_exchange_partners": sum(w["expanded"]["missing_exchange_partners"] for w in fresh),
        "all_base_candidates_preserved": all(w["base_preserved"] for w in fresh)}
    root = DEFAULT_BUNDLE.parents[1]
    sources = ("cost_model/decoder_candidate_coverage.py", "cost_model/placement_probes.py",
               "execution/decoder_reference.py", "simulator/model_graphs/decoder_stack.py")
    report = {"format": "fusion-placement-search.a2-candidate-coverage", "version": "1.0",
        "scope": "Label-free finite structural coverage only. No GPU, latency fitting, performance competition or SROP claim.",
        "summary": summary, "frozen_reference_id": frozen["freeze_id"], "frozen_single_layer_pools": old,
        "decoder_body_probes": fresh,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "frozen_input_sha256": hashlib.sha256((DEFAULT_BUNDLE / "freeze.json").read_bytes()).hexdigest(),
        "definitions": {"projection_pattern": "Actual physical bits for branch root projections.",
            "colocated_chain_pattern": "Physical bits, only when each pre-join branch chain stays on one device.",
            "isolated_branch_mask": "Branch bits XOR the common device of ALL nodes outside this layer's branch set; independent local witnesses.",
            "prefix_cut": "A split of deterministic graph topology order into two constant, different device regions; not tensor or pipeline parallelism.",
            "symmetry": "Exchange classes describe model symmetry; both actual physical assignments remain separate."}}
    write_json(args.output, report)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
