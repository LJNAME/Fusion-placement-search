"""Label-independent placement probes for fixed-pool development diagnostics.

These probes complement shallow prefix MCTS. They do not select workloads,
use latency labels, change fusion legality, or claim exhaustive coverage.
"""

from __future__ import annotations

import copy
from collections import defaultdict

from simulator import make_reference_plan, validate_fusion_plan

from .search_sensitivity import device_exchange_permutations, plan_id


COVERAGE_VERSION = "graph-wide-v1"


def graph_wide_candidates(graph, base_plans, device_classes):
    """Keep every base candidate and add deterministic whole-depth probes.

    Singleton motifs and fork-to-join chains are repeated at every layer.
    Six uniformly spaced layer anchors get a local chain probe, and three
    layer-boundary cuts test larger contiguous regions. No score truncation
    is applied, since it could discard the very depth coverage being tested.
    The current experiment deliberately supports exactly two devices.
    """

    if len(device_classes) != 2:
        raise ValueError("graph-wide-v1 requires exactly two devices")
    exchange = device_exchange_permutations(device_classes)
    plans = {plan_id(plan, exchange): copy.deepcopy(plan) for plan in base_plans}
    base_ids = set(plans)
    reference = make_reference_plan(graph, device=0)
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for edge in graph.edges:
        successors[edge.src].append(edge.dst)
        predecessors[edge.dst].append(edge.src)
    layers = sorted({int(node.extra["layer"]) for node in graph.nodes.values()
                     if node.extra.get("layer") is not None and int(node.extra["layer"]) >= 0
                     and int(node.extra["layer"]) < graph.extra.get("layer_end", float("inf"))})
    motifs = defaultdict(set)
    for node in graph.nodes.values():
        if node.op_type == "input":
            continue
        motifs[str(node.extra.get("local_name", node.id))].add(node.id)
    chains = defaultdict(set)
    for parent in sorted(successors):
        if len(successors[parent]) < 2:
            continue
        for child in sorted(successors[parent]):
            chain = [child]
            while len(successors[chain[-1]]) == 1:
                following = successors[chain[-1]][0]
                if len(predecessors[following]) != 1:
                    break
                chain.append(following)
            if len(chain) > 1:
                signature = tuple(str(graph.nodes[key].extra.get("local_name", key)) for key in chain)
                chains[signature].update(chain)

    proposals = []
    def add(kind, label, nodes):
        if not nodes:
            return
        plan = copy.deepcopy(reference)
        for group in plan.groups:
            if any(key in nodes for key in group.node_ids):
                group.device = 1
        plan.name = f"{COVERAGE_VERSION}:{kind}:{label}"
        plan.extra["placement_probe"] = {"version": COVERAGE_VERSION, "kind": kind, "label": label}
        identifier = plan_id(plan, exchange)
        proposals.append({"plan_id": identifier, "kind": kind, "label": label})
        if identifier not in plans:
            validate_fusion_plan(graph, plan, 2, require_executable=True)
            plans[identifier] = plan

    add("reference", "single-device", set(graph.nodes))
    for name, nodes in sorted(motifs.items()):
        add("all-depth-singleton", name, nodes)
    for signature, nodes in sorted(chains.items()):
        add("all-depth-chain", "+".join(signature), nodes)
    chain_items = sorted(chains.items())
    if layers and chain_items:
        anchors = sorted({layers[round(index * (len(layers) - 1) / 5)] for index in range(6)})
        for index, layer in enumerate(anchors):
            signature, nodes = chain_items[index % len(chain_items)]
            selected = {key for key in nodes if graph.nodes[key].extra.get("layer") == layer}
            add("depth-anchor-chain", f"{layer}:" + "+".join(signature), selected)
    if len(layers) > 1:
        for cut in sorted({max(1, len(layers) * numerator // 4) for numerator in (1, 2, 3)}):
            boundary = layers[cut]
            nodes = {key for key, node in graph.nodes.items()
                     if node.extra.get("layer") is not None and int(node.extra["layer"]) >= boundary}
            add("layer-boundary-cut", str(boundary), nodes)
    metadata = {
        "version": COVERAGE_VERSION,
        "selection": "all structural probes retained; no latency-dependent selection",
        "base_candidates": len(base_ids), "added_candidates": len(plans) - len(base_ids),
        "union_candidates": len(plans), "layer_ids": layers,
        "proposal_count_before_deduplication": len(proposals), "proposals": proposals,
    }
    return list(plans.values()), metadata
