"""Local checks for a proposed logical stage map, not a downstream PP adapter."""
import math

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.structure import analyze_plan_structure
from simulator.placement_artifacts import validate_placed_op_graph
from simulator.validate import validate_fusion_plan


def validate_stage_assignment(placed, stage_by_node, n_stages, *, device_by_node=None,
                              require_registry_executable=False):
    """Preserve nodes, dependencies, atomic groups and the fixed device labels."""
    validate_placed_op_graph(placed)
    if require_registry_executable:
        validate_fusion_plan(placed.graph, placed.plan, placed.hardware.n_devs, require_executable=True)
    nodes = set(placed.graph.nodes)
    if set(stage_by_node) != nodes:
        raise ValueError("stage assignment must cover exactly the original node IDs")
    if n_stages < 1 or any(type(s) is not int or not 0 <= s < n_stages for s in stage_by_node.values()):
        raise ValueError("stage IDs must be integers within the declared stage count")
    if set(stage_by_node.values()) != set(range(n_stages)):
        raise ValueError("logical stages must be nonempty")
    expected_devices = {n: g.device for g in placed.plan.groups for n in g.node_ids}
    if device_by_node is not None and device_by_node != expected_devices:
        raise ValueError("proposal changes the output graph's fixed device labels")
    for index, group in enumerate(placed.plan.groups):
        if len({stage_by_node[n] for n in group.node_ids}) != 1:
            raise ValueError(f"proposal cuts fusion group fg{index:04d}")
    if any(stage_by_node[e.src] > stage_by_node[e.dst] for e in placed.graph.edges):
        raise ValueError("proposal reverses an original dependency across stages")


def group_work(placed):
    structure = analyze_plan_structure(placed.graph, placed.plan, placed.hardware.n_devs)
    profile = AnalyticHardwareProfile(placed.hardware)
    return structure, [profile.estimate_group(placed.graph, group).duration for group in structure.groups]


def balanced_group_assignment(placed, n_stages):
    """A deterministic one/two-stage compute-work proxy, preserving group order."""
    if n_stages not in (1, 2):
        raise ValueError("diagnostic proposer supports one or two logical stages")
    structure, weights = group_work(placed)
    if n_stages == 1:
        return {n: 0 for n in placed.graph.nodes}
    if len(weights) < 2:
        raise ValueError("not enough atomic groups for two nonempty stages")
    prefix = [0.]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    cut = min(range(1, len(weights)), key=lambda i: (max(prefix[i], prefix[-1]-prefix[i]), i))
    return {n: int(i >= cut) for i, group in enumerate(structure.groups) for n in group.node_ids}


def equal_layer_assignment(placed, n_stages):
    """An actual equal-layer baseline; do not substitute equal operator counts."""
    layer_by_node = {n: node.extra.get("layer") for n, node in placed.graph.nodes.items()}
    if any(type(layer) is not int for layer in layer_by_node.values()):
        raise ValueError("not_applicable: missing integer layer metadata")
    layers = sorted(set(layer_by_node.values()))
    if not 1 <= n_stages <= len(layers):
        raise ValueError("not_applicable: fewer layers than requested nonempty stages")
    stage_by_layer = {layer: min(n_stages-1, i*n_stages//len(layers)) for i, layer in enumerate(layers)}
    return {n: stage_by_layer[layer] for n, layer in layer_by_node.items()}


def stage_metrics(placed, stage_by_node, n_stages):
    """Only valid proposals have load metrics; these are analytic work sums."""
    validate_stage_assignment(placed, stage_by_node, n_stages)
    structure, weights = group_work(placed)
    loads = [math.fsum(weight for group, weight in zip(structure.groups, weights)
                       if stage_by_node[group.node_ids[0]] == stage) for stage in range(n_stages)]
    mean = math.fsum(loads)/n_stages
    return dict(stage_group_work_s=loads, imbalance_ratio=max(loads)/mean-1 if mean else 0.,
        interstage_dependency_bytes=sum(dep.bytes for dep in structure.dependencies
            if stage_by_node[dep.src_node] != stage_by_node[dep.dst_node]),
        original_cross_device_bytes=structure.cross_device_bytes,
        scope="logical partition work proxy; devices preserved, no stage/rank mapping or PP runtime model")
