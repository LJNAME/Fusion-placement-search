"""B0 CPU identity/layout closure using the real Reference execution path.

No timings, latency labels, GPU execution, search, or admission decisions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from cost_model.fusion_kernel_profile import key_id
from cost_model.search_evaluation import build_evaluation_graph
from execution.reference import _ReferenceExecutor
from execution.reference_identity import ReferenceKernelKeys, observe_reference_runtime, tensor_observation, with_tensor_layouts
from simulator import make_reference_plan
from simulator.hardware import DeviceSpec, HardwareSpec
from simulator.placement_artifacts import PlacedOpGraph, graph_fingerprint

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = "L000.pre_mlp_norm"
NODES = ["L000.mlp." + name for name in ("gate_proj", "up_proj", "silu", "gated_mul", "down_proj")]


def prepare_cpu_identity():
    """Actual CPU observations are never projected onto BW1000 identities."""
    graph = build_evaluation_graph("layer")
    node_ids = {BOUNDARY, *NODES}
    graph.nodes = {name: copy.deepcopy(graph.nodes[name]) for name in [BOUNDARY, *NODES]}
    graph.edges = [copy.deepcopy(e) for e in graph.edges if e.src in node_ids and e.dst in node_ids]
    graph.name = "qwen3-layer0-reference-ffn-b0-cpu"
    graph.nodes[BOUNDARY].op_type = "input"
    graph.nodes[BOUNDARY].flops = graph.nodes[BOUNDARY].param_bytes = 0
    with torch.no_grad():
        observation = observe_reference_runtime("cpu")
        # Positive numeric schema placeholders only. Never a calibrated cost source.
        hardware = HardwareSpec([DeviceSpec(1., 1., 1., name=observation["device"]["name"],
            extra={"source": observation["device"], "physical_index": None, "identity_only": True})])
        plan = make_reference_plan(graph)
        executor = _ReferenceExecutor(PlacedOpGraph(graph, hardware, plan), ["cpu"], 1234)
        values = {}
        for node_id in graph.topological_order():
            values[node_id] = executor._execute_node(node_id, [values[e.src] for e in graph.in_edges(node_id)], torch.device("cpu"))
        layouts = {node_id: tensor_observation(value) for node_id, value in values.items()}
        parameters = {node_id: {"weight": tensor_observation(weight)} for (node_id, _), weight in executor._weights.items()}
        bound_graph = with_tensor_layouts(graph, layouts, parameters=parameters)
        keys = ReferenceKernelKeys(bound_graph, hardware, observation)
        groups = [group for group in plan.groups if group.node_ids[0] in NODES]
        records = []
        for group in groups:
            key, identifier = keys.require_complete(group)
            assert identifier == key_id(json.loads(json.dumps(key)))
            records.append({"node_id": group.node_ids[0], "kernel_key_id": identifier,
                            "kernel_key": key, "identity_gaps": [], "profile_admission": "not_requested"})
    return keys, groups, {"format": "fusion-placement-search.reference-identity-b0", "version": "1.0",
        "scope": "actual_local_cpu_reference_identity_only", "runtime": observation["runtime"],
        "device": observation["device"], "source_paths": observation["source_paths"],
        "source_graph_hash": graph_fingerprint(graph), "layout_observations": layouts,
        "parameter_observations": parameters, "records": records,
        "local_identity_closed": True, "gpu_identity_closed": False,
        "gpu_identity_requirement": "observe the executing GPU Torch build, device and Tensor layouts; never copy CPU Keys",
        "new_latency_profiles": 0, "accepted_profiles_created": 0, "new_gpu_measurements": 0,
        "search_cost_items_admitted": 0, "gpu_measurement_entry_open": False,
        "pending_entry_conditions": ["GPU runtime identity/layout verification", "B1", "B2", "B3"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reference_identity_b0_v1/identity.json")
    args = parser.parse_args()
    _, _, report = prepare_cpu_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "local_complete_keys": len(report["records"]),
                      "gpu_identity_closed": False, "new_latency_profiles": 0}))


if __name__ == "__main__":
    main()
