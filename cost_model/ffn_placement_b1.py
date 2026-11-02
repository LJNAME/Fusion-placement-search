"""B1 exhaustive structural FFN placement inventory; no scoring or GPU calls.

Run: python -m cost_model.ffn_placement_b1
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE, candidate_applicability
from cost_model.fusion_kernel_profile import canonical, key_id
from cost_model.kerv_profile import digest
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import build_evaluation_graph
from integration.flagscale.kerv_execution import load_kerv_registry
from simulator.dataset import graph_to_dict
from simulator.placement_artifacts import graph_fingerprint

ROOT = Path(__file__).resolve().parents[1]
INPUT = "L000.pre_mlp_norm"
OUTPUT = "L000.residual_add_2"
COMPUTE = tuple("L000.mlp." + name for name in ("gate_proj", "up_proj", "silu", "gated_mul", "down_proj"))
FUSED = (COMPUTE[2], COMPUTE[3])
FILES = ("identity.json", "T1_legal_placements.json", "T2_cross_device_edges.json", "T3_communication_union.json")
SYNC = "producer_complete -> transfer_complete -> consumer_ready; no overlap claim"


def serialized(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def assignment_key(groups):
    return sorted([list(group["node_ids"]), group["device"]] for group in groups)


def structure_class(groups):
    original = assignment_key(groups)
    swapped = sorted([list(g["node_ids"]), 1 - g["device"]] for g in groups)
    return key_id(min((original, swapped), key=canonical))


def build_inputs():
    full = build_evaluation_graph("model")
    selected = {INPUT, *COMPUTE}
    edges = [e for e in full.edges if e.src in selected and (e.dst in selected or e.dst == OUTPUT)]
    graph = copy.deepcopy(full)
    graph.name = "qwen3-layer0-ffn-b1"
    graph.nodes = {n: copy.deepcopy(full.nodes[n]) for n in [INPUT, *COMPUTE]}
    graph.edges = [copy.deepcopy(e) for e in edges if e.dst != OUTPUT]
    graph.extra = {"scope": "FFN structural action enumeration", "output_nodes": [COMPUTE[-1]]}
    graph.nodes[INPUT].op_type = "input"
    graph.nodes[INPUT].flops = graph.nodes[INPUT].param_bytes = 0
    registry, capability = load_kerv_registry(DEFAULT_BUNDLE / "capability.json", DEFAULT_FLAGSCALE)
    definitions = [{"edge_id": f"ffn.e{i:02d}", "producer": e.src, "consumer": e.dst,
                    "tensor_shape": list(full.nodes[e.src].extra["output_shape"]),
                    "tensor_bytes": int(e.bytes), "dtype": full.nodes[e.src].extra["dtype"],
                    "boundary": "ingress" if e.src == INPUT else "egress" if e.dst == OUTPUT else "internal"}
                   for i, e in enumerate(edges)]
    assert len(definitions) == 7
    return full, graph, registry, capability, definitions


def _check_restrictions(restrictions):
    result = {}
    for node, devices in (restrictions or {}).items():
        if node not in COMPUTE or not isinstance(devices, list) or any(type(d) is not int or d not in (0, 1) for d in devices):
            raise ValueError("invalid B1 node device restriction: " + node)
        result[node] = sorted(set(devices))
    return result


def enumerate_plans(graph, registry, input_device, output_device, restrictions=None):
    restrictions = _check_restrictions(restrictions)
    if input_device not in (0, 1) or output_device not in (0, 1):
        raise ValueError("B1 boundary devices must be 0 or 1")
    space = LegalFusionSearchSpace(graph, 2, registry=registry, require_executable=True)
    plans = []

    def visit(state):
        if space.is_terminal(state):
            plan = space.to_plan(state)  # Existing graph/pattern/placement legality gate.
            groups = [{"node_ids": list(g.node_ids), "device": g.device,
                       "pattern_id": g.extra["pattern_id"], "backend": g.extra["backend"],
                       "implementation_status": g.extra["implementation_status"]}
                      for g in plan.groups if g.node_ids != [INPUT]]
            identity = {"groups": assignment_key(groups), "input_device": input_device, "output_device": output_device}
            plans.append({"plan_id": key_id(identity), "fusion_enabled": any(len(g["node_ids"]) > 1 for g in groups),
                          "partition_id": key_id(sorted(list(g["node_ids"]) for g in groups)),
                          "groups": groups, "boundary": {INPUT: input_device, OUTPUT: output_device},
                          "structural_class_id": structure_class(groups), "legality": "passed_declared_constraints",
                          "scoring_performed": False})
            return
        for choice in space.actions(state):
            permitted = all(choice.device == input_device if node == INPUT else
                            choice.device in restrictions.get(node, [0, 1]) for node in choice.candidate.node_ids)
            if permitted:
                visit(space.apply(state, choice))

    visit(space.initial_state())
    return sorted(plans, key=lambda p: p["plan_id"]), space


def derive_t2(plans, edges):
    rows = []
    for plan in plans:
        devices = {node: group["device"] for group in plan["groups"] for node in group["node_ids"]}
        devices.update(plan["boundary"])
        membership = {node: i for i, group in enumerate(plan["groups"]) for node in group["node_ids"]}
        for edge in edges:
            src, dst = edge["producer"], edge["consumer"]
            eliminated = src in membership and dst in membership and membership[src] == membership[dst]
            cross = not eliminated and devices[src] != devices[dst]
            rows.append({"plan_id": plan["plan_id"], **edge, "source_gpu": devices[src], "destination_gpu": devices[dst],
                         "internal_to_fused_group": eliminated, "materialized_tensor": not eliminated,
                         "cross_device": cross, "transfer_required": cross,
                         "transfer_bytes": edge["tensor_bytes"] if cross else 0,
                         "layout": "unobserved; declaration/runtime binding pending B3/B4",
                         "transfer_mechanism": "unbound_B2", "synchronization": SYNC})
    return rows


def derive_t3(rows):
    buckets = {}
    for row in rows:
        if not row["cross_device"]:
            continue
        signature = {"tensor_bytes": row["tensor_bytes"], "source_gpu": row["source_gpu"],
                     "destination_gpu": row["destination_gpu"], "dtype": row["dtype"],
                     "layout": row["layout"], "transfer_mechanism": row["transfer_mechanism"],
                     "synchronization": row["synchronization"]}
        signature_key = canonical(signature)
        if signature_key not in buckets:
            buckets[signature_key] = {"requirement_id": key_id(signature), **signature,
                                      "tensor_shapes": [], "witnesses": [],
                                      "measurement_contract_id": None, "measurement_ready": False}
        item = buckets[signature_key]
        if row["tensor_shape"] not in item["tensor_shapes"]:
            item["tensor_shapes"].append(row["tensor_shape"])
        item["witnesses"].append({"plan_id": row["plan_id"], "edge_id": row["edge_id"]})
    for item in buckets.values():
        item["edge_occurrences"] = len(item["witnesses"])
        item["witness_plan_count"] = len({w["plan_id"] for w in item["witnesses"]})
    return sorted(buckets.values(), key=lambda item: (item["tensor_bytes"], item["source_gpu"], item["destination_gpu"], item["requirement_id"]))


def independent_expected(restrictions, input_device, output_device):
    """Independent closed-form product oracle for this one-pair FFN domain."""
    expected = set()
    for partition in ([[n] for n in COMPUTE], [[COMPUTE[0]], [COMPUTE[1]], list(FUSED), [COMPUTE[4]]]):
        for devices in itertools.product((0, 1), repeat=len(partition)):
            if any(device not in restrictions.get(node, [0, 1]) for group, device in zip(partition, devices) for node in group):
                continue
            groups = [{"node_ids": group, "device": device} for group, device in zip(partition, devices)]
            expected.add(key_id({"groups": assignment_key(groups), "input_device": input_device, "output_device": output_device}))
    return expected


def generate(input_device=0, output_device=0, restrictions=None):
    restrictions = _check_restrictions(restrictions)
    full, graph, registry, capability, definitions = build_inputs()
    plans, space = enumerate_plans(graph, registry, input_device, output_device, restrictions)
    rows = derive_t2(plans, definitions)
    union = derive_t3(rows)
    original = json.loads((ROOT / "data/kerv_profile_admission_v1/measurements/profile.json").read_text(encoding="utf-8"))
    fused_evidence = [{"case_id": r["case_id"], "median_us": r["statistics"]["median_us"],
                       "record_id": r["record_id"], "device": r["kernel_key"]["hardware"]}
                      for r in original["records"] if r["role"] == "fused_kernel"]
    symmetry = {"operation": "0 <-> 1", "structural_class_scope": "compute groups only; boundary excluded",
                "structural_classes": len({p["structural_class_id"] for p in plans}),
                "global_swap_preserves_boundary_constraints": False,
                "global_swap_preserves_all_constraints_and_cost": False,
                "cost_invariance_proven": False, "merge_for_scoring": False,
                "known_fused_terms_equal": len({r["median_us"] for r in fused_evidence}) == 1,
                "historical_local_fused_evidence": fused_evidence,
                "reason": "fixed boundary identities and unequal local measured terms; no full cost invariance certificate"}
    internal_native = sum(not p["fusion_enabled"] for p in plans)
    internal_fused = sum(p["fusion_enabled"] for p in plans)
    partitions = {}
    plan_ids = {p["plan_id"] for p in plans}
    for plan in plans:
        swapped = [{"node_ids": g["node_ids"], "device": 1 - g["device"]} for g in plan["groups"]]
        partner = key_id({"groups": assignment_key(swapped), "input_device": input_device, "output_device": output_device})
        plan["compute_swap_partner_id"] = partner if partner in plan_ids else None
        partition = partitions.setdefault(plan["partition_id"], {"partition_id": plan["partition_id"],
            "fusion_enabled": plan["fusion_enabled"], "groups": [
                {"node_ids": g["node_ids"], "allowed_devices": [d for d in (0, 1) if all(d in restrictions.get(n, [0, 1]) for n in g["node_ids"])]}
                for g in plan["groups"]], "labeled_placements": 0, "structure_class_ids": set()})
        partition["labeled_placements"] += 1
        partition["structure_class_ids"].add(plan["structural_class_id"])
    symmetry["compute_only_swap_preserves_declared_constraints"] = all(p["compute_swap_partner_id"] is not None for p in plans)
    for partition in partitions.values():
        partition["compute_swap_structure_classes"] = len(partition.pop("structure_class_ids"))
    candidates = [{"node_ids": list(c.node_ids), "pattern_id": c.pattern_id,
                   "allowed_devices": [d for d in (0, 1) if all(d in restrictions.get(n, [0, 1]) for n in c.node_ids)],
                   "implementation_status": c.implementation_status}
                  for c in space.candidates if c.node_ids != (INPUT,)]
    identity = {"format": "fusion-placement-search.ffn-placement-b1", "version": "1.0",
        "scope": "layer0 FFN declared structural action space; not cost/HBM/deployment certification",
        "source_graph_hash": graph_fingerprint(full), "subgraph_hash": graph_fingerprint(graph),
        "subgraph": graph_to_dict(graph), "compute_node_order": list(COMPUTE), "edges": definitions,
        "boundary_constraints": {"input_producer": INPUT, "input_device": input_device,
                                 "output_consumer": OUTPUT, "output_device": output_device,
                                 "authority": "user-confirmed GPU0/GPU0 default; CLI alternatives are separate scenarios"},
        "legal_fusion_candidates": candidates, "additional_node_device_restrictions": restrictions,
        "node_reachable_devices": {node: sorted({g["device"] for p in plans for g in p["groups"] if node in g["node_ids"]}) for node in COMPUTE},
        "constraints_enforced": ["exact disjoint coverage", "DAG/topology", "same-layer source-pinned executable pattern",
                                 "KERV alias/shape guards with runtime contiguous precondition", "GPU ordinal 0/1", "declared boundary and node device sets"],
        "not_certified": ["runtime layout", "HBM/workspace feasibility", "stage/stream implementation", "cost admission", "production captured graph"],
        "capability_id": capability["capability_id"], "kerv_applicability": candidate_applicability(graph, FUSED, layout_policy="require_contiguous_at_runtime"),
        "source_sha256": {path: digest(ROOT / path) for path in (
            "cost_model/ffn_placement_b1.py", "cost_model/legal_search.py", "simulator/validate.py", "simulator/fusion_rules.py",
            "simulator/model_graphs/dense_layer.py", "cost_model/search_evaluation.py", "integration/flagscale/kerv_execution.py",
            "calibration/kerv_silu_mul.py",
            "data/kerv_silu_mul_proxy_v1/capability.json", "data/kerv_profile_admission_v1/measurements/profile.json")},
        "new_gpu_measurements": 0, "cost_scoring_calls": 0, "mcts_runs": 0, "b2_contract_review_complete": False}
    t1 = {"summary": {"native_plans": internal_native, "fused_plans": internal_fused, "legal_labeled_plans": len(plans),
                      "compute_device_swap_structure_classes": symmetry["structural_classes"],
                      "labeled_plans_to_score": len(plans), "cost_scoring_calls": 0,
                      "unconstrained_theory": {"native": 2 ** 5, "fused": 2 ** 4, "total": 2 ** 5 + 2 ** 4,
                                               "compute_swap_classes": (2 ** 5 + 2 ** 4) // 2},
                      "additional_node_restrictions_active": bool(restrictions)}, "symmetry": symmetry,
          "fusion_partitions": sorted(partitions.values(), key=lambda p: p["partition_id"]), "plans": plans}
    t2 = {"scope": "all seven original logical edges per legal plan; eliminated edges have transfer_bytes=0", "rows": rows}
    t3 = {"scope": "B1-derived logical communication requirements; B2 contract binding still required",
          "requirements": union, "cost_admission": "not_open", "measurement_ready": False}
    return dict(zip(FILES, (identity, t1, t2, t3)))


def verify(bundle):
    identity, t1, t2, t3 = (bundle[name] for name in FILES)
    plans, rows, union = t1["plans"], t2["rows"], t3["requirements"]
    boundary = identity["boundary_constraints"]
    plan_ids = {p["plan_id"] for p in plans}
    assert len(plan_ids) == len(plans), "duplicate legal plans"
    assert plan_ids == independent_expected(identity["additional_node_device_restrictions"], boundary["input_device"], boundary["output_device"]), "missing/extra plans"
    expected_candidates = {((n,), "unfused") for n in COMPUTE}
    expected_candidates.add((FUSED, "silu_mul"))
    assert {(tuple(c["node_ids"]), c["pattern_id"]) for c in identity["legal_fusion_candidates"]} == expected_candidates, "domain differs from the one-pair FFN oracle"
    assert len(rows) == len(plans) * len(identity["edges"])
    assert derive_t2(plans, identity["edges"]) == rows, "T2 disagrees with plan/boundary assignments"
    assert t1["summary"]["legal_labeled_plans"] == len(plans)
    assert t1["summary"]["native_plans"] == sum(not p["fusion_enabled"] for p in plans)
    assert t1["summary"]["fused_plans"] == sum(p["fusion_enabled"] for p in plans)
    assert len({(r["plan_id"], r["edge_id"]) for r in rows}) == len(rows)
    assert all(r["cross_device"] == (not r["internal_to_fused_group"] and r["source_gpu"] != r["destination_gpu"]) for r in rows)
    assert derive_t3(rows) == union, "T3 is not derived exactly from T2"
    witnesses = {(w["plan_id"], w["edge_id"]) for u in union for w in u["witnesses"]}
    assert witnesses == {(r["plan_id"], r["edge_id"]) for r in rows if r["cross_device"]}
    internal = [r for r in rows if (r["producer"], r["consumer"]) == FUSED]
    native_internal = [r for r in internal if not r["internal_to_fused_group"] and r["cross_device"]]
    repeat = generate(boundary["input_device"], boundary["output_device"], identity["additional_node_device_restrictions"])
    assert all(serialized(bundle[name]) == serialized(repeat[name]) for name in FILES), "enumeration is not byte-reproducible"
    return {"pass_checks": True, "legal_labeled_plans": len(plans), "native_plans": t1["summary"]["native_plans"],
            "fused_plans": t1["summary"]["fused_plans"], "compute_structure_classes": t1["symmetry"]["structural_classes"],
            "labeled_plans_to_score": len(plans), "t2_rows": len(rows), "cross_device_edge_occurrences": sum(r["cross_device"] for r in rows),
            "t3_requirement_classes": len(union), "native_silu_mul_cross_device_occurrences": len(native_internal),
            "native_silu_mul_directions": sorted({(r["source_gpu"], r["destination_gpu"]) for r in native_internal}),
            "fused_silu_mul_eliminated_occurrences": sum(r["internal_to_fused_group"] for r in internal),
            "independent_product_oracle_matches": True, "duplicate_plans": 0, "all_edges_per_plan": True,
            "T3_exactly_derived_from_T2": True, "all_union_entries_have_witnesses": True, "repeat_byte_equivalent": True,
            "output_payload_ids": {name: key_id(value) for name, value in bundle.items()},
            "new_gpu_measurements": 0, "cost_scoring_calls": 0, "mcts_runs": 0, "b2_contract_review_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ffn_placement_b1")
    parser.add_argument("--input-device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--restrictions", type=Path, help="JSON mapping compute node IDs to allowed GPU lists")
    args = parser.parse_args()
    restrictions = json.loads(args.restrictions.read_text(encoding="utf-8")) if args.restrictions else None
    bundle = generate(args.input_device, args.output_device, restrictions)
    verification = verify(bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in {**bundle, "verification.json": verification}.items():
        with (args.output / name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized(value))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
