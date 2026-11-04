"""CPU-only LightOP cost-boundary and FFN plan coverage audit; no search scores."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from calibration.lightop_profile_calibration import review_receipt, read
from cost_model.ffn_placement_b1 import (COMPUTE, FUSED, INPUT, OUTPUT, derive_t2,
                                         derive_t3, enumerate_plans, independent_expected)
from cost_model.fusion_kernel_profile import key_id
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.measurement_labels import MeasurementContractError
from cost_model.structure import analyze_plan_structure
from integration.lightop.silu_mul import adapter_registry, declared_binding
from simulator.placement_artifacts import graph_fingerprint, load_placed_op_graph
from simulator.schema import FusionGroup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/lightop_profile_calibration_20261104_01/raw"


def query_id(nodes, device):
    return "+".join(nodes) + "@gpu" + str(device)


def ffn_domain(full):
    """Reuse B1 enumeration with the LightOP registry, never its KERV binding."""
    selected = {INPUT, *COMPUTE}
    edges = [e for e in full.edges if e.src in selected and (e.dst in selected or e.dst == OUTPUT)]
    graph = copy.deepcopy(full)
    graph.name = "qwen3-layer0-ffn-lightop-coverage"
    graph.nodes = {n: copy.deepcopy(full.nodes[n]) for n in [INPUT, *COMPUTE]}
    graph.nodes[INPUT].op_type = "input"
    graph.nodes[INPUT].flops = graph.nodes[INPUT].param_bytes = 0
    graph.edges = [copy.deepcopy(e) for e in edges if e.dst != OUTPUT]
    graph.extra = {"scope": "LightOP structural placement audit", "output_nodes": [COMPUTE[-1]]}
    definitions = [{"edge_id": f"ffn.e{i:02d}", "producer": e.src, "consumer": e.dst,
                    "tensor_shape": list(full.nodes[e.src].extra["output_shape"]),
                    "tensor_bytes": int(e.bytes), "dtype": full.nodes[e.src].extra["dtype"],
                    "boundary": "ingress" if e.src == INPUT else "egress" if e.dst == OUTPUT else "internal"}
                   for i, e in enumerate(edges)]
    plans, _ = enumerate_plans(graph, adapter_registry(), 0, 0)
    if {p["plan_id"] for p in plans} != independent_expected({}, 0, 0):
        raise ValueError("LightOP action space differs from the independent product oracle")
    rows = derive_t2(plans, definitions)
    return graph, plans, rows, derive_t3(rows)


def same_contract_diagnostic(placed, profile, index, raw):
    structure = analyze_plan_structure(placed.graph, placed.plan, placed.hardware.n_devs,
                                       registry=adapter_registry())
    group = structure.groups[index]
    old = AnalyticHardwareProfile(placed.hardware).estimate_group(placed.graph, group)
    prep = read(raw / "preparation.json")["one_call"]["tensors"]
    def nbytes(name):
        n = 2  # All four carriers have been strictly checked as BF16 by admission.
        for extent in prep[name]["shape"]:
            n *= extent
        return n
    inputs = nbytes("gate") + nbytes("up")
    packed, output = nbytes("packed"), nbytes("output")
    return {"status": "not_comparable_under_target_contract",
            "measured_complete_call_us": next(iter(profile.latencies.values())).duration_s * 1e6,
            "existing_analytic_group_diagnostic": {**asdict(old), "unit": "seconds"},
            "logical_tensor_io_bytes": {"ideal_fused_boundary": inputs + output,
                "pack_stage": inputs + packed, "lightop_stage": packed + output,
                "complete_adapter": inputs + 2 * packed + output,
                "not_a_measured_hbm_traffic_count": True},
            "old_launch_parameter_s": placed.hardware.launch_overhead,
            "old_rate_environment": read(ROOT / "data/bw1000_2card/raw_measurements.json")["environment"],
            "mismatches": ["ideal fused boundary omits the mandatory cat pack stage",
                "launch parameter came from synchronized eager scalar.add_ wall timing",
                "old rate calibration does not establish the current runtime Graph Replay contract"],
            "target_contract_analytic_prediction_us": None, "prediction_error_percent": None,
            "calibration_factor": None,
            "required_before_error_comparison": "model the complete call and explicitly bind its Graph Replay overhead/rates; independent validation cases still needed"}


def build(raw=RAW):
    raw = Path(raw)
    admission, profile = review_receipt(raw)
    if profile is None or admission["status"] != "accepted":
        raise MeasurementContractError("cost coverage requires the admitted single-group measurement")
    placed = load_placed_op_graph(raw / "placed.json", registry=adapter_registry())
    index = read(raw / "identity.json")["group_index"]
    graph, plans, rows, union = ffn_domain(placed.graph)
    groups = {}
    for p in plans:
        for g in p["groups"]:
            qid = query_id(g["node_ids"], g["device"])
            if qid in groups:
                continue
            fusion = len(g["node_ids"]) > 1
            group = FusionGroup(list(g["node_ids"]), g["device"], extra={"pattern_id": g["pattern_id"]})
            if fusion:
                group.extra["implementation"] = declared_binding()
            try:
                cost = profile.estimate_group(profile.keys.graph, group)
                measured, reason = cost.duration * 1e6, None
            except MeasurementContractError as error:
                measured, reason = None, str(error)
            groups[qid] = {"query_id": qid, "node_ids": g["node_ids"], "device": g["device"],
                "graph_node_specs": [{"node_id": n, "op_type": placed.graph.nodes[n].op_type,
                    "shape_parameters": placed.graph.nodes[n].extra.get("shape"),
                    "output_shape": placed.graph.nodes[n].extra.get("output_shape"),
                    "dtype": placed.graph.nodes[n].extra.get("dtype"),
                    "layout_declaration": placed.graph.nodes[n].extra.get("layout", "unknown")}
                    for n in g["node_ids"]],
                "implementation": declared_binding() if fusion else {"status": "Reference runtime identity/admission not bound in this LightOP path"},
                "accepted_query": measured is not None, "measured_us": measured,
                "rejection": reason, "candidate_plan_uses": 0}
    communications = {}
    by_edge = {}
    for item in union:
        qid = "copy/" + str(item["tensor_bytes"]) + "/" + str(item["source_gpu"]) + "->" + str(item["destination_gpu"])
        analytic = profile.estimate_transfer(item["source_gpu"], item["destination_gpu"], item["tensor_bytes"])
        if analytic.source != "analytic_link":
            raise ValueError("communication provider changed; review its admission before rerunning")
        communications[qid] = {"query_id": qid, "tensor_bytes": item["tensor_bytes"],
            "source_gpu": item["source_gpu"], "destination_gpu": item["destination_gpu"],
            "tensor_shapes": item["tensor_shapes"], "accepted_query": False, "measured_us": None,
            "existing_query_source": analytic.source, "existing_query_diagnostic_us": analytic.duration * 1e6,
            "reason": "selected-group strict=True delegates links to the analytic base; no admitted LightOP-scope link measurement",
            "transport_contract": "unbound; logical requirement only", "witnesses": item["witnesses"]}
        for w in item["witnesses"]:
            by_edge[(w["plan_id"], w["edge_id"])] = qid
    plan_rows = []
    for p in plans:
        required = [query_id(g["node_ids"], g["device"]) for g in p["groups"]]
        for qid in required:
            groups[qid]["candidate_plan_uses"] += 1
        link_items = [r for r in rows if r["plan_id"] == p["plan_id"] and r["cross_device"]]
        missing = [qid for qid in required if not groups[qid]["accepted_query"]]
        missing_links = [by_edge[(p["plan_id"], r["edge_id"])] for r in link_items]
        plan_rows.append({"plan_id": p["plan_id"], "fusion_enabled": p["fusion_enabled"],
            "groups": p["groups"], "boundary": p["boundary"], "required_compute_queries": required,
            "available_compute_queries": [q for q in required if groups[q]["accepted_query"]],
            "missing_compute_queries": missing, "missing_link_queries": sorted(set(missing_links)),
            "cross_device_edges": len(link_items), "priceable": not missing and not missing_links,
            "formal_plan_cost_us": None, "scoring_performed": False})
    universal_families = [{"node_id": node, "device_alternatives": [query_id([node], d) for d in (0, 1)],
                           "one_alternative_required_per_plan": True}
                          for node in COMPUTE if all(not groups[query_id([node], d)]["accepted_query"] for d in (0, 1))
                          and all(any(node in g["node_ids"] and len(g["node_ids"]) == 1 for g in p["groups"]) for p in plans)]
    compute_missing = sorted(q for q, v in groups.items() if not v["accepted_query"])
    zero_plans = [p for p in plan_rows if p["cross_device_edges"] == 0]
    gpu0_missing = sorted(set(q for p in zero_plans for q in p["missing_compute_queries"]))
    same_shape = [{"device": d, "query_ids": [query_id([n], d) for n in COMPUTE[:2]],
                   "reuse_authorized": False, "condition": "actual Canonical Keys, layouts, input/weight carriers and contract must match before coalescing"}
                  for d in (0, 1)]
    return {"scope": "LightOP GPU0 single-profile CPU cost boundary and structural FFN coverage audit",
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "admission_id": admission["admission_id"], "kernel_key_id": admission["kernel_key_id"],
        "measurement_contract_id": admission["measurement_contract_id"],
        "raw_receipt_sha256": admission["raw_file_sha256"], "subgraph_hash": graph_fingerprint(graph),
        "action_space": {"implementation": declared_binding(), "boundary_input_gpu": 0, "boundary_output_gpu": 0,
            "native_plans": sum(not p["fusion_enabled"] for p in plans), "fused_plans": sum(p["fusion_enabled"] for p in plans),
            "labeled_plans": len(plans), "structural_classes": len({p["structural_class_id"] for p in plans}),
            "merged_for_cost_scoring": False, "certification": "declared structural legality; no HBM/whole-plan execution certification"},
        "summary": {"accepted_selected_group_profiles": 1, "priceable_plans": sum(p["priceable"] for p in plan_rows),
            "total_plans": len(plans), "plans_with_the_local_fused_cost": sum(bool(p["available_compute_queries"]) for p in plan_rows),
            "unique_compute_queries": len(groups), "missing_compute_queries": len(compute_missing),
            "logical_link_query_classes": len(communications), "accepted_link_queries": 0,
            "logical_edge_rows": len(rows), "cross_device_edge_occurrences": sum(r["cross_device"] for r in rows),
            "zero_communication_plans": len(zero_plans), "uncovered_in_every_plan_families": universal_families},
        "compute_queries": sorted(groups.values(), key=lambda r: r["query_id"]),
        "communication_queries": sorted(communications.values(), key=lambda r: r["query_id"]),
        "plan_coverage": plan_rows, "same_contract_analytic_audit": same_contract_diagnostic(placed, profile, index, raw),
        "measurement_next_steps": {
            "first_scope": "GPU0/GPU0 boundary, both zero-communication plans; close compute costs before transport expansion",
            "first_scope_missing_queries": gpu0_missing, "gpu_measurement_tasks_frozen": False,
            "same_shape_reuse_candidates": same_shape,
            "full_48_plan_missing_compute_queries": compute_missing,
            "full_48_plan_missing_link_queries": sorted(communications),
            "task_count": None, "task_count_reason": "query slots are not measurement tasks; Reference identity/contract closure and reuse checks precede task freezing",
            "additional_validation": ["native SiLU->Mul chain versus separately measured operators under the same timing semantics",
                "representative FFN chain composition; standalone costs alone do not certify plan latency"],
            "historical_kerv_or_run05_promoted": False},
        "verification": {"independent_action_product_matches": True, "new_gpu_measurements": 0,
            "search_or_mcts_runs": 0, "model_error_or_calibration_factor_reported": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    report = build()
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
