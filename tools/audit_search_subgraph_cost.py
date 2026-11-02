"""CPU-only cost inventory; never measures, admits, or searches candidates.

Run from the repository root: python -m tools.audit_search_subgraph_cost
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from calibration.kerv_profile_workflow import demonstrate
from cost_model.fusion_kernel_profile import GroupKernelKeys, key_id
from cost_model.kerv_admission import USE, load_admitted_store
from cost_model.kerv_profile import digest, normalize_identity
from cost_model.measurement_labels import MeasurementContractError
from cost_model.search_evaluation import build_evaluation_graph
from integration.flagscale.kerv_execution import load_kerv_registry
from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from simulator import enumerate_fusion_candidates, make_reference_plan
from simulator.placement_artifacts import graph_fingerprint, load_placed_op_graph

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/kerv_profile_admission_v1"
OUTPUT = ROOT / "data/search_subgraph_cost_audit_v1/report.json"
PREFIX = "L000."
COMPUTE = [PREFIX + name for name in (
    "mlp.gate_proj", "mlp.up_proj", "mlp.silu", "mlp.gated_mul", "mlp.down_proj")]
BOUNDARY = PREFIX + "pre_mlp_norm"


def audit():
    sources = {}

    def read(path):
        path = Path(path)
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    graph = build_evaluation_graph("model")
    ff_ids = {BOUNDARY, *COMPUTE}
    edges = [e for e in graph.edges if e.src in ff_ids and e.dst in ff_ids]
    egress = [e for e in graph.edges if e.src in COMPUTE and e.dst not in ff_ids]
    assert len(edges) == 6 and len(egress) == 1
    signatures = {tuple(tuple(graph.nodes[f"L{layer:03d}.mlp.{name}"].extra["output_shape"])
                        for name in ("gate_proj", "up_proj", "silu", "gated_mul", "down_proj"))
                  for layer in range(28)}
    assert len(signatures) == 1
    registry, _ = load_kerv_registry(DEFAULT_BUNDLE / "capability.json", DEFAULT_FLAGSCALE)
    candidates = [c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                  if c.pattern_id == "silu_mul"]
    selected = [c for c in candidates if set(c.node_ids).issubset(ff_ids)]
    assert len(candidates) == 28 and len(selected) == 1
    store = load_admitted_store(RECEIPT / "measurements/profile.json", RECEIPT / "admission.json", usage_scope=USE)
    read(RECEIPT / "admission.json")
    read(RECEIPT / "measurements/profile.json")
    queries = demonstrate(RECEIPT / "measurements/profile.json", admission_path=RECEIPT / "admission.json")
    assert queries == read(RECEIPT / "strict_cost_queries.json")

    nodes = [{"node_id": nid, "role": "boundary_input" if nid == BOUNDARY else "compute",
              "original_op": graph.nodes[nid].op_type,
              "shape": graph.nodes[nid].extra["output_shape"],
              "dtype": graph.nodes[nid].extra["dtype"],
              "parameter_bytes_in_scope": 0 if nid == BOUNDARY else graph.nodes[nid].param_bytes,
              "matmul_shape": graph.nodes[nid].extra.get("shape") if graph.nodes[nid].op_type == "matmul" else None,
              "semantic_slice": graph.nodes[nid].extra.get("semantic_slice")}
             for nid in [BOUNDARY, *COMPUTE]]
    fused, native, singleton = [], [], []
    for record in store.records.values():
        identity = {field: record[field] for field in (
            "case_id", "role", "kernel_key_id", "measurement_contract_id", "record_id", "admission")}
        identity.update(median_us=record["statistics"]["median_us"],
                        hardware=record["kernel_key"]["hardware"],
                        admission_use=record.get("admission_use"), search_cost_admitted=False)
        if record["role"] == "native_chain":
            try:
                store.query(record["kernel_key"], record["measurement_contract_id"])
            except MeasurementContractError:
                identity["strict_query_rejected"] = True
            else:
                raise AssertionError("Native comparison unexpectedly admitted")
            native.append(identity)
            continue
        report_file = next(p for p in record["evidence"] if p.endswith("/result.json"))
        placed_file = next(p for p in record["evidence"] if p.endswith("/placed.json"))
        report = read(RECEIPT / "measurements" / report_file)
        read(RECEIPT / "measurements" / placed_file)
        placed = load_placed_op_graph(RECEIPT / "measurements" / placed_file)
        assert graph_fingerprint(graph) == placed.graph_hash
        keys, group, key, gaps = normalize_identity(placed, report)
        assert not gaps and key_id(key) == record["kernel_key_id"]
        assert list(group.node_ids) == list(selected[0].node_ids)
        store.query(key, record["measurement_contract_id"])
        identity.update(strict_query_matched=True, node_ids=list(group.node_ids),
                        full_ffn_composition_admitted=False)
        fused.append(identity)
        # Do not turn fused-expression metadata into a Native singleton layout.
        draft_keys = GroupKernelKeys(copy.deepcopy(graph), keys.hardware, runtime=keys.runtime)
        reference = make_reference_plan(graph, device=group.device)
        for single in reference.groups:
            if single.node_ids[0] not in COMPUTE:
                continue
            draft, missing = draft_keys.describe(single)
            assert missing
            singleton.append({"node_id": single.node_ids[0], "device": group.device,
                              "pattern_id": draft["pattern_id"], "draft_key_id": key_id(draft),
                              "identity_gaps": list(missing), "key": draft,
                              "identity_state": "raw_builder_without_executing_reference_runtime_observation",
                              "reference_operator": {"matmul": "torch.matmul (flatten input; resident weight)",
                                                     "silu": "torch.nn.functional.silu", "mul": "torch.Tensor.__mul__"}
                              [graph.nodes[single.node_ids[0]].op_type],
                              "accepted": False, "cost_source": "historical_diagnostic_or_analytic_only"})

    historical = []
    for device in (0, 1):
        path = ROOT / f"data/fusion_cost_validation_v1/inputs/gpu{device}_seq256/result.json"
        old = read(path)
        summary = old["summary"]
        historical.append({"device": device, "source": path.relative_to(ROOT).as_posix(),
                           "measurement_contract_id": old["measurement_contract_id"],
                           "admission": old["admission"], "rounds": len(summary["round_results"]),
                           "median_us": {arm: stats["median"] for arm, stats in
                                         summary["graph_batch_amortized_per_call_us"].items()},
                           "sum_single_minus_chain_paired_median_us": summary["additivity_residual_us"]["median"],
                           "sum_single_minus_chain_over_chain_paired_median": summary["additivity_residual_relative"]["median"],
                           "chain_minus_sum_single_paired_median_us": -summary["additivity_residual_us"]["median"],
                           "output_lifetime": "last final output only; 50 calls per graph",
                           "compatible_with_p2": False})
    raw = read(ROOT / "data/bw1000_2card/raw_measurements.json")
    legacy = read(ROOT / "data/fixed_profile_audit_v1/report.json")
    assert legacy["kernel_records"] == legacy["record_admissions"]["legacy_unbound"] == 36
    b1_root = ROOT / "data/ffn_placement_b1"
    b1_verification = read(b1_root / "verification.json")
    b1_payloads = {name: read(b1_root / name) for name in b1_verification["output_payload_ids"]}
    assert all(key_id(value) == b1_verification["output_payload_ids"][name] for name, value in b1_payloads.items())
    assert b1_verification["pass_checks"] and b1_payloads["identity.json"]["source_graph_hash"] == graph_fingerprint(graph)
    edge_rows = [{"src": e.src, "dst": e.dst, "bytes": e.bytes,
                  "source_device": None, "destination_device": None,
                  "tensor_shape": graph.nodes[e.src].extra["output_shape"], "tensor_bytes": e.bytes,
                  "transfer_required": None, "synchronization": "unresolved_until_B1_B2",
                  "kind": "ffn_internal" if e in edges else "external_egress",
                  "eliminated_when_fused": e.src.endswith("mlp.silu") and e.dst.endswith("mlp.gated_mul"),
                  "cross_device_cost_accepted": False} for e in edges + egress]
    potential_links = [{"src_device": a, "dst_device": b, "bytes": size, "measurement_required": None}
                      for size in sorted({e.bytes for e in edges}) for a, b in ((0, 1), (1, 0))]
    assert len(singleton) == 10 and len(fused) == 2 and len(native) == 2 and len(potential_links) == 4
    return {"format": "fusion-placement-search.search-subgraph-cost-audit", "version": "1.2",
            "execution_protocol": {"path": "cost_model/STAGE_B_PROTOCOL.md",
                                   "sha256": digest(ROOT / "cost_model/STAGE_B_PROTOCOL.md")},
            "scope": "reference-builder FFN layer 0; accepted coverage under current P2 decision",
            "graph_hash": graph_fingerprint(graph), "model_nodes": len(graph.nodes),
            "layers": 28, "ffn_shape_templates": len(signatures), "nodes": nodes,
            "edges": edge_rows, "fused_groups": fused, "native_comparisons": native,
            "singleton_cost_queries": singleton, "potential_link_query_classes": potential_links,
            "actual_transfer_measurement_cases": None,
            "ffn_b1_inventory": {"path": "data/ffn_placement_b1", "boundary": b1_payloads["identity.json"]["boundary_constraints"],
                                 "legal_labeled_plans": b1_verification["legal_labeled_plans"],
                                 "compute_structure_classes": b1_verification["compute_structure_classes"],
                                 "labeled_plans_to_score": b1_verification["labeled_plans_to_score"],
                                 "logical_communication_requirement_classes": b1_verification["t3_requirement_classes"],
                                 "b2_contract_review_complete": False},
            "chain_cost_queries": [{"device": device, "arm": arm, "scope": "full_ffn_fork_join",
                                    "canonical_key": None, "measurement_contract_id": None,
                                    "accepted": False, "evidence": "no matching FFN chain in current P2 bundle"}
                                   for device in (0, 1) for arm in ("native", "fused")],
            "historical_four_arm": historical,
            "historical_links": [{"src": link["src"], "dst": link["dst"],
                                  "points": len(link["points"]),
                                  "sizes_bytes": [p["size_bytes"] for p in link["points"]],
                                  "accepted_under_p2": False} for link in raw["links"]],
            "legacy_profile_audit_source": "data/fixed_profile_audit_v1/report.json",
            "legacy_kernel_records": legacy["kernel_records"],
            "legacy_record_admissions": legacy["record_admissions"],
            "evidence_coverage": {"accepted_paired_comparison_cases": 2, "reviewed_cases": 2,
                                  "use": USE, "standalone_search_cost_admission": False},
            "search_cost_coverage": {"admitted_cost_items": 0, "denominator": None,
                                     "reason": "singleton/link/whole-plan search use is not open"},
            "stage_b_entry": {"B0": "CPU identity workflow verified; actual BW1000 identity captured before B4 timing",
                              "B1": "complete for GPU0/GPU0 boundary: 48 labeled placements, 24 compute-only swap classes; no cost scoring",
                              "B2": "B1 union derived; transfer mechanism/layout/synchronization contract review pending",
                              "B3": "same-case/contract/recipe Fused-Native case matrix pending",
                              "B4": "BW1000 identity capture and paired admission pending B1-B3",
                              "B5.5": "controlled analytic-versus-Profile decision comparison pending B5",
                              "B6": "FFN-only Exact/Guided/Sequential/Joint comparison pending B5.5",
                              "C": "whole-graph mixed-coverage validation deferred",
                              "gpu_measurement_allowed": False},
            "stage_b_measurement_plan": [
                "B0: close singleton identity/layout provenance in the actual executing runtime.",
                "B1: enumerate legal fusion/placement actions, boundary constraints and jointly reachable device pairs.",
                "B2: union crossing edges over all legal plans; deduplicate by direction, Tensor and transfer contract.",
                "B3: define paired Fused/Native cases with equal case, contract and recipe; paths remain distinct.",
                "B4: after B0-B3, capture actual BW1000 identities before paired timing; admission is independent of speedup.",
                "B5: validate communication and sum(single) - chain under identical timing boundaries.",
                "B5.5: compare analytic versus admitted costs on identical graph/space/algorithm/parameters; cross-score both plans.",
                "B6: FFN-only Exact/Guided/Sequential/Joint under fixed admitted costs; retain Native actions.",
                "C: defer whole-graph validation with explicitly mixed cost sources."],
            "source_sha256": sources, "new_gpu_measurements": 0, "exact_runs": 0, "mcts_runs": 0,
            "ready_for_real_profile_search": False}


if __name__ == "__main__":
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "evidence_coverage": result["evidence_coverage"],
                      "search_cost_coverage": result["search_cost_coverage"],
                      "historical_four_arm": result["historical_four_arm"]}, indent=2))
