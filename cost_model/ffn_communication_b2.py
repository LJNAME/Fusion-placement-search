"""CPU B2 coverage and declared transfer contracts from immutable B1 tables."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cost_model.ffn_placement_b1 import FILES, SYNC, serialized, verify as verify_b1
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_profile import digest
from execution.ffn_transfer import IMPLEMENTATION, declared_carrier

ROOT = Path(__file__).resolve().parents[1]
B1 = ROOT / "data/ffn_placement_b1"
B2_FILES = ("identity.json", "communication_requests.json", "coverage.json")
SEMANTICS = {
    "implementation_id": IMPLEMENTATION, "operation": "preallocated_destination.copy_(source, non_blocking=True)",
    "mode": "graph_replay", "unit": "us/transfer", "allocation_included": False,
    "timing_boundary": "same destination-GPU Event pair around one Graph replay, divided by calls_per_round",
    "producer_boundary": "producer complete before isolated replay/Event interval; immutable sources throughout replay",
    "consumer_boundary": "destination readable after copy; consumer uses same destination stream or an explicit completion wait",
    "synchronization": "both endpoints synchronized outside capture/before replay; destination end Event synchronized after replay",
    "event_coverage": "B4 must prove target Event covers actual cross-device copy completion; never subtract events on different GPUs",
    "source_lifetime": "retained and unmodified until replay completion",
    "destination_lifetime": "independent preallocated destination per call; all outputs retained and checked",
    "copy_reuse": "one action per crossing graph edge; no cross-consumer cache or coalescing",
    "scope": "isolated transfer proxy with no other traffic; not whole-plan or overlap admission",
    "allowed_route": "peer-capable device-to-device; actual route must be observed in B4",
    "fallback": "no CPU staging, packing, eager timer or implicit Tensor.to allocation",
    "recipe_binding": "calls/rounds/warmup/order/statistic frozen in B3; actual environment/UUID and timing contract in B4",
}


def load_b1(root=B1):
    root = Path(root)
    bundle = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in FILES}
    saved = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if key_id(verify_b1(bundle)) != key_id(saved):
        raise ValueError("B1 verification or source identities changed")
    return bundle, {name: digest(root / name) for name in (*FILES, "verification.json")}


def derive(bundle):
    rows = bundle["T2_cross_device_edges.json"]["rows"]
    t3 = bundle["T3_communication_union.json"]["requirements"]
    by_edge = {(row["plan_id"], row["edge_id"]): row for row in rows}
    requests, assignments = [], []
    for item in t3:
        if (item["transfer_mechanism"] != "unbound_B2" or item["synchronization"] != SYNC
                or item["layout"] != "unobserved; declaration/runtime binding pending B3/B4"):
            raise ValueError("B1 has different transport/layout/dependency obligations; review instead of overwriting")
        shapes = {}
        for witness in item["witnesses"]:
            row = by_edge[(witness["plan_id"], witness["edge_id"])]
            shapes.setdefault(tuple(row["tensor_shape"]), []).append(witness)
        # Same bytes are insufficient to justify sharing differing shapes.
        for shape, witnesses in sorted(shapes.items()):
            carrier = declared_carrier(list(shape), item["dtype"])
            if carrier["tensor_bytes"] != item["tensor_bytes"]:
                raise ValueError("B1 Tensor bytes disagree with shape/dtype")
            payload = {"source_gpu": item["source_gpu"], "destination_gpu": item["destination_gpu"],
                       "source_carrier": carrier, "destination_carrier": copy.deepcopy(carrier),
                       "transfer_semantics": copy.deepcopy(SEMANTICS),
                       "implementation_source_sha256": digest(ROOT / "execution/ffn_transfer.py")}
            identifier = key_id(payload)
            requests.append({"request_id": identifier, **payload, "b1_requirement_id": item["requirement_id"],
                             "witnesses": witnesses, "edge_occurrences": len(witnesses),
                             "runtime_identity": {"source_uuid": None, "destination_uuid": None, "torch_build": None, "peer_access": None, "observed_route": None},
                             "measurement_contract_id": None, "b3_recipe_frozen": False, "b4_runtime_verified": False,
                             "measurement_ready": False, "admission": "not_requested"})
            assignments.extend({**w, "request_id": identifier} for w in witnesses)
    requests.sort(key=lambda r: (r["source_carrier"]["tensor_bytes"], r["source_gpu"], r["destination_gpu"], r["request_id"]))
    assignments.sort(key=lambda r: (r["plan_id"], r["edge_id"]))
    return requests, assignments


def build(b1_root=B1):
    bundle, hashes = load_b1(b1_root)
    requests, assignments = derive(bundle)
    raw = json.loads((ROOT / "data/bw1000_2card/raw_measurements.json").read_text(encoding="utf-8"))
    identity = {"format": "fusion-placement-search.ffn-communication-b2", "version": "1.0",
                "scope": "CPU declaration and exhaustive coverage review; no GPU path validation or cost admission",
                "b1_source_sha256": hashes,
                "source_sha256": {name: digest(ROOT / name) for name in (
                    "cost_model/ffn_communication_b2.py", "execution/ffn_transfer.py", "execution/reference.py",
                    "calibration/calibrate_bw1000.py", "cost_model/measured_profile.py", "data/bw1000_2card/raw_measurements.json")},
                "existing_paths": [
                    {"source": "execution/reference.py:255", "operation": "value.to(target)",
                     "allocation": "destination allocated by Tensor.to", "timer": "synchronized CPU wall, eager",
                     "compatible_with_selected_contract": False},
                    {"source": "calibration/calibrate_bw1000.py:582", "operation": "target.copy_(source, non_blocking=True)",
                     "allocation": "outside timing", "timer": "synchronized CPU wall eager loop",
                     "compatible_with_selected_contract": False}],
                "historical_link_tables": [{"source_gpu": link["src"], "destination_gpu": link["dst"],
                                            "sizes_bytes": [p["size_bytes"] for p in link["points"]],
                                            "new_contract_admission": False} for link in raw["links"]],
                "layout_policy": "constructor declarations are new carrier requirements, not inferred producer observations; mismatches reject",
                "current_executor_alignment": "new standalone copy binding; existing eager reference loop is not silently relabelled Graph Replay",
                "b2_declaration_review_complete": True, "b3_complete": False, "gpu_measurement_allowed": False,
                "new_gpu_measurements": 0, "accepted_link_profiles": 0, "cost_scoring_calls": 0}
    coverage = {"b1_plans": bundle["T1_legal_placements.json"]["summary"]["legal_labeled_plans"],
                "b1_original_edge_rows": len(bundle["T2_cross_device_edges.json"]["rows"]),
                "cross_device_edge_occurrences": len(assignments), "assignments": assignments,
                "zero_transfer_plans": sorted(p["plan_id"] for p in bundle["T1_legal_placements.json"]["plans"] if
                                               p["plan_id"] not in {r["plan_id"] for r in assignments}),
                "billing_policy": SEMANTICS["copy_reuse"], "hidden_layout_or_transport_costs_allowed": False}
    return dict(zip(B2_FILES, (identity, {"requests": requests, "status": "declared_not_measured"}, coverage)))


def verify(bundle, b1_root=B1):
    b1, hashes = load_b1(b1_root)
    identity, request_file, coverage = (bundle[name] for name in B2_FILES)
    if identity["b1_source_sha256"] != hashes:
        raise ValueError("B2 points to different B1 evidence")
    expected_requests, expected_assignments = derive(b1)
    if request_file["requests"] != expected_requests or coverage["assignments"] != expected_assignments:
        raise ValueError("B2 requests/coverage differ from the full B1 crossing-edge union")
    pairs = [(r["plan_id"], r["edge_id"]) for r in coverage["assignments"]]
    crossing = {(r["plan_id"], r["edge_id"]) for r in b1["T2_cross_device_edges.json"]["rows"] if r["cross_device"]}
    if len(pairs) != len(set(pairs)) or set(pairs) != crossing:
        raise ValueError("duplicate, missing or extra crossing-edge assignments")
    all_plans = {p["plan_id"] for p in b1["T1_legal_placements.json"]["plans"]}
    crossing_plans = {plan for plan, _ in crossing}
    if coverage["zero_transfer_plans"] != sorted(all_plans - crossing_plans):
        raise ValueError("zero-transfer plan coverage differs from B1")
    if len({r["request_id"] for r in expected_requests}) != len(expected_requests):
        raise ValueError("duplicate B2 request identities")
    repeat = build(b1_root)
    if any(serialized(bundle[name]) != serialized(repeat[name]) for name in B2_FILES):
        raise ValueError("B2 replay differs")
    return {"pass_checks": True, "b2_declaration_review_complete": True,
            "request_count": len(expected_requests), "covered_cross_device_edge_occurrences": len(pairs),
            "b1_total_plans": len(all_plans), "plans_requiring_communication": len(crossing_plans),
            "zero_transfer_plans": len(all_plans - crossing_plans),
            "all_crossing_edges_covered_once": True, "B1_sources_unchanged": True, "repeat_byte_equivalent": True,
            "output_payload_ids": {name: key_id(value) for name, value in bundle.items()},
            "gpu_capture_or_peer_route_validated": False, "measurement_ready_requests": 0,
            "new_gpu_measurements": 0, "accepted_link_profiles": 0, "cost_scoring_calls": 0,
            "next_stage": "B3 paired matrix; B4 target identity/route/Graph/event preflight before timing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, default=B1)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ffn_communication_b2")
    args = parser.parse_args()
    bundle = build(args.b1)
    verification = verify(bundle, args.b1)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in {**bundle, "verification.json": verification}.items():
        with (args.output / name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized(value))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
