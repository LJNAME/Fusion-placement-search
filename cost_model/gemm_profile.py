"""Reviewed per-GPU GEMM Profiles placed in front of a caller's analytic profile.

Entries are keyed by the full case identity (GPU device + software stack,
contract, recipe, shapes). A node is served only if the caller binds it to an
identity for that device and the node's m/n/k and dtype match; otherwise the
base (analytic) profile answers and the miss reason is recorded.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cost_model.fusion_kernel_profile import key_id
from cost_model.hardware_profile import GroupCost


class ReviewedGemmStore:
    def __init__(self, raw_directory, review_file):
        raw = Path(raw_directory)
        read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
        review = read(review_file)
        for name, expected in review["raw_sha256"].items():
            if Path(name).name != name or hashlib.sha256((raw / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Reviewed GEMM receipt changed: " + name)
        if read(raw / "identity.json") != review["identity"]:
            raise ValueError("Reviewed GEMM identities differ")
        self.records = {}
        for case_id, quality in review["cases"].items():
            if quality["status"] != "quality_passed":
                continue
            ident = review["identity"]["cases"][case_id]
            if key_id(ident["case_identity"]) != ident["case_identity_id"]:
                raise ValueError("GEMM identity hash differs")
            self.records[ident["case_identity_id"]] = {"case_id": case_id,
                "identity": copy.deepcopy(ident["case_identity"]), "gpu_key_id": ident["gpu_key_id"],
                "contract_id": ident["contract_id"], "duration_s": quality["statistics"]["median_us"] / 1e6,
                "statistics": copy.deepcopy(quality["statistics"]), "provenance": key_id(review["raw_sha256"]),
                "status": "reviewed_exact_gemm"}
        self.failed = sorted(c for c, q in review["cases"].items() if q["status"] != "quality_passed")

    def lookup(self, identity_id):
        record = self.records.get(identity_id)
        return copy.deepcopy(record) if record is not None else None


class MeasuredGemmFirstProfile:
    """bindings[(node_id, device)] = case_identity_id. Other groups go to base."""
    def __init__(self, base, store, bindings):
        self.base, self.store, self.bindings = base, store, dict(bindings)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def estimate_group(self, graph, group):
        if len(group.node_ids) != 1 or graph.nodes[group.node_ids[0]].op_type != "matmul":
            return self.base.estimate_group(graph, group)
        node = graph.nodes[group.node_ids[0]]
        identity_id = self.bindings.get((group.node_ids[0], group.device))
        record = self.store.lookup(identity_id) if identity_id else None
        reason = "unbound" if identity_id is None else "not_reviewed_or_quality_failed"
        if record is not None:
            case = record["identity"]["case"]
            shape = node.extra.get("shape") or {}
            if (case["device"] == group.device and case["node"] == group.node_ids[0]
                    and node.extra.get("dtype") == "bf16"
                    and (shape.get("m"), shape.get("n"), shape.get("k")) == (case["m"], case["n"], case["k"])):
                d = record["duration_s"]
                return GroupCost(d, d, 0., 0., .95, "measured_gemm_projection",
                                 ("measurement_context:isolated_gemm_shared_weight",))
            reason = "graph_or_shape_mismatch"
        cost = self.base.estimate_group(graph, group)
        return replace(cost, ood_reasons=tuple(sorted(set(cost.ood_reasons) | {"gemm_profile:" + reason})))
