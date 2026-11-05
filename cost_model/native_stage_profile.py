"""Use reviewed isolated-stage measurements before a caller's analytic profile.

Exact stage identities are supplied by the caller, never inferred from Shape
alone. The existing collector/reviewer and raw measurement files are unchanged.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics

from cost_model.fusion_kernel_profile import key_id
from cost_model.hardware_profile import GroupCost


class ReviewedStageStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
        # Consume the completed CPU review, not model feasibility checks or
        # plan() again. Changing an analytic model must not invalidate timings.
        result = read(self.directory.parent / "review/evaluation.json")
        for name, expected in result["raw_sha256"].items():
            if Path(name).name != name or hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Reviewed stage receipt changed: " + name)
        identities = read(self.directory / "identity.json")
        if identities != result["identity"]:
            raise ValueError("Reviewed stage identities differ")
        rows = read(self.directory / "result.json")["records"]
        self.records = {}
        provenance = key_id(result["raw_sha256"])
        for stage, quality in result["stages"].items():
            if quality["status"] != "quality_passed":
                continue
            identity = result["identity"]["stages"][stage]
            values = [r["timing"]["elapsed_ms_per_50_calls"] * 1000 / 50 for r in rows if r["stage"] == stage]
            if len(values) != 7 or statistics.median(values) != quality["statistics"]["median_us"]:
                raise ValueError("Reviewed stage latency differs from raw attempts")
            if key_id(identity["stage_identity"]) != identity["stage_identity_id"]:
                raise ValueError("Reviewed stage identity hash differs")
            self.records[identity["stage_identity_id"]] = {
                "stage": stage, "identity": copy.deepcopy(identity["stage_identity"]),
                "contract_id": identity["contract_id"], "duration_s": quality["statistics"]["median_us"] / 1e6,
                "statistics": copy.deepcopy(quality["statistics"]),
                "provenance": provenance, "status": "reviewed_exact_isolated_stage",
            }

    def lookup(self, identity):
        record = self.records.get(key_id(identity))
        return copy.deepcopy(record) if record is not None else None


def _tensor_matches(node, observed):
    extra = node.extra
    return (list(extra.get("output_shape") or ()) == observed["shape"]
            and extra.get("dtype") == observed["dtype"]
            and extra.get("layout") == observed["layout"]
            and list(extra.get("strides") or ()) == observed["strides"])


class NativeMeasuredFirstProfile:
    """Optional HardwareCostProfile adapter; other costs delegate to base.

    bindings[node_id] = {"identity": actual_stage_identity,
                         "operands": {"gate": producer_id, ...}}
    Graph nodes must explicitly declare the observed aten-out implementation.
    This preserves isolated-stage timing provenance in all aggregate estimates.
    """
    def __init__(self, graph, base, store, bindings):
        self.graph, self.base, self.store = graph, base, store
        self.bindings = copy.deepcopy(bindings)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def estimate_group(self, graph, group):
        if graph is not self.graph:
            raise ValueError("Stage profile is bound to a different graph snapshot")
        if len(group.node_ids) != 1:
            return self.base.estimate_group(graph, group)
        node_id = group.node_ids[0]
        node = graph.nodes[node_id]
        binding = self.bindings.get(node_id)
        record = self.store.lookup(binding["identity"]) if binding is not None else None
        reason = "unbound"
        if record is not None:
            identity = record["identity"]
            implementation = identity["implementation"]
            operands = binding["operands"]
            declared = {"id": implementation["recipe"]["id"], "version": implementation["source_sha256"]}
            valid = (node.op_type == record["stage"] and identity["device"]["ordinal"] == group.device
                     and node.extra.get("implementation") == declared
                     and _tensor_matches(node, implementation["tensors"]["output"])
                     and set(operands) == set(implementation["recipe"]["operands"])
                     and len(set(operands.values())) == len(operands)
                     and Counter(e.src for e in graph.edges if e.dst == node_id) == Counter(operands.values()))
            for role, producer in operands.items():
                valid = (valid and producer in graph.nodes
                         and _tensor_matches(graph.nodes[producer], implementation["tensors"][role]))
            if record["stage"] == "mul":
                valid = valid and graph.nodes[operands["left"]].op_type == "silu"
            if valid:
                duration = record["duration_s"]
                return GroupCost(duration, duration, 0., 0., .95, "measured_native_isolated_stage",
                                 ("measurement_context:isolated_stage",))
            reason = "graph_or_implementation_mismatch"
        elif binding is not None:
            reason = "identity_or_contract_miss"
        cost = self.base.estimate_group(graph, group)
        if node.op_type not in ("silu", "mul"):
            return cost
        return replace(cost, ood_reasons=tuple(sorted(set(cost.ood_reasons) | {"stage_profile:" + reason})))
