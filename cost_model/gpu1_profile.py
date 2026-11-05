"""Reviewed GPU1 SiLU/Mul/Fused/Down records in front of the existing profile chain.

A group is served only if (nodes, device) is bound to a reviewed, quality-
passed identity and the graph matches it: device, op types, output layout,
the implementation each stage node declares, Mul's left operand from SiLU,
GEMM m/n/k. Everything else delegates to the base chain unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cost_model.fusion_kernel_profile import key_id
from cost_model.hardware_profile import GroupCost

SOURCES = {"stage": "measured_gpu1_isolated_stage", "fused": "measured_gpu1_lightop_complete_call",
           "gemm": "measured_gemm_projection"}


class ReviewedGpu1Store:
    def __init__(self, raw_directory, review_file, *, excluded_cases=()):
        raw = Path(raw_directory)
        read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
        review = read(review_file)
        for name, expected in review["raw_sha256"].items():
            if Path(name).name != name or hashlib.sha256((raw / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Reviewed GPU1 receipt changed: " + name)
        if read(raw / "identity.json") != review["identity"]:
            raise ValueError("Reviewed GPU1 identities differ")
        self.records, self.bindings = {}, {}
        for case_id, quality in review["cases"].items():
            ident = review["identity"]["cases"][case_id]
            case = ident["case_identity"]["case"]
            if quality["status"] != "quality_passed" or case_id in excluded_cases:
                continue
            if key_id(ident["case_identity"]) != ident["case_identity_id"]:
                raise ValueError("GPU1 identity hash differs")
            self.records[ident["case_identity_id"]] = {"case_id": case_id, "case": copy.deepcopy(case),
                "identity": copy.deepcopy(ident["case_identity"]), "duration_s": quality["statistics"]["median_us"] / 1e6,
                "statistics": copy.deepcopy(quality["statistics"]), "provenance": key_id(review["raw_sha256"])}
            self.bindings[tuple(case["nodes"]), case["device"]] = ident["case_identity_id"]
        self.failed = sorted(c for c, q in review["cases"].items() if q["status"] != "quality_passed")
        self.excluded = sorted(excluded_cases)


def _layout_matches(node, observed):
    extra = node.extra
    return (list(extra.get("output_shape") or ()) == observed["shape"] and extra.get("dtype") == observed["dtype"]
            and list(extra.get("strides") or ()) == observed["strides"])


class MeasuredGpu1Profile:
    def __init__(self, base, store):
        self.base, self.store = base, store

    def __getattr__(self, name):
        return getattr(self.base, name)

    def estimate_group(self, graph, group):
        identity_id = self.store.bindings.get((tuple(group.node_ids), group.device))
        if identity_id is None:
            return self.base.estimate_group(graph, group)
        record = self.store.records[identity_id]
        case, nodes = record["case"], [graph.nodes[n] for n in group.node_ids]
        valid = case["device"] == group.device
        if case["kind"] == "stage":
            obs = record["identity"]["implementation"]
            node = nodes[0]
            declared = {"id": obs["recipe"]["id"], "version": obs["source_sha256"]}
            valid = (valid and node.op_type == case["stage"] and node.extra.get("implementation") == declared
                     and _layout_matches(node, obs["tensors"]["output"]))
            if case["stage"] == "mul":
                producers = [e.src for e in graph.edges if e.dst == group.node_ids[0]]
                valid = valid and any(graph.nodes[p].op_type == "silu" for p in producers)
        elif case["kind"] == "fused":
            valid = valid and [n.op_type for n in nodes] == ["silu", "mul"]
        else:
            shape = nodes[0].extra.get("shape") or {}
            valid = (valid and nodes[0].op_type == "matmul" and nodes[0].extra.get("dtype") == "bf16"
                     and (shape.get("m"), shape.get("n"), shape.get("k")) == (case["m"], case["n"], case["k"]))
        if valid:
            d = record["duration_s"]
            return GroupCost(d, d, 0., 0., .95, SOURCES[case["kind"]], ("measurement_context:isolated_case",))
        cost = self.base.estimate_group(graph, group)
        return replace(cost, ood_reasons=tuple(sorted(set(cost.ood_reasons) | {"gpu1_profile:graph_mismatch"})))
