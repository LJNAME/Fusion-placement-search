"""Group-level Kernel identities and explicit latency lookup for SROP experiments.

This namespace does not reinterpret the older shape-only BW1000 tables as
complete Kernel-Key measurements. Bindings and runtime metadata come from the
caller; an unbound implementation or unspecified layout stays visibly missing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from simulator import enumerate_fusion_candidates
from simulator.schema import group_boundary_io

from .hardware_profile import GroupCost
from .structure import GroupStructure


KEY_VERSION = "fusion-group-kernel.v1"
IDENTITY_ATTRIBUTES = {"layer", "local_name", "module_path", "implementation_group"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def key_id(key):
    return "sha256:" + hashlib.sha256(canonical(key).encode("utf-8")).hexdigest()


class GroupKernelKeys:
    """Describe single-output kernels on unchanged graph/hardware snapshots.

    Bindings are indexed by pattern_id and contain id/version, plus any compile
    and runtime flags. Node attributes other than graph-location annotations are
    retained conservatively. Input order and repeated-input aliases are retained;
    graph node names and consumer placement never become measurement identities.
    """

    def __init__(self, graph, hardware, bindings=None, runtime=None):
        self.graph = graph
        self.hardware = hardware
        self.bindings = copy.deepcopy(bindings or {})
        self.runtime = copy.deepcopy(runtime or {})
        self.incoming = {node: [] for node in graph.nodes}
        self.outgoing = {node: [] for node in graph.nodes}
        for edge in graph.edges:
            self.incoming[edge.dst].append(edge)
            self.outgoing[edge.src].append(edge)

    def describe(self, group):
        pattern_id = (group.pattern_id if hasattr(group, "pattern_id")
                      else str(group.extra["pattern_id"]))
        nodes = tuple(group.node_ids)
        if len(nodes) == 1 and self.graph.nodes[nodes[0]].op_type in {"input", "embedding"}:
            return None, ()
        if not 0 <= group.device < self.hardware.n_devs:
            raise ValueError("Kernel device is outside the hardware snapshot")
        binding = self.bindings.get(pattern_id, {})
        gaps = set()
        for field in ("id", "version"):
            if not binding.get(field):
                gaps.add("implementation." + field)
        for field in ("framework", "framework_version", "accelerator", "accelerator_version"):
            if not self.runtime.get(field):
                gaps.add("runtime." + field)

        def tensor(node_id):
            node = self.graph.nodes[node_id]
            result = {"shape": node.extra.get("output_shape"),
                      "dtype": node.extra.get("dtype"),
                      "layout": node.extra.get("layout"),
                      "strides": node.extra.get("strides")}
            for field in ("shape", "dtype"):
                if result[field] is None:
                    gaps.add("tensor." + field)
            if result["layout"] is None and result["strides"] is None:
                gaps.add("tensor.layout_or_strides")
            return result

        positions = {node: index for index, node in enumerate(nodes)}
        external = {}
        inputs = []
        operators = []
        for node_id in nodes:
            node = self.graph.nodes[node_id]
            operands = []
            for edge in self.incoming[node_id]:
                if edge.src in positions:
                    operand = {"internal": positions[edge.src]}
                else:
                    alias = (edge.src, canonical(edge.extra.get("src_port")))
                    if alias not in external:
                        external[alias] = len(inputs)
                        inputs.append(tensor(edge.src))
                    operand = {"external": external[alias]}
                operand.update(bytes=edge.bytes,
                               attributes={k: v for k, v in edge.extra.items() if k != "tensor_id"})
                operands.append(operand)
            operators.append({
                "operator": node.op_type, "output": tensor(node_id), "inputs": operands,
                "parameter_bytes": node.param_bytes,
                "attributes": {k: v for k, v in node.extra.items() if k not in IDENTITY_ATTRIBUTES},
            })
        device = self.hardware.devices[group.device]
        source = device.extra.get("source", self.hardware.extra.get("source"))
        if not source:
            gaps.add("hardware.source")
        key = {
            "key_version": KEY_VERSION, "pattern_id": pattern_id,
            "hardware": {"device_index": group.device, "name": device.name,
                         "class": device.extra.get("class"),
                         "physical_index": device.extra.get("physical_index", group.device),
                         "source": source},
            "runtime": self.runtime,
            "implementation": binding or {"id": None, "version": None},
            "operators": operators, "external_inputs": inputs,
            "outputs": [positions[node] for node in nodes if not self.outgoing[node]
                        or any(edge.dst not in positions for edge in self.outgoing[node])],
        }
        # Normalize tuples/lists so file round trips retain identical identities.
        return json.loads(canonical(key)), tuple(sorted(gaps))


@dataclass(frozen=True)
class KernelLatency:
    duration_s: float
    kind: str
    provenance: str

    def __post_init__(self):
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("Kernel latency must be finite and positive")
        if self.kind not in {"measured", "synthetic"} or not self.provenance:
            raise ValueError("Kernel latency needs a measured/synthetic kind and provenance")


class KeyedKernelProfile:
    """Use exact group identities, retaining a fixed link/memory base profile.

    Missing entries raise by default. Synthetic entries remain OOD, and cannot
    satisfy measured-coverage acceptance. This is a cost adapter, not an executor.
    """

    def __init__(self, keys, latencies: Mapping[str, KernelLatency], base, *, strict=True):
        self.keys = keys
        self.latencies = dict(latencies)
        self.base = base
        self.strict = strict

    @property
    def n_devs(self):
        return self.base.n_devs

    @property
    def overlap_send(self):
        return self.base.overlap_send

    @property
    def interference(self):
        return self.base.interference

    def max_memory(self, device):
        return self.base.max_memory(device)

    def estimate_transfer(self, source, target, nbytes):
        return self.base.estimate_transfer(source, target, nbytes)

    def link_key(self, source, target):
        return self.base.link_key(source, target)

    def estimate_group(self, graph, group):
        if graph is not self.keys.graph:
            raise ValueError("Kernel profile is bound to a different graph snapshot")
        key, gaps = self.keys.describe(group)
        if key is None:
            return GroupCost(0., 0., 0., 0., 1., "logical_input")
        identifier = key_id(key)
        entry = self.latencies.get(identifier)
        if entry is None:
            if self.strict:
                raise KeyError("missing group Kernel Key: " + identifier)
            return self.base.estimate_group(graph, group)
        if entry.kind == "measured" and gaps:
            raise ValueError("incomplete measured Kernel identity: " + ",".join(gaps))
        reasons = () if entry.kind == "measured" else ("synthetic_kernel:" + identifier,)
        return GroupCost(entry.duration_s, entry.duration_s, 0., 0.,
                         .95 if entry.kind == "measured" else 0.,
                         entry.kind + "_group_kernel", reasons)


def candidate_structure(graph, candidate, device):
    incoming, outgoing = group_boundary_io(graph, list(candidate.node_ids))
    return GroupStructure(
        0, "inventory-group", candidate.node_ids, device, candidate.pattern_id,
        candidate.backend, candidate.implementation_status,
        sum(graph.nodes[node].flops for node in candidate.node_ids),
        sum(graph.nodes[node].param_bytes for node in candidate.node_ids),
        incoming, outgoing, 0.,
    )


def inventory(keys, registry, *, latencies=None, legacy_profile=None, require_executable=False):
    """Enumerate local legal candidates x devices, not complete placed plans.

    Legacy lookup sources are diagnostic only: a shape-table hit does not prove
    implementation/layout identity or populate this namespace's measured entries.
    """
    latencies = latencies or {}
    rows = {}
    logical = 0
    candidates = enumerate_fusion_candidates(keys.graph, registry,
                                              require_executable=require_executable)
    for candidate in candidates:
        for device in range(keys.hardware.n_devs):
            group = candidate_structure(keys.graph, candidate, device)
            key, gaps = keys.describe(group)
            if key is None:
                logical += 1
                continue
            identifier = key_id(key)
            entry = latencies.get(identifier)
            if identifier not in rows:
                rows[identifier] = {
                    "kernel_id": identifier, "kernel_key": key,
                    "identity_complete": not gaps, "identity_gaps": list(gaps),
                    "pattern_executable": candidate.executable,
                    "latency_kind": entry.kind if entry else "missing",
                    "duration_s": entry.duration_s if entry else None,
                    "provenance": entry.provenance if entry else None,
                    "measured_covered": bool(entry and entry.kind == "measured" and not gaps),
                    "usages": [], "legacy_sources": [],
                }
            row = rows[identifier]
            row["usages"].append({"node_ids": list(candidate.node_ids), "device": device})
            if legacy_profile is not None:
                cost = legacy_profile.estimate_group(keys.graph, group)
                row["legacy_sources"].append({"source": cost.source,
                                              "ood_reasons": list(cost.ood_reasons)})
    values = [rows[key] for key in sorted(rows)]
    return {
        "scope": "local_legal_candidates_times_physical_devices",
        "require_executable": require_executable,
        "candidate_count": len(candidates), "kernel_key_count": len(values),
        "kernel_usages": sum(len(row["usages"]) for row in values),
        "logical_usages": logical,
        "multi_node_candidates": sum(len(row.node_ids) > 1 for row in candidates),
        "measured_key_count": sum(row["measured_covered"] for row in values),
        "missing_or_nonmeasured_key_count": sum(not row["measured_covered"] for row in values),
        "complete_measured_kernel_profile": bool(values) and all(row["measured_covered"] for row in values),
        "latency_kinds": dict(Counter(row["latency_kind"] for row in values)),
        "legacy_lookup_sources_by_usage": dict(Counter(
            item["source"] for row in values for item in row["legacy_sources"])),
        "entries": values,
    }
