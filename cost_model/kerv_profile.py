"""Exact KERV group identities, contract-scoped records and auditable costs.

Reuses fusion-group-kernel.v1 and the shared measurement admission gate. A
native SiLU/Mul chain is a comparison record, never a per-operator label.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from .fusion_kernel_profile import GroupKernelKeys, KernelLatency, KeyedKernelProfile, key_id
from .measurement_labels import MeasurementContractError, MeasurementLabel, MeasurementLabelLookup, measurement_contract_id
from .hardware_profile import AnalyticHardwareProfile


FORMAT = "fusion-placement-search.canonical-kerv-profile"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_identity(placed, report, role="fused_kernel"):
    """Copy snapshots and bind observed metadata; never mutate the source graph.

    The internal SiLU has no materialized tensor in the fused implementation.
    This is encoded explicitly instead of inventing observed strides for it.
    """
    graph, hardware = copy.deepcopy(placed.graph), copy.deepcopy(placed.hardware)
    group = copy.deepcopy(placed.plan.groups[report["group_index"]])
    if list(group.node_ids) != report["node_ids"] or group.device != report["logical_device"]:
        raise ValueError("report does not describe the selected placed group")
    if group.extra["pattern_id"] != "silu_mul":
        raise ValueError("only the selected silu_mul group is supported")
    shape, dtype = report["input"]["shape"], report["input"]["dtype"]
    expected_strides, stride = [], 1
    for size in reversed(shape):
        if type(size) is not int or size <= 0:
            raise ValueError("invalid observed tensor shape")
        expected_strides.insert(0, stride)
        stride *= size
    if (report["input"].get("layout") != "contiguous"
            or any(report["input"][field] != expected_strides for field in ("gate_strides", "up_strides"))):
        raise ValueError("observed layout is not the supported contiguous layout")
    for name in (report["gate_node"], report["up_node"], *group.node_ids):
        node = graph.nodes[name]
        if list(node.extra["output_shape"]) != shape or node.extra["dtype"] != dtype:
            raise ValueError("observed tensors disagree with graph metadata")
        node.extra["layout"] = "contiguous"
        node.extra["strides"] = list(report["input"]["gate_strides"])
    graph.nodes[report["up_node"]].extra["strides"] = list(report["input"]["up_strides"])
    if role == "fused_kernel":
        internal = graph.nodes[group.node_ids[0]]
        internal.extra["layout"] = {"materialization": "internal_fused_expression"}
        internal.extra["strides"] = None
    elif role != "native_chain":
        raise ValueError("unsupported record role")
    observed = report["observed_device"]
    environment = report["measurement_contract"]["environment"]
    provenance = report.get("loaded_source", {})
    gaps = []
    architecture = observed.get("architecture")
    driver = environment.get("driver_version")
    if not architecture:
        gaps.append("device.architecture")
    if not observed.get("uuid") or observed["uuid"] == "unavailable":
        gaps.append("device.uuid")
    if not driver:
        gaps.append("runtime.driver_version")
    if not provenance.get("module_sha256") or not provenance.get("git_revision"):
        gaps.append("implementation.loaded_source")
    if any(name not in report["measurement_contract"] for name in
           ("calls_per_round", "rounds", "warmup", "statistic", "synchronization")):
        gaps.append("measurement_contract.recipe_incomplete")
    dev = hardware.devices[group.device]
    dev.name = observed["name"]
    dev.max_mem = observed["total_memory"]
    dev.extra["physical_index"] = int(report["runtime_device"].split(":")[-1])
    dev.extra["source"] = {"kind": "runtime_observation", "uuid": observed["uuid"],
                           "architecture": architecture}
    runtime = {"framework": "torch", "framework_version": environment["torch"],
               "accelerator": "HIP" if environment.get("hip") else "CUDA",
               "accelerator_version": environment.get("hip") or environment.get("cuda"),
               "triton_version": environment["triton"], "driver_version": driver,
               "device_architecture": architecture, "device_uuid": observed["uuid"]}
    binding = copy.deepcopy(report["binding"])
    # Absolute installation paths are provenance, not Kernel identity.
    binding["loaded_source"] = {k: provenance.get(k) for k in ("module_sha256", "git_revision")}
    keys = GroupKernelKeys(graph, hardware, {"silu_mul": binding}, runtime)
    key, original_gaps = keys.describe(group)
    return keys, group, key, sorted(set(gaps) | set(original_gaps))


def make_record(placed, report, role, evidence, *, case_id):
    keys, group, key, gaps = normalize_identity(placed, report, role)
    contract = report["measurement_contract"]
    if measurement_contract_id(contract) != report["measurement_contract_id"]:
        raise ValueError("measurement contract hash mismatch")
    rows = report["rows"]
    if not rows or [row["round"] for row in rows] != list(range(len(rows))):
        raise ValueError("missing or unordered sampling rounds")
    values = []
    for row in rows:
        total, calls = row["graph_event_total_s"], row["graph_calls"]
        if type(calls) is not int or calls <= 0 or not math.isfinite(total) or total <= 0:
            raise ValueError("invalid timing sample")
        if not math.isclose(row["amortized_per_call_s"], total / calls, rel_tol=1e-12):
            raise ValueError("amortization mismatch")
        values.append(total / calls * 1e6)
    if "rounds" in contract and contract["rounds"] != len(rows):
        raise ValueError("round count disagrees with measurement contract")
    if "calls_per_round" in contract and any(row["graph_calls"] != contract["calls_per_round"] for row in rows):
        raise ValueError("call count disagrees with measurement contract")
    median = statistics.median(values)
    if not math.isclose(report["graph_batch_amortized_per_call_s"] * 1e6, median, rel_tol=1e-12):
        raise ValueError("median mismatch")
    payload = {"kernel_key": key, "kernel_key_id": key_id(key), "identity_gaps": gaps,
               "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract),
               "statistics": {"median_us": median, "min_us": min(values), "max_us": max(values),
                   "mad_rel": statistics.median(abs(v-median) for v in values) / median,
                   "n_rounds": len(rows), "n_calls": sum(row["graph_calls"] for row in rows)},
               "samples": rows, "case_id": case_id, "role": role, "evidence": evidence,
               "admission": "conditional_dev", "source_graph_hash": placed.graph_hash,
               "source_candidate_id": placed.candidate_id,
               "loaded_source": report.get("loaded_source", {}),
               "correctness": report["correctness"], "execution": report["execution"]}
    return {**payload, "record_id": key_id(payload)}, keys, group


class KervProfileStore:
    def __init__(self, records, *, evidence_root=None, approved_contract_ids=()):
        self.records = {}
        self.evidence_verified = evidence_root is not None
        self.approved_contract_ids = frozenset(approved_contract_ids)
        for record in records:
            payload = {k: v for k, v in record.items() if k != "record_id"}
            if record["record_id"] != key_id(payload) or record["kernel_key_id"] != key_id(record["kernel_key"]):
                raise ValueError("profile record identity mismatch")
            if record["measurement_contract_id"] != measurement_contract_id(record["measurement_contract"]):
                raise ValueError("profile contract identity mismatch")
            if record["admission"] not in {"conditional_dev", "accepted"}:
                raise ValueError("legacy labels are not KERV profile records")
            if evidence_root is not None:
                root = Path(evidence_root).resolve()
                for rel, expected in record["evidence"].items():
                    target = (root / rel).resolve()
                    if not target.is_relative_to(root) or digest(target) != expected:
                        raise ValueError("profile evidence missing or changed: " + rel)
                from simulator.placement_artifacts import load_placed_op_graph
                reports = [p for p in record["evidence"] if p.endswith("/result.json")]
                placed_files = [p for p in record["evidence"] if p.endswith("/placed.json")]
                if len(reports) != 1 or len(placed_files) != 1:
                    raise ValueError("each record requires one report and one placed artifact")
                report = json.loads((root / reports[0]).read_text(encoding="utf-8"))
                placed = load_placed_op_graph(root / placed_files[0])
                expected, _, _ = make_record(placed, report, record["role"], record["evidence"], case_id=record["case_id"])
                for field in ("kernel_key", "kernel_key_id", "identity_gaps", "measurement_contract", "statistics",
                              "samples", "correctness", "execution", "loaded_source"):
                    if record[field] != expected[field]:
                        raise ValueError("record differs from measurement evidence: " + field)
            index = (record["kernel_key_id"], record["measurement_contract_id"])
            if index in self.records:
                raise ValueError("duplicate key/contract; select one explicit measurement record")
            self.records[index] = copy.deepcopy(record)

    @classmethod
    def load(cls, path, *, approved_contract_ids=()):
        path = Path(path)
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle["format"] != FORMAT or bundle["version"] != "1.0":
            raise ValueError("unsupported KERV profile format")
        return cls(bundle["records"], evidence_root=path.parent,
                   approved_contract_ids=approved_contract_ids)

    def query(self, kernel_key, contract_id, *, allow_conditional_dev=False):
        identifier = key_id(kernel_key)
        try:
            record = self.records[(identifier, contract_id)]
        except KeyError:
            raise KeyError("no exact KERV key/contract record: " + identifier) from None
        if record["admission"] == "accepted":
            if not self.evidence_verified or contract_id not in self.approved_contract_ids or record["identity_gaps"]:
                raise MeasurementContractError("accepted KERV Profile requires reviewed contract and complete identity")
            counterpart = [r for r in self.records.values() if r["case_id"] == record["case_id"]
                           and r["measurement_contract_id"] == contract_id and r["role"] != record["role"]]
            if len(counterpart) != 1 or counterpart[0]["identity_gaps"]:
                raise MeasurementContractError("accepted KERV Profile requires same-contract Native/Fused evidence")
        label = MeasurementLabel(record["kernel_key_id"], contract_id, record["statistics"], record["admission"])
        MeasurementLabelLookup({identifier: label}).lookup(
            identifier, contract_id, allow_unbound=allow_conditional_dev)
        return copy.deepcopy(record)


class KervMeasuredProfile:
    """A measured selected-group overlay with explicit singleton fallback.

    Missing fusion records always fail. Native-chain totals cannot be assigned
    to the SiLU or Mul singleton. Graph amortized durations are consumed once.
    """
    def __init__(self, keys, store, contract_by_device, *, allow_conditional_dev=False):
        self.keys = keys
        self.store = store
        self.contract_by_device = dict(contract_by_device)
        self.allow_conditional_dev = allow_conditional_dev
        self.hardware = keys.hardware
        self.base = AnalyticHardwareProfile(self.hardware)
        self.usage = []

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

    def estimate_transfer(self, src, dst, nbytes):
        if getattr(self.store, "admission_use", None) and nbytes > 0:
            raise MeasurementContractError("selected-group admission does not admit communication costs")
        return self.base.estimate_transfer(src, dst, nbytes)

    def link_key(self, src, dst):
        return self.base.link_key(src, dst)

    def estimate_group(self, graph, group):
        if graph is not self.keys.graph:
            raise ValueError("KERV Profile is bound to a different graph snapshot")
        if len(group.node_ids) == 1:
            if (getattr(self.store, "admission_use", None)
                    and graph.nodes[group.node_ids[0]].op_type != "input"):
                raise MeasurementContractError("selected-group admission does not admit analytic singleton costs")
            cost = self.base.estimate_group(graph, group)
            self.usage.append({"node_ids": list(group.node_ids), "source": cost.source,
                               "measured": False, "scope": "logical input or explicit analytic singleton"})
            return cost
        key, _ = self.keys.describe(group)
        contract = self.contract_by_device[group.device]
        record = self.store.query(key, contract, allow_conditional_dev=self.allow_conditional_dev)
        if record["role"] != "fused_kernel":
            raise ValueError("Native chain labels are comparison-only, not fused Kernel costs")
        entry = KernelLatency(record["statistics"]["median_us"] / 1e6, "measured", record["record_id"],
                              contract, record["statistics"], record["admission"])
        adapter = KeyedKernelProfile(self.keys, {record["kernel_key_id"]: entry}, self.base,
                                    measurement_contract_id=contract, allow_unbound=self.allow_conditional_dev)
        cost = adapter.estimate_group(graph, group)
        self.usage.append({"node_ids": list(group.node_ids), "device": group.device,
            "source": cost.source, "measured": True, "kernel_key_id": record["kernel_key_id"],
            "measurement_contract_id": contract, "record_id": record["record_id"],
            "admission": record["admission"], "identity_gaps": record["identity_gaps"],
            "duration_s": cost.duration})
        if "admission_decision_id" in record:
            self.usage[-1].update(admission_decision_id=record["admission_decision_id"],
                                 source_record_id=record["source_record_id"], admission_use=record["admission_use"])
        return cost
