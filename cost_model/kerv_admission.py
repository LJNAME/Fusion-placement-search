"""Scoped admission decisions over immutable KERV measurement records."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .fusion_kernel_profile import key_id
from .kerv_profile import KervProfileStore, digest
from .measurement_labels import MeasurementContractError


FORMAT = "fusion-placement-search.kerv-profile-admission"
USE = "selected_fused_group_graph_batch_proxy"
SCOPE = {
    "use": USE, "executor": "placed_kerv_native_fused_graph_replay",
    "shape": [1, 256, 3072], "dtype": "bf16", "layout": "contiguous",
    "calls_per_round": 50, "rounds": 7, "warmup_replays": 10,
    "native_use": "comparison_only", "singleton_costs_admitted": False,
    "link_costs_admitted": False, "hardware_profile_sharing": False,
    "whole_plan_costs_admitted": False, "downstream_deployable": False,
}


def check_pair(fused, native):
    """Data eligibility does not depend on whether fusion is faster."""
    if fused["role"] != "fused_kernel" or native["role"] != "native_chain":
        raise MeasurementContractError("admission requires a fused Kernel and its Native comparison")
    if (fused["case_id"] != native["case_id"] or fused["measurement_contract_id"] != native["measurement_contract_id"]):
        raise MeasurementContractError("admission requires a same-case, same-contract pair")
    for record in (fused, native):
        if record["identity_gaps"] or record["admission"] != "conditional_dev":
            raise MeasurementContractError("admission requires complete original conditional evidence")
        contract = record["measurement_contract"]
        if (contract["executor"] != SCOPE["executor"] or contract.get("calls_per_round") != 50
                or contract.get("rounds") != 7 or contract.get("warmup", {}).get("replays_per_arm") != 10):
            raise MeasurementContractError("measurement recipe is outside the admitted scope")
        key = record["kernel_key"]
        if key["pattern_id"] != "silu_mul" or len(key["external_inputs"]) != 2:
            raise MeasurementContractError("admission only supports the selected SiLU-Mul expression")
        for tensor in key["external_inputs"]:
            if tensor["shape"] != SCOPE["shape"] or tensor["dtype"] != "bf16" or tensor["layout"] != "contiguous":
                raise MeasurementContractError("tensor is outside the admitted shape/dtype/layout")
        correct = record["correctness"]
        required = ("gate_alias", "gate_invalidated", "up_preserved") if record["role"] == "fused_kernel" else ("gate_preserved", "up_preserved")
        if (not all(correct.get(name) is True for name in required)
                or correct.get("all_replay_outputs_checked") != 350
                or correct.get("rtol") != .01 or correct.get("atol") != .001):
            raise MeasurementContractError("correctness/alias evidence does not meet the reviewed tolerance")
        expected_kernels = 1 if record["role"] == "fused_kernel" else 2
        if (record["execution"].get("device_kernel_count") != expected_kernels
                or record["execution"].get("synchronized_successfully") is not True
                or record["statistics"]["n_rounds"] != 7 or record["statistics"]["n_calls"] != 350):
            raise MeasurementContractError("dispatch or measurement coverage is incomplete")
        for row in record["samples"]:
            expected_order = int((record["role"] == "native_chain") == (row["round"] % 2 == 0))
            if row.get("arm_order_index") != expected_order:
                raise MeasurementContractError("paired arm order differs from the reviewed recipe")
    if fused["kernel_key"]["hardware"] != native["kernel_key"]["hardware"]:
        raise MeasurementContractError("comparison was measured on another device")


def load_admitted_store(profile_path, decision_path, *, usage_scope):
    """Materialize accepted records from an explicit, exact-record decision.

    The original profile and evidence are never rewritten. The derived record
    has its own content ID and retains the original measurement record ID.
    """
    if usage_scope != USE:
        raise MeasurementContractError("KERV admission cannot authorize this cost-model use")
    profile_path = Path(profile_path)
    source = KervProfileStore.load(profile_path)
    decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in decision.items() if k != "admission_id"}
    if (decision.get("format") != FORMAT or decision.get("version") != "1.0"
            or decision.get("admission_id") != key_id(payload)):
        raise MeasurementContractError("admission decision identity mismatch")
    if decision.get("scope") != SCOPE or decision.get("status") != "accepted":
        raise MeasurementContractError("admission decision is outside the supported scope")
    if decision.get("source_profile_sha256") != digest(profile_path):
        raise MeasurementContractError("admission decision is bound to another profile bundle")
    by_id = {r["record_id"]: r for r in source.records.values()}
    selected, contracts = {}, set()
    for entry in decision["entries"]:
        try:
            fused, native = by_id[entry["source_record_id"]], by_id[entry["native_record_id"]]
        except KeyError:
            raise MeasurementContractError("admission references an unknown measurement") from None
        check_pair(fused, native)
        if entry["kernel_key_id"] != fused["kernel_key_id"] or entry["measurement_contract_id"] != fused["measurement_contract_id"]:
            raise MeasurementContractError("admission Key/contract differs from its measurement")
        if fused["record_id"] in selected:
            raise MeasurementContractError("duplicate admission entry")
        accepted = copy.deepcopy(fused)
        accepted.update(admission="accepted", source_record_id=fused["record_id"],
                        admission_decision_id=decision["admission_id"], admission_use=USE)
        accepted["record_id"] = key_id({k: v for k, v in accepted.items() if k != "record_id"})
        selected[fused["record_id"]] = accepted
        contracts.add(fused["measurement_contract_id"])
    if not selected:
        raise MeasurementContractError("empty admission decision")
    records = [selected.get(r["record_id"], r) for r in source.records.values()]
    store = KervProfileStore(records, evidence_root=profile_path.parent, approved_contract_ids=contracts)
    store.admission_use = USE
    store.admission_id = decision["admission_id"]
    return store
