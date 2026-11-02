"""Review one complete paired batch for selected-group graph-cost admission."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from calibration.kerv_profile_workflow import demonstrate, save, verify_measurement
from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_admission import FORMAT, SCOPE, USE, check_pair, load_admitted_store
from cost_model.kerv_profile import KervProfileStore, digest
from cost_model.measurement_labels import MeasurementContractError


def review(profile_path, decision_path, *, reviewer, rationale,
           capability=DEFAULT_BUNDLE / "capability.json", flagscale_root=DEFAULT_FLAGSCALE):
    profile_path, decision_path = Path(profile_path), Path(decision_path)
    if decision_path.exists():
        raise FileExistsError("admission decision already exists")
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("record the project reviewer and intended use")
    original_hash = digest(profile_path)
    store = KervProfileStore.load(profile_path)
    if len(store.records) != 4:
        raise MeasurementContractError("this review requires both arms on two devices")
    pairs = {}
    for record in store.records.values():
        reports = [p for p in record["evidence"] if p.endswith("/result.json")]
        placed = [p for p in record["evidence"] if p.endswith("/placed.json")]
        verify_measurement(profile_path.parent / reports[0], profile_path.parent / placed[0], capability, flagscale_root)
        pair = pairs.setdefault(record["case_id"], {})
        if record["role"] in pair:
            raise MeasurementContractError("duplicate arm in the paired batch")
        pair[record["role"]] = record
    entries, devices, uuids = [], [], []
    for case in sorted(pairs):
        pair = pairs[case]
        if set(pair) != {"fused_kernel", "native_chain"}:
            raise MeasurementContractError("missing paired arm")
        fused, native = pair["fused_kernel"], pair["native_chain"]
        check_pair(fused, native)
        devices.append(fused["kernel_key"]["hardware"]["device_index"])
        uuids.append(fused["kernel_key"]["hardware"]["source"]["uuid"])
        entries.append({"source_record_id": fused["record_id"], "native_record_id": native["record_id"],
                        "kernel_key_id": fused["kernel_key_id"],
                        "measurement_contract_id": fused["measurement_contract_id"]})
    if sorted(devices) != [0, 1] or len(set(uuids)) != 2:
        raise MeasurementContractError("review requires two distinct device identities")
    payload = {"format": FORMAT, "version": "1.0", "status": "accepted", "scope": copy.deepcopy(SCOPE),
               "source_profile_sha256": original_hash, "entries": entries,
               "review": {"reviewer": reviewer, "rationale": rationale,
                   "data_checks": "source, trace, identity, pairing, numeric assertions and finite sampling",
                   "preference_used_as_admission_gate": False,
                   "numeric_evidence": "source-matched collector assertions; raw output tensors not replayed locally",
                   "downstream_execution_contract_confirmed": False}}
    decision = {**payload, "admission_id": key_id(payload)}
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    save(decision_path, decision)
    admitted = load_admitted_store(profile_path, decision_path, usage_scope=USE)
    query = demonstrate(profile_path, admission_path=decision_path)
    if any(row["out_of_distribution"] for row in query["results"]):
        raise ValueError("strict admitted query unexpectedly remained OOD")
    # Derived snapshot for inspection. The original profile stays byte-identical.
    source = json.loads(profile_path.read_text(encoding="utf-8"))
    source["records"] = list(admitted.records.values())
    source["admission_decision_id"] = decision["admission_id"]
    source["admission_use"] = USE
    save(profile_path.with_name("admitted_profile.json"), source)
    save(decision_path.with_name("strict_cost_queries.json"), query)
    if digest(profile_path) != original_hash:
        raise ValueError("review changed the source measurement bundle")
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--capability", type=Path, default=DEFAULT_BUNDLE / "capability.json")
    args = parser.parse_args(argv)
    result = review(args.profile, args.output, reviewer=args.reviewer, rationale=args.rationale,
                    capability=args.capability, flagscale_root=args.flagscale_root)
    print(json.dumps({"admission_id": result["admission_id"], "accepted_fused_records": len(result["entries"]),
                      "use": USE, "downstream_deployable": False}, indent=2))


if __name__ == "__main__":
    main()
