"""CPU-only frozen pilot and run05 identity export, with closed cost admission."""
import argparse
from pathlib import Path

from calibration.ffn_b4 import save
from calibration.ffn_anchor_control_protocol import frozen_protocol
from cost_model.ffn_profile_export import FFNExportStore, export_bundle, read, ROOT
from cost_model.measurement_labels import MeasurementContractError


def prepare(source_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = export_bundle(source_root,read(ROOT/"data/ffn_b4_run05_review/report.json"))
    save(output/"profiles.json",bundle)
    save(output/"anchor_protocol.json",frozen_protocol())
    save(output/"scope_manifest.json", {"format":"ffn-export-scope.v1", "records":[{
        "row_id":r["row_id"],"record_id":r["record_id"],"kernel_key_id":r["kernel_key_id"],
        "measurement_contract_id":r["measurement_contract_id"],"usage":r["usage"],
        "identity_gaps":r["identity_gaps"],"admission":r["admission"],"cost_use_open":False,
        "admission_blockers":r["admission_blockers"],
        "forbidden_uses":["whole_plan_cost","automatic_admission"]+
                         (["singleton_cost_substitution"] if r["kind"]=="native_chain" else [])+
                         (["certified_direct_P2P"] if r["kind"]=="transfer" else [])
    } for r in bundle["records"]]})
    store = FFNExportStore.load(output/"profiles.json",source_root)
    inspected, incomplete, rejected = [], [], []
    for record in bundle["records"]:
        try:
            store.inspect_exact(record["kernel_key"],record["measurement_contract_id"],usage=record["usage"])
            inspected.append(record["row_id"])
        except MeasurementContractError:
            incomplete.append({"row_id":record["row_id"],"identity_gaps":record["identity_gaps"]})
        try:
            store.query_cost(record["kernel_key"],record["measurement_contract_id"],usage=record["usage"])
        except MeasurementContractError:
            rejected.append(record["row_id"])
    if len(rejected) != len(bundle["records"]):
        raise ValueError("unadmitted record escaped strict cost gate")
    result = {"scope":"CPU preparation only", "exported_records":len(bundle["records"]),
              "canonical_key_count":len({r["kernel_key_id"] for r in bundle["records"]}),
              "exact_complete_identity_inspections":inspected,"incomplete_identities":incomplete,
              "strict_cost_rejected_records":rejected, "new_accepted_profiles":0,
              "new_gpu_measurements":0,"gpu_collector_ready":False,"gpu_entry_open":False,
              "priceable_plans":0,"legal_plans":48,"export_id":bundle["export_id"],
              "protocol_id":frozen_protocol()["protocol_id"]}
    save(output/"verification.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    print(prepare(args.source_root,args.output))


if __name__ == "__main__":
    main()
