"""Review one complete Native chain and compare with the existing KERV record."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from calibration.kerv_add_rmsnorm_probe import ROOT, prepare_adapter, save, sha
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from calibration.native_add_rmsnorm_timing import (
    FUSED, SOURCES, comparable_contract, native_identity, plan, summarize_native,
)
from cost_model.fusion_kernel_profile import key_id
from cost_model.hardware_profile import GroupCost
from cost_model.kerv_add_rmsnorm_profile import load_profile
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sources(frozen):
    revision=frozen["collector_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}",revision) or set(frozen["collector_sources"])!=set(SOURCES):
        raise MeasurementContractError("invalid Native collector provenance")
    for name,digest in frozen["collector_sources"].items():
        source=subprocess.check_output(["git","-C",str(ROOT),"show",revision+":"+name])
        if hashlib.sha256(source).hexdigest()!=digest:
            raise MeasurementContractError("Native collector differs from committed source: "+name)


def validate(directory, flagscale_root=DEFAULT_FLAGSCALE):
    directory=Path(directory)
    names={"plan.json","identity.json","runtime.json","result.json","progress.json","fatal_traceback.log"}
    names.update(f"round_{i}.json" for i in range(7))
    if not all((directory/n).is_file() for n in names) or (directory/"fatal_traceback.log").read_bytes() or (directory/"failure.json").exists():
        raise MeasurementContractError("incomplete or failed Native measurement")
    frozen,identity,runtime,result=(read(directory/n) for n in ("plan.json","identity.json","runtime.json","result.json"))
    current=plan(flagscale_root)
    for field in ("scope","adapter","shape","dtype","pool","contract_template","quality","graph_fingerprint",
                  "graph_contract","fused_profile_sha256","fused_kernel_key_id"):
        if frozen[field]!=current[field]:
            raise MeasurementContractError("Native measurement scope changed: "+field)
    verify_sources(frozen)
    rows=[read(directory/f"round_{i}.json") for i in range(7)]
    quality=summarize_native(rows)
    if quality["status"]!="quality_passed" or result["quality"]!=quality or result["records"]!=rows:
        raise MeasurementContractError("Native rounds did not pass")
    if (any(type(r[k]) is not int for r in rows for k in ("started_monotonic_ns","finished_monotonic_ns"))
        or any(r["started_monotonic_ns"]>=r["finished_monotonic_ns"] for r in rows)
        or any(a["finished_monotonic_ns"]>b["started_monotonic_ns"] for a,b in zip(rows,rows[1:]))):
        raise MeasurementContractError("Native round chronology invalid")
    if (not all(result[k] is True for k in ("complete","gpu_executed","graph_capture_verified"))
        or result["identity"]!=identity or result["fused_remeasurement"] is not False or result["automatic_admission"] is not False):
        raise MeasurementContractError("incomplete Native execution receipt")
    contract={**frozen["contract_template"],"environment":runtime["environment"]}
    if identity["measurement_contract"]!=contract or identity["measurement_contract_id"]!=measurement_contract_id(contract):
        raise MeasurementContractError("Native measurement contract mismatch")
    fused=read(FUSED/"review/profile.json"); fused_runtime=read(FUSED/"raw/runtime.json")
    if (runtime["runtime"]!=fused_runtime["runtime"] or runtime["environment"]!=fused_runtime["environment"]
        or comparable_contract(contract)!=comparable_contract(fused["measurement_contract"])):
        raise MeasurementContractError("Native/KERV comparison contract or runtime mismatch")
    if {k:v for k,v in runtime["device"].items() if k!="uuid"}!={k:v for k,v in fused_runtime["device"].items() if k!="uuid"}:
        raise MeasurementContractError("Native/KERV device class mismatch")
    bound,_=prepare_adapter(flagscale_root,"cuda:0")
    rebuilt=native_identity(bound,identity["tensor_observations"],runtime["runtime"],runtime["device"],identity["implementation"])
    if any(identity[k]!=v for k,v in rebuilt.items()):
        raise MeasurementContractError("Native canonical key mismatch")
    impl=identity["implementation"]
    if (impl.get("id")!="torch.native.add_then_rmsnorm.dual_output.v1"
        or impl.get("source_sha256")!=frozen["collector_sources"]["calibration/native_add_rmsnorm_timing.py"]
        or impl.get("version")!=runtime["runtime"]["framework_version"]
        or impl.get("eps")!=1e-6 or not impl.get("functional_sha256")):
        raise MeasurementContractError("Native implementation identity incomplete")
    return frozen,identity,runtime,quality,{n:sha(directory/n) for n in sorted(names)}


def review(directory, flagscale_root=DEFAULT_FLAGSCALE):
    directory=Path(directory)
    frozen,identity,runtime,quality,hashes=validate(directory,flagscale_root)
    fused=read(FUSED/"review/profile.json")
    native_us=quality["statistics"]["median_us"]
    fused_us=fused["latency"]["duration_s"]*1e6
    record={"scope":"GPU0 complete Native Add->RMSNorm chain; never singleton costs",
            "admission":"accepted","kernel_key":identity["kernel_key"],"kernel_key_id":identity["kernel_key_id"],
            "implementation":identity["implementation"],"runtime":runtime,
            "measurement_contract":identity["measurement_contract"],
            "measurement_contract_id":identity["measurement_contract_id"],"quality":quality,
            "duration_s":native_us/1e6,"raw_sha256":hashes,"verified_collector_revision":frozen["collector_revision"],
            "comparison":{"native_us":native_us,"kerv_us":fused_us,"kerv_minus_native_us":fused_us-native_us,
                "kerv_saving_fraction":(native_us-fused_us)/native_us,
                "preferred":"Native" if native_us<fused_us else "KERV" if fused_us<native_us else "tie",
                "same_experimental_boundary":True,"different_batches":True,
                "fused_profile_sha256":frozen["fused_profile_sha256"],"fused_remeasurement":False},
            "cross_layer_reuse_admitted":False,"whole_plan_admitted":False}
    (directory/"review").mkdir(exist_ok=True);save(directory/"review/profile.json",record)
    return record


def load_record(directory, flagscale_root=DEFAULT_FLAGSCALE):
    directory=Path(directory);record=read(directory/"review/profile.json")
    if record["admission"]!="accepted": raise MeasurementContractError("Native profile not accepted")
    for name,digest in record["raw_sha256"].items():
        if Path(name).name!=name or sha(directory/name)!=digest:
            raise MeasurementContractError("Native raw receipt changed")
    _,identity,runtime,quality,_=validate(directory,flagscale_root)
    if (record["kernel_key"]!=identity["kernel_key"] or record["kernel_key_id"]!=identity["kernel_key_id"]
        or record["measurement_contract"]!=identity["measurement_contract"]
        or record["measurement_contract_id"]!=identity["measurement_contract_id"]
        or record["runtime"]!=runtime or record["implementation"]!=identity["implementation"]
        or record["quality"]!=quality or record["duration_s"]!=quality["statistics"]["median_us"]/1e6):
        raise MeasurementContractError("Native accepted record differs from raw measurement")
    return record


def qualified_native_cost(graph,hardware,adapter,pair,device,context,fused_record,native_record):
    """Explicit target group evidence is still required; no layer-ID copying."""
    from cost_model.qwen25_add_rmsnorm import bind_candidate, qualify_context
    q=qualify_context(graph,hardware,adapter,pair,device,context,fused_record)
    if q["profile"]!="exact_equivalent": return None
    if comparable_contract(native_record["measurement_contract"])!=comparable_contract(context["measurement_contract"]):
        return None
    if native_record["runtime"]["runtime"]!=context["runtime"]:
        return None
    bound=bind_candidate(graph,hardware,adapter,pair,device)
    identity=native_identity(bound,context["observations"],context["runtime"],context["device"],native_record["implementation"])
    if identity["kernel_key"]!=native_record["kernel_key"]: return None
    duration=native_record["duration_s"]
    return GroupCost(duration,duration,0.,0.,1.,"measured_native_add_rmsnorm_chain")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("review",))
    parser.add_argument("directory",type=Path)
    parser.add_argument("--flagscale-root",type=Path,default=DEFAULT_FLAGSCALE)
    args=parser.parse_args(argv)
    print(json.dumps(review(args.directory,args.flagscale_root)["comparison"],indent=2))


if __name__=="__main__": main()
