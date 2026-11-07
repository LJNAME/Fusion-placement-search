"""One GPU0 Native Add->RMSNorm chain; reuse, never rerun, the KERV cost."""
import argparse
import copy
import faulthandler
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from calibration.kerv_add_rmsnorm_probe import (
    ROOT, EPS, SHAPE, TOLERANCE, bitwise_equal, prepare_adapter, references,
    save, sha, synthetic_inputs,
)
from calibration.kerv_add_rmsnorm_timing import CALLS, ROUNDS, WARMUP, SOURCES as KERV_SOURCES, summarize
from calibration.lightop_profile_smoke import POOL, memory_snapshot
from calibration.stage_size_calibration import timed
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import measurement_contract_id

FUSED = ROOT / "data/kerv_add_rmsnorm_timing_01"
CHECKS = {"restored_exact": True, "outputs_checked": CALLS, "references_per_output": 2,
          "sum_outputs_independent": True, "normalized_outputs_independent": True,
          "all_inputs_unchanged": True}
SOURCES = (*KERV_SOURCES, "calibration/native_add_rmsnorm_timing.py", "cost_model/native_add_rmsnorm_profile.py",
           "cost_model/qwen25_add_rmsnorm.py", "tools/run_native_add_rmsnorm_timing.sh")
# Only these fields describe implementation-dependent execution, not a shared
# experimental boundary. Everything else, including the environment, must match.
RECIPE_FIELDS = {"executor", "recipe", "restore", "allocation"}


def comparable_contract(contract):
    return {k:v for k,v in contract.items() if k not in RECIPE_FIELDS}


def native_call(x, residual, weight):
    import torch.nn.functional as F
    summed = x + residual
    return summed, F.rms_norm(summed, (SHAPE[-1],), weight, eps=EPS)


def binding():
    import torch
    import torch.nn.functional as F
    return {"id":"torch.native.add_then_rmsnorm.dual_output.v1", "version":str(torch.__version__),
            "entrypoints":["torch.Tensor.__add__", "torch.nn.functional.rms_norm"],
            "source_sha256":sha(__file__), "functional_sha256":sha(inspect.getsourcefile(F.rms_norm)),
            "eps":EPS, "semantics":"BF16 out-of-place add; aten RMSNorm; preserve sum output",
            "output_aliases":{"sum":"independent", "normalized":"independent"}}


def plan(flagscale_root):
    from cost_model.kerv_add_rmsnorm_profile import load_profile
    from cost_model.search_evaluation import qwen25_3b_spec
    from simulator.model_graphs import build_model_graph
    from simulator.placement_artifacts import graph_fingerprint
    from calibration.kerv_add_rmsnorm_probe import graph_contract
    load_profile(FUSED, flagscale_root)  # Read-only, existing acceptance only.
    fused = json.loads((FUSED/"review/profile.json").read_text(encoding="utf-8"))
    _, adapter = prepare_adapter(flagscale_root, "cuda:0")
    graph = build_model_graph(qwen25_3b_spec())
    contract = copy.deepcopy(fused["measurement_contract"])
    contract.pop("environment")
    contract.update(executor="native_add_then_aten_rmsnorm_graph_replay",
                    recipe="out_of_place_add_dual_output_graph_private_pool.v1",
                    restore="restore all x exactly and invalidate sum/normalized outputs before each replay; outside Events",
                    allocation="sum and normalized allocations captured in private Graph pool; no eager allocation cost")
    return {"scope":"one_native_add_rmsnorm_gpu0_layer0_graph_chain",
            "adapter":adapter, "shape":list(SHAPE), "dtype":"bf16", "pool":POOL,
            "contract_template":contract, "quality":{"attempts":ROUNDS, "extra_attempts":0,
                "sample_replacement":False, "iqr_method":"inclusive", "max_iqr_over_median":.10},
            "graph_fingerprint":graph_fingerprint(graph), "graph_contract":graph_contract(graph),
            "fused_profile_sha256":sha(FUSED/"review/profile.json"),
            "fused_kernel_key_id":fused["kernel_key_id"],
            "collector_revision":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
            "collector_sources":{p:sha(ROOT/p) for p in SOURCES},
            "gpu_executed":False, "fused_remeasurement":False, "automatic_admission":False}


def summarize_native(rows):
    translated = copy.deepcopy(rows)
    from calibration.kerv_add_rmsnorm_timing import CHECKS as fused_checks
    for row in translated:
        if row["checks"] != CHECKS:
            raise ValueError("Native chain correctness/restoration checks incomplete")
        row["checks"] = dict(fused_checks)
    return summarize(translated)


def native_identity(bound, observations, runtime, device, implementation):
    from calibration.kerv_add_rmsnorm_timing import kernel_identity
    # Use the existing graph/key factory, with a distinct Native implementation.
    keys = kernel_identity(bound, observations, runtime, device, return_keys=True)
    group = copy.deepcopy(bound.placed.plan.groups[bound.index])
    group.extra["pattern_id"] = "native_add_rmsnorm"
    keys.bindings = {"native_add_rmsnorm":implementation}
    key, gaps = keys.describe(group)
    if gaps:
        raise ValueError("incomplete Native chain key: " + repr(gaps))
    return {"kernel_key":key, "kernel_key_id":key_id(key)}


def run(flagscale_root, output, frozen, context):
    import torch
    from types import SimpleNamespace
    from execution.reference_identity import tensor_observation
    from integration.flagscale.kerv_profile_runtime import observe_environment
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU0 required; no CPU/eager fallback")
    import triton
    if any(v and (k in {"ROCP_TOOL_LIB","ROCPROFILER_TOOL_LIBRARIES","HSA_TOOLS_LIB"}
           or k=="LD_PRELOAD" and any(s in v.lower() for s in ("prof","trace"))) for k,v in os.environ.items()):
        raise RuntimeError("external profiling instrumentation enabled")
    bound, descriptor = prepare_adapter(flagscale_root,"cuda:0")
    if descriptor != frozen["adapter"]:
        raise ValueError("binding changed after plan")
    device=torch.device("cuda:0"); torch.cuda.set_device(device)
    stream=torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream), torch.no_grad():
        observed, environment=observe_environment(SimpleNamespace(triton=triton),device)
        environment={k:v for k,v in environment.items() if k!="device_uuid"}
        runtime={"framework":"torch", "framework_version":str(torch.__version__),
                 "accelerator":"HIP", "accelerator_version":torch.version.hip,
                 "torch_git_revision":torch.version.git_version, "torch_extension_sha256":sha(torch._C.__file__),
                 "torch_build_sha256":hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
                 "triton":environment["triton"], "driver_version":environment["driver_version"],
                 "grad_enabled":torch.is_grad_enabled(), "autocast_enabled":torch.is_autocast_enabled("cuda")}
        expected=json.loads((FUSED/"raw/runtime.json").read_text(encoding="utf-8"))
        if runtime!=expected["runtime"] or environment!=expected["environment"]:
            raise ValueError("Native/Fused runtime or environment changed; no automatic comparison")
        if {k:v for k,v in observed.items() if k!="uuid"}!={k:v for k,v in expected["device"].items() if k!="uuid"}:
            raise ValueError("GPU0 device class differs from the accepted KERV case")
        save(output/"runtime.json",{"device":observed,"environment":environment,"runtime":runtime})
        original,residual,weight=synthetic_inputs(device)
        snapshots=[t.clone() for t in (original,residual,weight)]
        xs=[original.clone() for _ in range(CALLS)]
        if len({t.untyped_storage().data_ptr() for t in [original,residual,weight,*xs]})!=CALLS+3:
            raise ValueError("input buffers alias")
        refs=references(original,residual,weight)
        context.update(stage="eager_preparation"); save(output/"progress.json",context)
        for x in xs: native_call(x,residual,weight)
        for x in xs: x.copy_(original)
        torch.cuda.synchronize(device)
        context.update(stage="capture",memory_before_capture=memory_snapshot(torch,device))
        save(output/"progress.json",context)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream,pool=None):
            calls=[native_call(x,residual,weight) for x in xs]
        specs={k:tensor_observation(t) for k,t in (("x",xs[0]),("residual",residual),("weight",weight),
               ("sum",calls[0][0]),("normalized",calls[0][1]))}
        implementation=binding()
        identity=native_identity(bound,specs,runtime,observed,implementation)
        contract={**frozen["contract_template"],"environment":environment}
        identity.update(measurement_contract=contract,measurement_contract_id=measurement_contract_id(contract),
                        tensor_observations=specs,implementation=implementation,device_uuid_provenance_only=observed["uuid"])
        save(output/"identity.json",identity)

        def restore():
            for x in xs: x.copy_(original)
            for summed,normalized in calls:
                summed.fill_(float("nan")); normalized.fill_(float("nan"))
            if not all(bitwise_equal(x,original) for x in xs):
                raise RuntimeError("x restore failed")

        def verify():
            if not all(bitwise_equal(a,b) for a,b in zip((original,residual,weight),snapshots)) or not all(bitwise_equal(x,original) for x in xs):
                raise RuntimeError("Native modified an input")
            pointers={t.untyped_storage().data_ptr() for t in [original,residual,weight,*xs]}
            for summed,normalized in calls:
                if not bitwise_equal(summed,refs[0]): raise RuntimeError("Native Add output incorrect")
                for reference in refs[1:]: torch.testing.assert_close(normalized,reference,**TOLERANCE)
                for tensor in (summed,normalized):
                    pointer=tensor.untyped_storage().data_ptr()
                    if pointer in pointers: raise RuntimeError("Native outputs alias other buffers")
                    pointers.add(pointer)
            return dict(CHECKS)

        context.update(stage="warmup"); save(output/"progress.json",context)
        for _ in range(WARMUP):
            restore(); graph.replay(); torch.cuda.synchronize(device); verify()
        entry={"graph":graph,"stream":stream,"timing_device":"cuda:0","devices":[0]}
        rows=[]
        for attempt in range(ROUNDS):
            context.update(stage="measurement",round=attempt); save(output/"progress.json",context)
            restore(); torch.cuda.synchronize(device)
            started=time.monotonic_ns(); timing=timed(torch,entry); checks=verify()
            row={"round":attempt,"timing":timing,"checks":checks,
                 "started_monotonic_ns":started,"finished_monotonic_ns":time.monotonic_ns()}
            rows.append(row); save(output/f"round_{attempt}.json",row)
        return {"scope":frozen["scope"],"complete":True,"gpu_executed":True,"graph_capture_verified":True,
                "identity":identity,"records":rows,"quality":summarize_native(rows),
                "fused_remeasurement":False,"automatic_admission":False,"accepted_profiles_added":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","run"))
    parser.add_argument("--flagscale-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=False)
    context={"stage":"plan","pool":POOL,"device":"cuda:0"}
    with (args.output/"fatal_traceback.log").open("w",encoding="utf-8") as fault:
        faulthandler.enable(file=fault,all_threads=True)
        try:
            frozen=plan(args.flagscale_root); save(args.output/"plan.json",frozen)
            if args.action=="run":
                result=run(args.flagscale_root,args.output,frozen,context); save(args.output/"result.json",result)
                print(json.dumps(result["quality"]),flush=True)
                return 0 if result["quality"]["status"]=="quality_passed" else 1
            print("CPU plan ready; one Native chain; KERV not rerun",flush=True)
            return 0
        except Exception as error:
            save(args.output/"failure.json",{"error":repr(error),"traceback":traceback.format_exc(),
                 "context":context,"accepted_profiles_added":0})
            raise
        finally:
            faulthandler.disable()


if __name__=="__main__":
    raise SystemExit(main())
