"""One GPU0 Adapter Graph-Replay measurement; no comparison or search."""
import argparse
import copy
from dataclasses import asdict
import faulthandler
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import os

from calibration.kerv_add_rmsnorm_probe import (
    ROOT, KERNEL, SHAPE, TOLERANCE, bitwise_equal, prepare_adapter, references,
    save, sha, source_path, synthetic_inputs,
)
from calibration.lightop_profile_smoke import POOL, memory_snapshot
from calibration.stage_size_calibration import timed
from cost_model.fusion_kernel_profile import GroupKernelKeys, KernelLatency, key_id
from cost_model.measurement_labels import measurement_contract_id

CALLS, ROUNDS, WARMUP = 50, 7, 10
CHECKS = {"restored_exact": True, "outputs_checked": CALLS, "references_per_output": 2,
          "sum_aliases_preserved": True, "normalized_outputs_independent": True,
          "residual_weight_and_source_unchanged": True}
SOURCES = ("calibration/kerv_add_rmsnorm_timing.py", "calibration/kerv_add_rmsnorm_probe.py",
           "integration/flagscale/kerv_add_rmsnorm.py", "execution/reference_identity.py",
           "calibration/stage_size_calibration.py", "calibration/lightop_profile_smoke.py",
           "integration/flagscale/kerv_profile_runtime.py", "tools/run_kerv_add_rmsnorm_timing.sh")


def plan(flagscale_root):
    import json
    _, adapter = prepare_adapter(flagscale_root, "cuda:0")
    evidence_path = ROOT / "data/kerv_add_rmsnorm_adapter_01/review/report.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not evidence["review_passed"] or not evidence["gpu_adapter_correctness_verified_for_case"]:
        raise ValueError("the existing Adapter correctness evidence must pass first")
    return {"scope": "kerv_add_rmsnorm_gpu0_graph_timing_candidate_v1", "adapter": adapter,
            "shape": list(SHAPE), "dtype": "bf16", "source_sha256": sha(source_path(flagscale_root)),
            "flagscale_revision": subprocess.check_output(["git", "-C", str(flagscale_root), "rev-parse", "HEAD"], text=True).strip(),
            "correctness_evidence_sha256": sha(evidence_path),
            "collector_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "collector_sources": {p: sha(ROOT / p) for p in SOURCES}, "pool": POOL,
            "contract_template": {
                "executor": "placed_kerv_add_rmsnorm_graph_replay", "submission_included": False,
                "allocation_included": False, "mode": "graph_replay",
                "granularity": "batch_amortized_complete_adapter_device_work", "unit": "us/call",
                "calls_per_replay": CALLS, "rounds": ROUNDS, "warmup_replays": WARMUP,
                "eager_preparation_calls": CALLS, "statistic": "median", "extra_attempts": 0,
                "recipe": "consume_x_dual_output_graph_private_pool.v1",
                "timing_boundary": "start Event -> one Graph replay -> end Event on the same stream",
                "synchronization": "restore, validation and device synchronization outside Events",
                "input_reuse": "50 independent x buffers; shared read-only residual/weight/source",
                "restore": "restore all x exactly and invalidate normalized outputs before each replay; outside Events",
                "allocation": "normalized allocations captured in private Graph pool; no eager allocation cost",
                "excluded": ["input restoration", "input generation", "validation", "host submission", "allocation"],
                "cache_context": "isolated fixed-case replay; no full-chain cache equivalence claim",
                "correctness": TOLERANCE},
            "quality": {"attempts": 7, "extra_attempts": 0, "sample_replacement": False,
                        "iqr_method": "inclusive", "max_iqr_over_median": .10,
                        "anchor_policy": "none, matching existing isolated-stage contract"},
            "automatic_admission": False, "gpu_executed": False}


def summarize(rows):
    if len(rows) != ROUNDS or [r["round"] for r in rows] != list(range(ROUNDS)):
        raise ValueError("exactly seven chronological attempts required; no replacement")
    values, reasons, streams = [], [], set()
    for row in rows:
        if row["checks"] != CHECKS:
            raise ValueError("incomplete correctness or restoration checks")
        timing = row["timing"]
        b = timing["event_boundary"]
        if any(b[k] != "cuda:0" for k in ("device", "start_device", "end_device")) or b["calls_per_replay"] != CALLS:
            raise ValueError("Event boundary does not match the GPU0 Graph")
        streams.add(b["stream"])
        elapsed = timing["elapsed_ms_per_50_calls"]
        if timing["error"] is not None or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
            reasons.append("invalid_timing_round_" + str(row["round"]))
        else:
            values.append(elapsed * 1000 / CALLS)
    if len(streams) != 1:
        raise ValueError("measurement stream changed")
    stats = None
    if not reasons:
        median = statistics.median(values)
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        stats = {"median_us": median, "min_us": min(values), "max_us": max(values),
                 "iqr_over_median": (q3-q1)/median,
                 "mad_rel": statistics.median(abs(v-median) for v in values)/median,
                 "n_rounds": ROUNDS, "n_calls": ROUNDS*CALLS}
        if stats["iqr_over_median"] > .10:
            reasons.append("iqr_over_median_exceeds_10_percent")
    return {"status": "quality_failed" if reasons else "quality_passed", "statistics": stats,
            "reasons": reasons, "samples_us": values}


def kernel_identity(bound, observations, runtime, device, *, return_keys=False):
    """Use the existing canonical factory; UUID remains separate provenance."""
    graph, hardware = copy.deepcopy(bound.placed.graph), copy.deepcopy(bound.placed.hardware)
    ids = {"x": bound.x_node, "residual": bound.residual_node,
           "sum": bound.node_ids[0], "normalized": bound.node_ids[1]}
    for role, node_id in ids.items():
        observed = observations[role]
        if (observed["shape"] != list(SHAPE) or observed["dtype"] != "bf16"
                or observed["layout"] != "contiguous" or observed["strides"] != [524288, 2048, 1]
                or observed["layout_provenance"]["device"] != "cuda:0"
                or observed["layout_provenance"]["kind"] != "runtime_tensor"):
            raise ValueError("actual Tensor layout differs from the fixed measured case")
        for field in ("layout", "strides", "layout_provenance"):
            graph.nodes[node_id].extra[field] = copy.deepcopy(observed[field])
    physical = hardware.devices[0]
    physical.name = device["name"]
    physical.extra.update(physical_index=0, source={"kind":"runtime_observation", "execution_device":"cuda:0",
                          **{k:v for k,v in device.items() if k != "uuid"}})
    binding = copy.deepcopy(bound.adapter.implementation)
    binding.update(weight=observations["weight"], output_aliases={"sum": "consumed_x", "normalized": "independent"},
                   execution_sources={p:sha(ROOT/p) for p in SOURCES[1:4]})
    keys = GroupKernelKeys(graph, hardware, {"add_rmsnorm": binding}, runtime)
    key, gaps = keys.describe(bound.placed.plan.groups[bound.index])
    if gaps:
        raise ValueError("incomplete canonical identity: " + ",".join(gaps))
    if return_keys:
        return keys
    return {"kernel_key": key, "kernel_key_id": key_id(key)}


def run(flagscale_root, output, frozen, context):
    import torch
    from execution.reference_identity import tensor_observation
    from integration.flagscale.kerv_profile_runtime import observe_environment
    context.update(stage="runtime_guard", device="cuda:0")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU0 required; no CPU/eager fallback")
    instrumentation = {k:v for k,v in os.environ.items() if v and (k in
        {"ROCP_TOOL_LIB", "ROCPROFILER_TOOL_LIBRARIES", "HSA_TOOLS_LIB"} or
        (k == "LD_PRELOAD" and any(s in v.lower() for s in ("prof", "trace"))))}
    if instrumentation:
        raise RuntimeError("external profiling instrumentation is enabled: " + repr(instrumentation))
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    bound, descriptor = prepare_adapter(flagscale_root, "cuda:0")
    if descriptor != frozen["adapter"]:
        raise ValueError("Adapter/placement changed after the plan")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream), torch.no_grad():
        original, residual, weight = synthetic_inputs(device)
        snapshots = [t.clone() for t in (original, residual, weight)]
        xs = [original.clone() for _ in range(CALLS)]
        if len({x.data_ptr() for x in xs}) != CALLS:
            raise ValueError("mutable x buffers must be independent")
        refs = references(original, residual, weight)

        def invoke(x):
            return bound({bound.x_node:x, bound.residual_node:residual}, weight)

        context.update(stage="eager_preparation", pool=POOL, device="cuda:0")
        save(output/"progress.json", context)
        for x in xs:
            invoke(x)  # compile before capture; no profiler is started in this process
        module = bound.adapter._module
        observed, environment = observe_environment(module, device)
        if not environment.get("driver_version") or not observed.get("architecture"):
            raise ValueError("missing driver/architecture identity")
        if not str(observed["architecture"]).startswith("gfx936"):
            raise ValueError("this measurement is scoped to the BW1000 gfx936 case")
        runtime = {"framework":"torch", "framework_version":str(torch.__version__),
                   "accelerator":"HIP", "accelerator_version":torch.version.hip,
                   "torch_git_revision":torch.version.git_version, "torch_extension_sha256":sha(torch._C.__file__),
                   "torch_build_sha256":hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
                   "triton":environment["triton"], "driver_version":environment["driver_version"],
                   "grad_enabled":torch.is_grad_enabled(), "autocast_enabled":torch.is_autocast_enabled("cuda")}
        if runtime["autocast_enabled"]:
            raise ValueError("autocast must be disabled for this BF16 contract")
        environment = {k:v for k,v in environment.items() if k != "device_uuid"}
        save(output/"runtime.json", {"device":observed, "environment":environment, "runtime":runtime})
        for x in xs: x.copy_(original)
        torch.cuda.synchronize(device)
        context.update(stage="capture", memory_before_capture=memory_snapshot(torch, device))
        save(output/"progress.json", context)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=None):
            calls = [invoke(x) for x in xs]
        specs = {k:tensor_observation(t) for k,t in (("x",xs[0]),("residual",residual),("weight",weight),
                 ("sum",calls[0].add_output),("normalized",calls[0].normalized_output))}
        identity = kernel_identity(bound, specs, runtime, observed)
        contract = {**frozen["contract_template"], "environment":environment}
        identity.update(measurement_contract=contract, measurement_contract_id=measurement_contract_id(contract),
                        device_uuid_provenance_only=observed["uuid"], tensor_observations=specs)
        save(output/"identity.json", identity)

        def restore():
            for x in xs: x.copy_(original)
            for call in calls: call.normalized_output.fill_(float("nan"))
            if not all(bitwise_equal(x, original) for x in xs):
                raise RuntimeError("x restoration failed")

        def verify():
            if not all(bitwise_equal(a,b) for a,b in zip((original,residual,weight),snapshots)):
                raise RuntimeError("read-only source/residual/weight changed")
            pointers = set()
            for x,call in zip(xs,calls):
                if call.add_output.data_ptr()!=x.data_ptr() or not bitwise_equal(call.add_output,refs[0]):
                    raise RuntimeError("sum output or alias mismatch")
                for reference in refs[1:]:
                    torch.testing.assert_close(call.normalized_output,reference,**TOLERANCE)
                pointers.add(call.normalized_output.data_ptr())
            if len(pointers)!=CALLS:
                raise RuntimeError("normalized outputs share storage")
            return dict(CHECKS)

        context.update(stage="warmup"); save(output/"progress.json",context)
        for _ in range(WARMUP):
            restore(); graph.replay(); torch.cuda.synchronize(device); verify()
        entry={"graph":graph,"stream":stream,"timing_device":"cuda:0","devices":[0]}
        rows=[]
        for attempt in range(ROUNDS):
            context.update(stage="measurement",round=attempt);save(output/"progress.json",context)
            restore();torch.cuda.synchronize(device)
            started=time.monotonic_ns()
            timing=timed(torch,entry)
            checks=verify()
            row={"round":attempt,"timing":timing,"checks":checks,
                 "started_monotonic_ns":started,"finished_monotonic_ns":time.monotonic_ns()}
            rows.append(row);save(output/f"round_{attempt}.json",row)
        quality=summarize(rows)
        candidate=None
        if quality["status"]=="quality_passed":
            stats=quality["statistics"]
            candidate=asdict(KernelLatency(stats["median_us"]/1e6,"measured","KERV GPU0 Adapter Graph Replay",
                              identity["measurement_contract_id"],stats,"conditional_dev"))
        return {"scope":frozen["scope"],"complete":True,"gpu_executed":True,"graph_capture_verified":True,
                "timing_attempted":True,"quality":quality,"records":rows,
                "identity":identity,"profile_candidate":candidate,"automatic_admission":False,
                "accepted_profiles_added":0,"full_graph_executed":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","run"))
    parser.add_argument("--flagscale-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    context={"stage":"plan","pool":POOL}
    with (args.output/"fatal_traceback.log").open("w",encoding="utf-8") as fault:
        faulthandler.enable(file=fault,all_threads=True)
        try:
            frozen=plan(args.flagscale_root);save(args.output/"plan.json",frozen)
            if args.action=="run":
                result=run(args.flagscale_root,args.output,frozen,context);save(args.output/"result.json",result)
                print(result["quality"],flush=True)
                return 0 if result["quality"]["status"]=="quality_passed" else 1
            print("CPU plan ready; one GPU0 case; GPU not executed",flush=True)
            return 0
        except Exception as error:
            save(args.output/"failure.json",{"error":repr(error),"traceback":traceback.format_exc(),
                 "context":context,"accepted_profiles_added":0})
            raise
        finally:
            faulthandler.disable()


if __name__=="__main__":
    raise SystemExit(main())
