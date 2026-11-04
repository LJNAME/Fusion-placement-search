"""Separate hipprof process for the already-smoked GPU0 adapter; no timing labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from calibration.lightop_profile_smoke import (POOL, check_outputs, invalidate_buffers,
                                               memory_snapshot, placed_case, save)
from cost_model.fusion_kernel_profile import key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
from cost_model.measurement_labels import measurement_contract_id
from integration.lightop.silu_mul import RECIPE, SHAPE, declared_binding
from simulator.placement_artifacts import save_placed_op_graph

SCOPE = "lightop_gpu0_dispatch_evidence_not_latency_admission"
MARKERS = (257, 263)
BASELINE = ROOT / "data/lightop_profile_smoke_20261104_02_review/report.json"
SOURCES = ("calibration/lightop_dispatch.py", "calibration/lightop_profile_smoke.py",
           "integration/lightop/silu_mul.py", "integration/lightop/catalog.py",
           "integration/lightop/profile_identity.py", "cost_model/lightop_profile.py",
           "tools/run_lightop_dispatch.sh")


def collector_identity():
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {"git_revision": revision,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}}


def plan():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    return {"scope": SCOPE, "device": "cuda:0", "shape": list(SHAPE), "dtype": "bf16",
            "implementation": declared_binding(), "recipe": RECIPE,
            "baseline_smoke_archive_sha256": baseline["archive"]["sha256"],
            "expected_kernel_key_id": baseline["identity"]["kernel_key_id"],
            "expected_smoke_contract_id": baseline["identity"]["measurement_contract_id"],
            "calls_per_replay": 50, "eager_warmup_calls": 50, "warmup_replays": 10,
            "target_replays": 1, "marker_copy_bytes": list(MARKERS),
            "marker_semantics": "pinned-host to GPU0 copies, device synchronized before/after the target replay",
            "target_region": "end of 257-byte copy -> start of 263-byte copy",
            "profiler": "hipprof --hip-trace", "attempts": 1, "pool": POOL,
            "numerics": {"references": ["BF16 Native", "FP32 Reference"], "rtol": .01, "atol": .001},
            "collector": collector_identity(), "timing_labels_emitted": False,
            "automatic_admission": False, "full_graph_executed": False}


def marked_replay(torch, graph, stream, device, host_markers, device_markers):
    # No fixture, validation, NaN fill or other work belongs between these copies.
    torch.cuda.synchronize(device)
    with torch.cuda.stream(stream):
        device_markers[0].copy_(host_markers[0], non_blocking=True)
        torch.cuda.synchronize(device)
        graph.replay()
        torch.cuda.synchronize(device)
        device_markers[1].copy_(host_markers[1], non_blocking=True)
        torch.cuda.synchronize(device)


def worker(output, context):
    import torch
    import torch.nn.functional as F
    from integration.lightop.profile_identity import observe_runtime
    from integration.lightop.silu_mul import LightopSiluMulAdapter

    frozen = plan()
    save(output / "plan.json", frozen)
    context.update(stage="runtime_guard", device="cuda:0", pool=POOL)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU0 required; no eager/CPU/native substitute")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = LightopSiluMulAdapter()
    placed, index = placed_case(0)
    bound = adapter.bind(placed, index, {0: device})
    save_placed_op_graph(output / "placed.json", placed)
    actual = observe_runtime(adapter, device)
    save(output / "runtime.json", actual)
    context["stage"] = "buffer_preparation"
    generator = torch.Generator(device=device).manual_seed(20261104)
    gate = torch.randn(SHAPE, dtype=torch.bfloat16, device=device, generator=generator)
    up = torch.randn(SHAPE, dtype=torch.bfloat16, device=device, generator=generator)
    snapshots = [x.view(torch.uint8).clone() for x in (gate, up)]
    calls = [bound.prepare({bound.gate_node: gate, bound.up_node: up}) for _ in range(50)]
    if any(call.observation != calls[0].observation for call in calls):
        raise ValueError("prepared calls differ in execution identity")
    keys = LightopKernelKeys(placed, index, calls[0].observation, actual)
    key, gaps = keys.describe(placed.plan.groups[index])
    smoke_contract = measurement_contract(actual["environment"])
    identity = {"kernel_key": key, "kernel_key_id": key_id(key), "identity_gaps": gaps,
                "smoke_contract": smoke_contract, "smoke_contract_id": measurement_contract_id(smoke_contract),
                "group_index": index, "collector": frozen["collector"]}
    save(output / "identity.json", identity)
    save(output / "preparation.json", {"one_call": calls[0].observation, "call_count": len(calls)})
    if (gaps or identity["kernel_key_id"] != frozen["expected_kernel_key_id"]
            or identity["smoke_contract_id"] != frozen["expected_smoke_contract_id"]):
        raise ValueError("runtime differs from the accepted Smoke case; inspect recorded identity, do not relabel")
    # Allocate and initialize markers before capture. No new package/API is needed.
    host_markers = [torch.zeros(size, dtype=torch.uint8, pin_memory=True) for size in MARKERS]
    device_markers = [torch.empty(size, dtype=torch.uint8, device=device) for size in MARKERS]
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    context["stage"] = "side_stream_eager_warmup"
    save(output / "last_stage.json", context)
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.synchronize(device)
    context.update(stage="capture", memory_before_capture=memory_snapshot(torch, device))
    save(output / "last_stage.json", context)
    print(json.dumps(context), flush=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=None):
        for call in calls:
            call()
    torch.cuda.synchronize(device)
    with torch.cuda.stream(stream):
        for _ in range(10):
            graph.replay()
        invalidate_buffers(calls)
    torch.cuda.synchronize(device)
    context.update(stage="marked_graph_replay", pid=os.getpid(), marker_copy_bytes=list(MARKERS))
    save(output / "last_stage.json", context)
    marked_replay(torch, graph, stream, device, host_markers, device_markers)
    context["stage"] = "validation_after_end_marker"
    save(output / "last_stage.json", context)
    references = (F.silu(gate) * up, F.silu(gate.float()) * up.float())
    checks = check_outputs(torch, calls, gate, up, snapshots, references, frozen["numerics"])
    result = {"scope": SCOPE, "complete": True, "pid": os.getpid(), "device_index": 0,
              "target_replays": 1, "calls_per_replay": 50, "marker_copy_bytes": list(MARKERS),
              "checks": checks, "kernel_key_id": identity["kernel_key_id"],
              "collector": frozen["collector"], "timing_labels_emitted": False,
              "automatic_admission": False, "dispatch_audit_pending": True}
    save(output / "profile_worker.json", result)
    return result


def run(output, launcher, context, command=subprocess.run):
    frozen = plan()
    save(output / "plan.json", frozen)
    context["stage"] = "profiler_preflight"
    profiler = shutil.which("hipprof")
    if not profiler:
        raise RuntimeError("hipprof not found in the existing environment; no install or fallback")
    trace = output / "trace"
    trace.mkdir()
    args = [profiler, "--hip-trace", launcher, "-X", "utf8", "-X", "faulthandler",
            str(Path(__file__).resolve()), "worker", "--output", str(trace)]
    save(output / "profiler.json", {"path": profiler, "sha256": hashlib.sha256(Path(profiler).read_bytes()).hexdigest(),
                                    "argv": args, "attempts": 1})
    context["stage"] = "profile_worker"
    with (trace / "worker.log").open("w", encoding="utf-8") as log:
        completed = command(args, cwd=trace, stdout=log, stderr=subprocess.STDOUT, check=False)
    summary = {"scope": SCOPE, "returncode": completed.returncode, "attempts": 1,
               "pipeline_passed": False, "timing_labels_emitted": False, "automatic_admission": False}
    save(output / "summary.json", summary)
    if completed.returncode:
        return summary
    context["stage"] = "cpu_trace_review"
    from tools.review_lightop_dispatch import review
    try:
        report, timeline = review(trace)
        save(output / "dispatch_review.json", report)
        save(output / "target_kernel_timeline.json", timeline)
        summary["pipeline_passed"] = report["dispatch_observed"]
    except Exception:
        summary["trace_audit_error"] = traceback.format_exc()
    save(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher")
    args = parser.parse_args(argv)
    if args.phase == "run" and not args.launcher:
        parser.error("run requires the existing --launcher")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed = "worker.log" if args.phase == "worker" else "process.log"
    if any(path.name != allowed for path in output.iterdir()):
        parser.error("fresh output required; preserve earlier receipts")
    context = {"stage": args.phase, "device": "cuda:0", "pool": POOL}
    try:
        if args.phase == "plan":
            save(output / "plan.json", plan())
        elif args.phase == "worker":
            print(json.dumps(worker(output, context)), flush=True)
        else:
            summary = run(output, str(Path(args.launcher).resolve()), context)
            print(json.dumps(summary), flush=True)
            if not summary["pipeline_passed"]:
                raise SystemExit(1)
    except BaseException as error:
        snapshot = None
        if args.phase == "worker" and "torch" in sys.modules:
            snapshot = memory_snapshot(sys.modules["torch"], context["device"])
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
                                      "context": context, "pool": POOL, "memory_at_failure": snapshot,
                                      "automatic_admission": False, "timing_labels_emitted": False})
        raise


if __name__ == "__main__":
    main()
