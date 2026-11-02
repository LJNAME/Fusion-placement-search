"""Small crash-isolated diagnostics; no formal Profile samples or admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from calibration.ffn_b4 import save, failure
from calibration.ffn_b4_runtime import Runtime, GraphCase, sync, E, H
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from cost_model.kerv_profile import digest

ROOT = Path(__file__).resolve().parents[1]
CASES = ("timing_control", "timing_profile", "chain_allocating_1", "chain_preallocated_1", "chain_preallocated_50")


def diagnostic_plan():
    return {"version": "ffn-b4-diagnostics.v1", "cases": list(CASES), "worker_attempts": 1,
            "process_scope": "CPU-only parent; fresh HIP context per worker; no retained parent smoke runtime",
            "timing": {"calls_per_replay": 50, "warmup": 10, "pre_samples": 3, "post_samples": 3,
                       "intervention": "one replay with profiler versus one replay without profiler",
                       "wall_boundary": "before start.record through all-device completion; excludes reset/check",
                       "wall_is_diagnostic_proxy": True},
            "chain": {"calls": [1, 50], "preallocated_50_requires_preallocated_1_pass": True,
                      "preallocation_scope": "explicit producer/consumer outputs only; BLAS internals not certified"},
            "formal_measurements": False, "automatic_admission": False,
            "source_sha256": {name: digest(ROOT/name) for name in (
                "calibration/ffn_b4_diagnostics.py", "calibration/ffn_b4_runtime.py",
                "integration/flagscale/kerv_execution.py", "data/ffn_measurement_b3/recipe.json")}}


def paired_sample(case, count):
    case.mark("diagnostic_sample_reset")
    case.reset(0)
    sync(case.devices)
    with torch.cuda.device(case.device), torch.cuda.stream(case.stream):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter_ns()
        start.record(case.stream)
        case.graph.replay()
        end.record(case.stream)
        end.synchronize()
        event_wait_done = time.perf_counter_ns()
    sync(case.devices)
    all_done = time.perf_counter_ns()
    elapsed_ms = start.elapsed_time(end)
    checks = case.check(case.outputs, 0)
    if not all(value is True for value in checks.values()):
        raise AssertionError("diagnostic numerical/invariant check failed")
    case.mark("diagnostic_sample_complete")
    return {"event_batch_ms": elapsed_ms, "event_us_per_call": elapsed_ms*1000/count,
            "event_wait_wall_us": (event_wait_done-begin)/1000,
            "all_device_completion_wall_us": (all_done-begin)/1000,
            "extra_device_sync_us": (all_done-event_wait_done)/1000,
            "event_device": str(start.device), "end_device": str(end.device),
            "stream": int(case.stream.cuda_stream), "calls": count, "checks": checks,
            "formal_sample": False}


def projection(value, weight, output=None):
    if output is None:
        return value @ weight
    return torch.matmul(value, weight, out=output)


def chain_case(runtime, count, preallocated):
    values = runtime.tensor("producer_V", E, 0).unsqueeze(0).repeat((count, 1, 1, 1))
    wd = runtime.tensor("W_down", (3072, 1024), 0)
    wg = runtime.tensor("W_gate", (1024, 3072), 1)
    targets = torch.empty((count, *H), device="cuda:1", dtype=torch.bfloat16)
    producer_out = torch.empty((count, 256, 1024), device="cuda:0", dtype=torch.bfloat16) if preallocated else None
    consumer_out = torch.empty((count, 256, 3072), device="cuda:1", dtype=torch.bfloat16) if preallocated else None
    expected = (runtime.y(1).float().reshape(256, 1024) @ wg.float()).to(torch.bfloat16).reshape(E)
    producer_stream, consumer_stream = torch.cuda.Stream(device=0), torch.cuda.Stream(device=1)
    begin = torch.cuda.Event()
    ready, consumed = [torch.cuda.Event() for _ in range(count)], [torch.cuda.Event() for _ in range(count)]
    produced, outputs = [], []
    def reset(_):
        targets.fill_(float("nan"))
        for output in outputs:
            output.fill_(float("nan"))
        if producer_out is not None:
            producer_out.fill_(float("nan"))
        sync((0, 1))
    def body():
        produced.clear()
        outputs.clear()
        begin.record(consumer_stream)
        producer_stream.wait_event(begin)
        for i, value in enumerate(values.unbind(0)):
            if i:
                producer_stream.wait_event(consumed[i-1])
            with torch.cuda.device(0), torch.cuda.stream(producer_stream):
                tensor = projection(value.reshape(256, 3072), wd, producer_out[i] if preallocated else None).reshape(H)
                produced.append(tensor)
                ready[i].record(producer_stream)
            consumer_stream.wait_event(ready[i])
            with torch.cuda.stream(producer_stream):
                with torch.cuda.device(1), torch.cuda.stream(consumer_stream):
                    targets[i].copy_(tensor, non_blocking=True)
                    outputs.append(projection(targets[i].reshape(256, 1024), wg,
                                              consumer_out[i] if preallocated else None).reshape(E))
                    consumed[i].record(consumer_stream)
        return list(outputs)
    def check(result, _):
        for output in result:
            torch.testing.assert_close(output, expected, rtol=.01, atol=.001)
        return {"all_outputs": len(result) == count, "stale_outputs_excluded": True}
    if not runtime.peer["0->1"]:
        raise RuntimeError("peer path unavailable; no host fallback")
    case = GraphCase("diagnostic_chain", torch.device("cuda:1"), (0, 1), body, reset, check,
                     {**runtime.identity((0,1), {"operator": "diagnostic matmul/copy/matmul", "explicit_outputs_preallocated": preallocated}),
                      "calls": count, "formal_profile": False}, consumer_stream)
    case.progress = runtime.checkpoint
    return case


def worker(name, output, flagscale_root):
    runtime = None
    try:
        recipe = json.loads((ROOT/"data/ffn_measurement_b3/recipe.json").read_text(encoding="utf-8"))
        with torch.no_grad():
            runtime = Runtime(output, recipe, flagscale_root)
            save(output/"runtime_identity.json", runtime.identity_record)
            if name.startswith("timing_"):
                case = runtime.make("silu_mul/fused@gpu0")
                case.capture()
                for _ in range(10):
                    case.replay(0)
                before = [paired_sample(case, 50) for _ in range(3)]
                if name == "timing_profile":
                    intervention = case.trace(output/"intervention_trace.json")
                else:
                    case.reset(0)
                    sync(case.devices)
                    with torch.cuda.device(case.device), torch.cuda.stream(case.stream):
                        case.graph.replay()
                    sync(case.devices)
                    case.check(case.outputs, 0)
                    intervention = {"profiler_enabled": False, "replays": 1}
                after = [paired_sample(case, 50) for _ in range(3)]
                result = {"status": "passed", "before": before, "after": after, "intervention": intervention}
            else:
                count = 50 if name.endswith("_50") else 1
                case = chain_case(runtime, count, "preallocated" in name)
                runtime.context["memory_before_capture"] = runtime.snapshots()
                case.capture()
                sample = paired_sample(case, count)
                result = {"status": "passed", "calls": count, "sample": sample}
            save(output/"identity.json", case.identity)
            save(output/"result.json", {**result, "automatic_admission": False, "formal_profile": False})
    except Exception as error:
        save(output/"failure.json", failure(error, runtime, diagnostic=name))
        raise


def collect(output, flagscale_root, run_worker=subprocess.run):
    records = {}
    save(output/"plan.json", diagnostic_plan())
    for name in CASES:
        if name == "chain_preallocated_50" and records["chain_preallocated_1"]["status"] != "passed":
            records[name] = {"status": "blocked", "reason": "one-call preallocated chain did not pass", "worker_attempts": 0}
        else:
            directory = output/name
            directory.mkdir()
            command = [sys.executable, "-X", "faulthandler", "-m", "calibration.ffn_b4_diagnostics",
                       "--output", str(directory), "--flagscale-root", str(flagscale_root), "--worker", name]
            print(json.dumps({"diagnostic_start": name}), flush=True)
            returncode = None
            try:
                with (directory/"worker.log").open("w", encoding="utf-8") as log:
                    completed = run_worker(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                returncode = completed.returncode
                result = json.loads((directory/"result.json").read_text(encoding="utf-8")) if completed.returncode == 0 else {}
                if completed.returncode != 0 or result.get("status") != "passed":
                    raise RuntimeError(f"diagnostic worker failed, returncode={completed.returncode}")
                records[name] = {"status": "passed", "worker_attempts": 1, "returncode": 0}
            except Exception as error:
                records[name] = {"status": "failed", "worker_attempts": 1, "returncode": returncode, "error": repr(error)}
        save(output/"summary.json", {"cases": records, "formal_measurements": False, "automatic_admission": False})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--worker", choices=CASES)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output, args.flagscale_root)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        collect(args.output, args.flagscale_root)


if __name__ == "__main__":
    main()
