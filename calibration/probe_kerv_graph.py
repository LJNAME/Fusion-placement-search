"""Conditional development diagnostic: single-stream graph batch replay.

No measurement is exported to a model. Failure records are part of the output.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import time
import traceback

from .kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from .probe_kerv_silu_mul import load_kernel_module, checked_launch_probe
from cost_model.measurement_labels import measurement_contract_id


POOL = {"selection": "per_arm_default_private_pool", "pool_argument": None, "shared_between_arms": False}


def save(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def memory_snapshot(torch, device):
    try:
        return {name: int(getattr(torch.cuda, name)(device)) for name in (
            "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")}
    except Exception as error:
        return {"unavailable": repr(error)}


def write_failure(path, error, context, torch=None):
    # Save the original exception before any best-effort allocator inspection.
    trace = traceback.format_exc()
    snapshot = memory_snapshot(torch, context.get("device")) if torch is not None else {"unavailable": "Torch not loaded"}
    save(path, {"status": "failed", "error": repr(error), "traceback": trace,
        "context": context, "attempted_pool_config": dict(POOL), "memory_at_failure": snapshot,
        "admission": "conditional_dev", "automatic_model_writeback": False})


def restore(torch, gates, original):
    gates.copy_(original.unsqueeze(0).expand_as(gates))
    torch.cuda.synchronize()


def capture_arm(torch, function, gates, original, up, context):
    device = gates.device
    views = list(gates.unbind(0))
    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current)
    context.update(stage="side_stream_warmup", capture_attempted=False)
    with torch.cuda.stream(stream):
        gates.copy_(original.unsqueeze(0).expand_as(gates))
        for gate in views:
            result = function(gate, up)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    del result
    restore(torch, gates, original)
    torch.cuda.reset_peak_memory_stats(device)
    context.update(stage="capture", capture_attempted=True, memory_before_capture=memory_snapshot(torch, device))
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    started = time.perf_counter()
    with torch.cuda.graph(graph, pool=None, stream=stream):
        for gate in views:
            result = function(gate, up)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    record = {"capture_wall_s": time.perf_counter() - started,
        "memory_before_capture": context["memory_before_capture"],
        "memory_after_capture": memory_snapshot(torch, device), "attempted_pool_config": dict(POOL),
        "retained_outputs": "last_final_output_only", "single_stream": True}
    return graph, result, record


def run_case(torch, F, module, device, seq, output, context):
    torch.cuda.set_device(device)
    context.clear()
    context.update(case_id=f"gpu{device.index}_seq{seq}", device=str(device), stage="input_setup", capture_attempted=False)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    shape = (1, seq, 3072)
    gate_cpu = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    up_cpu = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    original, up = gate_cpu.to(device), up_cpu.to(device)
    gates = original.unsqueeze(0).repeat((50, 1, 1, 1))
    reference = (gate_cpu.float() * torch.sigmoid(gate_cpu.float()) * up_cpu.float()).to(torch.bfloat16).to(device)
    eager_reference = (F.silu(gate_cpu) * up_cpu).to(device)
    torch.testing.assert_close(checked_launch_probe(module, gates[0], up), reference, rtol=.01, atol=.001)
    functions = {"fused": module.kerv_silu_mul, "unfused": lambda gate, up: F.silu(gate) * up}
    captures = {}
    report = {"case_id": context["case_id"], "shape": list(shape), "dtype": "bf16",
        "gate_pool_bytes": gates.numel() * gates.element_size(), "captures": {}, "rows": []}
    contract = {"executor": "graph_replay", "submission_included": True, "allocation_included": False,
        "granularity": "graph_replay", "unit": "throughput_per_call",
        "environment": {"torch": str(torch.__version__), "hip": torch.version.hip,
            "triton": str(module.triton.__version__), "device_model": torch.cuda.get_device_name(device)}}
    report.update(measurement_contract=contract, measurement_contract_id=measurement_contract_id(contract),
        admission="conditional_dev", submission_scope="one graph-launch boundary remains inside each Event interval",
        allocation_scope="allocator work occurs at capture, outside replay timing; device writes are included")
    for method, function in functions.items():
        context["method"] = method
        graph, result, record = capture_arm(torch, function, gates, original, up, context)
        captures[method] = (graph, result)
        report["captures"][method] = record
        save(output / "partial.json", report)
    context["stage"] = "replay_warmup"
    for method, (graph, result) in captures.items():
        context["method"] = method
        for _ in range(10):
            restore(torch, gates, original)
            graph.replay()
            torch.cuda.synchronize()
    context["stage"] = "timed_replay"
    for round_id in range(7):
        order = ("fused", "unfused") if round_id % 2 == 0 else ("unfused", "fused")
        for method in order:
            context.update(method=method, round=round_id)
            graph, result = captures[method]
            restore(torch, gates, original)
            torch.cuda.reset_peak_memory_stats(device)
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            started = time.perf_counter()
            begin.record()
            graph.replay()
            end.record()
            submitted = time.perf_counter() - started
            torch.cuda.synchronize()
            completed = time.perf_counter() - started
            total_s = begin.elapsed_time(end) / 1000
            if not math.isfinite(total_s) or total_s <= 0:
                raise ValueError("invalid graph Event duration")
            context["stage"] = "replay_numeric_check"
            if method == "fused":
                torch.testing.assert_close(gates, reference.unsqueeze(0).expand_as(gates), rtol=.01, atol=.001)
            else:
                torch.testing.assert_close(result, eager_reference, rtol=.01, atol=.001)
            torch.testing.assert_close(up, up_cpu.to(device), rtol=0, atol=0)
            report["rows"].append({"round": round_id, "method": method, "graph_calls": 50,
                "graph_event_total_s": total_s, "graph_batch_amortized_per_call_us": total_s * 1e6 / 50,
                "graph_submission_wall_s": submitted, "graph_completion_wall_s": completed,
                "memory_after_replay": memory_snapshot(torch, device), "numeric_passed": True,
                "numeric_coverage": "all_50_gate_outputs" if method == "fused" else "last_final_output_only"})
            save(output / "partial.json", report)
            context["stage"] = "timed_replay"
    f, u = ([r["graph_batch_amortized_per_call_us"] for r in report["rows"] if r["method"] == method]
            for method in ("fused", "unfused"))
    delta = [a - b for a, b in zip(f, u)]
    relative = [d / b for d, b in zip(delta, u)]
    preference = "inconclusive"
    if all(v < 0 for v in delta) and st.median(relative) <= -.01:
        preference = "fused_under_this_graph_contract"
    elif all(v > 0 for v in delta) and st.median(relative) >= .01:
        preference = "unfused_under_this_graph_contract"
    report["summary"] = {"graph_batch_amortized_per_call_us": {
        "fused": st.median(f), "unfused": st.median(u)}, "paired_differences_us": delta,
        "paired_relative_differences": relative, "preference": preference,
        "statistical_significance_claim": False, "automatic_model_writeback": False}
    save(output / "result.json", report)
    print(report["case_id"], json.dumps(report["summary"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    context = {"stage": "environment", "capture_attempted": False}
    torch = None
    save(args.out_dir / "plan.json", {"format": "fusion-placement-search.kerv-graph-diagnostic",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_order": ["gpu0_seq256", "gpu0_seq1", "gpu1_seq1", "gpu1_seq256"],
        "rounds": 7, "calls_per_graph": 50, "warmup_replays_per_arm": 10,
        "timed_calls": 2800, "attempted_pool_config": dict(POOL), "admission": "conditional_dev",
        "metric": "graph_batch_amortized_per_call_us", "automatic_model_writeback": False})
    try:
        import torch
        import torch.nn.functional as F
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("two GPUs required; no CPU/eager fallback")
        module, cap = load_kernel_module(args.flagscale_root, DEFAULT_BUNDLE / "capability.json")
        module.configure_recording(None, False)
        module.configure_backend("triton")
        module.configure_enabled_operators({"kerv_silu_mul"})
        save(args.out_dir / "environment.json", {"capability": cap, "torch": str(torch.__version__),
            "hip": torch.version.hip, "triton": str(module.triton.__version__),
            "allocator_config": {name: os.environ.get(name) for name in (
                "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}})
        with torch.no_grad():
            for index, seq in ((0, 256), (0, 1), (1, 1), (1, 256)):
                output = args.out_dir / f"gpu{index}_seq{seq}"
                output.mkdir()
                run_case(torch, F, module, torch.device(f"cuda:{index}"), seq, output, context)
    except Exception as error:
        write_failure(args.out_dir / "failure.json", error, context, torch)
        raise


if __name__ == "__main__":
    main()
