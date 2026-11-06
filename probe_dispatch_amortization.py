"""Diagnostic only: per-call Events versus one Event pair for 50 eager calls.

Run from the Fusion repository root with the existing pinned probe available.
No label export, launch subtraction, graph capture, or automatic rerun.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def measure(torch, function, gates, original, up, mode, expected, fused):
    # Both timing modes have the same working set and fresh in-place inputs.
    block_start = time.perf_counter()
    gates.copy_(original.unsqueeze(0).expand_as(gates))
    torch.cuda.synchronize()
    views = list(gates.unbind(0))
    events = []
    wall_start = time.perf_counter()
    if mode == "batch_event":
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        submit_start = time.perf_counter()
        begin.record()
        for gate in views:
            result = function(gate, up)
        end.record()
        submit_s = time.perf_counter() - submit_start
        events.append((begin, end))
    else:
        submit_start = time.perf_counter()
        for gate in views:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = function(gate, up)
            end.record()
            events.append((begin, end))
        submit_s = time.perf_counter() - submit_start
    torch.cuda.synchronize()
    completion_s = time.perf_counter() - wall_start
    values = [a.elapsed_time(b) / 1000 for a, b in events]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("invalid Event timing")
    block_wall_s = time.perf_counter() - block_start
    torch.testing.assert_close(result, expected, rtol=.01, atol=.001)
    if fused:
        torch.testing.assert_close(gates, expected.unsqueeze(0).expand_as(gates), rtol=.01, atol=.001)
    return {
        "mode": mode, "calls": len(views), "event_durations_s": values,
        "event_mean_per_call_s": sum(values) / len(views),
        "per_call_event_median_s": st.median(values) if mode == "per_call_event" else None,
        "submission_wall_s": submit_s, "completion_wall_s": completion_s,
        "block_wall_s": block_wall_s,
        "numeric_check": "all_mutated_gate_buffers" if fused else "last_unfused_output",
        "passed": True,
    }


def run_case(torch, F, module, device, seq, output):
    torch.cuda.set_device(device)
    shape = (1, seq, 3072)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    gate_cpu = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    up_cpu = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    original, up = gate_cpu.to(device), up_cpu.to(device)
    gates = original.unsqueeze(0).repeat((50, 1, 1, 1))
    reference = (gate_cpu.float() * torch.sigmoid(gate_cpu.float()) * up_cpu.float()).to(torch.bfloat16).to(device)
    eager_reference = (F.silu(gate_cpu) * up_cpu).to(device)
    from calibration.probe_kerv_silu_mul import checked_launch_probe
    result = checked_launch_probe(module, gates[0], up)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, reference, rtol=.01, atol=.001)
    torch.testing.assert_close(result, eager_reference, rtol=.01, atol=.001)
    methods = {"fused": module.kerv_silu_mul, "unfused": lambda gate, up: F.silu(gate) * up}
    with torch.no_grad():
        # Same warmup count for both call paths; no repeated mutated input.
        for _ in range(10):
            for function in methods.values():
                gates[0].copy_(original)
                result = function(gates[0], up)
        torch.cuda.synchronize()
        summary = {"case_id": f"gpu{device.index}_seq{seq}", "device": str(device),
            "shape": list(shape), "dtype": "bf16", "layout": "contiguous",
            "working_gate_buffers": 50, "working_gate_bytes": gates.numel() * gates.element_size(),
            "rows": []}
        save(output / "partial.json", summary)
        for round_id in range(7):
            modes = ("per_call_event", "batch_event")
            if round_id % 2:
                modes = modes[::-1]
            for mode in modes:
                mode_id = int(mode == "batch_event")
                order = ("fused", "unfused") if (round_id + mode_id) % 2 == 0 else ("unfused", "fused")
                for method in order:
                    row = measure(torch, methods[method], gates, original, up, mode,
                                  reference if method == "fused" else eager_reference, method == "fused")
                    row.update(round=round_id, method=method, first_method=order[0])
                    summary["rows"].append(row)
                    save(output / "partial.json", summary)
        summary["summary"] = {}
        for mode in ("per_call_event", "batch_event"):
            summary["summary"][mode] = {}
            for method in methods:
                rows = [r for r in summary["rows"] if r["mode"] == mode and r["method"] == method]
                summary["summary"][mode][method] = {
                    "median_of_round_event_means_s": st.median(r["event_mean_per_call_s"] for r in rows),
                    "round_event_means_s": [r["event_mean_per_call_s"] for r in rows],
                }
    torch.testing.assert_close(up, up_cpu.to(device), rtol=0, atol=0)
    summary["up_preserved"] = True
    save(output / "result.json", summary)
    print(summary["case_id"], json.dumps(summary["summary"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagscale-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    probe = repo / "calibration/probe_kerv_silu_mul.py"
    if digest(probe) != "f5703c31e2984e71f43b3ae8ed6298937ef860e170e4d71ab5ba80ecf1f62718":
        raise ValueError("original probe changed; this diagnostic expects the frozen source")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = {"scope": "dispatch_and_event_granularity_diagnostic_not_kernel_ground_truth",
        "source_sha256": digest(__file__), "original_probe_sha256": digest(probe),
        "case_order": ["gpu0_seq256", "gpu0_seq1", "gpu1_seq1", "gpu1_seq256"],
        "rounds": 7, "calls_per_arm_per_round": 50, "total_timed_calls": 5600,
        "modes": ["per_call_event", "batch_event"],
        "input_contract": "50 independent contiguous gate buffers, identical initial values; same up; restore all gates and synchronize before each timed block",
        "working_set_change": "prefill gate pool ~75MiB; matched across new modes, different from original single-buffer probe",
        "comparison_statistic": "median of 7 round Event means per call in both modes; do not compare batch means only to old per-call medians",
        "limitations": ["batch eager still includes Python/driver submission gaps",
            "wall minus Event is not pure host time", "no constant overhead subtraction",
            "allocation behavior retained", "no evaluator label export", "no automatic rerun"]}
    save(args.out_dir / "plan.json", plan)
    try:
        import torch
        import torch.nn.functional as F
        from calibration.probe_kerv_silu_mul import load_kernel_module
        from calibration.kerv_silu_mul import DEFAULT_BUNDLE
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("two GPUs required; CPU fallback is forbidden")
        module, capability = load_kernel_module(args.flagscale_root, DEFAULT_BUNDLE / "capability.json")
        module.configure_recording(None, False)
        module.configure_backend("triton")
        module.configure_enabled_operators({"kerv_silu_mul"})
        save(args.out_dir / "environment.json", {"torch": str(torch.__version__),
            "hip": torch.version.hip, "triton": str(module.triton.__version__),
            "capability_id": capability["capability_id"], "source_sha256": capability["source_sha256"],
            "devices": [{"index": i, "name": torch.cuda.get_device_name(i)} for i in (0, 1)]})
        for index, seq in ((0, 256), (0, 1), (1, 1), (1, 256)):
            case_out = args.out_dir / f"gpu{index}_seq{seq}"
            case_out.mkdir()
            run_case(torch, F, module, torch.device(f"cuda:{index}"), seq, case_out)
    except Exception as error:
        save(args.out_dir / "failure.json", {"error": repr(error),
            "scope": "diagnostic failed; preserve completed/partial cases; no retry"})
        raise


if __name__ == "__main__":
    main()
