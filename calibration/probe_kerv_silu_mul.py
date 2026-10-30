"""Single-Kernel GPU probe for the pinned KERV development capability.

This probes Kernel dispatch/correctness/latency only. It cannot execute an
operator graph, accept a CPU fallback, or promote a downstream capability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

from .kerv_silu_mul import (
    CORE, DEFAULT_BUNDLE, DEFAULT_FLAGSCALE, PACKAGE, check_capability,
)


def load_kernel_module(flagscale_root, capability_path):
    manifest = check_capability(capability_path, flagscale_root)
    runtime_parent = Path(flagscale_root) / PACKAGE.parent
    sys.path.insert(0, str(runtime_parent))
    module = importlib.import_module("KERVRuntimeOptimization.embodied_ops.kerv_ops")
    if Path(module.__file__).resolve() != (Path(flagscale_root) / CORE).resolve():
        raise ValueError("KERV imported from another source tree")
    return module, manifest


def checked_fused_call(module, gate, up):
    """Reject every known native fallback condition before the public call."""
    if gate.shape != up.shape or gate.device != up.device or gate.dtype != up.dtype:
        raise ValueError("KERV probe requires same shape/device/dtype")
    if not gate.is_cuda or not up.is_cuda:
        raise RuntimeError("GPU-only fusion probe; CPU fallback is forbidden")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("noncontiguous tensors would take the native fallback")
    if gate.ndim < 1 or any(size <= 0 for size in gate.shape) or gate.shape[-1] > 8192:
        raise ValueError("shape would take the native fallback")
    if str(gate.dtype).removeprefix("torch.") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported probe dtype")
    if module._BACKEND != "triton" or module.triton is None or not module._use_triton("kerv_silu_mul"):
        raise RuntimeError("explicit enabled Triton backend is required; auto/native is forbidden")
    return module.kerv_silu_mul(gate, up)


def checked_launch_probe(module, gate, up):
    """Observe the JIT dispatch once; do not instrument timed iterations."""
    original = module._silu_mul_kernel
    launches = []

    class DispatchProbe:
        def __getitem__(self, grid):
            launch = original[grid]
            def invoke(*args, **kwargs):
                launches.append(1)
                return launch(*args, **kwargs)
            return invoke

    module._silu_mul_kernel = DispatchProbe()
    try:
        result = checked_fused_call(module, gate, up)
    finally:
        module._silu_mul_kernel = original
    if len(launches) != 1:
        raise RuntimeError("public KERV call did not dispatch exactly one fused JIT launch")
    return result


def run_probe(capability_path, flagscale_root=DEFAULT_FLAGSCALE, *, shape=(1, 256, 3072),
              dtype="bf16", device="cuda:0", warmup=10, samples=3, iterations=20):
    started = time.perf_counter()
    if not str(device).startswith("cuda"):
        raise RuntimeError("GPU-only fusion probe; CPU fallback is forbidden")
    if warmup < 0 or samples <= 0 or iterations <= 0:
        raise ValueError("invalid measurement counts")
    if not shape or any(size <= 0 for size in shape) or shape[-1] > 8192:
        raise ValueError("shape is outside the pinned single-Kernel guard")
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU available; fused probe was not run")
    module, manifest = load_kernel_module(flagscale_root, capability_path)
    module.configure_recording(None, False)
    module.configure_backend("triton")
    module.configure_enabled_operators({"kerv_silu_mul"})
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    target = torch.device(device)
    torch.cuda.set_device(target)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    gate_cpu = torch.randn(shape, dtype=torch_dtype, generator=generator)
    up_cpu = torch.randn(shape, dtype=torch_dtype, generator=generator)
    gate_original, up = gate_cpu.to(target), up_cpu.to(target)
    gate = gate_original.clone()
    reference = (gate_cpu.float() * torch.sigmoid(gate_cpu.float()) * up_cpu.float()).to(torch_dtype).to(target)
    unfused_reference = (F.silu(gate_cpu) * up_cpu).to(target)
    contract = manifest["numeric_contract"]
    rtol = contract["fp32_rtol" if dtype == "fp32" else "bf16_fp16_rtol"]
    atol = contract["fp32_atol" if dtype == "fp32" else "bf16_fp16_atol"]
    with torch.no_grad():
        result = checked_launch_probe(module, gate, up)
        torch.cuda.synchronize(target)
        if result.data_ptr() != gate.data_ptr():
            raise RuntimeError("KERV no longer aliases gate as declared")
        if not torch.isfinite(result).all().item() or not torch.equal(up, up_cpu.to(target)):
            raise RuntimeError("nonfinite output or mutated up input")
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
        torch.testing.assert_close(result, unfused_reference, rtol=rtol, atol=atol)
        max_error = (result.float() - reference.float()).abs().max().item()
        unfused_error = (result.float() - unfused_reference.float()).abs().max().item()
        methods = {"fused": lambda value: module.kerv_silu_mul(value, up),
                   "unfused": lambda value: F.silu(value) * up}
        for _ in range(warmup):
            for function in methods.values():
                gate.copy_(gate_original)
                function(gate)
        torch.cuda.synchronize(target)
        timings = {label: [] for label in methods}
        for sample in range(samples):
            for label in (("fused", "unfused") if sample % 2 == 0 else ("unfused", "fused")):
                block_started = time.perf_counter()
                events = []
                for _ in range(iterations):
                    # The mutating kernel must receive original inputs each time.
                    # Restore is outside the Kernel event region, but its cost is
                    # included in the complete profiling wall clock below.
                    gate.copy_(gate_original)
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    result = methods[label](gate)
                    end.record()
                    events.append((begin, end))
                torch.cuda.synchronize(target)
                values = [begin.elapsed_time(end) / 1000 for begin, end in events]
                timings[label].append({"sample": sample, "kernel_samples_s": values,
                                       "kernel_median_s": statistics.median(values),
                                       "block_wall_s": time.perf_counter() - block_started})
    props = torch.cuda.get_device_properties(target)
    return {
        "format": "fusion-placement-search.kerv-single-kernel-probe", "version": "1.0",
        "scope": "development_Kernel_probe_not_full_graph_or_SROP_benefit",
        "capability_id": manifest["capability_id"], "binding": manifest["binding"],
        "source_sha256": manifest["source_sha256"],
        "collector_source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("probe_kerv_silu_mul.py", "kerv_silu_mul.py")},
        "runtime": {"torch": str(torch.__version__), "hip": torch.version.hip,
                    "cuda": torch.version.cuda, "triton": str(module.triton.__version__),
                    "device": str(target), "device_name": props.name,
                    "device_total_memory": props.total_memory},
        "input": {"shape": list(shape), "dtype": dtype, "strides": list(gate.stride()),
                  "layout": "contiguous", "seed": 1234},
        "dispatch": {"backend": "triton", "observed_JIT_dispatches_per_probe": 1,
                     "native_fallback_allowed": False},
        "correctness": {"rtol": rtol, "atol": atol,
                        "max_abs_error_fp32_reference": max_error,
                        "max_abs_error_unfused_reference": unfused_error,
                        "gate_alias": True, "up_preserved": True, "finite": True},
        "warmup": warmup, "samples": samples, "iterations": iterations,
        "timings": timings,
        "kernel_median_s": {label: statistics.median(row["kernel_median_s"] for row in rows)
                            for label, rows in timings.items()},
        "profiling_wall_s": time.perf_counter() - started,
        "timing_scope": "Kernel events exclude input restore; profiling wall includes source load, setup, compile, checks, restore, warmup and both methods",
        "downstream_deployable": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", type=Path, default=DEFAULT_BUNDLE / "capability.json")
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--shape", default="1,256,3072")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_BUNDLE / "probe_results")
    args = parser.parse_args(argv)
    result_path = args.out_dir / "probe.json"
    failure_path = args.out_dir / "failure.json"
    if result_path.exists() or failure_path.exists():
        raise FileExistsError("probe output exists; choose a fresh --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = run_probe(args.capability, args.flagscale_root,
            shape=tuple(int(size) for size in args.shape.split(",")), dtype=args.dtype,
            device=args.device, warmup=args.warmup, samples=args.samples, iterations=args.iterations)
    except Exception as error:
        failure_path.write_text(json.dumps({"status": "failed_or_not_run", "error": str(error),
                                           "latency_measured": False}, indent=2) + "\n", encoding="utf-8")
        raise
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("KERV single-Kernel GPU probe: PASS", report["kernel_median_s"])
    print("Result:", result_path)


if __name__ == "__main__":
    main()
