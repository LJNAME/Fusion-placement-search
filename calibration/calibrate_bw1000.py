"""Calibrate one homogeneous two-card BW1000 mesh with PyTorch/HIP.

The script measures only costs owned by Fusion Placement Search:

* per-device launch, memory-copy, Qwen3 GEMM and reference-op latency;
* directional peer-to-peer latency/bandwidth for device 0 <-> device 1;
* a HardwareSpec-compatible JSON profile plus lossless raw measurements.

It deliberately does not benchmark TP/DP/PP, collectives, pipeline bubbles, or
other downstream auto-parallel choices.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on the GPU server
    torch = None
    F = None


MIB = 1024**2
GIB = 1024**3
QWEN3_06B = {
    "hidden_size": 1024,
    "ffn_hidden_size": 3072,
    "num_attention_heads": 16,
    "num_query_groups": 8,
    "head_dim": 128,
    "seq_length": 256,
    "micro_batch_size": 1,
}


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("cannot summarize an empty sequence")
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "median": float(statistics.median(vals)),
        "mean": float(mean),
        "std": float(std),
        "p10": float(_percentile(vals, 0.10)),
        "p90": float(_percentile(vals, 0.90)),
        "samples": vals,
    }


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return result


def _sync(devices: Iterable[int]) -> None:
    for device in sorted(set(int(d) for d in devices)):
        torch.cuda.synchronize(device)


def time_loop(
    fn: Callable[[], object],
    *,
    devices: Sequence[int],
    warmup: int,
    iterations: int,
    samples: int,
) -> dict:
    if warmup < 0 or iterations <= 0 or samples <= 0:
        raise ValueError("invalid timing counts")
    for _ in range(warmup):
        fn()
    _sync(devices)
    durations = []
    for _ in range(samples):
        _sync(devices)
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        _sync(devices)
        durations.append((time.perf_counter() - started) / iterations)
    result = summarize(durations)
    result["warmup"] = int(warmup)
    result["iterations_per_sample"] = int(iterations)
    return result


def _dtype_from_name(name: str):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _new_tensor(shape: Sequence[int], *, device: int, dtype):
    return torch.empty(tuple(shape), device=device, dtype=dtype).normal_()


def benchmark_matmul(
    device: int,
    *,
    name: str,
    m: int,
    n: int,
    k: int,
    dtype,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict:
    with torch.cuda.device(device):
        left = _new_tensor((m, k), device=device, dtype=dtype)
        right = _new_tensor((k, n), device=device, dtype=dtype)

        def op():
            return torch.mm(left, right)

        timing = time_loop(
            op,
            devices=[device],
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
    seconds = timing["median"]
    flops = 2.0 * m * n * k
    return {
        "name": name,
        "op_type": "matmul",
        "shape": {"m": m, "n": n, "k": k},
        "flops": flops,
        "duration_s": timing,
        "throughput_tflops": float(flops / seconds / 1e12),
    }


def benchmark_bmm(
    device: int,
    *,
    name: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict:
    with torch.cuda.device(device):
        left = _new_tensor((batch, m, k), device=device, dtype=dtype)
        right = _new_tensor((batch, k, n), device=device, dtype=dtype)

        def op():
            return torch.bmm(left, right)

        timing = time_loop(
            op,
            devices=[device],
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
    seconds = timing["median"]
    flops = 2.0 * batch * m * n * k
    return {
        "name": name,
        "op_type": "batched_matmul",
        "shape": {"batch": batch, "m": m, "n": n, "k": k},
        "flops": flops,
        "duration_s": timing,
        "throughput_tflops": float(flops / seconds / 1e12),
    }


def benchmark_reference_ops(
    device: int,
    *,
    dtype,
    warmup: int,
    iterations: int,
    samples: int,
) -> list[dict]:
    tokens = QWEN3_06B["seq_length"] * QWEN3_06B["micro_batch_size"]
    hidden = QWEN3_06B["hidden_size"]
    ffn = QWEN3_06B["ffn_hidden_size"]
    with torch.cuda.device(device):
        hidden_x = _new_tensor((tokens, hidden), device=device, dtype=dtype)
        hidden_y = _new_tensor((tokens, hidden), device=device, dtype=dtype)
        gate = _new_tensor((tokens, ffn), device=device, dtype=dtype)
        up = _new_tensor((tokens, ffn), device=device, dtype=dtype)
        weight = torch.ones(hidden, device=device, dtype=dtype)
        cases = [
            ("rmsnorm_reference", lambda: F.rms_norm(hidden_x, (hidden,), weight, 1e-6)),
            ("silu", lambda: F.silu(gate)),
            ("elementwise_mul", lambda: gate * up),
            ("residual_add", lambda: hidden_x + hidden_y),
            # This is an eager reference expression, not a measured fused kernel.
            ("silu_mul_eager_reference", lambda: F.silu(gate) * up),
        ]
        results = []
        for name, op in cases:
            timing = time_loop(
                op,
                devices=[device],
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            )
            results.append(
                {
                    "name": name,
                    "duration_s": timing,
                    "implementation_status": "reference",
                }
            )
    return results


def benchmark_device(
    device: int,
    *,
    dtype_name: str,
    copy_mib: int,
    warmup: int,
    iterations: int,
    samples: int,
    launch_iterations: int,
) -> dict:
    dtype = _dtype_from_name(dtype_name)
    props = torch.cuda.get_device_properties(device)
    tokens = QWEN3_06B["seq_length"] * QWEN3_06B["micro_batch_size"]
    hidden = QWEN3_06B["hidden_size"]
    ffn = QWEN3_06B["ffn_hidden_size"]
    q_size = QWEN3_06B["num_attention_heads"] * QWEN3_06B["head_dim"]
    kv_size = QWEN3_06B["num_query_groups"] * QWEN3_06B["head_dim"]

    matmul_shapes = [
        ("q_proj", tokens, q_size, hidden),
        ("k_proj", tokens, kv_size, hidden),
        ("v_proj", tokens, kv_size, hidden),
        ("attn_out_proj", tokens, hidden, q_size),
        ("gate_proj", tokens, ffn, hidden),
        ("up_proj", tokens, ffn, hidden),
        ("down_proj", tokens, hidden, ffn),
    ]
    matmuls = [
        benchmark_matmul(
            device,
            name=name,
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        for name, m, n, k in matmul_shapes
    ]
    heads = QWEN3_06B["num_attention_heads"]
    seq = QWEN3_06B["seq_length"]
    head_dim = QWEN3_06B["head_dim"]
    matmuls.extend(
        [
            benchmark_bmm(
                device,
                name="qk_matmul",
                batch=heads,
                m=seq,
                n=seq,
                k=head_dim,
                dtype=dtype,
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            ),
            benchmark_bmm(
                device,
                name="av_matmul",
                batch=heads,
                m=seq,
                n=head_dim,
                k=seq,
                dtype=dtype,
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            ),
        ]
    )

    element_size = torch.empty((), dtype=dtype).element_size()
    copy_elements = copy_mib * MIB // element_size
    with torch.cuda.device(device):
        source = torch.empty(copy_elements, device=device, dtype=dtype).normal_()
        target = torch.empty_like(source)
        copy_timing = time_loop(
            lambda: target.copy_(source),
            devices=[device],
            warmup=warmup,
            iterations=max(5, iterations // 2),
            samples=samples,
        )
        moved_bytes = 2.0 * copy_elements * element_size
        copy_bandwidth = moved_bytes / copy_timing["median"]

        scalar = torch.zeros(1, device=device, dtype=dtype)
        launch_timing = time_loop(
            lambda: scalar.add_(1),
            devices=[device],
            warmup=max(50, warmup),
            iterations=launch_iterations,
            samples=samples,
        )

    operator_results = benchmark_reference_ops(
        device,
        dtype=dtype,
        warmup=warmup,
        iterations=iterations,
        samples=samples,
    )
    return {
        "index": int(device),
        "name": str(props.name),
        "total_memory_bytes": int(props.total_memory),
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
        "dtype": dtype_name,
        "copy": {
            "size_mib": int(copy_mib),
            "bytes_counting_read_and_write": moved_bytes,
            "duration_s": copy_timing,
            "bandwidth_bytes_per_s": float(copy_bandwidth),
        },
        "launch": {
            "operation": "one_element_add_in_place",
            "duration_s": launch_timing,
        },
        "matmul": matmuls,
        "reference_ops": operator_results,
    }


def _peer_access(src: int, dst: int) -> bool | None:
    checker = getattr(torch.cuda, "can_device_access_peer", None)
    if checker is None:
        return None
    try:
        return bool(checker(src, dst))
    except RuntimeError:
        return None


def benchmark_link_direction(
    src: int,
    dst: int,
    *,
    dtype,
    sizes_kib: Sequence[int],
    warmup: int,
    max_iterations: int,
    samples: int,
) -> dict:
    element_size = torch.empty((), dtype=dtype).element_size()
    points = []
    for size_kib in sizes_kib:
        nbytes = int(size_kib) * 1024
        elements = max(1, nbytes // element_size)
        actual_bytes = elements * element_size
        with torch.cuda.device(src):
            source = torch.empty(elements, device=src, dtype=dtype).normal_()
        with torch.cuda.device(dst):
            target = torch.empty(elements, device=dst, dtype=dtype)
        # Keep each timing sample around 256 MiB but run at least five copies.
        iterations = min(max_iterations, max(5, int((256 * MIB) / actual_bytes)))
        with torch.cuda.device(dst):
            timing = time_loop(
                lambda: target.copy_(source, non_blocking=True),
                devices=[src, dst],
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            )
        points.append(
            {
                "size_bytes": int(actual_bytes),
                "duration_s": timing,
                "bandwidth_bytes_per_s": float(actual_bytes / timing["median"]),
            }
        )
    fit = fit_link_model(points)
    return {
        "src": int(src),
        "dst": int(dst),
        "peer_access": _peer_access(src, dst),
        "points": points,
        "fit": fit,
    }


def fit_link_model(points: Sequence[dict]) -> dict:
    if not points:
        raise ValueError("link fit requires points")
    large = [p for p in points if float(p["size_bytes"]) >= MIB]
    if len(large) < 2:
        large = list(points)
    xs = [float(p["size_bytes"]) for p in large]
    ys = [float(p["duration_s"]["median"]) for p in large]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
        if denom > 0
        else 0.0
    )
    if slope > 0:
        bandwidth = 1.0 / slope
        fitted_intercept = y_mean - slope * x_mean
    else:
        bandwidth = max(float(p["bandwidth_bytes_per_s"]) for p in points)
        fitted_intercept = 0.0
    smallest = min(points, key=lambda p: float(p["size_bytes"]))
    observed_small = float(smallest["duration_s"]["median"])
    latency = max(0.0, min(observed_small, max(0.0, fitted_intercept)))
    if latency == 0.0:
        latency = max(0.0, observed_small - float(smallest["size_bytes"]) / bandwidth)
    return {
        "bandwidth_bytes_per_s": float(bandwidth),
        "latency_s": float(latency),
        "fitted_intercept_s": float(fitted_intercept),
        "fit_point_count": len(large),
    }


def _matmul_efficiency_points(devices: Sequence[dict]) -> list[list[float]]:
    per_device_peak = {
        int(device["index"]): max(float(item["throughput_tflops"]) for item in device["matmul"])
        for device in devices
    }
    by_k: dict[int, list[float]] = {}
    for device in devices:
        peak = per_device_peak[int(device["index"])]
        for item in device["matmul"]:
            k = int(item["shape"]["k"])
            by_k.setdefault(k, []).append(min(1.0, float(item["throughput_tflops"]) / peak))
    return [[k, float(statistics.median(values))] for k, values in sorted(by_k.items())]


def build_hardware_config(raw: dict) -> dict:
    devices = raw["devices"]
    links = raw["links"]
    if len(devices) != 2 or len(links) != 2:
        raise ValueError("BW1000 mesh calibration requires two devices and two directions")
    index_to_local = {int(device["index"]): local for local, device in enumerate(devices)}
    link_matrix: list[list[float | None]] = [[None, None], [None, None]]
    bandwidths = []
    latencies = []
    for link in links:
        src = index_to_local[int(link["src"])]
        dst = index_to_local[int(link["dst"])]
        bandwidth = float(link["fit"]["bandwidth_bytes_per_s"])
        latency = float(link["fit"]["latency_s"])
        link_matrix[src][dst] = bandwidth
        bandwidths.append(bandwidth)
        latencies.append(latency)

    launch = statistics.median(
        float(device["launch"]["duration_s"]["median"]) for device in devices
    )
    config_devices = []
    for device in devices:
        peak_tflops = max(float(item["throughput_tflops"]) for item in device["matmul"])
        config_devices.append(
            {
                "name": device["name"],
                "flops": peak_tflops * 1e12,
                "mem_bw": float(device["copy"]["bandwidth_bytes_per_s"]),
                "max_mem": float(device["total_memory_bytes"]),
                "class": "BW1000",
                "dtype": raw["dtype"],
                "calibrated": True,
                "source": raw["source_id"],
                "physical_index": int(device["index"]),
            }
        )
    return {
        "name": "bw1000-2card-calibrated",
        "comm_bw": float(min(bandwidths)),
        "comm_latency": float(statistics.median(latencies)),
        "launch_overhead": float(launch),
        # No real fused kernel is measured by this script.
        "fusion_overhead": 0.0,
        "link_matrix": link_matrix,
        "interference": 0.0,
        "overlap_send": False,
        "devices": config_devices,
        "calibrated": True,
        "calibration_scope": {
            "device_compute": True,
            "device_memory": True,
            "launch": True,
            "point_to_point_links": True,
            "real_fused_kernels": False,
            "compute_communication_interference": False,
            "downstream_parallelism": False,
        },
        "shape_model": {
            "matmul": {
                "compute": {
                    "curve": "table",
                    "points": _matmul_efficiency_points(devices),
                }
            }
        },
        "operator_latency_source": "raw_measurements.json",
        "source": raw["source_id"],
    }


def _environment() -> dict:
    return {
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--devices", type=parse_int_csv, default=parse_int_csv("0,1"))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--copy-mib", type=int, default=256)
    parser.add_argument(
        "--link-sizes-kib",
        type=parse_int_csv,
        default=parse_int_csv("4,64,1024,16384,65536,262144"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--link-iterations", type=int, default=100)
    parser.add_argument("--launch-iterations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--allow-non-hip",
        action="store_true",
        help="allow a CUDA build for development only; BW1000 results require torch.version.hip",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if torch is None:
        raise SystemExit("PyTorch is required")
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false; start an instance with BW1000 attached")
    if torch.version.hip is None and not args.allow_non_hip:
        raise SystemExit("torch.version.hip is empty; this is not the BW1000 HIP build")
    if len(args.devices) != 2 or len(set(args.devices)) != 2:
        raise SystemExit("--devices must select exactly two distinct devices")
    count = torch.cuda.device_count()
    if any(device >= count for device in args.devices):
        raise SystemExit(f"selected devices {args.devices}, but PyTorch sees only {count}")
    names = [torch.cuda.get_device_name(device) for device in args.devices]
    if len(set(names)) != 1:
        raise SystemExit(f"the selected mesh is not homogeneous: {names}")

    warmup = 2 if args.quick else args.warmup
    iterations = 5 if args.quick else args.iterations
    link_iterations = 10 if args.quick else args.link_iterations
    launch_iterations = 200 if args.quick else args.launch_iterations
    samples = 3 if args.quick else args.samples
    env = _environment()
    source_id = (
        f"{env['hostname']}:{env['timestamp']}:torch-{env['torch']}:"
        f"hip-{env['torch_hip']}:{args.dtype}"
    )

    print(f"host={env['hostname']} torch={env['torch']} hip={env['torch_hip']}")
    print(f"devices={args.devices} names={names} dtype={args.dtype}")
    measured_devices = []
    for device in args.devices:
        print(f"[device {device}] measuring Qwen3 operators")
        result = benchmark_device(
            device,
            dtype_name=args.dtype,
            copy_mib=args.copy_mib,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
            launch_iterations=launch_iterations,
        )
        measured_devices.append(result)
        peak = max(item["throughput_tflops"] for item in result["matmul"])
        print(
            f"  peak={peak:.3f} TFLOP/s "
            f"mem={result['copy']['bandwidth_bytes_per_s'] / 1e9:.3f} GB/s "
            f"launch={result['launch']['duration_s']['median'] * 1e6:.3f} us"
        )

    dtype = _dtype_from_name(args.dtype)
    measured_links = []
    for src, dst in ((args.devices[0], args.devices[1]), (args.devices[1], args.devices[0])):
        print(f"[link {src}->{dst}] measuring {args.link_sizes_kib} KiB")
        result = benchmark_link_direction(
            src,
            dst,
            dtype=dtype,
            sizes_kib=args.link_sizes_kib,
            warmup=warmup,
            max_iterations=link_iterations,
            samples=samples,
        )
        measured_links.append(result)
        print(
            f"  bandwidth={result['fit']['bandwidth_bytes_per_s'] / 1e9:.3f} GB/s "
            f"latency={result['fit']['latency_s'] * 1e6:.3f} us "
            f"peer_access={result['peer_access']}"
        )

    raw = {
        "format": "fusion-placement-search.bw1000-calibration",
        "version": "1.0",
        "source_id": source_id,
        "environment": env,
        "dtype": args.dtype,
        "model": {"name": "qwen3-0.6b", **QWEN3_06B},
        "timing": {
            "warmup": warmup,
            "iterations": iterations,
            "link_iterations_cap": link_iterations,
            "launch_iterations": launch_iterations,
            "samples": samples,
            "quick": bool(args.quick),
        },
        "devices": measured_devices,
        "links": measured_links,
        "limitations": [
            "reference ops are eager PyTorch operations, not fused kernels",
            "compute/communication interference is not measured",
            "downstream TP/DP/PP behavior is outside this calibration",
        ],
    }
    config = build_hardware_config(raw)
    out_dir = Path(args.out_dir)
    raw_path = out_dir / "raw_measurements.json"
    profile_path = out_dir / "bw1000_2card.calibrated.json"
    _write_json(raw_path, raw)
    _write_json(profile_path, config)
    print(f"raw measurements: {raw_path}")
    print(f"hardware profile: {profile_path}")
    print("real fused kernels remain uncalibrated by design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
