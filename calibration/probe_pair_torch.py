"""两卡异构机的资源探针（torch 版；A6000 + Titan 首发）。

测出来的东西直接对应 configs/README.md 的字段：
  devices[].max_mem / flops / mem_bw   —— 显存、GEMM 实测峰值、大拷贝带宽
  shape_model.matmul.compute.points    —— GEMM 效率随 K 的曲线
  link_matrix / link_bw / comm_latency —— 双卡 ping-pong
  launch_overhead                      —— 小 kernel 启动
用法（项目根目录，机器上）：
  python calibration/probe_pair_torch.py --out data/a6000_titan/probe.json
  python calibration/probe_pair_torch.py --quick      # 少跑几轮
只用 torch + nvidia-smi，不需要编译；卡数不足 2 张时报错退出。
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

try:
    import torch
except ImportError:  # 项目本身不依赖 torch；只有上机跑探针才需要
    torch = None

MB = 1024 * 1024


def nvidia_smi_text():
    out = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        out["gpus"] = [s.strip() for s in r.stdout.strip().splitlines()]
    except Exception as e:  # pragma: no cover
        out["gpus"] = [f"<nvidia-smi 失败: {e}>"]
    try:
        r = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                           text=True, timeout=30)
        out["topo"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        out["topo"] = f"<nvidia-smi topo -m 失败: {e}>"
    return out


def copy_bw_gbps(dev, mb=256, iters=20, warmup=3):
    """大数组拷贝：返回 (读+写合计 GB/s, 单次耗时 s)。"""
    n = mb * MB // 4
    with torch.cuda.device(dev):
        src = torch.empty(n, dtype=torch.float32, device=dev).normal_()
        dst = torch.empty_like(src)
        for _ in range(warmup):
            dst.copy_(src)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return 2.0 * n * 4 / dt / 1e9, dt


def gemm_tflops(dev, m, n, k, iters=20, warmup=5):
    with torch.cuda.device(dev):
        a = torch.randn(m, k, dtype=torch.float16, device=dev)
        b = torch.randn(k, n, dtype=torch.float16, device=dev)
        c = None
        for _ in range(warmup):
            c = a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return 2.0 * m * n * k / dt / 1e12, dt


def pingpong(d0, d1, mb=64, iters=200, warmup=10):
    """双卡 ping-pong：返回 (单程带宽 GB/s, 单程耗时 us)。"""
    n = mb * MB // 4
    a = torch.empty(n, dtype=torch.float32, device=d0).normal_()
    b = torch.empty(n, dtype=torch.float32, device=d1)
    for _ in range(warmup):
        b.copy_(a)
        a.copy_(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
        a.copy_(b)
    torch.cuda.synchronize()
    hop = (time.perf_counter() - t0) / (2 * iters)
    return n * 4 / hop / 1e9, hop * 1e6


def small_latency_us(d0, d1, iters=1000, warmup=50):
    """4 KB 小包的往返延迟 -> 单程 us（含启动开销）。"""
    n = 1024  # 4 KB
    a = torch.empty(n, dtype=torch.float32, device=d0).normal_()
    b = torch.empty(n, dtype=torch.float32, device=d1)
    for _ in range(warmup):
        b.copy_(a)
        a.copy_(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
        a.copy_(b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (2 * iters) * 1e6


def launch_overhead_us(dev, iters=2000, warmup=50):
    """单元素 add_ 的逐次耗时（启动开销上界，含 python 循环开销）。"""
    with torch.cuda.device(dev):
        x = torch.zeros(1, device=dev)
        for _ in range(warmup):
            x.add_(1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            x.add_(1.0)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="每项少跑几轮")
    ap.add_argument("--copy-mb", type=int, default=256)
    ap.add_argument("--link-mb", type=int, default=64)
    args = ap.parse_args()

    if torch is None:
        raise SystemExit('这台机器没有 torch；探针要在装了 torch 的 GPU 机器上跑')
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda 不可用；这个脚本要在有 GPU 的机器上跑")
    n_dev = torch.cuda.device_count()
    if n_dev < 2:
        raise SystemExit(f"只看到 {n_dev} 张卡，双卡链路测不了")

    iters = 8 if args.quick else 20
    link_iters = 50 if args.quick else 200

    print(f"host={socket.gethostname()}  torch={torch.__version__}  "
          f"cuda={torch.version.cuda}  devices={n_dev}")

    info = nvidia_smi_text()
    print("--- nvidia-smi -L ---")
    for line in info["gpus"]:
        print("  " + line)
    print("--- nvidia-smi topo -m ---")
    print(info["topo"])

    devices = []
    for i in range(n_dev):
        prop = torch.cuda.get_device_properties(i)
        copy_bw, copy_dt = copy_bw_gbps(i, mb=args.copy_mb, iters=iters)
        peak, _ = gemm_tflops(i, 8192, 8192, 8192, iters=max(5, iters // 2))
        curve = []
        for k in (256, 512, 1024, 2048, 4096):
            tf, _ = gemm_tflops(i, 4096, 4096, k, iters=iters)
            curve.append([k, round(tf, 2)])
        launch = launch_overhead_us(i) if not args.quick else None
        devices.append({
            "index": i,
            "name": prop.name,
            "capability": f"{prop.major}.{prop.minor}",
            "sm_count": prop.multi_processor_count,
            "total_mem_gb": prop.total_memory / 1e9,
            "copy_bw_gbps_readwrite": round(copy_bw, 1),
            "gemm_peak_tflops_fp16": round(peak, 1),
            "gemm_curve_tflops_by_k": curve,
            "launch_overhead_us": None if launch is None else round(launch, 2),
        })
        print(f"[dev{i}] {prop.name}: copy {copy_bw:.1f} GB/s(读+写)  "
              f"gemm峰值 {peak:.1f} TFLOPS  curve {curve}  "
              f"launch {launch if launch is None else round(launch, 2)} us")

    bw, hop_us = pingpong(0, 1, mb=args.link_mb, iters=link_iters)
    lat = small_latency_us(0, 1, iters=1000 if not args.quick else 200)
    print(f"[link] 0<->1 单程 {bw:.2f} GB/s  {hop_us:.1f} us（{args.link_mb}MB）；"
          f"4KB 单程 {lat:.1f} us")

    out = {
        "host": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi": info,
        "devices": devices,
        "link": {
            "pair": [0, 1],
            "pingpong_bw_gbps": round(bw, 2),
            "pingpong_hop_us": round(hop_us, 2),
            "small_msg_latency_us": round(lat, 2),
            "msg_size_mb": args.link_mb,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"写到 {args.out}")


if __name__ == "__main__":
    main()