"""微基准编排 + 聚合：测量 -> 生成 calibrated.json + 校准报告。

用法（海光 K100，ROCm 环境）：
  python calibration/run_calibration.py \
      --nominal configs/hygon_4cards.example.json \
      --build-cmd "hipcc -O2 -o build/kernels_hip calibration/kernels_hip.cpp -lrocblas" \
      --binary build/kernels_hip \
      --out-config configs/hygon_k100_4cards.calibrated.json \
      --report calibration_report.json

本地无卡验证整条链路（合成数据）：
  python calibration/run_calibration.py --nominal configs/hygon_4cards.example.json --dry-run --out-config /tmp/calib.json --report /tmp/calib_report.json

摩尔卡：--build-cmd 换成 musa 编译器 + 对应 kernel 源文件，聚合逻辑不变。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_json(binary: str, args: List[str]) -> dict:
    try:
        proc = subprocess.run(
            [binary] + args, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[calib] 命令失败: {[binary] + args}\n")
        sys.stderr.write(f"[calib] stderr:\n{e.stderr or '(空)'}\n")
        raise
    out = proc.stdout
    line = [l for l in out.strip().splitlines() if l.strip()][-1]
    return json.loads(line)


def _synthetic() -> Dict[str, dict]:
    """无卡时的合成测量：数值接近海光 K100 的合理占位。"""

    peak = 137.0  # TFLOPS（FP16 大 GEMM，≈ 196T × 0.7）
    gemm = []
    for k in (128, 256, 512, 1024, 2048, 4096, 8192):
        eta = 0.9 * k / (k + 512.0)
        gemm.append({"k": k, "tflops": peak * eta})
    return {
        "launch": {"per_kernel_ms": 0.02},
        "copy": {"gbps": 627.0},
        "triad": {"gbps": 750.0},
        "fusion": {"per_extra_op_ms": 0.001},
        "gemm": gemm,
        "pingpong_bw": [
            {"dev0": i, "dev1": j, "bw_gbps": 18.0 if (i + j) % 2 == 1 else 9.0}
            for i in range(4)
            for j in range(i + 1, 4)
        ],
        "pingpong_lat": {"latency_us": 10.0},
        "concurrency": {"ratio": 1.9},
        "overlap": {"interference": 0.12},
    }


def _measure(binary: str, n_gpus: int, dtype: str, gemm_ks: List[int]) -> Dict[str, dict]:
    meas: Dict[str, dict] = {}
    meas["launch"] = _run_json(binary, ["--mode", "launch", "--iters", "2000"])
    meas["copy"] = _run_json(
        binary, ["--mode", "copy", "--bytes", str(int(1e9)), "--iters", "20"]
    )
    meas["triad"] = _run_json(
        binary, ["--mode", "triad", "--n", str(1 << 26), "--iters", "20"]
    )
    meas["fusion"] = _run_json(binary, ["--mode", "fusion"])
    meas["gemm"] = [
        _run_json(
            binary,
            [
                "--mode", "gemm", "--m", "4096", "--n", "4096",
                "--k", str(k), "--dtype", dtype, "--iters", "10",
            ],
        )
        for k in gemm_ks
    ]
    bw, lat = [], {}
    for i in range(n_gpus):
        for j in range(i + 1, n_gpus):
            bw.append(
                _run_json(
                    binary,
                    ["--mode", "pingpong", "--dev0", str(i), "--dev1", str(j)],
                )
            )
    meas["pingpong_bw"] = bw
    meas["pingpong_lat"] = _run_json(
        binary,
        ["--mode", "pingpong", "--dev0", "0", "--dev1", "1", "--latency", "1"],
    )
    meas["concurrency"] = _run_json(binary, ["--mode", "concurrency"])
    meas["overlap"] = _run_json(binary, ["--mode", "overlap"])
    return meas


def _aggregate(nominal: dict, meas: Dict[str, dict], machine: str) -> dict:
    """把测量值聚合成 calibrated 配置。"""

    cfg = json.loads(json.dumps(nominal))  # 深拷贝

    # ---- 单卡计算：峰值算力 + 形状曲线 ----
    gemm = sorted(meas["gemm"], key=lambda r: r["k"])
    peak_tflops = max(r["tflops"] for r in gemm)
    eta_points = [[r["k"], r["tflops"] / peak_tflops] for r in gemm]
    nominal_flops = cfg["devices"][0]["flops"]
    eff_flops = peak_tflops * 1e12 / nominal_flops

    # ---- 显存带宽 ----
    nominal_mem_bw = cfg["devices"][0]["mem_bw"]
    # 用 Stream triad（kernel 访存）作为有效带宽的代表；D2D copy 作参考
    mem_bw_gbps = meas.get("triad", {}).get("gbps") or meas["copy"]["gbps"]
    eff_mem_bw = mem_bw_gbps * 1e9 / nominal_mem_bw

    n_devs = len(cfg["devices"])
    for d in cfg["devices"]:
        d["efficiency"] = {"flops": round(eff_flops, 4), "mem_bw": round(eff_mem_bw, 4)}
        d["calibrated"] = True
        d["source"] = (
            f"实测校准 {machine} {datetime.date.today().isoformat()} "
            f"(gemm峰值 {peak_tflops:.1f} TFLOPS / triad带宽 {mem_bw_gbps:.1f} GB/s / "
            f"copy带宽 {meas['copy']['gbps']:.1f} GB/s)"
        )

    # ---- 形状曲线：table（真机数据点） ----
    cfg["shape_model"] = {
        "_说明": "实测形状效率曲线（逐 K 扫描），matmul 用 table 插值。",
        "matmul": {
            "compute": {
                "curve": "table",
                "points": [[int(k), round(eta, 4)] for k, eta in eta_points],
            }
        },
    }

    # ---- 链路：逐对带宽 + 延迟 ----
    mat = [[None] * n_devs for _ in range(n_devs)]
    bw_list = []
    for r in meas["pingpong_bw"]:
        i, j, bw = r["dev0"], r["dev1"], r["bw_gbps"]
        mat[i][j] = round(bw * 1e9, 3)
        mat[j][i] = round(bw * 1e9, 3)
        bw_list.append(bw)
    cfg["link_bw"] = round(sum(bw_list) / len(bw_list) * 1e9, 3)
    cfg["link_matrix"] = mat
    cfg["comm_latency"] = round(meas["pingpong_lat"]["latency_us"] * 1e-6, 10)

    # ---- 调度开销 ----
    cfg["launch_overhead"] = round(meas["launch"]["per_kernel_ms"] * 1e-3, 9)
    cfg["fusion_overhead"] = round(meas["fusion"]["per_extra_op_ms"] * 1e-3, 9)

    # ---- 干扰 ----
    cfg["interference"] = round(max(0.0, meas["overlap"]["interference"]), 4)

    cfg["_说明"] = (
        "海光 K100 四卡实测校准产物（由 calibration/run_calibration.py 生成）；"
        "标称值保留在 devices[].extra.nominal，效率系数为实测/标称。"
    )
    return cfg


def _report(nominal: dict, meas: Dict[str, dict], cfg: dict, machine: str) -> dict:
    peak = max(r["tflops"] for r in meas["gemm"])
    mem_bw_gbps = meas.get("triad", {}).get("gbps") or meas["copy"]["gbps"]
    ratio = meas["concurrency"]["ratio"]
    verdict = (
        "链路独立（可并发）"
        if ratio >= 1.4
        else ("共享总线/交换机（并发无加速）" if ratio <= 1.2 else "部分共享（不确定）")
    )
    return {
        "machine": machine,
        "date": datetime.date.today().isoformat(),
        "nominal_config": nominal.get("name", ""),
        "parameters": {
            "flops_peak_tflops": {"value": peak, "unit": "TFLOPS", "method": "rocblas gemm M=N=4096 逐 K 扫描", "mode": "gemm"},
            "flops_efficiency": {"value": cfg["devices"][0]["efficiency"]["flops"], "method": "实测峰值 / 官方标称"},
            "mem_bw_gbps": {"value": meas["copy"]["gbps"], "unit": "GB/s", "method": "同设备 D2D copy 1GB", "mode": "copy"},
            "mem_bw_triad_gbps": {"value": mem_bw_gbps, "unit": "GB/s", "method": "Stream triad（读2+写1）", "mode": "triad"},
            "mem_bw_efficiency": {"value": cfg["devices"][0]["efficiency"]["mem_bw"], "method": "实测 / 官方标称"},
            "launch_overhead_s": {"value": cfg["launch_overhead"], "method": "空 kernel 连续启动", "mode": "launch"},
            "fusion_overhead_s": {"value": cfg["fusion_overhead"], "method": "add+scale 融合 vs 未融合", "mode": "fusion"},
            "link_matrix_bps": {"value": cfg["link_matrix"], "method": "双卡 ping-pong 逐对", "mode": "pingpong"},
            "link_bw_bps": {"value": cfg["link_bw"], "method": "逐对带宽均值"},
            "comm_latency_s": {"value": cfg["comm_latency"], "method": "小消息 ping-pong", "mode": "pingpong-latency"},
            "interference": {"value": cfg["interference"], "method": "同卡并发 GEMM+拷贝 降速比-1", "mode": "overlap"},
        },
        "interconnect_verdict": {
            "concurrency_ratio": ratio,
            "verdict": verdict,
            "note": "ratio>=1.4 用逐链路模型；<=1.2 建议退回共享总线/把 link_matrix 视为保守值。",
        },
        "shape_model_points": cfg["shape_model"]["matmul"]["compute"]["points"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True, help="占位 config（模板）")
    ap.add_argument("--build-cmd", default="", help="编译 kernel 的命令（可选，先执行）")
    ap.add_argument("--binary", default="build/kernels_hip", help="微基准可执行文件")
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="无卡时用合成数据跑通聚合链路")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--dtype", default="fp32",
                    help="预留：当前统一用 rocblas_sgemm(FP32)，DTK/rocblas 兼容最稳")
    ap.add_argument("--gemm-k", default="128,256,512,1024,2048,4096,8192")
    args = ap.parse_args()

    with open(args.nominal, encoding="utf-8") as f:
        nominal = json.load(f)
    machine = os.uname().nodename if hasattr(os, "uname") else "local"

    if args.dry_run:
        meas = _synthetic()
        print("[dry-run] 使用合成测量值，未调用真机。")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.binary)), exist_ok=True)
        if args.build_cmd:
            subprocess.run(args.build_cmd, shell=True, check=True)
        ks = [int(x) for x in args.gemm_k.split(",")]
        meas = _measure(args.binary, args.n_gpus, args.dtype, ks)

    cfg = _aggregate(nominal, meas, machine)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_config)), exist_ok=True)
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    rep = _report(nominal, meas, cfg, machine)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

    print(f"校准配置已生成: {args.out_config}")
    print(f"  flops效率={cfg['devices'][0]['efficiency']['flops']}  "
          f"mem_bw效率={cfg['devices'][0]['efficiency']['mem_bw']}")
    print(f"  link_bw={cfg['link_bw']:.3e} B/s  latency={cfg['comm_latency']:.3e} s  "
          f"interference={cfg['interference']}")
    print(f"  shape table 点数={len(cfg['shape_model']['matmul']['compute']['points'])}")
    print(f"  互联判定: {rep['interconnect_verdict']['verdict']} (ratio={rep['interconnect_verdict']['concurrency_ratio']:.2f})")


if __name__ == "__main__":
    main()
