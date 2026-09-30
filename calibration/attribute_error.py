"""组合层验证的残差归因：自动判断误差是"结构问题"还是"数值问题"。

输入 measured.json（真机 makespan / 逐卡峰值显存），对每个方案重跑
仿真器，提取组合层特征（组数、跨卡传输量、链路繁忙、重叠度、显存利用
率等），然后：
  - 全局：MAPE + log 比值的 std -> 若 std 小，说明是"常数偏移"
    （数值问题），给出全局修正因子 median(real/sim)；
  - 逐特征：log 误差与特征的 Spearman -> 若某特征强相关，说明是
    "结构问题"候选（该特征对应的模型机制错了），给出提示。

运行：
  python calibration/attribute_error.py \
      --measured measured.json --hw configs/hygon_k100_4cards.calibrated.json \
      --out attribution_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.execution import simulate
from simulator.hardware import load_hardware


FEATURE_HINTS = {
    "n_groups": "组数相关 -> 启动/融合开销或设备内调度问题",
    "n_transfers": "跨卡传输数相关 -> 链路/拷贝调度问题",
    "total_transfer_bytes": "传输字节相关 -> 链路带宽/分块模型问题",
    "comm_share": "通信占比相关 -> 争用/重叠模型问题",
    "overlap_ratio": "重叠度相关 -> 通信-计算重叠/干扰模型问题",
    "mem_util": "显存利用率相关 -> 显存生命周期/分配器问题",
}


def _spearman(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _features(res, graph, plan, hw) -> Dict[str, float]:
    total_work = sum(gs.duration for gs in res.group_stats) + res.bus_busy
    mem_peak = max(res.peak_mem.values(), default=0.0)
    max_mem = max(d.max_mem for d in hw.devices)
    return {
        "n_nodes": float(len(graph.nodes)),
        "n_groups": float(len(plan.groups)),
        "n_transfers": float(len(res.transfer_stats)),
        "total_transfer_bytes": float(sum(t.bytes for t in res.transfer_stats)),
        "comm_share": res.bus_busy / max(res.makespan, 1e-12),
        "overlap_ratio": res.makespan / max(total_work, 1e-12),
        "mem_util": mem_peak / max(max_mem, 1e-12),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", required=True, help="真机测量 JSON")
    ap.add_argument("--hw", required=True, help="校准后的硬件配置")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hw)
    with open(args.measured, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for r in data["plans"]:
        g = graph_from_dict(r["graph"])
        p = plan_from_dict(r["plan"])
        res = simulate(g, p, hw, collect_timeline=False)
        real_mk = r["makespan"]
        mk_ape = abs(res.makespan - real_mk) / max(abs(real_mk), 1e-12)
        log_ratio = np.log(max(res.makespan, 1e-12) / max(real_mk, 1e-12))
        peak_real = {int(k): v for k, v in r.get("peak_mem", {}).items()}
        pk_ape = 0.0
        for d in range(hw.n_devs):
            rr = peak_real.get(d, 0.0)
            ss = res.peak_mem.get(d, 0.0)
            pk_ape = max(pk_ape, abs(ss - rr) / max(abs(rr), 1e-12))
        feat = _features(res, g, p, hw)
        rows.append(
            {
                "name": r.get("name", ""),
                "makespan_real": real_mk,
                "makespan_sim": res.makespan,
                "mk_ape": float(mk_ape),
                "pk_ape": float(pk_ape),
                "log_ratio": float(log_ratio),
                **feat,
            }
        )

    log_ratios = [x["log_ratio"] for x in rows]
    mk_apes = [x["mk_ape"] for x in rows]
    pk_apes = [x["pk_ape"] for x in rows]
    median_ratio = float(np.exp(np.median(log_ratios)))
    std_log = float(np.std(log_ratios))

    feature_analysis = {}
    for feat in FEATURE_HINTS:
        vals = [x[feat] for x in rows]
        corr = _spearman(vals, log_ratios)
        med = float(np.median(vals))
        lo = [abs(x["log_ratio"]) for x in rows if x[feat] <= med]
        hi = [abs(x["log_ratio"]) for x in rows if x[feat] > med]
        feature_analysis[feat] = {
            "spearman_log_ratio": corr,
            "mean_abs_err_low": float(np.mean(lo)) if lo else 0.0,
            "mean_abs_err_high": float(np.mean(hi)) if hi else 0.0,
            "hint": FEATURE_HINTS[feat],
        }

    strong = [
        f for f, a in feature_analysis.items()
        if abs(a["spearman_log_ratio"]) > 0.5
    ]
    verdict = []
    if std_log < 0.2:
        verdict.append(
            f"误差分布集中（log std={std_log:.3f}）：以数值偏差为主，"
            f"建议全局效率修正 factor={median_ratio:.3f}"
        )
    else:
        verdict.append(
            f"误差分布分散（log std={std_log:.3f}）：存在结构性问题，"
            "按下方强相关特征定位模型机制"
        )
    for f in strong:
        verdict.append(f"强相关特征 [{f}]：{feature_analysis[f]['hint']}")

    report = {
        "hw": args.hw,
        "n_plans": len(rows),
        "makespan_mape": float(np.mean(mk_apes)),
        "peak_mem_mape": float(np.mean(pk_apes)),
        "log_ratio_std": std_log,
        "global_correction_factor": median_ratio,
        "feature_analysis": feature_analysis,
        "verdict": verdict,
        "per_plan": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"归因报告: {args.out}")
    print(f"  makespan MAPE = {report['makespan_mape']:.4f}  "
          f"peak MAPE = {report['peak_mem_mape']:.4f}  "
          f"log std = {std_log:.3f}  factor = {median_ratio:.3f}")
    for v in verdict:
        print(f"  - {v}")
    print("  特征 | Spearman(特征, log误差) | 低/高均值|log误差|")
    for f, a in feature_analysis.items():
        print(
            f"  {f:<22} {a['spearman_log_ratio']:>8.3f}  "
            f"{a['mean_abs_err_low']:>8.3f} / {a['mean_abs_err_high']:.3f}"
        )


if __name__ == "__main__":
    main()
