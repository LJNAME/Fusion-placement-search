"""端到端验证：真机跑一组方案，对比校准后的仿真器。

输入 measured.json（与标签同构，由你在真机上执行方案后填写）：
{
  "plans": [
    {
      "name": "plan-A",
      "graph": { ...graph_to_dict... },
      "plan":  { ...plan_to_dict... },
      "makespan": 0.123,
      "peak_mem": {"0": 1.2e10, "1": 1.1e10, "2": 1.3e10, "3": 1.0e10}
    }
  ]
}

运行：
  python calibration/validate_real.py \
      --hw configs/hygon_k100_4cards.calibrated.json \
      --measured measured.json --out validation_report.json

输出 makespan MAPE / Spearman、每设备峰值显存 MAPE，以及逐方案明细。
这是"最终裁决留在真机"的校准闭环收尾。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.execution import simulate
from simulator.hardware import load_hardware


def _spearman(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return 1.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _mape(a: List[float], b: List[float]) -> float:
    return float(np.mean([abs(x - y) / max(abs(y), 1e-12) for x, y in zip(a, b)]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True, help="校准后的硬件配置")
    ap.add_argument("--measured", required=True, help="真机测量 JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hw)
    with open(args.measured, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    mk_real, mk_sim = [], []
    pk_real, pk_sim = [], []
    for r in data["plans"]:
        g = graph_from_dict(r["graph"])
        p = plan_from_dict(r["plan"])
        res = simulate(g, p, hw, collect_timeline=False)
        real_mk = r["makespan"]
        sim_mk = res.makespan
        mk_real.append(real_mk)
        mk_sim.append(sim_mk)
        peak_real = {int(k): v for k, v in r.get("peak_mem", {}).items()}
        for d in range(hw.n_devs):
            pk_real.append(peak_real.get(d, 0.0))
            pk_sim.append(res.peak_mem.get(d, 0.0))
        rows.append(
            {
                "name": r.get("name", ""),
                "makespan_real": real_mk,
                "makespan_sim": sim_mk,
                "makespan_ape": abs(sim_mk - real_mk) / max(abs(real_mk), 1e-12),
                "peak_real": peak_real,
                "peak_sim": {str(d): v for d, v in res.peak_mem.items()},
                "overflow_sim": res.mem_overflow,
            }
        )

    report = {
        "hw": args.hw,
        "n_plans": len(rows),
        "makespan_mape": _mape(mk_sim, mk_real),
        "makespan_spearman": _spearman(mk_sim, mk_real),
        "peak_mem_mape": _mape(pk_sim, pk_real),
        "per_plan": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"验证报告: {args.out}")
    print(f"  makespan MAPE = {report['makespan_mape']:.4f}  "
          f"Spearman = {report['makespan_spearman']:.4f}")
    print(f"  峰值显存 MAPE = {report['peak_mem_mape']:.4f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
