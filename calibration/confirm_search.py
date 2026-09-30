"""端到端搜索方案的真机确认。

把 eval_search.py 选出的方案（search_results.json 里带 graph/plan 的行）
在真实多卡上重跑，给出每个 beam 策略选方案的**真机 makespan** 对比：
  - 改善 vs 真机启发式基线（同一执行框架下相对比较，主机开销对所有方案一致）；
  - 仿真器自评 vs 真机的 MAPE（确认"自评"的水分）。

运行（海光 4 卡）：
  python calibration/confirm_search.py \
      --search data/search_results.json \
      --build-cmd "hipcc -O2 --offload-arch=gfx928 -o build/plan_executor_hip build/plan_executor_hip.cpp -lrocblas" \
      --binary build/plan_executor_hip \
      --out data/search_real.json

本地自检（合成真机数据）：
  python calibration/confirm_search.py --search data/search_results.json \
      --out /tmp/search_real.json --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import gen_plan_executor  # noqa: E402
from simulator.dataset import graph_from_dict, plan_from_dict  # noqa: E402
from simulator.execution import simulate  # noqa: E402
from simulator.hardware import load_hardware  # noqa: E402


def _parse_executor_output(text: str):
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return {r["plan"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True, help="eval_search 输出的 search_results.json")
    ap.add_argument("--executor-out", default="build/plan_executor_hip.cpp")
    ap.add_argument("--build-cmd", default="")
    ap.add_argument("--binary", default="build/plan_executor_hip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hw", required=True, help="校准后的硬件配置（用于无计算仿真基线）")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.search, encoding="utf-8") as f:
        data = json.load(f)
    rows = [r for r in data["rows"] if r.get("graph") and r.get("plan")]
    if not rows:
        raise SystemExit(
            "search_results.json 里没有方案（graph/plan）。"
            "请用更新后的 eval_search.py 重新生成（它会保存每个选出的方案）。"
        )

    plans = [
        {
            "name": f"{r['approach']}::{r['graph_id']}",
            "graph": r["graph"],
            "plan": r["plan"],
        }
        for r in rows
    ]
    tmp_plans = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_confirm_plans.json")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(tmp_plans, "w", encoding="utf-8") as f:
        json.dump({"plans": plans}, f, ensure_ascii=False)

    gen_plan_executor.main(
        ["--plans", tmp_plans, "--out", args.executor_out, "--n-gpus", str(args.n_gpus)]
    )

    if args.dry_run:
        rng = random.Random(args.seed)
        real = {}
        empty = {}
        for p in plans:
            real[p["name"]] = {
                "plan": p["name"],
                "makespan_s": 0.02 + rng.random() * 0.3,
                "peak_mem": {str(d): 1e9 for d in range(args.n_gpus)},
            }
            empty[p["name"]] = {
                "plan": p["name"],
                "makespan_s": 0.005 + rng.random() * 0.1,
                "peak_mem": {str(d): 1e9 for d in range(args.n_gpus)},
            }
        print("[dry-run] 合成真机数据")
    else:
        if args.build_cmd:
            subprocess.run(args.build_cmd, shell=True, check=True)
        try:
            out = subprocess.run(
                [args.binary], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"[confirm] 执行器失败，stderr:\n{e.stderr or '(空)'}\n")
            raise
        real = _parse_executor_output(out.stdout)
        env = dict(os.environ)
        env["EXEC_EMPTY_KERNELS"] = "1"
        out2 = subprocess.run(
            [args.binary], capture_output=True, text=True, check=True, env=env
        )
        empty = _parse_executor_output(out2.stdout)

    by_name = {p["name"]: p for p in plans}
    hw = load_hardware(args.hw)
    hw_nc = copy.deepcopy(hw)
    for d in hw_nc.devices:
        d.flops *= 1e6   # 计算≈0
        d.mem_bw *= 1e6  # IO≈0，只剩传输+调度
    hw_nc.launch_overhead = 0.0
    hw_nc.fusion_overhead = 0.0
    hw_nc.interference = 0.0

    stat = {}
    for r in rows:
        key = f"{r['approach']}::{r['graph_id']}"
        rr = real.get(key)
        er = empty.get(key)
        if rr is None or er is None:
            continue
        g = graph_from_dict(r["graph"])
        p = plan_from_dict(r["plan"])
        sim_nc = simulate(g, p, hw_nc, collect_timeline=False).makespan
        stat.setdefault(r["approach"], []).append(
            {
                "graph_id": r["graph_id"],
                "makespan_sim": r["makespan"],
                "makespan_real": rr["makespan_s"],
                "makespan_empty": er["makespan_s"],
                "sim_no_compute": sim_nc,
                "ape": abs(rr["makespan_s"] - r["makespan"])
                / max(abs(r["makespan"]), 1e-12),
            }
        )

    summary = {}
    for approach, vals in stat.items():
        mk_real = [v["makespan_real"] for v in vals]
        mk_sim = [v["makespan_sim"] for v in vals]
        mk_empty = [v["makespan_empty"] for v in vals]
        mk_nc = [v["sim_no_compute"] for v in vals]
        framework = [max(v["makespan_empty"] - v["sim_no_compute"], 0.0) for v in vals]
        compute_est = [v["makespan_real"] - v["makespan_empty"] for v in vals]
        summary[approach] = {
            "n": len(vals),
            "mean_real": float(np.mean(mk_real)),
            "mean_empty": float(np.mean(mk_empty)),
            "mean_compute_est": float(np.mean(compute_est)),
            "mean_framework_overhead": float(np.mean(framework)),
            "framework_share": float(np.mean(framework)) / max(np.mean(mk_real), 1e-12),
            "mean_sim": float(np.mean(mk_sim)),
            "mape_sim_vs_real": float(np.mean([v["ape"] for v in vals])),
        }
    if "heuristic" in summary:
        ref = summary["heuristic"]["mean_real"]
        for k in list(summary):
            if k.startswith("beam_"):
                summary[k]["improve_vs_heuristic_real"] = round(
                    1.0 - summary[k]["mean_real"] / max(ref, 1e-12), 4
                )

    result = {"summary": summary, "per_plan": stat}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    header = f"{'approach':<16}{'n':>4}{'real':>9}{'empty':>9}{'计算估计':>9}{'框架开销':>9}{'框架占比':>8}{'改善':>8}"
    print(header)
    print("-" * len(header))
    for approach, v in sorted(summary.items()):
        imp = v.get("improve_vs_heuristic_real", float("nan"))
        imp_s = f"{imp:>7.1%}" if "improve_vs_heuristic_real" in v else f"{'—':>7}"
        print(
            f"{approach:<16}{v['n']:>4}{v['mean_real']:>9.3f}"
            f"{v['mean_empty']:>9.3f}{v['mean_compute_est']:>9.3f}"
            f"{v['mean_framework_overhead']:>9.3f}{v['framework_share']:>8.1%}{imp_s}"
        )
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
