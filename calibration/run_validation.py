"""组合层真机验证编排：生成执行器 -> 编译 -> 真机跑 -> 对比仿真器。

流程：
  1. gen_validation_plans.py    生成接近真实规模的 (图, 方案) 集
  2. gen_plan_executor.py       生成 HIP 执行器（本脚本内部调用）
  3. hipcc 编译
  4. 真机运行 -> measured.json（真机 makespan / 逐卡峰值显存）
  5. validate_real.py           与校准后的仿真器对比 -> validation_report.json

运行（海光 4 卡）：
  python calibration/run_validation.py \
      --plans validation_plans.json \
      --build-cmd "hipcc -O2 -o build/plan_executor_hip build/plan_executor_hip.cpp -lrocblas" \
      --binary build/plan_executor_hip \
      --hw configs/hygon_k100_4cards.calibrated.json \
      --measured measured.json --report validation_report.json

本地无卡自检（合成真机数据）：
  python calibration/run_validation.py --plans validation_plans.json \
      --hw configs/hygon_4cards.example.json \
      --measured /tmp/measured.json --report /tmp/validation_report.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import gen_plan_executor  # noqa: E402
from calibration import validate_real  # noqa: E402


def _parse_executor_output(text: str):
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True, help="validation_plans.json")
    ap.add_argument("--executor-out", default="build/plan_executor_hip.cpp")
    ap.add_argument("--build-cmd", default="")
    ap.add_argument("--binary", default="build/plan_executor_hip")
    ap.add_argument("--hw", required=True, help="校准后的硬件配置")
    ap.add_argument("--measured", required=True, help="输出的真机测量 JSON")
    ap.add_argument("--report", required=True, help="对比报告输出")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 1. 生成执行器
    gen_plan_executor.main(
        ["--plans", args.plans, "--out", args.executor_out, "--n-gpus", str(args.n_gpus)]
    )

    # 2+3. 编译并运行（或 dry-run 合成）
    if args.dry_run:
        with open(args.plans, encoding="utf-8") as f:
            plans = json.load(f)["plans"]
        rng = random.Random(args.seed)
        rows = []
        for i, entry in enumerate(plans):
            rows.append(
                {
                    "plan": entry["name"],
                    "makespan_s": 0.05 + rng.random() * 0.5,
                    "peak_mem": {
                        str(d): 4e9 + rng.random() * 2e9 for d in range(args.n_gpus)
                    },
                }
            )
        print("[dry-run] 使用合成真机数据，未调用执行器。")
    else:
        if args.build_cmd:
            subprocess.run(args.build_cmd, shell=True, check=True)
        try:
            out = subprocess.run(
                [args.binary], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"[validation] 执行器失败，stderr:\n{e.stderr or '(空)'}\n")
            raise
        rows = _parse_executor_output(out.stdout)
        # 内存探针：单独跑一遍（组边界同步+释放），只取峰值显存
        env = dict(os.environ)
        env["EXEC_MEM_PROBE"] = "1"
        out2 = subprocess.run(
            [args.binary], capture_output=True, text=True, check=True, env=env
        )
        rows2 = {r["plan"]: r for r in _parse_executor_output(out2.stdout)}
        for r in rows:
            if r["plan"] in rows2:
                r["peak_mem"] = rows2[r["plan"]]["peak_mem"]

    # 4. 合并成 measured.json（validate_real 的输入格式）
    with open(args.plans, encoding="utf-8") as f:
        plans = json.load(f)["plans"]
    by_name = {p["name"]: p for p in plans}
    measured = {"plans": []}
    for r in rows:
        src = by_name[r["plan"]]
        measured["plans"].append(
            {
                "name": r["plan"],
                "graph": src["graph"],
                "plan": src["plan"],
                "makespan": r["makespan_s"],
                "peak_mem": r["peak_mem"],
            }
        )
    with open(args.measured, "w", encoding="utf-8") as f:
        json.dump(measured, f, ensure_ascii=False, indent=2)

    # 5. 与仿真器对比
    validate_real.main(
        ["--hw", args.hw, "--measured", args.measured, "--out", args.report]
    )


if __name__ == "__main__":
    main()
