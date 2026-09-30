"""把标签重生成到"贪心拓扑序可达"的方案空间（供 value/policy 头使用）。

做法：从现有 labels.jsonl 里取出所有图（按 graph_id 去重），用
simulator.generators_greedy 生成方案并跑仿真，输出新的 jsonl。
所有新方案的融合组都按贪心序连续分块，因此与 beam/MCTS 的决策空间一致。

用法（项目根目录）：

  python cost_model/regenerate_labels_greedy.py \
      --labels data/hygon_k100/labels.jsonl \
      --hw configs/hygon_k100_4cards.calibrated.json \
      --out data/hygon_k100/labels_greedy.jsonl \
      --n-random 8 --n-device-variants 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.plan_utils import is_greedy_compatible  # noqa: E402
from simulator import (  # noqa: E402
    graph_from_dict,
    graph_to_dict,
    greedy_topological_order,
    load_hardware,
    plan_to_dict,
    simulate,
)
from simulator.generators_greedy import generate_plans_greedy  # noqa: E402


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="现有标签（只用来取图）")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-device-variants", type=int, default=4)
    ap.add_argument("--n-perturb", type=int, default=8, help="单边界扰动方案数")
    ap.add_argument(
        "--no-probes",
        action="store_true",
        help="不加系统性探针方案（默认加：逐组换设备 + 逐边界合并/拆分）",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    records = load_records(args.labels)
    graphs = {}
    for r in records:
        gid = r["graph_id"]
        if gid not in graphs:
            graphs[gid] = graph_from_dict(r["graph"])

    out = []
    compatible = 0
    for gi, (gid, graph) in enumerate(sorted(graphs.items())):
        plans = generate_plans_greedy(
            graph,
            hw,
            n_random=args.n_random,
            n_device_variants=args.n_device_variants,
            n_perturb=args.n_perturb,
            seed=args.seed + gi,
            include_probes=not args.no_probes,
        )
        for plan in plans:
            res = simulate(graph, plan, hw, collect_timeline=False)
            ok = is_greedy_compatible(greedy_topological_order(graph), plan)
            compatible += int(ok)
            out.append(
                {
                    "graph_id": gid,
                    "graph": graph_to_dict(graph),
                    "plan": plan_to_dict(plan),
                    "makespan": res.makespan,
                    "peak_mem": {str(d): v for d, v in res.peak_mem.items()},
                    "mem_overflow": res.mem_overflow,
                    "overflow_devices": res.overflow_devices,
                    "num_groups": len(plan.groups),
                    "num_nodes": len(graph.nodes),
                    "generator": "greedy",
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"graphs={len(graphs)}  records={len(out)}  "
        f"greedy-compatible={compatible}/{len(out)}  -> {args.out}"
    )


if __name__ == "__main__":
    main()
