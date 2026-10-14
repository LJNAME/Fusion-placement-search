"""两卡机上的层分配穷举：前 k 层给设备 0，其余给设备 1，逐个仿真取最优。

用法（项目根目录）：
  python cost_model/sweep_layer_split.py \
      --hw configs/hetero_a6000_titan.json --layers 28 \
      --d-model 1024 --seq-len 256 --batch 2 --ffn-dim 3072 \
      --out data/a6000_titan/sweep.json

为什么不用 beam 搜索：两卡、每卡一个 mesh（tp=cp=dp=1）时，"放置"只剩一维
（层怎么分给两张卡），穷举 k 就是精确解；beam_search 的 rollout 在 400+ 节点图上太慢。
融合粒度这里取"一层一个融合组"，于是跨卡只有层边界那一条边。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.run_hw_search import build_stacked_graph  # noqa: E402
from simulator import FusionGroup, Plan, load_hardware, plan_to_dict, simulate  # noqa: E402


def layer_nodes(graph, layer):
    return [nid for nid, node in graph.nodes.items() if node.extra.get("layer") == layer]


def make_plan(graph, layers, split_k, n_devs):
    groups = []
    for i in range(layers):
        dev = 0 if i < split_k else min(1, n_devs - 1)
        groups.append(FusionGroup(layer_nodes(graph, i), device=dev))
    return Plan(groups=groups, name=f"split-{split_k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ffn-dim", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    graph = build_stacked_graph(
        args.layers, args.d_model, args.seq_len, args.batch, args.ffn_dim, args.heads
    )
    print(f"hw={hw.name} devs={hw.n_devs}（{', '.join(d.name for d in hw.devices)}）")
    print(f"graph={graph.name} nodes={len(graph.nodes)} 层数={args.layers}")

    rows = []
    t0 = time.time()
    for k in range(args.layers + 1):
        plan = make_plan(graph, args.layers, k, hw.n_devs)
        res = simulate(graph, plan, hw, collect_timeline=False)
        busy = {int(i): float(v) for i, v in res.device_busy.items()}
        rows.append(
            {
                "k": k,
                "makespan": float(res.makespan),
                "steady": max(busy.values()) if busy else float("inf"),
                "device_busy": busy,
                "mem_overflow": bool(res.mem_overflow),
                "peak_mem": {int(i): float(v) for i, v in res.peak_mem.items()},
            }
        )
    dt = time.time() - t0
    best = min(rows, key=lambda r: r["makespan"])
    best_steady = min(rows, key=lambda r: r["steady"])
    worst = max(rows, key=lambda r: r["makespan"])

    print(f"--- 层分配扫描（{len(rows)} 个分法，{dt:.1f} s）---")
    print(f"{'k(卡0层数)':>10} {'makespan(s)':>12} {'稳态(s)':>10} {'卡0忙碌':>10} {'卡1忙碌':>10} {'峰值显存MB 0/1':>22} {'溢出':>5}")
    for r in rows:
        pk = r["peak_mem"]
        bs = r["device_busy"]
        print(f"{r['k']:>10} {r['makespan']:>12.6f} {r['steady']:>10.6f} {bs.get(0,0):>10.6f} {bs.get(1,0):>10.6f} "
              f"{pk.get(0,0)/1e6:>10.1f} /{pk.get(1,0)/1e6:>10.1f} {'是' if r['mem_overflow'] else '否':>6}")

    print("--- 结论 ---")
    print(f"[单次遍历口径] 卡0 拿前 {best['k']} 层、卡1 拿后 {args.layers - best['k']} 层；makespan={best['makespan']:.6f} s")
    print(f"  全放卡0 {rows[args.layers]['makespan']:.6f} s；全放卡1 {rows[0]['makespan']:.6f} s；最差 {worst['makespan']:.6f} s")
    print(f"[流水线稳态口径] 卡0 拿前 {best_steady['k']} 层、卡1 拿后 {args.layers - best_steady['k']} 层；稳态节拍={best_steady['steady']:.6f} s")
    print(f"  全放卡0 稳态 {rows[args.layers]['steady']:.6f} s；全放卡1 稳态 {rows[0]['steady']:.6f} s")
    print(f"  切分收益（稳态）：{rows[args.layers]['steady'] / best_steady['steady']:.3f}×")

    out = {
        "hw": {"name": hw.name, "n_devs": hw.n_devs, "devices": [d.name for d in hw.devices]},
        "graph": {"name": graph.name, "layers": args.layers, "nodes": len(graph.nodes)},
        "sweep": rows,
        "best": best,
        "best_steady": best_steady,
        "layers_per_device": {0: best["k"], 1: args.layers - best["k"]},
        "device_order": [d.name for d in hw.devices],
        "plan": plan_to_dict(make_plan(graph, args.layers, best["k"], hw.n_devs)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"写到 {args.out}")


if __name__ == "__main__":
    main()