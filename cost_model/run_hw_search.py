"""在给定硬件配置上跑一次「融合 + 放置」搜索，产出方案、仿真指标与层分配。

用法（项目根目录）：
  python cost_model/run_hw_search.py \
      --hw configs/hetero_a6000_titan.json \
      --layers 28 --d-model 1024 --seq-len 256 --batch 2 --ffn-dim 3072 \
      --beam 8 --seed 0 --out data/a6000_titan/search.json

图用 N 个 Transformer 块串联合成，每个算子带 extra["layer"]（第几层），
这样下游才能把「算子→设备」聚合成「层→设备」。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.roofline import RooflineCostModel  # noqa: E402
from cost_model.search import beam_search  # noqa: E402
from simulator import (  # noqa: E402
    OpGraph,
    OpNode,
    TensorEdge,
    heuristic_plan,
    load_hardware,
    plan_to_dict,
    random_plan,
    simulate,
)
from simulator.graphs import make_transformer_block  # noqa: E402


def build_stacked_graph(layers, d_model, seq_len, batch, ffn_dim, heads):
    g = OpGraph(name=f"transformer-{layers}L")
    prev_out = None
    for i in range(layers):
        blk = make_transformer_block(
            d_model=d_model, seq_len=seq_len, batch=batch, ffn_dim=ffn_dim, heads=heads, seed=i
        )
        for nid, node in blk.nodes.items():
            extra = dict(node.extra)
            extra["layer"] = i
            g.add_node(
                OpNode(
                    f"L{i}_{nid}",
                    op_type=node.op_type,
                    flops=node.flops,
                    out_bytes=node.out_bytes,
                    param_bytes=node.param_bytes,
                    extra=extra,
                )
            )
        for e in blk.edges:
            g.add_edge(TensorEdge(f"L{i}_{e.src}", f"L{i}_{e.dst}", bytes=e.bytes))
        if prev_out is not None:
            g.add_edge(TensorEdge(prev_out, f"L{i}_input"))
        prev_out = f"L{i}_ln2"
    return g


def evaluate(graph, plan, hw):
    res = simulate(graph, plan, hw, collect_timeline=False)
    return float(res.makespan), bool(res.mem_overflow), {int(k): float(v) for k, v in res.peak_mem.items()}


def layer_assignment(graph, plan):
    per_layer = {}
    for g in plan.groups:
        for nid in g.node_ids:
            per_layer.setdefault(graph.nodes[nid].extra.get("layer", -1), set()).add(int(g.device))
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ffn-dim", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--random-n", type=int, default=5)
    ap.add_argument("--require-all-devices", action="store_true", help="每张卡至少一个算子（否则搜索可能把全部算子放一张卡）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    graph = build_stacked_graph(
        args.layers, args.d_model, args.seq_len, args.batch, args.ffn_dim, args.heads
    )
    print(f"hw={hw.name} devs={hw.n_devs} graph={graph.name} nodes={len(graph.nodes)}")

    cost = RooflineCostModel(hw)
    UNUSED_PENALTY = 1e3

    def scorer(plan):
        v = float(cost.predict(graph, plan)[0])
        if args.require_all_devices and len({g.device for g in plan.groups}) < hw.n_devs:
            v += UNUSED_PENALTY
        return v


    t0 = time.time()
    plans = beam_search(graph, scorer, n_devs=hw.n_devs, beam=args.beam, seed=args.seed)
    search_s = time.time() - t0

    scored = []
    for p in plans:
        mk, ovf, peak = evaluate(graph, p, hw)
        scored.append((mk, ovf, peak, p))
    scored.sort(key=lambda r: r[0])
    best_mk, best_ovf, best_peak, best = scored[0]

    base = heuristic_plan(graph, hw, mode="chain")
    base_mk, base_ovf, _ = evaluate(graph, base, hw)

    rng = random.Random(args.seed)
    rnd_mk = min(evaluate(graph, random_plan(graph, hw, rng=rng), hw)[0] for _ in range(args.random_n))

    per_layer = layer_assignment(graph, best)
    span = {L: sorted(d) for L, d in per_layer.items() if len(d) > 1}
    counts = {}
    for L, devs in per_layer.items():
        counts[devs.copy().pop() if len(devs) == 1 else -1] = counts.get(devs.copy().pop() if len(devs) == 1 else -1, 0) + 1

    print("--- 结果（makespan 越小越好）---")
    print(f"搜索最优   {best_mk:.6f} s   overflow={best_ovf}")
    print(f"启发式链式 {base_mk:.6f} s   overflow={base_ovf}")
    print(f"随机最优   {rnd_mk:.6f} s   (n={args.random_n})")
    print(f"搜索加速比 {base_mk / best_mk:.3f}×（vs 启发式）")
    print(f"每卡层数   {counts}   跨卡层 {len(span)} 层（>0 表示该层算子被切开）")
    for L in sorted(span):
        print(f"  第 {L} 层跨卡：devices={span[L]}")

    out = {
        "hw": {"name": hw.name, "n_devs": hw.n_devs, "devices": [d.name for d in hw.devices]},
        "graph": {"name": graph.name, "layers": args.layers, "nodes": len(graph.nodes), "edges": len(graph.edges)},
        "search": {"beam": args.beam, "seed": args.seed, "time_s": search_s, "n_candidates": len(plans), "require_all_devices": bool(args.require_all_devices)},
        "metrics": {
            "search_best": {"makespan": best_mk, "mem_overflow": best_ovf, "peak_mem": best_peak},
            "heuristic_chain": {"makespan": base_mk, "mem_overflow": base_ovf},
            "random_best": {"makespan": rnd_mk, "n": args.random_n},
        },
        "layers_per_device": counts,
        "cross_device_layers": {str(L): span[L] for L in sorted(span)},
        "plan": plan_to_dict(best),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"写到 {args.out}")


if __name__ == "__main__":
    main()