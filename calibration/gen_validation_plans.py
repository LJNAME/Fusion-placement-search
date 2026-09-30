"""生成"接近真实规模"的组合验证方案集。

产出 validation_plans.json：{plans: [{name, graph, plan}]}，
供 gen_plan_executor.py 生成真机可执行的 HIP 代码。
图是接近真实规模的 Transformer 块 / 深 MLP / 大 matmul 链 / 菱形，
方案覆盖 all / chain / threshold / alternating / random，让组合层的
调度、通信、显存效应都能被真机检验。

运行：
  python calibration/gen_validation_plans.py --out validation_plans.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.generators import alternating_plan, generate_plans, random_plan
from simulator.graphs import make_mlp_block, make_transformer_block
from simulator.hardware import load_hardware
from simulator.schema import OpGraph, OpNode, TensorEdge


def _make_matmul_chain(n_ops: int = 16, size: int = 2048, seed: int = 0) -> OpGraph:
    """大 matmul + gelu 交替链（接近真实 MLP 的单卡/多卡压力）。"""

    rng = random.Random(seed)
    g = OpGraph(name=f"mm-chain-{n_ops}")
    prev = "input"
    g.add_node(
        OpNode("input", op_type="input", flops=0.0, out_bytes=size * size * 4,
               extra={"shape": {"n": size * size}})
    )
    for i in range(n_ops):
        mm = f"mm{i}"
        g.add_node(
            OpNode(mm, op_type="matmul", flops=2.0 * size**3,
                   out_bytes=size * size * 4, param_bytes=(size * size + size) * 4,
                   extra={"shape": {"m": size, "n": size, "k": size}})
        )
        g.add_edge(TensorEdge(prev, mm, bytes=g.nodes[prev].out_bytes))
        prev = mm
        ge = f"ge{i}"
        g.add_node(
            OpNode(ge, op_type="gelu", flops=5.0 * size * size,
                   out_bytes=size * size * 4, extra={"shape": {"n": size * size}})
        )
        g.add_edge(TensorEdge(prev, ge, bytes=g.nodes[prev].out_bytes))
        prev = ge
    return g


def _make_diamond_ops(stages: int = 3, branch: int = 2, size: int = 1024, seed: int = 0) -> OpGraph:
    """菱形（split -> 双分支 matmul/gelu -> add 汇聚），压关键路径与通信。"""

    rng = random.Random(seed)
    g = OpGraph(name=f"diamond-ops-{stages}")
    prev: list = []
    for st in range(stages):
        split = f"s{st}"
        g.add_node(
            OpNode(split, op_type="gelu", flops=5.0 * size * size,
                   out_bytes=size * size * 4, extra={"shape": {"n": size * size}})
        )
        for p in prev:
            g.add_edge(TensorEdge(p, split, bytes=g.nodes[p].out_bytes))
        prev = [split]
        for br in range(branch):
            cur = split
            for k in range(branch):
                mm = f"b{st}_{br}_{k}_mm"
                g.add_node(
                    OpNode(mm, op_type="matmul", flops=2.0 * size**3,
                           out_bytes=size * size * 4,
                           param_bytes=(size * size + size) * 4,
                           extra={"shape": {"m": size, "n": size, "k": size}})
                )
                g.add_edge(TensorEdge(cur, mm, bytes=g.nodes[cur].out_bytes))
                cur = mm
            prev.append(cur)
        join = f"j{st}"
        g.add_node(
            OpNode(join, op_type="add", flops=2.0 * size * size,
                   out_bytes=size * size * 4, extra={"shape": {"n": size * size}})
        )
        for p in prev:
            g.add_edge(TensorEdge(p, join, bytes=g.nodes[p].out_bytes))
        prev = [join]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation_plans.json")
    ap.add_argument("--hw", default="configs/hygon_4cards.example.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-plans", type=int, default=3)
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    trans = make_transformer_block(
        d_model=2048, seq_len=1024, batch=2, ffn_dim=8192, heads=32, seed=1
    )
    # 把 layernorm/softmax 修正为按行展开：(rows, n)，供执行器并行归约
    for _nid, nd in trans.nodes.items():
        if nd.op_type == "layernorm":
            nd.extra["shape"] = {"n": 2048, "rows": 2 * 1024}
        elif nd.op_type == "softmax":
            nd.extra["shape"] = {"n": 1024, "rows": 2 * 1024}
    graphs = [
        ("transformer", trans),
        ("mlp", make_mlp_block(hidden_sizes=(4096, 16384, 4096), batch=128, seed=2)),
        ("mm-chain", _make_matmul_chain(n_ops=12, size=2048, seed=3)),
        ("diamond", _make_diamond_ops(stages=3, branch=2, size=1024, seed=4)),
    ]

    rng = random.Random(args.seed)
    plans_out = []
    for name, g in graphs:
        plans = generate_plans(
            g, hw, n_random=args.n_plans,
            heuristic_modes=("all", "chain", "threshold"),
            seed=rng.randrange(2**31),
        )
        plans.append(alternating_plan(g, hw))
        plans.append(random_plan(g, hw, rng, merge_prob=0.5, name="rand-0.5"))
        for p in plans:
            plans_out.append(
                {"name": f"{name}::{p.name}", "graph": graph_to_dict(g),
                 "plan": plan_to_dict(p)}
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"plans": plans_out}, f, ensure_ascii=False, indent=2)
    print(f"验证方案已生成: {args.out} ({len(plans_out)} 个)")


if __name__ == "__main__":
    main()
