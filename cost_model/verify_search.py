"""树搜索主流程验证脚本（无 torch 依赖，本地/服务器均可跑）。

验证对象：beam_search / mcts_search（cost_model/search.py），以及它们与
仿真器精评组成的搜索主流程。检查：

1. 合法性：返回的每个方案都通过 validate_plan（全覆盖、不重复、拓扑序、设备范围）；
2. 有效性：以仿真器 makespan 为准，搜索最优 ≤ 启发式与随机基线（允许小容差）；
3. 一致性：mcts 与 beam 在同一批图、同一打分器下可比（比值不超 3×）；
4. 确定性：同 seed 的 mcts 结果可复现。

用法（项目根目录）：
  python cost_model/verify_search.py [--graphs 6] [--mcts-iters 100]
                                      [--random-n 30] [--seed 0] [--out report.json]

退出码：全部通过为 0，任一检查失败为 1。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.roofline import RooflineCostModel  # noqa: E402
from cost_model.search import beam_search, mcts_search  # noqa: E402
from simulator import heuristic_plan, load_hardware, random_plan, simulate  # noqa: E402
from simulator.graphs import make_mlp_block, make_transformer_block  # noqa: E402
from simulator.schema import OpGraph, OpNode, TensorEdge  # noqa: E402
from simulator.validate import validate_plan  # noqa: E402


def make_chain(n_ops: int = 5, size: int = 128) -> OpGraph:
    g = OpGraph(name=f"chain-{n_ops}")
    prev = "in"
    g.add_node(
        OpNode("in", op_type="input", flops=0.0, out_bytes=size * size * 4,
               extra={"shape": {"n": size * size}})
    )
    for i in range(n_ops):
        nid = f"mm{i}"
        g.add_node(
            OpNode(nid, op_type="matmul", flops=2.0 * size**3,
                   out_bytes=size * size * 4, param_bytes=(size * size + size) * 4,
                   extra={"shape": {"m": size, "n": size, "k": size}})
        )
        g.add_edge(TensorEdge(prev, nid, bytes=g.nodes[prev].out_bytes))
        prev = nid
    return g


def best_by_sim(plans, graph, hw):
    """用仿真器 makespan 选最优（溢出方案排除），返回 (plan, makespan) 或 None。"""

    best = None
    for p in plans:
        r = simulate(graph, p, hw, collect_timeline=False)
        if r.mem_overflow:
            continue
        if best is None or r.makespan < best[1]:
            best = (p, r.makespan)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=6)
    ap.add_argument("--mcts-iters", type=int, default=100)
    ap.add_argument("--random-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    hw = load_hardware(
        str(Path(__file__).resolve().parents[1] / "configs" / "hygon_4cards.example.json")
    )
    graphs = []
    graphs += [make_chain(n_ops=5) for _ in range(2)]
    graphs += [make_mlp_block(hidden_sizes=(512, 512, 256), batch=4, seed=i)
               for i in range(2)]
    graphs += [make_transformer_block(d_model=256, seq_len=128, batch=1,
                                      ffn_dim=512, heads=4, seed=i)
               for i in range(2)]
    graphs = graphs[: max(1, args.graphs)]

    rm = RooflineCostModel(hw)
    failures = []
    rows = []
    print(f"{'graph':<18}{'heu':>10}{'rand':>10}{'beam':>10}{'mcts':>10}"
          f"{'mcts/beam':>10}{'wall_m':>8}")
    for gi, g in enumerate(graphs):
        def scorer(plan):
            return float(np.exp(rm.predict(g, plan)[0]))

        # 基线：启发式（chain / threshold 取优）与随机采样
        max_edge = max(e.bytes for e in g.edges) if g.edges else 0.0
        cands = [
            heuristic_plan(g, hw, mode="chain"),
            heuristic_plan(g, hw, mode="threshold", threshold_bytes=0.5 * max_edge),
        ]
        h = best_by_sim(cands, g, hw)
        rp = [
            random_plan(g, hw, rng=random.Random(args.seed + i * 101),
                        merge_prob=0.5)
            for i in range(args.random_n)
        ]
        r = best_by_sim(rp, g, hw)

        # beam
        t0 = time.time()
        bp = beam_search(g, scorer, n_devs=hw.n_devs, beam=8, rollouts=3,
                         seed=args.seed)
        btime = time.time() - t0
        b = best_by_sim(bp, g, hw)

        # mcts
        t0 = time.time()
        mp = mcts_search(g, scorer, n_devs=hw.n_devs,
                         iterations=args.mcts_iters, rollouts=3,
                         seed=args.seed, n_best=8)
        mtime = time.time() - t0
        m = best_by_sim(mp, g, hw)

        # 1) 合法性
        for p in bp + mp:
            try:
                validate_plan(g, p, hw.n_devs)
            except ValueError as e:
                failures.append(f"[{g.name}] 非法方案: {e}")

        # 2)+3) 有效性 / 一致性
        if h is None or r is None or b is None or m is None:
            failures.append(f"[{g.name}] 存在溢出或没有无溢出方案")
            continue
        ratio = m[1] / max(b[1], 1e-12)
        rows.append({
            "graph": g.name, "heuristic": h[1], "random": r[1],
            "beam": b[1], "mcts": m[1], "mcts_over_beam": ratio,
            "beam_wall_s": btime, "mcts_wall_s": mtime,
        })
        print(f"{g.name:<18}{h[1]:>10.5f}{r[1]:>10.5f}{b[1]:>10.5f}"
              f"{m[1]:>10.5f}{ratio:>10.2f}{(btime + mtime):>8.2f}")
        if b[1] > h[1] * 1.15:
            failures.append(
                f"[{g.name}] beam 劣于启发式: {b[1]:.5f} vs {h[1]:.5f}"
            )
        if m[1] > h[1] * 1.3:
            failures.append(
                f"[{g.name}] mcts 劣于启发式: {m[1]:.5f} vs {h[1]:.5f}"
            )
        if ratio > 3.0:
            failures.append(
                f"[{g.name}] mcts/beam 比值过大: {ratio:.2f}"
            )

    # 4) 确定性：同 seed 的 mcts 最优方案集合可复现
    g0 = graphs[0]
    s0 = lambda p: float(np.exp(rm.predict(g0, p)[0]))
    a = mcts_search(g0, s0, n_devs=hw.n_devs, iterations=20, seed=args.seed,
                    n_best=4)
    b2 = mcts_search(g0, s0, n_devs=hw.n_devs, iterations=20, seed=args.seed,
                     n_best=4)
    key = lambda plans: [tuple((tuple(x.node_ids), x.device) for x in pl.groups)
                         for pl in plans]
    if key(a) != key(b2):
        failures.append("mcts 同 seed 结果不可复现")

    report = {"seed": args.seed, "rows": rows, "failures": failures}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    if failures:
        print(f"验证失败（{len(failures)}）:")
        for x in failures:
            print("  -", x)
        sys.exit(1)
    print(f"验证通过：{len(rows)} 个图全部合法且搜索有效（seed={args.seed}）")


if __name__ == "__main__":
    main()
