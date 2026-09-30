"""扫描"仿真器（组合层）与 roofline（公式层）"的排序分歧。

对每个图族（链/菱形/Transformer/随机 DAG）生成一批方案，同时用
roofline 闭式模型和事件仿真器打分，报告：
  - makespan / 峰值显存的 Spearman 排序一致性（越低 = 分歧越大）；
  - top1 / top3 重叠（roofline 挑出的最优方案和仿真器是否一致）；
  - 分歧最大的反例（供人工检查"组合层改变了哪类排序"）。

纯 Python + numpy，无需 torch，本地可跑：
  python -B simulator/scan_differentiation.py \
      --hw configs/hetero_4cards.example.json --out scan.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

import numpy as np

# 允许从任意 cwd 直接运行：python simulator/scan_differentiation.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model.roofline import RooflineCostModel
from simulator.execution import simulate
from simulator.generators import (
    alternating_plan,
    generate_plans,
    random_plan,
)
from simulator.graphs import (
    make_chain_graph,
    make_diamond_graph,
    make_random_dag,
    make_transformer_block,
)
from simulator.hardware import load_hardware


def _spearman(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return 1.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _top_overlap(best_roo: List[str], best_sim: List[str], k: int) -> float:
    n = len(best_roo)
    if n == 0:
        return 1.0
    hit = 0
    for r, s in zip(best_roo, best_sim):
        hit += len(set(r[:k]) & set(s[:k])) / k
    return hit / n


def _make_family_graphs(family: str, n: int, seed: int, rng: random.Random):
    graphs = []
    for i in range(n):
        s = seed + i * 101
        if family == "chain":
            graphs.append(
                make_chain_graph(
                    n_ops=rng.randint(8, 18), seed=s
                )
            )
        elif family == "diamond":
            graphs.append(
                make_diamond_graph(
                    stages=rng.randint(3, 5),
                    branch_len=rng.randint(2, 4),
                    seed=s,
                )
            )
        elif family == "transformer":
            graphs.append(
                make_transformer_block(
                    d_model=rng.choice([512, 1024]),
                    seq_len=rng.choice([256, 512]),
                    batch=rng.choice([1, 2]),
                    ffn_dim=rng.choice([2048, 4096]),
                    heads=16,
                    seed=s,
                )
            )
        elif family == "dag":
            graphs.append(
                make_random_dag(
                    n_nodes=rng.randint(12, 22),
                    edge_prob=rng.uniform(0.2, 0.45),
                    seed=s,
                )
            )
        else:
            raise ValueError(f"未知图族: {family}")
    return graphs


def _collect_records(graphs, hw, n_random: int, rng: random.Random):
    rm = RooflineCostModel(hw)
    records = []
    for g in graphs:
        plans = generate_plans(
            g,
            hw,
            n_random=n_random,
            heuristic_modes=("all", "chain", "threshold"),
            seed=rng.randrange(2**31),
        )
        plans.append(alternating_plan(g, hw))
        for merge_prob in (0.25, 0.75):
            plans.append(
                random_plan(g, hw, rng, merge_prob=merge_prob, name=f"rand-{merge_prob}")
            )
        for p in plans:
            roo = rm.predict(g, p)
            res = simulate(g, p, hw, collect_timeline=False)
            records.append(
                {
                    "graph": g.name,
                    "plan": p.name,
                    "roo_mk": float(roo[0]),
                    "sim_mk": float(np.log(max(res.makespan, 1e-12))),
                    "roo_pk": float(roo[1]),
                    "sim_pk": float(
                        np.log(max(max(res.peak_mem.values(), default=0.0), 1e-12))
                    ),
                    "mem_overflow": res.mem_overflow,
                }
            )
    return records


def _summarize(family: str, records: List[dict]) -> dict:
    if not records:
        return {"n": 0}
    mk_r = [r["roo_mk"] for r in records]
    mk_s = [r["sim_mk"] for r in records]
    pk_r = [r["roo_pk"] for r in records]
    pk_s = [r["sim_pk"] for r in records]
    by_graph: Dict[str, List[dict]] = {}
    for r in records:
        by_graph.setdefault(r["graph"], []).append(r)
    best_roo, best_sim = [], []
    for gname, recs in by_graph.items():
        best_roo.append(sorted(recs, key=lambda x: x["roo_mk"])[:3])
        best_sim.append(sorted(recs, key=lambda x: x["sim_mk"])[:3])
    return {
        "n": len(records),
        "n_graphs": len(by_graph),
        "makespan_spearman": _spearman(mk_r, mk_s),
        "peak_mem_spearman": _spearman(pk_r, pk_s),
        "top1_overlap": _top_overlap(
            [[r["plan"] for r in x] for x in best_roo],
            [[r["plan"] for r in x] for x in best_sim],
            1,
        ),
        "top3_overlap": _top_overlap(
            [[r["plan"] for r in x] for x in best_roo],
            [[r["plan"] for r in x] for x in best_sim],
            3,
        ),
        "counterexamples": sorted(
            records,
            key=lambda x: abs(x["roo_mk"] - x["sim_mk"]),
            reverse=True,
        )[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True, help="硬件配置 JSON")
    ap.add_argument("--out", default="", help="可选：结果 JSON 输出路径")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-family", type=int, default=8)
    ap.add_argument("--n-plans", type=int, default=16)
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    rng = random.Random(args.seed)
    families = ["chain", "diamond", "transformer", "dag"]
    summary: Dict[str, dict] = {}
    for fam in families:
        graphs = _make_family_graphs(fam, args.n_per_family, args.seed, rng)
        records = _collect_records(graphs, hw, args.n_plans, rng)
        summary[fam] = _summarize(fam, records)

    overall = {
        "makespan_spearman": np.mean(
            [summary[f]["makespan_spearman"] for f in families]
        ),
        "peak_mem_spearman": np.mean(
            [summary[f]["peak_mem_spearman"] for f in families]
        ),
        "top1_overlap": np.mean([summary[f]["top1_overlap"] for f in families]),
        "top3_overlap": np.mean([summary[f]["top3_overlap"] for f in families]),
    }

    header = f"{'family':<12}{'n':>5}{'mk_spear':>10}{'pk_spear':>10}{'top1':>7}{'top3':>7}"
    print(header)
    print("-" * len(header))
    for fam in families:
        s = summary[fam]
        print(
            f"{fam:<12}{s['n']:>5}{s['makespan_spearman']:>10.3f}"
            f"{s['peak_mem_spearman']:>10.3f}{s['top1_overlap']:>7.3f}"
            f"{s['top3_overlap']:>7.3f}"
        )
    print("-" * len(header))
    o = overall
    print(
        f"{'overall':<12}{'':>5}{o['makespan_spearman']:>10.3f}"
        f"{o['peak_mem_spearman']:>10.3f}{o['top1_overlap']:>7.3f}"
        f"{o['top3_overlap']:>7.3f}"
    )

    print("\n分歧最大的反例（|log roofline - log sim|）：")
    for fam in families:
        for ce in summary[fam]["counterexamples"][:2]:
            print(
                f"  [{fam}] {ce['graph']}/{ce['plan']} "
                f"roo={ce['roo_mk']:.3f} sim={ce['sim_mk']:.3f} "
                f"diff={abs(ce['roo_mk'] - ce['sim_mk']):.3f}"
            )

    result = {"summary": summary, "overall": overall}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
