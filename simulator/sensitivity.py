"""参数敏感性分析：哪些硬件参数最影响方案排序（决定校准优先级）。

方法：以一份基线 config 跑一组 (图, 方案) 的仿真 makespan/峰值显存，
然后对每个参数做 ± 扰动，重算全部方案，比较扰动前后的：
  - makespan / 峰值显存 Spearman（排序一致性，越低 = 越敏感）；
  - top1 / top3 重叠（最优方案会不会变）；
  - 溢出翻转率（显存上限附近方案溢出状态是否改变）。

纯 Python + numpy，无需 torch，本地可跑（不需要真卡——这是仿真器端的
分析；跑完后按"最敏感参数优先"安排真卡上的微基准）：
  python simulator/sensitivity.py \
      --hw configs/hygon_4cards.example.json --out sensitivity.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from simulator.hardware import HardwareSpec, load_hardware


def _spearman(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return 1.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranked_names(recs: List[dict], key: str) -> List[str]:
    return [r["plan"] for r in sorted(recs, key=lambda x: x[key])]


def _top_overlap(a: List[List[str]], b: List[List[str]], k: int) -> float:
    if not a:
        return 1.0
    return sum(len(set(x[:k]) & set(y[:k])) / k for x, y in zip(a, b)) / len(a)


def _make_graphs(family: str, n: int, seed: int, rng: random.Random):
    graphs = []
    for i in range(n):
        s = seed + i * 101
        if family == "chain":
            graphs.append(make_chain_graph(n_ops=rng.randint(8, 18), seed=s))
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
    return graphs


def _collect(hw: HardwareSpec, graphs, n_plans: int, rng: random.Random):
    """返回 {graph_name: [plan_rec]}；plan_rec = {plan, mk, pk, overflow}。"""

    out: Dict[str, List[dict]] = {}
    for g in graphs:
        plans = generate_plans(
            g,
            hw,
            n_random=n_plans,
            heuristic_modes=("all", "chain", "threshold"),
            seed=rng.randrange(2**31),
        )
        plans.append(alternating_plan(g, hw))
        for mp in (0.25, 0.75):
            plans.append(random_plan(g, hw, rng, merge_prob=mp, name=f"rand-{mp}"))
        recs = []
        for p in plans:
            res = simulate(g, p, hw, collect_timeline=False)
            recs.append(
                {
                    "plan": p.name,
                    "mk": res.makespan,
                    "pk": max(res.peak_mem.values(), default=0.0),
                    "overflow": res.mem_overflow,
                }
            )
        out[g.name] = recs
    return out


def _metrics(base: Dict[str, List[dict]], pert: Dict[str, List[dict]]):
    mk_b = [r["mk"] for g in base.values() for r in g]
    mk_p = [r["mk"] for g in pert.values() for r in g]
    pk_b = [r["pk"] for g in base.values() for r in g]
    pk_p = [r["pk"] for g in pert.values() for r in g]
    top1 = _top_overlap(
        [_ranked_names(g, "mk") for g in base.values()],
        [_ranked_names(g, "mk") for g in pert.values()],
        1,
    )
    top3 = _top_overlap(
        [_ranked_names(g, "mk") for g in base.values()],
        [_ranked_names(g, "mk") for g in pert.values()],
        3,
    )
    ovfl_flip = 0.0
    n = 0
    for gname in base:
        for r0, r1 in zip(base[gname], pert[gname]):
            ovfl_flip += r0["overflow"] != r1["overflow"]
            n += 1
    return {
        "mk_spearman": _spearman(mk_b, mk_p),
        "pk_spearman": _spearman(pk_b, pk_p),
        "top1_overlap": top1,
        "top3_overlap": top3,
        "overflow_flip": ovfl_flip / max(n, 1),
        "disruption": 1.0 - _spearman(mk_b, mk_p),
    }


def _scale_dev(attr: str, factor: float) -> Callable[[HardwareSpec], None]:
    def f(hw: HardwareSpec) -> None:
        for d in hw.devices:
            setattr(d, attr, getattr(d, attr) * factor)

    return f


def _scale_links(factor: float) -> Callable[[HardwareSpec], None]:
    def f(hw: HardwareSpec) -> None:
        if hw.link_bw is not None:
            hw.link_bw *= factor
        if hw.link_matrix is not None:
            hw.link_matrix = [
                [None if v is None else v * factor for v in row]
                for row in hw.link_matrix
            ]

    return f


def _set_shape(key: str, val: float) -> Callable[[HardwareSpec], None]:
    def f(hw: HardwareSpec) -> None:
        sm = hw.extra.setdefault("shape_model", {})
        comp = sm.setdefault("matmul", {}).setdefault("compute", {})
        comp[key] = val

    return f


def _set_interference(v: float) -> Callable[[HardwareSpec], None]:
    def f(hw: HardwareSpec) -> None:
        hw.interference = v

    return f


def _sweeps() -> List[Tuple[str, Callable[[HardwareSpec], None]]]:
    return [
        ("flops x0.5", _scale_dev("flops", 0.5)),
        ("flops x1.5", _scale_dev("flops", 1.5)),
        ("mem_bw x0.5", _scale_dev("mem_bw", 0.5)),
        ("mem_bw x1.5", _scale_dev("mem_bw", 1.5)),
        ("link_bw x0.5", _scale_links(0.5)),
        ("link_bw x1.5", _scale_links(1.5)),
        ("comm_latency x10", lambda h: setattr(h, "comm_latency", h.comm_latency * 10)),
        ("comm_latency x0.1", lambda h: setattr(h, "comm_latency", h.comm_latency * 0.1)),
        ("launch_overhead x10", lambda h: setattr(h, "launch_overhead", h.launch_overhead * 10)),
        ("fusion_overhead x10", lambda h: setattr(h, "fusion_overhead", h.fusion_overhead * 10)),
        ("interference 0.0", _set_interference(0.0)),
        ("interference 0.3", _set_interference(0.3)),
        ("eta_max 0.6", _set_shape("eta_max", 0.6)),
        ("k_half 2048", _set_shape("k_half", 2048.0)),
        ("max_mem x0.8", _scale_dev("max_mem", 0.8)),
        ("max_mem x1.2", _scale_dev("max_mem", 1.2)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-family", type=int, default=6)
    ap.add_argument("--n-plans", type=int, default=12)
    args = ap.parse_args()

    base_hw = load_hardware(args.hw)
    rng = random.Random(args.seed)
    families = ["chain", "diamond", "transformer", "dag"]
    graphs = []
    for fam in families:
        graphs += _make_graphs(fam, args.n_per_family, args.seed, rng)

    base = _collect(base_hw, graphs, args.n_plans, rng)
    rows = []
    for name, apply_fn in _sweeps():
        hw = copy.deepcopy(base_hw)
        apply_fn(hw)
        pert = _collect(hw, graphs, args.n_plans, rng)
        m = _metrics(base, pert)
        m["sweep"] = name
        rows.append(m)

    rows.sort(key=lambda r: r["disruption"], reverse=True)
    header = (
        f"{'sweep':<22}{'mk_spear':>9}{'pk_spear':>9}{'top1':>7}"
        f"{'top3':>7}{'ovfl_flip':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['sweep']:<22}{r['mk_spearman']:>9.3f}{r['pk_spearman']:>9.3f}"
            f"{r['top1_overlap']:>7.3f}{r['top3_overlap']:>7.3f}"
            f"{r['overflow_flip']:>10.3f}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"baseline_hw": args.hw, "sweeps": rows}, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
