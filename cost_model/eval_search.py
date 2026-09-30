"""端到端搜索评估：不同代价模型驱动 beam 搜索，最终方案由仿真器精评。

对比项：启发式（无搜索）、随机基线、roofline 打分、Ridge 打分、GNN 打分
（后两者需要先用 compare_models.py 训练）。

用法（在项目根目录）：
  python cost_model/eval_search.py \
      --hw data/hw.json \
      --extrap-labels data/labels_extrapolation.jsonl \
      --ridge-model data/compare/ridge_model.json \
      --gnn-model data/compare \
      --out data/search_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.baseline import RidgeLogRegressor  # noqa: E402
from cost_model.predict import RidgeCostModel  # noqa: E402
from cost_model.roofline import RooflineCostModel  # noqa: E402
from cost_model.search import beam_search, mcts_search  # noqa: E402
from cost_model.split import load_records, records_by_graph, split_by_graph  # noqa: E402
from simulator import (  # noqa: E402
    graph_to_dict,
    graph_from_dict,
    heuristic_plan,
    load_hardware,
    plan_to_dict,
    random_plan,
    simulate,
)


def best_sim_result(graph, plan, hw):
    r = simulate(graph, plan, hw, collect_timeline=False)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True)
    ap.add_argument("--labels", default=None, help="同分布标签（--eval-set test 时需要）")
    ap.add_argument("--extrap-labels", default=None, help="外推标签（--eval-set extrap 时需要）")
    ap.add_argument("--eval-set", choices=["extrap", "test"], default="extrap")
    ap.add_argument("--ridge-model", default=None)
    ap.add_argument("--gnn-model", default=None, help="compare_models 输出目录")
    ap.add_argument("--n-graphs", type=int, default=5)
    ap.add_argument(
        "--search-algo",
        choices=["beam", "mcts"],
        default="beam",
        help="搜索算法：beam（宽度受限，默认）或 mcts（AlphaZero 式 UCB 树搜索）",
    )
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--merge-prob", type=float, default=0.5)
    ap.add_argument("--mcts-iters", type=int, default=500, help="MCTS 迭代轮数")
    ap.add_argument("--mcts-c", type=float, default=1.5, help="MCTS 探索系数 C")
    ap.add_argument("--random-n", type=int, default=50, help="随机基线采样数")
    ap.add_argument("--threshold-factor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/search_results.json")
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    if args.eval_set == "extrap":
        if not args.extrap_labels:
            raise SystemExit("--eval-set extrap 需要 --extrap-labels")
        eval_records = load_records(args.extrap_labels)
    else:
        if not args.labels:
            raise SystemExit("--eval-set test 需要 --labels")
        _, _, test = split_by_graph(load_records(args.labels), seed=args.seed)
        eval_records = test

    items = list(records_by_graph(eval_records).items())
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.n_graphs]

    models = {"roofline": RooflineCostModel(hw)}
    if args.ridge_model:
        models["ridge"] = RidgeCostModel(RidgeLogRegressor.load(args.ridge_model), hw)
    if args.gnn_model:
        import torch

        from cost_model.gnn import load_gnn_model  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gnn_model, _ = load_gnn_model(args.gnn_model, device=device)
        models["gnn"] = gnn_model

    def make_scorer(model):
        return lambda plan: float(np.exp(model.predict(graph, plan)[0]))

    rows = []
    for gid, record in items:
        graph = graph_from_dict(record["graph"])
        max_edge_bytes = max(
            (e.get("bytes") or graph.nodes[e["src"]]["out_bytes"])
            for e in record["graph"]["edges"]
        ) if record["graph"]["edges"] else 0.0

        # 参考基线：启发式（chain + threshold 取优）与随机采样（保留方案供真机确认）
        refs = {}
        candidates = [
            heuristic_plan(graph, hw, mode="chain"),
            heuristic_plan(
                graph, hw, mode="threshold",
                threshold_bytes=args.threshold_factor * max_edge_bytes,
            ),
        ]
        best_plan, best = min(
            ((p, best_sim_result(graph, p, hw)) for p in candidates),
            key=lambda ps: (ps[1].mem_overflow, ps[1].makespan),
        )
        refs["heuristic"] = (best, best_plan)

        best_random_plan = None
        best_random = None
        for i in range(args.random_n):
            p = random_plan(graph, hw, rng=random.Random(args.seed + i * 101),
                            merge_prob=args.merge_prob)
            r = best_sim_result(graph, p, hw)
            if best_random is None or (r.mem_overflow, r.makespan) < (
                best_random.mem_overflow, best_random.makespan
            ):
                best_random = r
                best_random_plan = p
        refs["random"] = (best_random, best_random_plan)

        for ref_name, (ref, ref_plan) in refs.items():
            rows.append(
                {
                    "graph_id": gid,
                    "approach": ref_name,
                    "makespan": ref.makespan,
                    "peak_mem": max(ref.peak_mem.values()),
                    "overflow": ref.mem_overflow,
                    "wall_s": 0.0,
                    "graph": graph_to_dict(graph),
                    "plan": plan_to_dict(ref_plan),
                }
            )

        for mname, model in models.items():
            scorer = make_scorer(model)
            t0 = time.time()
            if args.search_algo == "mcts":
                plans = mcts_search(
                    graph, scorer, n_devs=hw.n_devs,
                    iterations=args.mcts_iters, rollouts=args.rollouts,
                    merge_prob=args.merge_prob, seed=args.seed,
                    C=args.mcts_c, n_best=args.beam,
                )
            else:
                plans = beam_search(
                    graph, scorer, n_devs=hw.n_devs, beam=args.beam,
                    rollouts=args.rollouts, merge_prob=args.merge_prob,
                    seed=args.seed,
                )
            wall = time.time() - t0
            sims = [best_sim_result(graph, p, hw) for p in plans]
            valid = [(p, s) for p, s in zip(plans, sims) if not s.mem_overflow]
            if valid:
                best_plan, best = min(valid, key=lambda ps: ps[1].makespan)
            else:
                best_plan, best = min(
                    zip(plans, sims), key=lambda ps: ps[1].makespan
                )
            rows.append(
                {
                    "graph_id": gid,
                    "approach": f"beam_{mname}",
                    "makespan": best.makespan,
                    "peak_mem": max(best.peak_mem.values()),
                    "overflow": best.mem_overflow,
                    "wall_s": round(wall, 4),
                    "graph": graph_to_dict(graph),
                    "plan": plan_to_dict(best_plan),
                }
            )
    print(f"graph {gid}: 完成（{len(plans)} 个搜索方案精评）")

    # 汇总
    summary = {}
    for approach in sorted({r["approach"] for r in rows}):
        vals = [r for r in rows if r["approach"] == approach]
        mk = [r["makespan"] for r in vals]
        summary[approach] = {
            "n": len(mk),
            "mean_makespan": round(float(np.mean(mk)), 6),
            "median_makespan": round(float(np.median(mk)), 6),
            "overflow_rate": round(sum(1 for r in vals if r["overflow"]) / max(len(vals), 1), 3),
            "mean_wall_s": round(float(np.mean([r["wall_s"] for r in vals])), 4),
        }
    if "heuristic" in summary:
        ref = summary["heuristic"]["mean_makespan"]
        for k in list(summary):
            if k.startswith("beam_"):
                summary[k]["improve_vs_heuristic"] = round(
                    1.0 - summary[k]["mean_makespan"] / max(ref, 1e-12), 4
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, ensure_ascii=False, indent=2)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        csv_rows = [
            {k: v for k, v in r.items() if k not in ("graph", "plan")} for r in rows
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=list(csv_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n汇总（makespan 越小越好）：")
    for k, v in summary.items():
        extra = f"  改善 {v['improve_vs_heuristic']:.1%}" if "improve_vs_heuristic" in v else ""
        print(
            f"  {k:<14} mean={v['mean_makespan']:.6f}s  "
            f"overflow={v['overflow_rate']:.1%}  wall={v['mean_wall_s']:.3f}s{extra}"
        )


if __name__ == "__main__":
    main()
