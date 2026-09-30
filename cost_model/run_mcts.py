"""跑 MCTS 并与 heuristic / random / beam 对比（真值用 simulator 复算）。

用法（项目根目录）：

  python cost_model/run_mcts.py \
      --labels data/hygon_k100/labels_greedy2.jsonl \
      --hw configs/hygon_k100_4cards.calibrated.json \
      --cost-model data/hygon_k100/compare \
      --heads data/hygon_k100/heads_value4 \
      --iters 200 --graphs 8 --beam 8 \
      --out data/hygon_k100/mcts_eval.json \
      --selfplay-out data/hygon_k100/selfplay.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.heads_io import load_search_net  # noqa: E402
from cost_model.mcts import MCTS  # noqa: E402
from cost_model.roofline import RooflineCostModel  # noqa: E402
from cost_model.search import beam_search  # noqa: E402
from simulator import (  # noqa: E402
    graph_from_dict,
    heuristic_plan,
    load_hardware,
    plan_to_dict,
    random_plan,
    simulate,
)


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def makespan_of(graph, plan, hw):
    res = simulate(graph, plan, hw, collect_timeline=False)
    return float(res.makespan), bool(res.mem_overflow)


def safe_makespan(graph, plan, hw):
    """跑仿真并容错：非法方案记 inf，不让整轮对比崩掉。"""

    try:
        mk, overflow = makespan_of(graph, plan, hw)
        return mk, overflow, None
    except Exception as exc:  # noqa: BLE001
        return float("inf"), True, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--graphs", type=int, default=8)
    ap.add_argument(
        "--graph-ids",
        default=None,
        help="逗号分隔的 graph_id 列表（优先于 --graphs 的排序截断）",
    )
    ap.add_argument(
        "--val-only",
        action="store_true",
        help="只跑 heads_meta.json 里记录的验证集图（用于泛化评估）",
    )
    ap.add_argument(
        "--exclude-val",
        action="store_true",
        help="排除验证集图（用于生成训练用自博弈数据）",
    )
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--beam-rollouts", type=int, default=3)
    ap.add_argument(
        "--beam-scorers",
        default="cost_model,roofline",
        help="逗号分隔：cost_model / roofline（可同时跑，便于公平对比）",
    )
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument(
        "--anchor-every",
        type=int,
        default=0,
        help="每 K 次迭代做一次终态锚定（0=关闭；value 为主 + 少量真实代价）",
    )
    ap.add_argument("--anchor-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selfplay-out", default=None)
    ap.add_argument(
        "--select-by",
        choices=("visits", "cost_model", "roofline", "blend"),
        default="visits",
        help="终态再排序：从访问最多的前 k 个完整方案里按该打分器重选",
    )
    ap.add_argument("--topk-terminal", type=int, default=8)
    ap.add_argument(
        "--completion-samples",
        type=int,
        default=2,
        help="每个前缀的采样补完次数（贪心之外的多样性来源）",
    )
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument(
        "--no-include-beam",
        action="store_true",
        help="不把 beam 的方案放进重排候选池",
    )
    ap.add_argument("--dump-plans", default=None, help="把选中的方案写成 JSON")
    ap.add_argument(
        "--diagnose",
        action="store_true",
        help="打印每个终态候选的方案访问数、模型/roofline 预测与仿真真值",
    )
    ap.add_argument(
        "--rerank-sim",
        action="store_true",
        help="用仿真器复核 top-k 候选并重排（离线评估/上界参考）",
    )
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    net, cost_model, meta = load_search_net(
        args.cost_model, args.heads, device=args.device
    )
    n_devs = int(meta["n_devs"])
    if n_devs != hw.n_devs:
        print(f"[warn] meta n_devs={n_devs} 与 hw n_devs={hw.n_devs} 不一致，用 hw")
        n_devs = hw.n_devs

    records = load_records(args.labels)
    graphs = {}
    for r in records:
        gid = r["graph_id"]
        if gid not in graphs:
            graphs[gid] = r["graph"]
    val_ids = set()
    heads_meta_path = Path(args.heads) / "heads_meta.json"
    if heads_meta_path.exists():
        heads_meta = json.loads(heads_meta_path.read_text(encoding="utf-8"))
        val_ids = set(int(x) for x in heads_meta.get("val_graph_ids", []))
    if not val_ids:
        # 兼容旧 checkpoint：按训练脚本 split_by_graph 的同一逻辑复算
        order = sorted(graphs)
        rng = np.random.RandomState(args.split_seed)
        rng.shuffle(order)
        n_val = max(1, int(len(order) * args.val_ratio))
        val_ids = set(order[:n_val])
    all_gids = sorted(graphs)
    if args.graph_ids:
        gids = [int(x) for x in args.graph_ids.split(",") if x.strip()]
        gids = [g for g in gids if g in graphs]
    elif args.val_only:
        gids = [g for g in all_gids if g in val_ids]
    else:
        gids = list(all_gids)
    if args.exclude_val:
        gids = [g for g in gids if g not in val_ids]
    gids = gids[: args.graphs]
    print(
        f"[split] graphs={len(gids)}  val_ids={sorted(val_ids)}  "
        f"using={gids[:8]}{'...' if len(gids) > 8 else ''}"
    )

    per_graph = []
    selfplay_rows = []
    dumped = []
    for gid in gids:
        graph = graph_from_dict(graphs[gid])
        row = {"graph_id": gid}

        base_plans = {
            "heuristic_chain": heuristic_plan(graph, hw, mode="chain"),
            "heuristic_threshold": heuristic_plan(graph, hw, mode="threshold"),
            "random": random_plan(graph, hw, name="random"),
        }
        for name, plan in base_plans.items():
            mk, overflow, err = safe_makespan(graph, plan, hw)
            row[name] = mk
            row[name + "_overflow"] = overflow
            if err:
                row[name + "_error"] = err

        roofline = RooflineCostModel(hw)
        beam_variants = {}
        beam_plan_pool = []
        for scorer_name in [s.strip() for s in args.beam_scorers.split(",") if s.strip()]:
            if scorer_name == "cost_model":
                scorer = lambda p: float(cost_model.predict(graph, p)[0])  # noqa: E731
            elif scorer_name == "roofline":
                scorer = lambda p: float(roofline.predict(graph, p)[0])  # noqa: E731
            else:
                raise ValueError(f"未知 beam scorer: {scorer_name}")
            t0 = time.time()
            beam_plans = beam_search(
                graph,
                scorer,
                n_devs,
                beam=args.beam,
                rollouts=args.beam_rollouts,
                merge_prob=0.5,
                seed=args.seed,
            )
            beam_plan_pool.extend(beam_plans)
            best_mk = float("inf")
            for plan in beam_plans:
                mk, overflow, _err = safe_makespan(graph, plan, hw)
                if not overflow and mk < best_mk:
                    best_mk = mk
            row[f"beam_{scorer_name}"] = best_mk
            row[f"beam_{scorer_name}_wall_s"] = time.time() - t0
            beam_variants[scorer_name] = best_mk
        row["beam"] = min(beam_variants.values())

        mcts = MCTS(
            net, cost_model, graph, hw, meta,
            device=args.device, c_puct=args.c_puct, seed=args.seed,
        )
        t0 = time.time()
        result = mcts.run(
            n_iters=args.iters,
            anchor_every=args.anchor_every,
            anchor_mix=args.anchor_mix,
        )
        row["mcts_wall_s"] = time.time() - t0
        chosen_plan = result["plan"]
        chosen_source = "visits"
        candidates = (
            mcts.candidate_plans(
                args.topk_terminal,
                samples_per_prefix=args.completion_samples,
            )
            if args.topk_terminal > 0
            else []
        )
        if not args.no_include_beam:
            for plan in beam_plan_pool:
                candidates.append({"prefix": [], "plan": plan, "visits": -1})
        visits_prefix = tuple(result["actions"])
        if not any(tuple(c["prefix"]) == visits_prefix for c in candidates):
            # visits 选中的方案不一定在最常访问的终态里，必须一起参与重排
            candidates.append(
                {
                    "prefix": list(visits_prefix),
                    "plan": result["plan"],
                    "visits": 0,
                }
            )
        # 去重（按方案结构）并截断
        deduped, seen_sigs = [], set()
        for cand in candidates:
            sig = tuple(
                (tuple(g.node_ids), int(g.device)) for g in cand["plan"].groups
            )
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            deduped.append(cand)
        candidates = deduped[: args.max_candidates]
        row["mcts_candidates"] = len(candidates)
        if args.rerank_sim and candidates:
            truths = []
            for cand in candidates:
                mk_i, overflow_i, _err_i = safe_makespan(graph, cand["plan"], hw)
                truths.append(float("inf") if overflow_i else mk_i)
            idx = int(np.argmin(truths))
            chosen_plan = candidates[idx]["plan"]
            chosen_source = f"sim#{idx}"
        elif args.select_by != "visits" and candidates:
            plans = [c["plan"] for c in candidates]
            cm = [float(cost_model.predict(graph, p)[0]) for p in plans]
            roo = [float(roofline.predict(graph, p)[0]) for p in plans]
            if args.select_by == "cost_model":
                scores = cm
            elif args.select_by == "roofline":
                scores = roo
            else:  # blend：模型与解析分歧大时更保守
                base_cm, base_roo = min(cm), min(roo)
                scores = [
                    0.5 * (cm[i] - base_cm) + 0.5 * (roo[i] - base_roo)
                    for i in range(len(plans))
                ]
            idx = int(np.argmin(scores))
            chosen_plan = plans[idx]
            chosen_source = f"{args.select_by}#{idx}"
        row["mcts_select"] = args.select_by
        row["mcts_chosen"] = chosen_source
        row["mcts_pred_log_mk"] = float(cost_model.predict(graph, chosen_plan)[0])
        mk, overflow, err = safe_makespan(graph, chosen_plan, hw)
        row["mcts"] = mk
        row["mcts_overflow"] = overflow
        row["mcts_cost_calls"] = result["n_cost_calls"]
        if err:
            row["mcts_error"] = err
        if args.dump_plans:
            dumped.append(
                {
                    "graph_id": gid,
                    "select": chosen_source,
                    "pred_log_mk": row["mcts_pred_log_mk"],
                    "makespan": mk,
                    "plan": plan_to_dict(chosen_plan),
                }
            )
        if args.diagnose and candidates:
            print(f"  [diagnose] graph {gid} candidates:")
            for i, cand in enumerate(candidates):
                pred_cm = float(cost_model.predict(graph, cand["plan"])[0])
                pred_roo = float(roofline.predict(graph, cand["plan"])[0])
                true_mk, overflow, _err = safe_makespan(graph, cand["plan"], hw)
                mark = " <== chosen" if cand["plan"] is chosen_plan else ""
                print(
                    f"    #{i} visits={cand['visits']:4d} "
                    f"cm={pred_cm:+.3f} roo={pred_roo:+.3f} "
                    f"true={true_mk:.4f} overflow={overflow}{mark}"
                )

        base = row["heuristic_chain"]
        for name in list(beam_variants) + ["beam", "mcts", "random"]:
            key = f"beam_{name}" if name in beam_variants else name
            value = row.get(key, float("inf"))
            if base > 0 and np.isfinite(value):
                row[key + "_impr"] = (base - value) / base
            else:
                row[key + "_impr"] = -1.0
        per_graph.append(row)

        if args.selfplay_out:
            selfplay_rows.extend(mcts.export_selfplay(result, gid))
        beam_str = " ".join(
            f"beam[{name}]={row[f'beam_{name}']:.4f}" for name in beam_variants
        )
        print(
            f"graph {gid}: heur={base:.4f} {beam_str} beam_best={row['beam']:.4f} "
            f"mcts={row['mcts']:.4f} (mcts_impr={row['mcts_impr']:+.1%}, "
            f"sel={row.get('mcts_chosen')})"
        )

    def mean(key):
        vals = [
            r[key]
            for r in per_graph
            if key in r and np.isfinite(r[key])
        ]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "n_graphs": len(per_graph),
        "iters": args.iters,
        "beam_width": args.beam,
        "beam_rollouts": args.beam_rollouts,
        "makespan": {
            k: mean(k)
            for k in (
                "heuristic_chain",
                "heuristic_threshold",
                "random",
                "beam_cost_model",
                "beam_roofline",
                "beam",
                "mcts",
            )
            if any(k in r for r in per_graph)
        },
        "improvement_vs_heuristic_chain": {
            k: mean(k + "_impr")
            for k in (
                "beam_cost_model",
                "beam_roofline",
                "beam",
                "mcts",
                "random",
            )
            if any(k + "_impr" in r for r in per_graph)
        },
        "mcts_wins_vs_beam": int(
            sum(1 for r in per_graph if r["mcts"] < r["beam"])
        ),
        "mean_wall_s": {
            k: mean(k)
            for k in ("beam_cost_model_wall_s", "beam_roofline_wall_s", "mcts_wall_s")
            if any(k in r for r in per_graph)
        },
        "mean_mcts_cost_calls": mean("mcts_cost_calls"),
    }
    out = {"summary": summary, "per_graph": per_graph}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print("结果已保存:", args.out)

    if args.selfplay_out:
        Path(args.selfplay_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.selfplay_out, "w", encoding="utf-8") as f:
            for row in selfplay_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"自博弈样本 {len(selfplay_rows)} 条 -> {args.selfplay_out}")

    if args.dump_plans:
        Path(args.dump_plans).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_plans).write_text(
            json.dumps(dumped, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"选中方案 {len(dumped)} 个 -> {args.dump_plans}")


if __name__ == "__main__":
    main()
