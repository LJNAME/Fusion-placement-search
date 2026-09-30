"""检查监督标签是否与搜索/回放用的拓扑序一致。

背景：random_plan() 用随机拓扑序建融合组，而 beam/MCTS 与
actions_from_plan 回放都用贪心拓扑序。若二者不一致，某个 plan 的
"动作前缀 -> makespan"就不是同一个方案，value/policy 头学到的标签是错的。

本脚本不需要 torch，直接跑在标签文件上：

  python cost_model/diagnose_head_labels.py --labels data/hygon_k100/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.plan_utils import is_greedy_compatible  # noqa: E402
from simulator import graph_from_dict, greedy_topological_order, plan_from_dict  # noqa: E402


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_family(name: str) -> str:
    return str(name).rstrip("0123456789") or "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--show-examples", type=int, default=5)
    args = ap.parse_args()

    records = load_records(args.labels)
    by_graph = defaultdict(list)
    for r in records:
        by_graph[r["graph_id"]].append(r)

    total = compatible = 0
    by_family_total, by_family_ok = Counter(), Counter()
    topk_total = topk_ok = 0
    no_good_supervision = []
    examples = []
    for gid, recs in by_graph.items():
        recs = sorted(recs, key=lambda r: float(r["makespan"]))
        top = recs[: args.top_k] if args.top_k else recs
        ok_in_topk = 0
        for r in recs:
            graph = graph_from_dict(r["graph"])
            plan = plan_from_dict(r["plan"])
            order = greedy_topological_order(graph)
            ok = is_greedy_compatible(order, plan)
            fam = plan_family(getattr(plan, "name", "?"))
            total += 1
            by_family_total[fam] += 1
            if ok:
                compatible += 1
                by_family_ok[fam] += 1
            elif len(examples) < args.show_examples:
                examples.append(
                    (
                        gid,
                        getattr(graph, "name", "?"),
                        getattr(plan, "name", "?"),
                        float(r["makespan"]),
                    )
                )
        for r in top:
            graph = graph_from_dict(r["graph"])
            plan = plan_from_dict(r["plan"])
            ok = is_greedy_compatible(greedy_topological_order(graph), plan)
            topk_total += 1
            topk_ok += int(ok)
            ok_in_topk += int(ok)
        if args.top_k and ok_in_topk == 0:
            no_good_supervision.append(gid)

    print(f"records={total}  greedy-compatible={compatible} ({compatible / max(total,1):.1%})")
    print("by plan family:")
    for fam, n in by_family_total.most_common():
        print(f"  {fam:10s} {by_family_ok[fam]:5d}/{n:<5d} ({by_family_ok[fam] / max(n,1):.1%})")
    print(
        f"top-{args.top_k} plans: {topk_ok}/{topk_total} compatible "
        f"({topk_ok / max(topk_total,1):.1%}); "
        f"graphs with no compatible top-k plan: {len(no_good_supervision)}/{len(by_graph)}"
    )
    if examples:
        print("incompatible examples (graph_id, graph, plan, makespan):")
        for e in examples:
            print("  ", e)


if __name__ == "__main__":
    main()
