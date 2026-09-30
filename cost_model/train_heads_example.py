"""训练 value / policy 头的示例脚本（cost 头仍由代价模型负责）。

关键点：同一个图有多份 plan，直接"一 plan 一样本"会把同一个部分状态标注成
互相矛盾的动作/回报，两个头只能学到多数类或均值。本脚本默认做两层清洗：

1. 只保留每个图最好的 top-k 个方案，并按质量加权
   w = exp(-(log(mk) - log(mk_best)) / tau)；
2. 按"动作前缀"聚合状态：同一个部分状态的动作目标 = 各方案权重归一化后的分布，
   value 目标 = 各方案 log(mk/mk_best) 的加权平均。

这两个目标与 MCTS 自博弈的 (visit 分布, 回报) 语义一致，后续换成自博弈数据时
只需替换 target 的生成方式。

用法（项目根目录）
------------------
  python cost_model/train_heads_example.py \
      --labels data/hygon_k100/labels.jsonl \
      --cost-model data/hygon_k100/compare \
      --outdir data/hygon_k100/heads \
      --top-k 4 --tau 0.25 --epochs 30

  # 想看旧行为（不做聚合）时加 --no-aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.gnn import load_gnn_model  # noqa: E402
from cost_model.search import PartialState  # noqa: E402
from cost_model.search_heads import (  # noqa: E402
    SearchNet,
    actions_from_plan,
    apply_action,
    build_partial_sample,
    collate,
    is_greedy_compatible,
    replay_trajectory,
    state_meta,
    value_target,
)
from simulator import (  # noqa: E402
    graph_from_dict,
    greedy_topological_order,
    plan_from_dict,
)


def load_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_graph(records: Sequence[Dict], val_ratio: float, seed: int):
    unique = sorted({r["graph_id"] for r in records})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    val_ids = set(unique[:n_val])
    tr = [r for r in records if r["graph_id"] not in val_ids]
    va = [r for r in records if r["graph_id"] in val_ids]
    return tr, va


def graph_ref_costs(records: Sequence[Dict]) -> Dict[int, float]:
    ref: Dict[int, float] = {}
    for r in records:
        gid = r["graph_id"]
        ref[gid] = min(ref.get(gid, float("inf")), float(r["makespan"]))
    return ref


def _make_item(
    graph,
    order: Sequence[str],
    state,
    vocab: Sequence[str],
    n_devs: int,
    target_dist: torch.Tensor,
    value_z: float,
    target_mix: float = 0.0,
) -> Dict:
    meta = state_meta(order, state, n_devs)
    meta["total_nodes"] = len(order)
    mask = torch.ones(1 + n_devs, dtype=torch.bool)
    if not meta["can_merge"]:
        mask[0] = False
    raw = target_dist.clamp(min=0.0) * mask.float()
    # 分层统计用"混合前"的真实动作数（混合后所有合法动作都有质量）
    meta["n_actions"] = int(((raw > 1e-8).sum()).item())
    dist = raw
    if float(dist.sum()) <= 0:
        dist = mask.float()
    dist = dist / dist.sum().clamp(min=1e-9)
    if target_mix > 0:
        uniform = mask.float() / mask.float().sum().clamp(min=1.0)
        dist = (1.0 - target_mix) * dist + target_mix * uniform
    return {
        "sample": build_partial_sample(graph, order, state, vocab, n_devs),
        "meta": meta,
        "target_dist": dist,
        "value_target": torch.tensor(float(value_z), dtype=torch.float32),
        "action_mask": mask,
    }


def _transform_z(z_raw: float, clip: float, mode: str) -> float:
    """value 目标压缩：探针方案可能比最优差很多（z 能到 2~3），
    log1p 压缩动态范围、clip 抑制离群，回归更容易学。"""

    z = max(float(z_raw), 0.0)
    if clip and clip > 0:
        z = min(z, float(clip))
    if mode == "log1p":
        return float(math.log1p(z))
    if mode == "none":
        return z
    raise ValueError(f"未知 value_transform: {mode}")


def build_head_dataset(
    records: Sequence[Dict],
    ref_costs: Dict[int, float],
    vocab: Sequence[str],
    n_devs: int,
    top_k: int = 4,
    tau: float = 0.25,
    aggregate: bool = True,
    require_compatible: bool = True,
    value_target_mode: str = "best",
    policy_beta: float = 0.1,
    target_mix: float = 0.1,
    value_clip: float = 2.0,
    value_transform: str = "log1p",
    cost_model=None,
    target_source: str = "sim",
    value_ref: str = "best",
) -> Tuple[List[Dict], Dict]:
    """把标签记录展开成 (部分状态, 动作分布, value 目标) 样本列表。

    require_compatible=True 时，只保留能被贪心拓扑序回放复现的 plan
    （random_plan 用随机拓扑序建组，和搜索/回放的顺序不一致，会污染标签）。
    """

    n_devs = int(n_devs)
    by_graph: Dict[int, List[Dict]] = defaultdict(list)
    for rec in records:
        by_graph[rec["graph_id"]].append(rec)

    stats = {
        "records": 0,
        "kept": 0,
        "skipped": 0,
        "skipped_by_name": Counter(),
        "graphs": 0,
        "graphs_with_supervision": 0,
        "graphs_without_supervision": 0,
    }
    items: List[Dict] = []
    for gid, recs in by_graph.items():
        stats["graphs"] += 1
        usable = []
        for rec in recs:
            graph = graph_from_dict(rec["graph"])
            plan = plan_from_dict(rec["plan"])
            order = greedy_topological_order(graph)
            stats["records"] += 1
            if require_compatible and not is_greedy_compatible(order, plan):
                stats["skipped"] += 1
                stats["skipped_by_name"][
                    str(getattr(plan, "name", "?")).rstrip("0123456789")
                ] += 1
                continue
            if target_source == "cm" and cost_model is not None:
                cost = float(math.exp(float(cost_model.predict(graph, plan)[0])))
            else:
                cost = float(rec["makespan"])
            stats["kept"] += 1
            usable.append(
                (rec, graph, plan, order, cost, str(getattr(plan, "name", "")).lower())
            )
        if not usable:
            stats["graphs_without_supervision"] += 1
            continue
        stats["graphs_with_supervision"] += 1
        # chain 方案（heuristic 参考）可能在 top-k 之外，先记住它
        chain_entry = next((e for e in usable if e[5].startswith("chain")), None)
        usable.sort(key=lambda e: e[4])
        if top_k and top_k > 0:
            usable = usable[:top_k]
        if value_ref == "heuristic" and chain_entry is not None:
            ref = chain_entry[4]
        else:
            ref = min(e[4] for e in usable)
        agg: Dict[Tuple, Dict] = {}
        for rec, graph, plan, order, cost, _name in usable:
            actions = actions_from_plan(order, plan, n_devs)
            if len(actions) != len(order):
                continue
            log_gap = math.log(cost) - math.log(ref)
            weight = math.exp(-log_gap / max(tau, 1e-6))
            z_raw = value_target(cost, ref)
            z = _transform_z(z_raw, value_clip, value_transform)
            traj, _final = replay_trajectory(order, actions, PartialState(), n_devs)
            if not aggregate:
                dist = torch.zeros(1 + n_devs)
                for state, action in traj:
                    dist.zero_()
                    dist[int(action)] = 1.0
                    items.append(
                        _make_item(
                            graph,
                            order,
                            state,
                            vocab,
                            n_devs,
                            dist.clone(),
                            z,
                            target_mix=target_mix,
                        )
                    )
                continue
            for t, (state, action) in enumerate(traj):
                key = (gid, tuple(actions[:t]))
                entry = agg.setdefault(
                    key,
                    {
                        "graph": graph,
                        "order": order,
                        "state": state,
                        "act_w": defaultdict(float),
                        "act_wz": defaultdict(float),
                        "act_minz": {},
                        "sum_w": 0.0,
                        "sum_wz": 0.0,
                    },
                )
                a = int(action)
                entry["act_w"][a] += weight
                entry["act_wz"][a] += weight * z
                entry["act_minz"][a] = min(entry["act_minz"].get(a, float("inf")), z)
                entry["sum_w"] += weight
                entry["sum_wz"] += weight * z
        if aggregate:
            for entry in agg.values():
                actions_sorted = sorted(entry["act_w"])
                # 动作价值：该动作后续能达到的最好回报（越小越好）
                scores = torch.tensor(
                    [entry["act_minz"][a] for a in actions_sorted]
                )
                if policy_beta and policy_beta > 0:
                    p = torch.softmax(-(scores - scores.min()) / policy_beta, dim=-1)
                else:
                    w = torch.tensor([entry["act_w"][a] for a in actions_sorted])
                    p = w / w.sum().clamp(min=1e-9)
                dist = torch.zeros(1 + n_devs)
                dist[torch.tensor(actions_sorted)] = p
                if value_target_mode == "best":
                    z = float(min(entry["act_minz"].values()))
                else:
                    z = entry["sum_wz"] / max(entry["sum_w"], 1e-9)
                items.append(
                    _make_item(
                        entry["graph"],
                        entry["order"],
                        entry["state"],
                        vocab,
                        n_devs,
                        dist,
                        z,
                        target_mix=target_mix,
                    )
                )
    return items, stats


def collate_train(items: Sequence[Dict]) -> Dict:
    batch = collate([it["sample"] for it in items])
    batch["meta"] = [it["meta"] for it in items]
    batch["target_dist"] = torch.stack([it["target_dist"] for it in items])
    batch["value_target"] = torch.stack([it["value_target"] for it in items])
    batch["action_mask"] = torch.stack([it["action_mask"] for it in items])
    return batch


def build_selfplay_dataset(
    rows: Sequence[Dict],
    graphs: Dict[int, Dict],
    vocab: Sequence[str],
    n_devs: int,
    target_mix: float = 0.0,
) -> List[Dict]:
    """把 run_mcts.py 导出的 (prefix, visit 分布, z) 转成训练样本。"""

    items: List[Dict] = []
    for row in rows:
        graph_dict = graphs.get(row["graph_id"])
        if graph_dict is None:
            continue
        graph = graph_from_dict(graph_dict)
        order = greedy_topological_order(graph)
        state = PartialState()
        ok = True
        for action in row["prefix"]:
            if state.i >= len(order):
                break
            try:
                state = apply_action(state, int(action), order, n_devs)
            except Exception:
                ok = False
                break
        if not ok:
            continue
        dist = torch.tensor(row["visits"], dtype=torch.float32)
        items.append(
            _make_item(
                graph,
                order,
                state,
                vocab,
                n_devs,
                dist,
                float(row["z"]),
                target_mix=target_mix,
            )
        )
    return items


@torch.no_grad()
def evaluate(
    net: SearchNet,
    loader,
    device,
    value_ref_mean: float = 0.0,
) -> Dict[str, float]:
    """验证指标：value MAE、policy top1/CE，并按目标熵分成 trivial/decision。

    - single：该状态只有一种动作出现过（标签近似 one-hot），top1 高不代表会决策；
    - multi：该状态有多个候选动作（真正的分叉点），才是考验策略的状态；
    - policy_ce_marginal：只预测"平均目标分布"的基线 CE（目标熵）。
    """

    net.eval()
    v_err, v_err_mean, hit, total, ce_sum = 0.0, 0.0, 0, 0, 0.0
    v_t_sum, v_t_sq = 0.0, 0.0
    groups = {
        "single": {"n": 0, "hit": 0, "ce": 0.0},
        "multi": {"n": 0, "hit": 0, "ce": 0.0},
    }
    dist_sum = None
    for batch in loader:
        batch_t = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        enc = net.encode(batch_t)
        v = net.value_batch(enc, batch_t["meta"])
        logits, mask = net.policy_batch(
            enc, batch_t["meta"], masks=batch_t["action_mask"]
        )
        target = batch_t["target_dist"]
        v_err += float((v - batch_t["value_target"]).abs().sum())
        v_err_mean += float((value_ref_mean - batch_t["value_target"]).abs().sum())
        v_t_sum += float(batch_t["value_target"].sum())
        v_t_sq += float((batch_t["value_target"] ** 2).sum())
        pred = logits.masked_fill(~mask, -1e9).argmax(dim=-1)
        target_best = target.masked_fill(~mask, 0.0).argmax(dim=-1)
        hits = (pred == target_best).float()
        logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
        ce = -(target * logp).sum(dim=-1)
        multi = torch.tensor(
            [int(m.get("n_actions", 1)) >= 2 for m in batch_t["meta"]],
            dtype=torch.bool,
            device=device,
        )
        hit += int(hits.sum())
        ce_sum += float(ce.sum())
        total += int(target.size(0))
        if dist_sum is None:
            dist_sum = target.sum(dim=0)
        else:
            dist_sum += target.sum(dim=0)
        for name, sel in (
            ("single", ~multi),
            ("multi", multi),
        ):
            if not bool(sel.any()):
                continue
            groups[name]["n"] += int(sel.sum())
            groups[name]["hit"] += int(hits[sel].sum())
            groups[name]["ce"] += float(ce[sel].sum())
    n = max(total, 1)
    if dist_sum is not None and float(dist_sum.sum()) > 0:
        pbar = dist_sum / dist_sum.sum()
        ce_marginal = float(-(pbar * torch.log(pbar.clamp(min=1e-12))).sum())
    else:
        ce_marginal = 0.0
    out = {
        "value_mae": v_err / n,
        "value_mae_meanpred": v_err_mean / n,
        "policy_top1": hit / n,
        "policy_ce": ce_sum / n,
        "policy_ce_marginal": ce_marginal,
        "n_multi": float(groups["multi"]["n"]),
        "value_target_mean": v_t_sum / n,
        "value_target_std": max(
            (v_t_sq / n - (v_t_sum / n) ** 2), 0.0
        )
        ** 0.5,
    }
    for name in ("single", "multi"):
        denom = max(groups[name]["n"], 1)
        out[f"policy_top1_{name}"] = groups[name]["hit"] / denom
        out[f"policy_ce_{name}"] = groups[name]["ce"] / denom
    return out


def majority_baseline(items: Sequence[Dict]) -> float:
    if not items:
        return 0.0
    targets = [
        int(it["target_dist"].argmax()) for it in items if it["target_dist"].sum() > 0
    ]
    if not targets:
        return 0.0
    counter = Counter(targets)
    return counter.most_common(1)[0][1] / len(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument(
        "--selfplay",
        default=None,
        help="用 run_mcts.py 导出的自博弈数据训练（需同时给 --labels 取图）",
    )
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--outdir", default="data/heads")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3, help="trunk 解冻后的学习率")
    ap.add_argument("--head-lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=4, help="每图只用最好的 k 个方案")
    ap.add_argument("--tau", type=float, default=0.25, help="方案质量权重温度")
    ap.add_argument(
        "--value-target",
        choices=("best", "mean"),
        default="best",
        help="value 目标：best=该状态后续可达的最优回报；mean=加权平均",
    )
    ap.add_argument(
        "--policy-beta",
        type=float,
        default=0.1,
        help="policy 目标温度：softmax(-动作价值差/beta)，<=0 时退回动作频率",
    )
    ap.add_argument(
        "--target-mix",
        type=float,
        default=0.1,
        help="policy 目标与均匀合法动作分布的混合系数（探索用）",
    )
    ap.add_argument(
        "--value-clip",
        type=float,
        default=2.0,
        help="value 目标 z 的上限（<=0 表示不裁剪）",
    )
    ap.add_argument(
        "--value-transform",
        choices=("log1p", "none"),
        default="log1p",
        help="value 目标变换：log1p 压缩动态范围；none 用原始 z",
    )
    ap.add_argument(
        "--target-source",
        choices=("sim", "cm"),
        default="sim",
        help="训练目标来源：sim=标签里的仿真 makespan；cm=cost 模型预测（与 MCTS 一致）",
    )
    ap.add_argument(
        "--value-ref",
        choices=("best", "heuristic"),
        default="best",
        help="value 参考：best=候选里最优；heuristic=chain 方案（与 MCTS 的 ref 一致）",
    )
    ap.add_argument("--no-aggregate", action="store_true", help="退回一 plan 一样本")
    ap.add_argument(
        "--keep-incompatible",
        action="store_true",
        help="保留随机拓扑序 plan（默认过滤，只用贪心序可复现的 plan）",
    )
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--unfreeze-trunk", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    records = load_records(args.labels)
    ref_costs = graph_ref_costs(records)
    cost_model, meta = load_gnn_model(args.cost_model, device="cpu")
    vocab, n_devs = meta["vocab"], int(meta["n_devs"])

    if args.selfplay:
        rows = load_records(args.selfplay)
        graphs = {r["graph_id"]: r["graph"] for r in records}
        gids = sorted({r["graph_id"] for r in rows})
        rng = np.random.RandomState(args.seed)
        rng.shuffle(gids)
        n_val = max(1, int(len(gids) * args.val_ratio))
        val_gids = set(gids[:n_val])
        train_rows = [r for r in rows if r["graph_id"] not in val_gids]
        val_rows = [r for r in rows if r["graph_id"] in val_gids]
        train_items = build_selfplay_dataset(
            train_rows, graphs, vocab, n_devs, target_mix=args.target_mix
        )
        val_items = build_selfplay_dataset(
            val_rows, graphs, vocab, n_devs, target_mix=args.target_mix
        )
        val_gids = sorted(val_gids)
        train_stats = val_stats = {
            "records": len(train_rows),
            "kept": len(train_items),
            "skipped": 0,
            "skipped_by_name": Counter(),
            "graphs": len(gids),
            "graphs_with_supervision": len(gids),
            "graphs_without_supervision": 0,
        }
    else:
        train_recs, val_recs = split_by_graph(records, args.val_ratio, args.seed)
        train_items, train_stats = build_head_dataset(
            train_recs,
            ref_costs,
            vocab,
            n_devs,
            top_k=args.top_k,
            tau=args.tau,
            aggregate=not args.no_aggregate,
            require_compatible=not args.keep_incompatible,
            value_target_mode=args.value_target,
            policy_beta=args.policy_beta,
            target_mix=args.target_mix,
            value_clip=args.value_clip,
            value_transform=args.value_transform,
            cost_model=cost_model,
            target_source=args.target_source,
            value_ref=args.value_ref,
        )
        val_items, val_stats = build_head_dataset(
            val_recs,
            ref_costs,
            vocab,
            n_devs,
            top_k=args.top_k,
            tau=args.tau,
            aggregate=not args.no_aggregate,
            require_compatible=not args.keep_incompatible,
            value_target_mode=args.value_target,
            policy_beta=args.policy_beta,
            target_mix=args.target_mix,
            value_clip=args.value_clip,
            value_transform=args.value_transform,
            cost_model=cost_model,
            target_source=args.target_source,
            value_ref=args.value_ref,
        )
        val_gids = sorted({r["graph_id"] for r in val_recs})
    base = majority_baseline(val_items)
    print(
        f"train samples={len(train_items)}  val samples={len(val_items)}  "
        f"majority_baseline={base:.4f}"
    )
    for split, st in (("train", train_stats), ("val", val_stats)):
        skip = st["skipped_by_name"].most_common(5)
        print(
            f"[{split}] plans kept={st['kept']}/{st['records']} "
            f"skipped(incompatible)={st['skipped']} "
            f"graphs_with_supervision={st['graphs_with_supervision']}/{st['graphs']} "
            f"top_skip={skip}"
        )

    train_loader = torch.utils.data.DataLoader(
        train_items, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train
    )
    val_loader = torch.utils.data.DataLoader(
        val_items, batch_size=args.batch_size, shuffle=False, collate_fn=collate_train
    )
    train_mean_value = float(
        torch.stack([it["value_target"] for it in train_items]).mean()
    )

    net = SearchNet(cost_model.model, mid=128, dropout=0.1)
    if not args.unfreeze_trunk:
        net.trunk.freeze_trunk(True)
    net = net.to(args.device)

    head_params = list(net.value.parameters()) + list(net.policy.parameters())
    groups = [{"params": head_params, "lr": args.head_lr}]
    if args.unfreeze_trunk:
        groups.append({"params": net.trunk.parameters(), "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    best = None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = 0.0
        for batch in train_loader:
            batch_t = {
                k: v.to(args.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            enc = net.encode(batch_t)
            v = net.value_batch(enc, batch_t["meta"])
            logits, mask = net.policy_batch(
                enc, batch_t["meta"], masks=batch_t["action_mask"]
            )
            v_loss = F.smooth_l1_loss(v, batch_t["value_target"])
            logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
            p_loss = (-(batch_t["target_dist"] * logp).sum(dim=-1)).mean()
            loss = args.value_weight * v_loss + args.policy_weight * p_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], 5.0
            )
            opt.step()
            total_loss += float(loss.detach()) * batch_t["target_dist"].size(0)

        metrics = evaluate(
            net, val_loader, args.device, value_ref_mean=train_mean_value
        )
        print(
            f"epoch {epoch:3d}  loss={total_loss / max(len(train_items), 1):.5f}  "
            f"val_value_mae={metrics['value_mae']:.4f}  "
            f"(mean-pred={metrics['value_mae_meanpred']:.4f})  "
            f"target={metrics['value_target_mean']:.3f}+-{metrics['value_target_std']:.3f}  "
            f"val_policy_top1={metrics['policy_top1']:.4f}  "
            f"top1_single={metrics['policy_top1_single']:.4f}  "
            f"top1_multi={metrics['policy_top1_multi']:.4f} "
            f"(n_multi={int(metrics['n_multi'])})  "
            f"ce={metrics['policy_ce']:.4f} "
            f"(marginal={metrics['policy_ce_marginal']:.4f})"
        )
        score = metrics["value_mae"] - metrics["policy_top1_multi"]
        if best is None or score < best:
            best = score
            torch.save(net.state_dict(), outdir / "heads_model.pt")
            (outdir / "heads_meta.json").write_text(
                json.dumps(
                    {
                        "vocab": list(vocab),
                        "n_devs": n_devs,
                        "hidden": net.trunk.hidden,
                        "mid": 128,
                        "trunk_frozen": not args.unfreeze_trunk,
                        "top_k": args.top_k,
                        "tau": args.tau,
                        "aggregate": not args.no_aggregate,
                        "cost_model_dir": str(args.cost_model),
                        "val_graph_ids": list(val_gids),
                        "metrics": {k: float(v) for k, v in metrics.items()},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(f"best score={best:.4f}  模型已保存到 {outdir}")


if __name__ == "__main__":
    main()
