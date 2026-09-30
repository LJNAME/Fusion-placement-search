"""三模型统一对比：roofline（解析式）/ Ridge（手写特征）/ GNN（结构感知）。

同一划分、同一随机种子、多 seed 可重复。GNN 需要 torch（服务器）；
本地没有 torch 时自动跳过 GNN，仍输出 roofline + Ridge 的结果。

用法（在项目根目录）：
  python cost_model/compare_models.py \
      --labels data/labels.jsonl \
      --extrap-labels data/labels_extrapolation.jsonl \
      --hw data/hw.json \
      --outdir data/compare
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.baseline import RidgeLogRegressor  # noqa: E402
from cost_model.features import make_targets  # noqa: E402
from cost_model.metrics import evaluate_metrics, mape, ranking_metrics  # noqa: E402
from cost_model.predict import RidgeCostModel  # noqa: E402
from cost_model.roofline import RooflineCostModel  # noqa: E402
from cost_model.split import build_feature_arrays, load_records, split_by_graph  # noqa: E402
from simulator import graph_from_dict, load_hardware, plan_from_dict  # noqa: E402


def collect_predictions(predictor, records):
    log_true = np.array([make_targets(r) for r in records], dtype=float)
    log_pred = np.array([predictor(r) for r in records], dtype=float)
    return log_true, log_pred


def report(name, records, log_true, log_pred):
    gids = np.array([r["graph_id"] for r in records], dtype=int)
    out = {
        "model": name,
        "n": int(len(records)),
        "n_graphs": int(len(set(gids.tolist()))),
    }
    for j, tname in enumerate(["makespan", "peak_mem"]):
        m = evaluate_metrics(log_true[:, j], log_pred[:, j])
        out[f"{tname}_mape"] = round(m["mape"], 4)
        out[f"{tname}_rmse_log"] = round(m["rmse_log"], 4)
        out[f"{tname}_spearman"] = round(m["spearman"], 4)
    rk = ranking_metrics(
        (int(gids[i]), float(log_pred[i, 0]), float(log_true[i, 0]))
        for i in range(len(records))
    )
    out["top1_acc"] = round(rk["top1_acc"], 4)
    out["top3_overlap"] = round(rk["top3_overlap"], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--extrap-labels", default=None)
    ap.add_argument("--outdir", default="data/compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=1e-2,
                    help="峰值显存头单独的学习率（默认高于主干，让显存头快速收敛）")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=12)
    args = ap.parse_args()

    np.random.seed(args.seed)
    hw = load_hardware(args.hw)
    records = load_records(args.labels)
    train, val, test = split_by_graph(records, seed=args.seed)
    extrap = load_records(args.extrap_labels) if args.extrap_labels else []

    # ---- 模型 A：解析式 roofline ----
    rm = RooflineCostModel(hw)

    def roofline_predictor(r):
        return rm.predict(graph_from_dict(r["graph"]), plan_from_dict(r["plan"]))

    # ---- 模型 B：手写特征 Ridge（在验证集上选 alpha）----
    X_tr, y_tr, _ = build_feature_arrays(train, hw)
    X_va, y_va, _ = build_feature_arrays(val, hw)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha, best_mape = alphas[0], float("inf")
    for a in alphas:
        m = RidgeLogRegressor(alpha=a)
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)
        mp = mape(np.exp(y_va[:, 0]), np.exp(p[:, 0]))
        if mp < best_mape:
            best_mape, best_alpha = mp, a
    print(f"Ridge alpha 选择: alpha={best_alpha}  val makespan MAPE={best_mape:.4f}")
    ridge = RidgeLogRegressor(alpha=best_alpha)
    ridge.fit(X_tr, y_tr)
    ridge_model = RidgeCostModel(ridge, hw)

    def ridge_predictor(r):
        return ridge_model.predict(graph_from_dict(r["graph"]), plan_from_dict(r["plan"]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ridge.save(str(outdir / "ridge_model.json"))

    # ---- 模型 C：GNN（无 torch 时跳过）----
    gnn_model = None
    try:
        import torch

        from cost_model.gnn import (  # noqa: F401
            CostGNN,
            GNNCostModel,
            LabelGraphDataset,
            collate,
        )

        has_torch = True
    except ImportError:
        print("警告: 未安装 torch，跳过 GNN")
        has_torch = False

    def gnn_predictor(r):
        return gnn_model.predict(graph_from_dict(r["graph"]), plan_from_dict(r["plan"]))

    if has_torch:
        torch.manual_seed(args.seed)
        vocab = sorted(
            {nd["op_type"] for r in train for nd in r["graph"]["nodes"]}
        )
        vocab.append("UNK")
        n_devs = (
            max(gd["device"] for r in records for gd in r["plan"]["groups"]) + 1
        )
        train_ds = LabelGraphDataset(train, vocab, n_devs)
        val_ds = LabelGraphDataset(val, vocab, n_devs)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CostGNN(
            vocab_size=len(vocab), n_devs=n_devs,
            hidden=args.hidden, num_layers=args.layers,
        ).to(device)
        # 峰值显存头的 dev_bytes 权重从 0 学到 ~1 需要 ~1000 步 Adam 更新，
        # 单独给它一个更高的学习率，避免 60 个 epoch 内根本收敛不完
        backbone_params = [
            p for n, p in model.named_parameters() if not n.startswith("head_pk")
        ]
        head_pk_params = [
            p for n, p in model.named_parameters() if n.startswith("head_pk")
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": args.lr},
                {"params": head_pk_params, "lr": args.head_lr},
            ]
        )
        # 双头目标量纲差异大，按目标方差倒数加权，避免峰值显存头主导梯度
        mk_logs = [math.log(float(r["makespan"])) for r in train]
        pk_logs = [
            math.log(max(float(v), 1.0))
            for r in train
            for v in r["peak_mem"].values()
        ]
        w_mk = 1.0 / (float(np.var(mk_logs)) + 1e-6)
        w_pk = 1.0 / (float(np.var(pk_logs)) + 1e-6)
        mean_w = (w_mk + w_pk) / 2.0
        w_mk /= mean_w
        w_pk /= mean_w
        loss_fn = torch.nn.MSELoss()

        @torch.no_grad()
        def gnn_evaluate(loader):
            model.eval()
            lt, lp = [], []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    batch["node_feats"], batch["node_mask"], batch["edge_index"],
                    batch["edge_attr"], batch["group_assign"],
                    batch["group_feats"], batch["group_mask"],
                    batch["dev_bytes"],
                )
                lt.append(batch["target"].cpu())
                lp.append(out.cpu())
            lt = torch.cat(lt)
            lp = torch.cat(lp)
            m_mk = evaluate_metrics(lt[:, 0].numpy(), lp[:, 0].numpy())
            m_pk = evaluate_metrics(lt[:, 1:].numpy(), lp[:, 1:].numpy())
            mk_mse = float(((lt[:, 0] - lp[:, 0]) ** 2).mean())
            pk_mse = float(((lt[:, 1:] - lp[:, 1:]) ** 2).mean())
            score = w_mk * mk_mse + w_pk * pk_mse
            return m_mk, m_pk, score, lt, lp

        best_score, best_state, bad = float("inf"), None, 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            mk_loss_sum = 0.0
            pk_loss_sum = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    batch["node_feats"], batch["node_mask"], batch["edge_index"],
                    batch["edge_attr"], batch["group_assign"],
                    batch["group_feats"], batch["group_mask"],
                    batch["dev_bytes"],
                )
                mk_loss = loss_fn(out[:, 0], batch["target"][:, 0])
                pk_loss = loss_fn(out[:, 1:], batch["target"][:, 1:])
                loss = w_mk * mk_loss + w_pk * pk_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n = len(batch["target"])
                total_loss += loss.item() * n
                mk_loss_sum += mk_loss.item() * n
                pk_loss_sum += pk_loss.item() * n
            vm_mk, vm_pk, val_score, _, _ = gnn_evaluate(val_loader)
            n_train = max(len(train_ds), 1)
            print(
                f"GNN epoch {epoch + 1:3d}  loss={total_loss / n_train:.5f}  "
                f"(mk={mk_loss_sum / n_train:.4f} pk={pk_loss_sum / n_train:.4f})  "
                f"val_mk={vm_mk['mape']:.3f}  val_pk={vm_pk['mape']:.3f}"
            )
            if val_score < best_score:
                best_score = val_score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        torch.save(best_state, str(outdir / "gnn_model.pt"))
        with open(outdir / "gnn_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"vocab": vocab, "n_devs": n_devs, "hidden": args.hidden,
                 "layers": args.layers, "best_val_score": best_score},
                f, ensure_ascii=False, indent=2,
            )
        gnn_model = GNNCostModel(model, vocab, n_devs, device=device)

    # ---- 统一评估 ----
    models = {"roofline": roofline_predictor, "ridge": ridge_predictor}
    if gnn_model is not None:
        models["gnn"] = gnn_predictor

    splits = {"val": val, "test": test}
    if extrap:
        splits["extrap"] = extrap

    all_reports = []
    header = f"{'model':<10} {'split':<8} {'n':>5} {'mk_mape':>8} {'mk_spear':>8} " \
             f"{'pk_mape':>8} {'top1':>6} {'top3':>6}"
    print("\n" + header)
    print("-" * len(header))
    for split_name, split_records in splits.items():
        for mname, predictor in models.items():
            lt, lp = collect_predictions(predictor, split_records)
            rep = report(mname, split_records, lt, lp)
            rep["split"] = split_name
            all_reports.append(rep)
            print(
                f"{mname:<10} {split_name:<8} {rep['n']:>5} "
                f"{rep['makespan_mape']:>8} {rep['makespan_spearman']:>8} "
                f"{rep['peak_mem_mape']:>8} {rep['top1_acc']:>6} {rep['top3_overlap']:>6}"
            )

    with open(outdir / "compare_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"seed": args.seed, "reports": all_reports},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n结果已保存: {outdir / 'compare_results.json'}")


if __name__ == "__main__":
    main()
