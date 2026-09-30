"""训练 GNN 双头代价模型（需要 PyTorch，服务器上运行）。

用法（在项目根目录）：
  python cost_model/train_gnn.py --labels data/labels.jsonl --outdir data/gnn_model
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.gnn import CostGNN, LabelGraphDataset, collate  # noqa: E402
from cost_model.metrics import evaluate_metrics, ranking_metrics  # noqa: E402


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_graph(records, val_ratio: float, seed: int):
    unique = sorted({r["graph_id"] for r in records})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    val_ids = set(unique[:n_val])
    tr = [r for r in records if r["graph_id"] not in val_ids]
    va = [r for r in records if r["graph_id"] in val_ids]
    return tr, va


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    log_true, log_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(
            batch["node_feats"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["group_assign"],
            batch["group_feats"],
            batch["group_mask"],
            batch["dev_bytes"],
        )
        log_true.append(batch["target"].cpu())
        log_pred.append(pred.cpu())
    lt = torch.cat(log_true)
    lp = torch.cat(log_pred)
    m_mk = evaluate_metrics(lt[:, 0].numpy(), lp[:, 0].numpy())
    m_pk = evaluate_metrics(lt[:, 1:].numpy(), lp[:, 1:].numpy())
    return m_mk, m_pk, lt, lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", default="data/gnn_model")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=1e-2,
                    help="峰值显存头单独的学习率（默认高于主干，让显存头快速收敛）")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    records = load_records(args.labels)
    vocab = sorted(
        {nd["op_type"] for r in records for nd in r["graph"]["nodes"]}
    )
    if "UNK" not in vocab:
        vocab.append("UNK")
    n_devs = max(gd["device"] for r in records for gd in r["plan"]["groups"]) + 1

    train_recs, val_recs = split_by_graph(records, args.val_ratio, args.seed)
    # 双头目标量纲差异大（log makespan ~ -3，log 峰值显存 ~ 21）。
    # 用训练集目标方差的倒数加权，避免峰值显存头主导梯度、makespan 头欠拟合。
    mk_logs = [math.log(float(r["makespan"])) for r in train_recs]
    pk_logs = [
        math.log(max(float(v), 1.0))
        for r in train_recs
        for v in r["peak_mem"].values()
    ]
    w_mk = 1.0 / (float(np.var(mk_logs)) + 1e-6)
    w_pk = 1.0 / (float(np.var(pk_logs)) + 1e-6)
    mean_w = (w_mk + w_pk) / 2.0
    w_mk /= mean_w
    w_pk /= mean_w
    print(
        f"目标加权: makespan 头 w={w_mk:.2f}（var={np.var(mk_logs):.3f}），"
        f"peak_mem 头 w={w_pk:.2f}（var={np.var(pk_logs):.3f}）"
    )

    train_ds = LabelGraphDataset(train_recs, vocab, n_devs)
    val_ds = LabelGraphDataset(val_recs, vocab, n_devs)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CostGNN(
        vocab_size=len(vocab),
        n_devs=n_devs,
        hidden=args.hidden,
        num_layers=args.layers,
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
    loss_fn = torch.nn.MSELoss()

    best_score = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        mk_loss_sum = 0.0
        pk_loss_sum = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(
                batch["node_feats"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["group_assign"],
                batch["group_feats"],
                batch["group_mask"],
                batch["dev_bytes"],
            )
            mk_loss = loss_fn(pred[:, 0], batch["target"][:, 0])
            pk_loss = loss_fn(pred[:, 1:], batch["target"][:, 1:])
            loss = w_mk * mk_loss + w_pk * pk_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = len(batch["target"])
            total_loss += loss.item() * n
            mk_loss_sum += mk_loss.item() * n
            pk_loss_sum += pk_loss.item() * n

        val_mk, val_pk, lt, lp = evaluate(model, val_loader, device)
        mk_mse = float(((lt[:, 0] - lp[:, 0]) ** 2).mean())
        pk_mse = float(((lt[:, 1:] - lp[:, 1:]) ** 2).mean())
        val_score = w_mk * mk_mse + w_pk * pk_mse
        n_train = max(len(train_ds), 1)
        print(
            f"epoch {epoch + 1:3d}  loss={total_loss / n_train:.5f}  "
            f"(mk={mk_loss_sum / n_train:.4f} pk={pk_loss_sum / n_train:.4f})  "
            f"val_mk={val_mk['mape']:.3f}  val_pk={val_pk['mape']:.3f}  "
            f"val_Spearman={val_mk['spearman']:.3f}"
        )
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop @ epoch {epoch + 1}")
                break

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, str(outdir / "model.pt"))
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab": vocab,
                "n_devs": n_devs,
                "hidden": args.hidden,
                "layers": args.layers,
                "best_val_score": best_score,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"best val score={best_score:.4f}，模型已保存到 {outdir}")


if __name__ == "__main__":
    main()
