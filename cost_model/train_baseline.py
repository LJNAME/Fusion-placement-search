"""训练手写特征基线代价模型（纯 numpy，本地可跑）。

用法（在项目根目录）：
  python cost_model/train_baseline.py --labels data/labels.jsonl --hw data/hw.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_model.baseline import RidgeLogRegressor  # noqa: E402
from cost_model.features import compute_plan_features, make_targets  # noqa: E402
from cost_model.metrics import evaluate_metrics, ranking_metrics  # noqa: E402
from simulator import graph_from_dict, load_hardware, plan_from_dict  # noqa: E402


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_graph(records, val_ratio: float, seed: int):
    unique = sorted({r["graph_id"] for r in records})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    val_ids = set(unique[:n_val])
    tr = [i for i, r in enumerate(records) if r["graph_id"] not in val_ids]
    va = [i for i, r in enumerate(records) if r["graph_id"] in val_ids]
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/baseline_model.json")
    args = ap.parse_args()

    records = load_records(args.labels)
    hw = load_hardware(args.hw)

    X, y, gids = [], [], []
    for r in records:
        graph = graph_from_dict(r["graph"])
        plan = plan_from_dict(r["plan"])
        X.append(compute_plan_features(graph, plan, hw))
        y.append(make_targets(r))
        gids.append(r["graph_id"])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    gids = np.asarray(gids)

    tr, va = split_by_graph(records, args.val_ratio, args.seed)
    model = RidgeLogRegressor(alpha=args.alpha)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])

    names = ["makespan", "peak_mem"]
    print(f"样本: 训练 {len(tr)} / 验证 {len(va)}，特征维度 {X.shape[1]}")
    for j, name in enumerate(names):
        m = evaluate_metrics(y[va, j], pred[:, j])
        print(f"[{name}] MAPE={m['mape']:.3f}  RMSE(log)={m['rmse_log']:.4f}  "
              f"Spearman={m['spearman']:.3f}")

    rk = ranking_metrics(
        (int(gids[i]), float(pp), float(tt))
        for i, pp, tt in zip(va, pred[:, 0], y[va, 0])
    )
    print(f"[排名] top1={rk['top1_acc']:.3f}  top3重叠={rk['top3_overlap']:.3f}  "
          f"图数={int(rk['n_graphs'])}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out), extra={"n_features": int(X.shape[1])})
    print(f"模型已保存: {out}")


if __name__ == "__main__":
    main()
