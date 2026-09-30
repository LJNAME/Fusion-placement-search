"""生成合成图 + 随机/启发式方案 + 仿真器标签。

用法（在项目根目录）：
  python cost_model/make_dataset.py --out data/labels.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import (
    generate_label_dataset,
    load_hardware,
    save_hardware,
    make_dataset_graphs,
    make_extrapolation_graphs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成代价模型训练数据集")
    ap.add_argument("--hw", default="configs/hetero_4cards.example.json")
    ap.add_argument("--out", default="data/labels.jsonl")
    ap.add_argument("--graphs-per-type", type=int, default=20)
    ap.add_argument("--plans-per-graph", type=int, default=12)
    ap.add_argument("--random-ratio", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extrap-out", default=None, help="外推测试集输出路径")
    ap.add_argument("--extrap-graphs-per-type", type=int, default=10)
    ap.add_argument("--extrap-id-offset", type=int, default=100000)
    args = ap.parse_args()

    hw = load_hardware(args.hw)
    graphs = make_dataset_graphs(n_per_type=args.graphs_per_type, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = generate_label_dataset(
        graphs,
        hw,
        str(out),
        plans_per_graph=args.plans_per_graph,
        random_ratio=args.random_ratio,
        seed=args.seed,
    )

    hw_path = out.with_name("hw.json")
    save_hardware(hw, str(hw_path))

    overflow = sum(1 for r in records if r["mem_overflow"])
    makespans = [r["makespan"] for r in records]
    print(f"图数量: {len(graphs)}")
    print(f"标签数量: {len(records)}")
    print(f"显存溢出方案占比: {overflow / max(len(records), 1):.1%}")
    print(f"makespan 范围: {min(makespans):.4f}s ~ {max(makespans):.4f}s")
    print(f"输出: {out}（硬件配置已存 {hw_path}）")

    if args.extrap_out:
        extrap_graphs = make_extrapolation_graphs(
            n_per_type=args.extrap_graphs_per_type, seed=args.seed + 10**6
        )
        extrap_path = Path(args.extrap_out)
        extrap_path.parent.mkdir(parents=True, exist_ok=True)
        extrap_records = generate_label_dataset(
            extrap_graphs,
            hw,
            str(extrap_path),
            plans_per_graph=args.plans_per_graph,
            random_ratio=args.random_ratio,
            seed=args.seed,
            graph_id_offset=args.extrap_id_offset,
        )
        extrap_overflow = sum(1 for r in extrap_records if r["mem_overflow"])
        print(f"外推图数量: {len(extrap_graphs)}")
        print(f"外推标签数量: {len(extrap_records)}")
        print(f"外推显存溢出占比: {extrap_overflow / max(len(extrap_records), 1):.1%}")
        print(f"外推输出: {extrap_path}（graph_id 从 {args.extrap_id_offset} 起）")


if __name__ == "__main__":
    main()
