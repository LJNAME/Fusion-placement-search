"""把「层→设备」的搜索结果翻译成 FlagScale 异构配置片段。

两卡机的映射：
- 每张卡一个 mesh：`[tp,cp,ep,dp,pp] = [1,1,1,1,1]`（每卡一段流水）；
- 全局 `pipeline_model_parallel_size` = mesh 数 = 2；
- `hetero_pipeline_layer_split` = 各 mesh 分到的层数（按 mesh 顺序，即设备顺序）；
- `hetero_device_types` / `hetero_current_device_type` = 两张卡的型号串。

用法：
  python integration/flagscale/plan_to_conf.py \
      --result data/a6000_titan/sweep.json \
      --device-types A6000,TITAN --out configs/a6000_titan_hetero_block.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_block(layer_split, device_types, standalone_embedding=False):
    meshes = ",".join("1,1,1,1,1" for _ in layer_split)
    types = ", ".join(f'"{t}"' for t in device_types)
    lines = [
        "  hetero:",
        "    enable_hetero: true",
        f"    standalone_embedding_stage: {standalone_embedding}",
        f"    hetero_pipeline_layer_split: [{', '.join(str(x) for x in layer_split)}]",
        f"    hetero_process_meshes: [{meshes}]",
        f"    hetero_device_types: [{types}]",
        f'    hetero_current_device_type: "{device_types[0]}"',
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="sweep_layer_split.py 的输出 JSON")
    ap.add_argument("--device-types", required=True, help="按设备顺序，逗号分隔，如 A6000,TITAN")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    # 优先用流水线稳态口径（微批重叠下的节拍）；没有就退回单次遍历口径
    chosen = data.get("best_steady") or data.get("best")
    metric = "steady" if data.get("best_steady") else "makespan"
    lpd = {"0": chosen["k"], "1": data["graph"]["layers"] - chosen["k"]} if chosen else data.get("layers_per_device")
    if not lpd:
        raise SystemExit("结果 JSON 里没有 layers_per_device（请用 sweep_layer_split.py 重跑）")
    n = max(int(k) for k in lpd) + 1
    layer_split = [int(lpd.get(str(i), lpd.get(i, 0))) for i in range(n)]
    types = [t.strip() for t in args.device_types.split(",")]
    if len(types) != len(layer_split):
        raise SystemExit(f"device-types 数量 {len(types)} 与 mesh 数 {len(layer_split)} 不一致")

    block = build_block(layer_split, types)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(block, encoding="utf-8")

    print(f"设备顺序: {types}")
    print(f"层切分: {layer_split}（合计 {sum(layer_split)} 层，全局 pp={len(layer_split)}）")
    print(f"口径={metric}  值={chosen['makespan'] if metric == 'makespan' else chosen['steady']:.6f} s（来源 {args.result}）")
    print(f"片段写到 {args.out}：")
    print(block)


if __name__ == "__main__":
    main()