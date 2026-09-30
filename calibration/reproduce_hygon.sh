#!/usr/bin/env bash
# 海光 K100 四卡：标定结果复现脚本（组会汇报前跑一遍，产出全套报告）
#
# 运行环境：海光服务器，py310（numpy/pandas/torch），ROCm/DTK + rocblas，
#          且本仓库已同步到最新（至少 calibration/ cost_model/ simulator/）。
#
# 用法：
#   bash calibration/reproduce_hygon.sh               # 用服务器上已有的 calibrated.json
#   bash calibration/reproduce_hygon.sh --recalibrate # 连微基准标定一起重跑
#
# 产物（默认 data/hygon_k100/）：
#   calibration_report.json  -> 汇报 3.3 参数表
#   validation_report.json   -> 汇报 3.3 组合层验证表（Spearman/MAPE/峰值显存）
#   attribution_report.json  -> 汇报 3.3 归因修正（overlap_send=false 依据）
#   compare_results.json     -> 汇报 3.1 预测精度表（跑完请把输出贴回文档）
#   search_results.json      -> 汇报 3.2 端到端自评表
#   search_real.json         -> 汇报 3.2 真机裁决表
#   scan_calibrated.json     -> 汇报 3.3 差异化扫描表
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

HW="configs/hygon_k100_4cards.calibrated.json"
OUT="data/hygon_k100"
BUILD_CMD="hipcc -O2 --offload-arch=gfx928 -o build/plan_executor_hip build/plan_executor_hip.cpp -lrocblas"

mkdir -p build "$OUT"

echo "==> 0. 环境自检"
python - <<'PY'
import sys
import numpy as np
print(f"  python {sys.version.split()[0]}  numpy {np.__version__}")
try:
    import torch
    print(f"  torch {torch.__version__}")
except ImportError:
    print("  !! 无 torch：GNN 相关步骤会失败，先装 torch 或只跑 roofline/ridge")
PY

if [ "${1:-}" = "--recalibrate" ] || [ ! -f "$HW" ]; then
  echo "==> 1. 微基准标定（重跑，生成 calibrated.json + calibration_report.json）"
  python calibration/run_calibration.py \
    --nominal configs/hygon_4cards.example.json \
    --build-cmd "hipcc -O2 -o build/kernels_hip calibration/kernels_hip.cpp -lrocblas" \
    --binary build/kernels_hip \
    --out-config "$HW" \
    --report "$OUT/calibration_report.json"
else
  echo "==> 1. 使用已有标定配置 $HW（跳过微基准；如需重测加 --recalibrate）"
fi

echo "==> 2. 组合层真机验证（32 个近真实规模方案）"
python calibration/gen_validation_plans.py --hw "$HW" --out "$OUT/validation_plans.json"
python calibration/run_validation.py \
  --plans "$OUT/validation_plans.json" \
  --build-cmd "$BUILD_CMD" \
  --binary build/plan_executor_hip \
  --hw "$HW" \
  --n-gpus 4 \
  --measured "$OUT/measured.json" \
  --report "$OUT/validation_report.json"
echo "  预期：makespan Spearman ~0.97 / MAPE ~0.19 / 峰值显存 MAPE ~0.11"

echo "==> 3. 误差归因（overlap_send=false 修正依据）"
python calibration/attribute_error.py \
  --measured "$OUT/measured.json" --hw "$HW" --out "$OUT/attribution_report.json"

echo "==> 4. 生成数据集（calibrated 配置，8 图族 × 12 方案 + 外推集）"
python cost_model/make_dataset.py \
  --hw "$HW" \
  --out "$OUT/labels.jsonl" \
  --extrap-out "$OUT/labels_extrapolation.jsonl"

echo "==> 5. 三模型统一对比（compare_models 内部会训 Ridge + GNN 并保存模型）"
python cost_model/compare_models.py \
  --labels "$OUT/labels.jsonl" \
  --extrap-labels "$OUT/labels_extrapolation.jsonl" \
  --hw "$OUT/hw.json" \
  --outdir "$OUT/compare"
echo "  >> 把上面打印的表格贴回《组会汇报-进展总结.md》3.1 节"

echo "==> 6. 端到端搜索（beam，代价模型打分 + 仿真器精评，自评表）"
python cost_model/eval_search.py \
  --hw "$OUT/hw.json" \
  --extrap-labels "$OUT/labels_extrapolation.jsonl" \
  --ridge-model "$OUT/compare/ridge_model.json" \
  --gnn-model "$OUT/compare" \
  --out "$OUT/search_results.json"

echo "==> 7. 真机确认（beam 选中方案在真实 4 卡重跑，裁决表）"
python calibration/confirm_search.py \
  --search "$OUT/search_results.json" \
  --build-cmd "$BUILD_CMD" \
  --binary build/plan_executor_hip \
  --hw "$HW" \
  --n-gpus 4 \
  --out "$OUT/search_real.json"

echo "==> 8. 差异化扫描（可选，3.3 表：roofline vs 仿真器排序分歧）"
python simulator/scan_differentiation.py --hw "$HW" --out "$OUT/scan_calibrated.json"

echo ""
echo "==> 全部完成，产物在 $OUT/"
ls -la "$OUT"
