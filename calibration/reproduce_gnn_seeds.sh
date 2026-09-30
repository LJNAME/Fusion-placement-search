#!/usr/bin/env bash
# GNN 多 seed 稳定性实验（海光服务器，py310 + torch）
#
# 目的：量化"GNN 搜索质量对重训的稳定性"。协议要点：
#   - 复用 data/hygon_k100/ 的 labels / extrap / hw（不重新生成，保证数据一致）；
#   - compare_models --seed 0 固定数据划分，--model-seed N 只变 GNN 初始化/训练随机性；
#   - eval_search / confirm_search 固定 --seed 0（同一批 5 个外推图、同一 beam 随机性）。
#
# 用法：bash calibration/reproduce_gnn_seeds.sh ["0 1 2 3"]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT="data/hygon_k100"
LABELS="$OUT/labels.jsonl"
EXTRAP="$OUT/labels_extrapolation.jsonl"
HW="$OUT/hw.json"
CFG_HW="configs/hygon_k100_4cards.calibrated.json"
BUILD_CMD="hipcc -O2 --offload-arch=gfx928 -o build/plan_executor_hip build/plan_executor_hip.cpp -lrocblas"
SEEDS="${1:-0 1 2 3}"

for S in $SEEDS; do
  echo "==> model_seed=$S（数据划分 seed=0 固定）"
  python cost_model/compare_models.py \
    --labels "$LABELS" --extrap-labels "$EXTRAP" --hw "$HW" \
    --seed 0 --model-seed "$S" --outdir "$OUT/compare_s$S"

  python cost_model/eval_search.py \
    --hw "$HW" --extrap-labels "$EXTRAP" \
    --ridge-model "$OUT/compare_s$S/ridge_model.json" \
    --gnn-model "$OUT/compare_s$S" \
    --seed 0 --out "$OUT/search_s$S.json"

  python calibration/confirm_search.py \
    --search "$OUT/search_s$S.json" \
    --build-cmd "$BUILD_CMD" --binary build/plan_executor_hip \
    --hw "$CFG_HW" --n-gpus 4 \
    --out "$OUT/search_real_s$S.json"
done

echo ""
echo "==> GNN 多 seed 汇总（预测精度 / 自评改善 / 真机改善）"
python - <<PY
import glob, json, os, statistics as st

out = "$OUT"
rows = []
for cpath in sorted(glob.glob(os.path.join(out, "compare_s*/compare_results.json"))):
    s = int(os.path.basename(os.path.dirname(cpath)).replace("compare_s", ""))
    c = json.load(open(cpath, encoding="utf-8"))
    g_ext = next(r for r in c["reports"] if r["model"] == "gnn" and r["split"] == "extrap")
    g_val = next(r for r in c["reports"] if r["model"] == "gnn" and r["split"] == "val")
    sp = os.path.join(out, f"search_s{s}.json")
    sr = os.path.join(out, f"search_real_s{s}.json")
    self_imp = None
    real_imp = None
    real_mk = None
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        self_imp = d["summary"].get("beam_gnn", {}).get("improve_vs_heuristic")
    if os.path.exists(sr):
        d = json.load(open(sr, encoding="utf-8"))
        real_imp = d["summary"].get("beam_gnn", {}).get("improve_vs_heuristic_real")
        real_mk = d["summary"].get("beam_gnn", {}).get("mean_real")
    rows.append({
        "seed": s,
        "val_mape": g_val["makespan_mape"], "val_spear": g_val["makespan_spearman"],
        "ext_mape": g_ext["makespan_mape"], "ext_spear": g_ext["makespan_spearman"],
        "self_imp": self_imp, "real_imp": real_imp, "real_mk": real_mk,
    })
    def pct(x):
        return f"{x:+.1%}" if x is not None else "N/A"
    mk_s = f"{real_mk:.3f}s" if real_mk is not None else "N/A"
    print(
        f"seed={s}  val MAPE={g_val['makespan_mape']:.3f} Spear={g_val['makespan_spearman']:.3f} | "
        f"extrap MAPE={g_ext['makespan_mape']:.3f} Spear={g_ext['makespan_spearman']:.3f} | "
        f"自评改善={pct(self_imp)} | 真机改善={pct(real_imp)} 真机={mk_s}"
    )

def rng(key):
    vs = [r[key] for r in rows if r[key] is not None]
    return vs

print()
for key, label in [("ext_spear", "外推 Spearman"), ("ext_mape", "外推 MAPE"),
                   ("self_imp", "自评改善"), ("real_imp", "真机改善")]:
    vs = rng(key)
    if vs:
        print(f"{label}: min={min(vs):.4f}  median={st.median(vs):.4f}  max={max(vs):.4f}  "
              f"(n={len(vs)})")
PY
