#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
flagscale_root="${FLAGSCALE_ROOT:-../FlagScale-main/FlagScale-main}"
out_dir="${1:-data/group_key_mlp_bw1000_v1}"

exec "$python_bin" -m calibration.run_group_key_remeasurement \
  --flagscale-root "$flagscale_root" --out-dir "$out_dir" --order-seed 0
