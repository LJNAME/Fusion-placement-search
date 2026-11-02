#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash tools/run_kerv_adapter_validation.sh /fresh/output/directory\n' >&2
    exit 2
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the pinned FlagScale checkout}"
task_python="${FUSION_PYTHON:-python}"
task_device="${FUSION_KERV_DEVICE:-cuda:0}"
task_output="$1"
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$task_output" != /* ]]; then
    task_output="$PWD/$task_output"
fi
if [[ -e "$task_output" ]]; then
    printf 'Output already exists: %s\n' "$task_output" >&2
    exit 2
fi
cd "$task_root"
mkdir -p "$task_output"
"$task_python" -m calibration.validate_kerv_adapter prepare \
    --flagscale-root "$FLAGSCALE_ROOT" --out-dir "$task_output/prepare"
"$task_python" -m calibration.validate_kerv_adapter run \
    --flagscale-root "$FLAGSCALE_ROOT" \
    --placed "$task_output/prepare/placed.json" --device "$task_device" \
    --rounds 7 --calls 50 --warmup 10 --out-dir "$task_output/gpu"
printf 'Return the complete result directory: %s\n' "$task_output"
