#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
    printf 'Usage: bash tools/run_kerv_add_rmsnorm_probe.sh /fresh/output [plan|run] [cuda:0]\n' >&2
    exit 2
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the audited FlagScale checkout}"
task_python="${FUSION_PYTHON:-python}"
task_output="$1"
task_action="${2:-plan}"
task_device="${3:-cuda:0}"
if [[ "$task_output" != /* ]]; then
    task_output="$PWD/$task_output"
fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
trap 'printf "Return the complete output, including failures: %s\n" "$task_output"' EXIT
"$task_python" -u -m calibration.kerv_add_rmsnorm_probe "$task_action" \
    --flagscale-root "$FLAGSCALE_ROOT" --output "$task_output" --device "$task_device"
