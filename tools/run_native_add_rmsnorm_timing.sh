#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash tools/run_native_add_rmsnorm_timing.sh /fresh/output [plan|run]\n' >&2
    exit 2
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the audited FlagScale checkout}"
task_output="$1"
task_action="${2:-plan}"
if [[ "$task_action" == run ]]; then
    : "${FUSION_PYTHON:?Set FUSION_PYTHON to the validated GPU Python launcher}"
fi
task_python="${FUSION_PYTHON:-python}"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
trap 'printf "Return the complete output, including failures: %s\n" "$task_output"' EXIT
"$task_python" -u -m calibration.native_add_rmsnorm_timing "$task_action" \
    --flagscale-root "$FLAGSCALE_ROOT" --output "$task_output"
