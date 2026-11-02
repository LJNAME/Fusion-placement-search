#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: bash tools/run_kerv_profile_stage3.sh /fresh/output/directory\n' >&2
    exit 2
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the pinned FlagScale checkout}"
task_python="${FUSION_PYTHON:-python}"
task_output="$1"
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then
    printf 'Output already exists: %s\n' "$task_output" >&2
    exit 2
fi
cd "$task_root"
mkdir -p "$task_output"
"$task_python" -m calibration.validate_kerv_adapter prepare \
    --flagscale-root "$FLAGSCALE_ROOT" --out-dir "$task_output/prepare"
"$task_python" -m calibration.measure_kerv_profile_pair \
    --flagscale-root "$FLAGSCALE_ROOT" --placed "$task_output/prepare/placed.json" \
    --out-dir "$task_output/paired"
"$task_python" -m calibration.kerv_profile_workflow export \
    --flagscale-root "$FLAGSCALE_ROOT" --source "$task_output/paired" --output "$task_output/profile"
"$task_python" -m calibration.kerv_profile_workflow query \
    --profile "$task_output/profile/profile.json" --allow-conditional-dev \
    --output "$task_output/cost_queries.json"
printf 'Return the complete directory: %s\n' "$task_output"
