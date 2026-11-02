#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash tools/run_ffn_anchor_conditioning.sh /fresh/output [plan|run]\n' >&2
    exit 2
fi
task_phase="${2:-plan}"
if [[ "$task_phase" != plan && "$task_phase" != run ]]; then
    printf 'Phase must be plan or run\n' >&2
    exit 2
fi
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the validated Python launcher}"
if [[ "$task_phase" == run ]]; then
    : "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the pinned FlagScale checkout}"
fi
task_output="$1"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then
    printf 'Output already exists: %s\n' "$task_output" >&2
    exit 2
fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'printf "Return the complete output, including failures: %s\n" "$task_output"' EXIT
cd "$task_root"
task_args=("$task_phase" --output "$task_output")
if [[ "$task_phase" == run ]]; then task_args+=(--flagscale-root "$FLAGSCALE_ROOT"); fi
"$FUSION_PYTHON" -X faulthandler -m calibration.ffn_anchor_conditioning "${task_args[@]}"
