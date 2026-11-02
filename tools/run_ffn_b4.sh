#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash tools/run_ffn_b4.sh /fresh/output/directory [smoke|run]\n' >&2
    exit 2
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the pinned FlagScale checkout}"
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the validated dcu-kernel310-20261031/run_python.sh launcher}"
task_python="$FUSION_PYTHON"
task_output="$1"
task_phase="${2:-smoke}"
if [[ "$task_phase" != smoke && "$task_phase" != run ]]; then
    printf 'Phase must be smoke or run\n' >&2
    exit 2
fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then
    printf 'Output already exists: %s\n' "$task_output" >&2
    exit 2
fi
trap 'printf "Return the complete output, including failures: %s\n" "$task_output"' EXIT
cd "$task_root"
"$task_python" -X faulthandler -m calibration.ffn_b4 "$task_phase" --flagscale-root "$FLAGSCALE_ROOT" --output "$task_output"
