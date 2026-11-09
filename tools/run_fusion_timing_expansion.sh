#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: bash tools/run_fusion_timing_expansion.sh /fresh/output [plan|run]\n' >&2
  exit 2
fi
task_output="$1"
task_action="${2:-plan}"
if [[ "$task_action" != plan && "$task_action" != run ]]; then exit 2; fi
task_python="${FUSION_PYTHON:-python}"
if [[ "$task_action" == run ]]; then
  : "${FUSION_PYTHON:?Set FUSION_PYTHON to the validated vendor Python launcher}"
  : "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the existing FlagScale checkout}"
  source "$FLAGSCALE_ROOT/tools/install/hygon/env.sh"
  ulimit -c 0
fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then printf 'Output already exists: %s\n' "$task_output" >&2; exit 2; fi
trap 'printf "Return complete output, including failures: %s\n" "$task_output"' EXIT
"$task_python" -u -X faulthandler -m calibration.fusion_timing_expansion "$task_action" --output "$task_output"
