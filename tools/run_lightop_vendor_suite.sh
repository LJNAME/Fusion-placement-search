#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
    printf 'Usage: bash tools/run_lightop_vendor_suite.sh /fresh/output [plan|run] [Python options]\n' >&2
    exit 2
fi
task_output="$1"
task_phase="${2:-plan}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
if [[ "$task_phase" != plan && "$task_phase" != run ]]; then exit 2; fi
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the existing vendor Python launcher}"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then printf 'Output already exists: %s\n' "$task_output" >&2; exit 2; fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$task_phase" == run ]]; then
    : "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the existing checkout}"
    source "$FLAGSCALE_ROOT/tools/install/hygon/env.sh"
fi
trap 'printf "Return all output including failures: %s\n" "$task_output"' EXIT
cd "$task_root"
"$FUSION_PYTHON" -X faulthandler "$task_root/calibration/lightop_vendor_suite.py" "$task_phase" --output "$task_output" --launcher "$FUSION_PYTHON" "$@"
