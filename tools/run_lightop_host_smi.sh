#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash tools/run_lightop_host_smi.sh /fresh/output [plan|run]\n' >&2
    exit 2
fi
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the existing vendor Python launcher}"
task_output="$1"
task_phase="${2:-plan}"
if [[ "$task_phase" != plan && "$task_phase" != run ]]; then exit 2; fi
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then printf 'Output already exists: %s\n' "$task_output" >&2; exit 2; fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'printf "Return the complete output including failures: %s\n" "$task_output"' EXIT
if [[ "$task_phase" == run ]]; then
    : "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the existing checkout}"
    source "$FLAGSCALE_ROOT/tools/install/hygon/env.sh"
    ulimit -c 0
fi
cd "$task_root"
task_args=()
if [[ -n "${LIGHTOP_VENDOR_WHEEL:-}" ]]; then task_args+=(--wheel "$LIGHTOP_VENDOR_WHEEL"); fi
"$FUSION_PYTHON" -X faulthandler "$task_root/tools/run_lightop_host_smi.py" "$task_phase" \
    --output "$task_output" "${task_args[@]}"
