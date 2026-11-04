#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash tools/run_lightop_local_pair.sh /fresh/output [plan|run]\n' >&2
    exit 2
fi
task_phase="${2:-plan}"
if [[ "$task_phase" != plan && "$task_phase" != run ]]; then exit 2; fi
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the validated vendor Python launcher}"
task_output="$1"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then printf 'Output already exists: %s\n' "$task_output" >&2; exit 2; fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$task_phase" == run ]]; then
    : "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the existing checkout}"
    source "$FLAGSCALE_ROOT/tools/install/hygon/env.sh"
fi
mkdir -p "$task_output"
trap 'printf "Return the complete output, including failures: %s\n" "$task_output"' EXIT
cd "$task_root"
set +e
"$FUSION_PYTHON" -X utf8 -X faulthandler -m calibration.lightop_local_pair "$task_phase" --output "$task_output" 2>&1 | tee "$task_output/process.log"
task_statuses=("${PIPESTATUS[@]}")
set -e
printf '{"python_returncode":%d,"log_returncode":%d,"automatic_admission":false}\n' "${task_statuses[0]}" "${task_statuses[1]}" > "$task_output/process.json"
if [[ "${task_statuses[0]}" -ne 0 ]]; then exit "${task_statuses[0]}"; fi
exit "${task_statuses[1]}"
