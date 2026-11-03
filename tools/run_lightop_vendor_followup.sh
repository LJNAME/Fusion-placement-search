#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash tools/run_lightop_vendor_followup.sh /fresh/output [plan|run]\n' >&2
    exit 2
fi
task_output="$1"
task_phase="${2:-plan}"
if [[ "$task_phase" != plan && "$task_phase" != run ]]; then exit 2; fi
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the existing vendor Python launcher}"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then printf 'Output already exists: %s\n' "$task_output" >&2; exit 2; fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$task_output"
trap 'printf "Return the complete directory including install failures: %s\n" "$task_output"' EXIT
cd "$task_root"
if [[ "$task_phase" == plan ]]; then
    "$FUSION_PYTHON" "$task_root/tools/prepare_lightop_bench_followup.py" plan --output "$task_output/preparation"
    exit 0
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the existing checkout}"
source "$FLAGSCALE_ROOT/tools/install/hygon/env.sh"
ulimit -c 0
"$FUSION_PYTHON" -X faulthandler "$task_root/tools/prepare_lightop_bench_followup.py" install --output "$task_output/preparation"
mapfile -t task_jobs < <("$FUSION_PYTHON" -c 'from tools.prepare_lightop_bench_followup import manifest; print("\n".join(manifest()["jobs"]))')
if [[ ${#task_jobs[@]} -ne 16 ]]; then printf 'Expected the 16 dependency-failed jobs\n' >&2; exit 2; fi
bash "$task_root/tools/run_lightop_vendor_suite.sh" "$task_output/measurements" run --jobs "${task_jobs[@]}"
