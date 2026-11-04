#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash tools/recover_lightop_smi.sh /receipt03 /fresh/recovery_output\n' >&2
    exit 2
fi
: "${FLAGSCALE_ROOT:?Set FLAGSCALE_ROOT to the existing FlagScale checkout}"
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the existing vendor Python launcher}"
task_receipt="$1"
task_output="$2"
if [[ "$task_receipt" != /* ]]; then task_receipt="$PWD/$task_receipt"; fi
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then printf 'Output already exists: %s\n' "$task_output" >&2; exit 2; fi
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'printf "Return the complete recovery directory: %s\n" "$task_output"' EXIT
source "$FLAGSCALE_ROOT/tools/install/hygon/env.sh"
ulimit -c 0
cd "$task_root"
"$FUSION_PYTHON" -X faulthandler "$task_root/tools/recover_lightop_smi.py" recover \
    --receipt "$task_receipt" --output "$task_output"
