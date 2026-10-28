#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bundle="${1:-data/bw1000_gpu_holdout_v1}"
results="${2:-data/bw1000_gpu_holdout_v1/results}"

echo "Checking the sealed predictions and source revision"
python -m calibration.gpu_holdout check --bundle "$bundle"

echo "Profiling formal unseen kernels on physical GPUs 0 and 1 (resumable)"
python -m calibration.profile_e2_bw1000 \
  --manifest "$bundle/manifest.json" --out-dir "$results" \
  --devices 0,1 --assignment replicate

echo "Scoring GPU measurements against sealed zero-shot predictions"
python -m calibration.evaluate_gpu_holdout \
  --bundle "$bundle" --raw "$results/raw_measurements.jsonl" \
  --environments "$results/environments.json" \
  --output "$results/zero_shot_report.json"

echo "Executing sealed single-layer candidates on two GPUs (resumable)"
python -m execution.gpu_holdout \
  --bundle "$bundle" --devices 0,1 --out-dir "$results"

echo "GPU collection and reference execution finished; inspect both reports, including failed targets."
