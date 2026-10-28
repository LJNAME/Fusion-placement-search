#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bundle="data/reference_validation_v1"
results="data/reference_validation_v1_results"
python -m calibration.reference_validation check --bundle "$bundle"
python -m calibration.profile_e2_bw1000 --manifest "$bundle/manifest.json" \
  --out-dir "$results" --devices 0,1 --assignment replicate
python -m calibration.evaluate_gpu_holdout --bundle "$bundle" \
  --raw "$results/raw_measurements.jsonl" --environments "$results/environments.json" \
  --output "$results/zero_shot_report.json"
python -m execution.gpu_holdout --bundle "$bundle" --devices 0,1 --out-dir "$results"
python -m execution.evaluate_reference_validation --bundle "$bundle" --results "$results"
