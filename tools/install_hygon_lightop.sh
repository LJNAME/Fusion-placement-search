#!/usr/bin/env bash
# Install only the statically inspected vendor wheel into the selected private Python.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: bash tools/install_hygon_lightop.sh /fresh/install-report\n' >&2
    exit 2
fi
: "${FUSION_PYTHON:?Set FUSION_PYTHON to the existing private Hygon Python launcher}"
task_output="$1"
if [[ "$task_output" != /* ]]; then task_output="$PWD/$task_output"; fi
if [[ -e "$task_output" ]]; then
    printf 'Output already exists: %s\n' "$task_output" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_output")"
mkdir "$task_output"
trap 'printf "Return this install report on failure: %s\n" "$task_output"' EXIT
"$FUSION_PYTHON" - "$task_output/before.json" 2>&1 <<'PY' | tee "$task_output/preflight.log"
import json, platform, sys
from importlib.metadata import version, PackageNotFoundError
import torch
def installed(name):
    try: return version(name)
    except PackageNotFoundError: return None
record = {"executable": sys.executable, "prefix": sys.prefix, "python": sys.version,
    "platform": platform.platform(), "torch": installed("torch"), "triton": installed("triton"),
    "lightop": installed("lightop"), "hip": torch.version.hip,
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)}
with open(sys.argv[1], "w") as stream: json.dump(record, stream, indent=2)
assert sys.version_info[:2] == (3, 10), "requires Python 3.10"
assert platform.system() == "Linux" and platform.machine() == "x86_64", "requires Linux x86_64"
assert record["torch"].startswith("2.9.0+das.") and "dtk2604" in record["torch"], "requires vendor Torch2.9/DTK26.04"
assert record["hip"] and record["cxx11_abi"], "requires HIP Torch with abi1"
assert record["lightop"] in (None, "0.6.0+das.dtk2604.torch290.2605220936.g6bfaca"), "existing different LightOP: preserve it and review first"
print(json.dumps(record), flush=True)
PY
task_filename='lightop-0.6.0+das.dtk2604.torch290.2605220936.g6bfaca-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
task_cache="$HOME/private_data/LJ/lightop_wheels"
mkdir -p "$task_cache"
task_wheel="$task_cache/$task_filename"
if [[ ! -f "$task_wheel" ]]; then
    curl --fail --location --show-error --max-time 600 \
        --output "$task_wheel.part" \
        "https://pypi.sourcefind.cn/das_release/das1.8.2/+f/61e/f2672bbca7419/$task_filename"
    mv -- "$task_wheel.part" "$task_wheel"
fi
printf '61ef2672bbca7419a59ef14d41d1fcd883a972d57d51a2070df98e18d98a5795  %s\n' "$task_wheel" \
    | sha256sum -c - | tee "$task_output/checksum.log"
"$FUSION_PYTHON" -m pip install --no-deps --no-index "$task_wheel" 2>&1 | tee "$task_output/pip.log"
"$FUSION_PYTHON" - "$task_output" 2>&1 <<'PY' | tee "$task_output/import.log"
import importlib.metadata as metadata, json, pathlib, sys
output = pathlib.Path(sys.argv[1])
before = json.loads((output / "before.json").read_text())
after = {n: metadata.version(n) for n in ("torch", "triton", "lightop")}
assert all(after[n] == before[n] for n in ("torch", "triton")), "Torch/Triton changed unexpectedly"
(output / "after.json").write_text(json.dumps(after, indent=2) + "\n")
import lightop
assert callable(getattr(lightop.op, "silu_and_mul_opt", None)), "installed optimized binding unavailable"
print("LightOP installed; Torch/Triton versions unchanged; optimized export available", flush=True)
PY
