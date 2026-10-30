"""Run the SCNet development microbenchmark workflow; stop at the first failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_workflow(out_dir, flagscale_root, *, order_seed=0):
    started = time.perf_counter()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, "-m", "calibration.group_key_remeasurement"]
    source = ["--flagscale-root", str(Path(flagscale_root).resolve())]
    smoke, formal = out_dir / "smoke", out_dir / "formal"
    seed = ["--order-seed", str(order_seed)]
    steps = [
        ("check", [*common, "check", *source]),
        ("smoke_collect", [*common, "collect", *source, "--mode", "smoke", *seed, "--out-dir", str(smoke)]),
        ("smoke_verify", [*common, "verify", "--bundle", str(smoke)]),
        ("formal_collect", [*common, "collect", *source, "--mode", "formal", *seed, "--out-dir", str(formal)]),
        ("formal_verify", [*common, "verify", "--bundle", str(formal), "--out", str(formal / "verified_profile.json")]),
    ]
    source_names = ("calibration/run_group_key_remeasurement.py", "tools/run_group_key_remeasurement.sh")
    plan = {
        "format": "fusion-placement-search.group-key-workflow", "version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development_Kernel_microbenchmarks_not_full_graph_execution",
        "working_directory": str(ROOT), "order_seed": order_seed,
        "device_policy": "existing two-device draft; visible HIP logical devices 0 and 1",
        "steps": [{"name": name, "argv": argv, "log": f"{name}.log"} for name, argv in steps],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
        "automatic_retries": False, "automatic_profile_writeback": False,
    }
    write_new(out_dir / "workflow_plan.json", plan)
    results = []
    exit_code = 0
    try:
        for name, argv in steps:
            print(f"[{name}] started; log: {out_dir / (name + '.log')}", flush=True)
            phase_started = time.perf_counter()
            with (out_dir / f"{name}.log").open("x", encoding="utf-8", newline="\n") as log:
                completed = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            results.append({"name": name, "exit_code": completed.returncode,
                            "subprocess_wall_s": time.perf_counter() - phase_started})
            if completed.returncode != 0:
                exit_code = completed.returncode if completed.returncode > 0 else 1
                print(f"[{name}] failed; later steps were not run. Keep this directory.", flush=True)
                break
            print(f"[{name}] passed", flush=True)
    except Exception as error:
        exit_code = 1
        results.append({"name": name, "error": f"{type(error).__name__}: {error}"})
    summary = {
        "format": plan["format"], "version": "1.0", "complete": exit_code == 0,
        "exit_code": exit_code, "steps": results,
        "workflow_wall_s": time.perf_counter() - started,
        "wall_scope": "parent workflow entry through child completion; includes every child process startup, check, collection, verification and log IO; excludes parent startup and final summary write",
        "nested_timer_policy": "child collection/setup/block timers are components; do not add them again",
        "automatic_profile_writeback": False,
    }
    write_new(out_dir / "workflow_summary.json", summary)
    print("Results:", out_dir, flush=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagscale-root", type=Path,
                        default=ROOT.parent / "FlagScale-main/FlagScale-main")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/group_key_mlp_bw1000_v1")
    parser.add_argument("--order-seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run_workflow(args.out_dir, args.flagscale_root, order_seed=args.order_seed)


if __name__ == "__main__":
    raise SystemExit(main())
