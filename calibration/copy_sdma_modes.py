"""Four cross-GPU Copies under one SDMA mode per process.

Two execution contracts: `sdma_default` (HSA_ENABLE_SDMA unset; the FFN
default and the primary object) and `sdma_disabled` (HSA_ENABLE_SDMA=0; a
control). Each mode runs in its own process and directory; the mode is part
of the contract so the two never share a contract ID. Batch-01 Copy records
stay as they are. Outliers are counted, never removed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from calibration import stage_size_calibration as base
from calibration.lightop_profile_smoke import POOL, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "cross_gpu_copy_sdma_modes_v1"
MODES = {"sdma_default": None, "sdma_disabled": "0"}
OUTLIER_RULES = {"over_3x_median": 3.0, "over_1ms_us": 1000.0}
SOURCES = ("calibration/copy_sdma_modes.py", "tools/run_copy_sdma_modes.sh", *base.SOURCES)


def case_ids():
    return [f"copy/{size}/{src}->{dst}" for size, src, dst in base.COPIES]


def plan(mode):
    if mode not in MODES:
        raise ValueError("Unknown SDMA mode")
    ids = case_ids()
    contract = {**base.contract(), "sdma_mode": mode, "HSA_ENABLE_SDMA": MODES[mode],
                "cache_context": "four Copies interleaved per round; one SDMA mode per process"}
    return {"scope": SCOPE, "mode": mode, "role": "primary" if mode == "sdma_default" else "control",
        "architecture": "gfx936", "seed": 20261109, "dtype": "bf16",
        "cases": {c: base.describe(c) for c in ids}, "recipes": {"copy": copy.deepcopy(base.RECIPES["copy"])},
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contract": contract, "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL),
        "outlier_rules": dict(OUTLIER_RULES),
        "cost_model_use": ("may replace FFN default link proxies if quality_passed" if mode == "sdma_default"
                           else "control only; never replaces default-SDMA communication costs"),
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "batch01_copy_records_preserved": True, "automatic_admission": False}


def check_environment(mode):
    observed = os.environ.get("HSA_ENABLE_SDMA")
    if observed != MODES[mode]:
        raise RuntimeError(f"HSA_ENABLE_SDMA={observed!r} does not match mode {mode}")
    return {"HSA_ENABLE_SDMA": observed}


def outliers(samples):
    if not samples:
        return None
    median = statistics.median(samples)
    return {"n": len(samples),
            "over_3x_median": sum(v > OUTLIER_RULES["over_3x_median"] * median for v in samples),
            "over_1ms": sum(v > OUTLIER_RULES["over_1ms_us"] for v in samples), "max_us": max(samples)}


def evaluate(frozen, records):
    result = base.evaluate(frozen, records)
    for q in result["cases"].values():
        q["outliers"] = outliers(q["samples_us"])
    every = [v for q in result["cases"].values() for v in q["samples_us"]]
    result.update(scope=SCOPE, mode=frozen["mode"], role=frozen["role"],
                  contract_id=key_id(frozen["contract"]),
                  mode_outliers={"over_3x_case_median": sum(q["outliers"]["over_3x_median"] for q in result["cases"].values()),
                                 "over_1ms": sum(v > OUTLIER_RULES["over_1ms_us"] for v in every), "samples": len(every)},
                  data_role="mode-specific calibration candidates; modes never pooled")
    return result


def run(output, frozen, context):
    import torch
    environment = check_environment(frozen["mode"])
    if not torch.version.hip or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 and GPU1 required")
    torch.cuda.set_device(0)
    with torch.no_grad():
        runtime = base.observe_runtime(torch)
        runtime["environment"] = environment
        save(output / "runtime.json", runtime)
        entries, prepared, warmup = base.copy_graphs(torch, frozen, output, context)
        save(output / "preparation.json", prepared)
        save(output / "identity.json", base.identity(frozen, prepared, runtime))
        save(output / "warmup.json", warmup)
        records = []
        for attempt, order in enumerate(frozen["order"]):
            for case_id in order:
                entry = entries[case_id]
                context.update(stage="measurement", case=case_id, round=attempt)
                save(output / "last_stage.json", context)
                started = time.monotonic_ns()
                entry["invalidate"]()
                for d in entry["devices"]:
                    torch.cuda.synchronize(d)
                timing = base.timed(torch, entry)
                checks = entry["verify"]()
                row = {"case": case_id, "round": attempt, "started_monotonic_ns": started,
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks}
                records.append(row)
                save(output / f"attempt_{len(records)-1:03d}.json", row)
                print(json.dumps(row), flush=True)
        save(output / "result.json", {"scope": SCOPE, "mode": frozen["mode"], "complete": True,
             "records": records, "automatic_admission": False})
        return evaluate(frozen, records)


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen = base.read(directory / "plan.json")
    current = plan(frozen["mode"])
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan or source identity changed")
    runtime = base.read(directory / "runtime.json")
    if runtime["environment"] != {"HSA_ENABLE_SDMA": MODES[frozen["mode"]]}:
        raise MeasurementContractError("Recorded SDMA environment differs from the mode")
    process, result = base.read(directory / "process.json"), base.read(directory / "result.json")
    if process["python_returncode"] not in (0, 1) or result["complete"] is not True or result["mode"] != frozen["mode"]:
        raise MeasurementContractError("Incomplete process or result")
    if base.read(directory / "warmup.json") != {c: base.expected_checks(k) for c, k in frozen["cases"].items()}:
        raise MeasurementContractError("Incomplete warmup checks")
    count = sum(len(o) for o in frozen["order"])
    files = sorted(directory.glob("attempt_*.json"))
    if [p.name for p in files] != [f"attempt_{i:03d}.json" for i in range(count)]:
        raise MeasurementContractError("Missing or extra attempts")
    records = [base.read(p) for p in files]
    if records != result["records"]:
        raise MeasurementContractError("Raw attempts differ from summary")
    replay = base.identity(frozen, base.read(directory / "preparation.json"), runtime)
    if replay != base.read(directory / "identity.json"):
        raise MeasurementContractError("Case identity or contract differs")
    evaluated = evaluate(frozen, records)
    if process["python_returncode"] != (0 if evaluated["status"] == "calibration_candidates_ready" else 1):
        raise MeasurementContractError("Process result differs from recomputed quality")
    evaluated.update(identity=replay, plan_sha256=base.digest(directory / "plan.json"),
                     raw_sha256={p.name: base.digest(p) for p in sorted(directory.glob("*.json"))})
    return evaluated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run", "review"))
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "review" and args.input is None:
        parser.error("review requires --input")
    if args.phase != "review" and args.mode is None:
        parser.error(args.phase + " requires --mode")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "process.log" for p in output.iterdir()):
        parser.error("Fresh output directory required")
    context = {"stage": args.phase, "mode": args.mode, "pool": POOL}
    try:
        if args.phase == "review":
            evaluated = review(args.input)
            save(output / "evaluation.json", evaluated)
        else:
            frozen = plan(args.mode)
            save(output / "plan.json", frozen)
            if args.phase == "plan":
                check_environment(args.mode)
                print(json.dumps({"phase": "plan", "mode": args.mode, "cases": len(frozen["cases"]),
                                  "contract_id": key_id(frozen["contract"])}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"mode": evaluated["mode"], "status": evaluated["status"],
                          "quality_failed": evaluated["quality_failed"], "outliers": evaluated["mode_outliers"]}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error),
                                       "traceback": traceback.format_exc(), "context": context})
        raise


if __name__ == "__main__":
    sys.exit(main())
