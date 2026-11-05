"""Independent Mul holdout at S=320/416/640/896 for the frozen linear vs two-segment predictions.

Predictions come from data/mul_piecewise_v1/report.json and are pinned by
prediction_id in plan.json before GPU work. Same Mul recipe and Graph
contract as batches 01-03, same host-load gate as batch-03. Nothing is
refitted; the evaluation compares the frozen predictions only.
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
from calibration.stage_size_batch03 import LOAD_GATE, wait_for_quiet_host
from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "gpu0_mul_independent_holdout_v1"
SIZES = (320, 416, 640, 896)
PREDICTIONS = "data/mul_piecewise_v1/report.json"
SOURCES = ("calibration/mul_holdout.py", "tools/run_mul_holdout.sh", "calibration/stage_size_batch03.py",
           *base.SOURCES)


def frozen_predictions():
    report = json.loads((ROOT / PREDICTIONS).read_text(encoding="utf-8"))
    predictions = report["frozen_holdout_predictions"]
    if key_id(predictions) != report["frozen_holdout_prediction_id"]:
        raise MeasurementContractError("Frozen Mul predictions changed")
    if set(predictions) != {f"mul/S{s}" for s in SIZES}:
        raise MeasurementContractError("Holdout sizes differ from the frozen predictions")
    return predictions, report["frozen_holdout_prediction_id"]


def describe(case_id):
    size = int(case_id.split("S")[1])
    return {"kind": "stage", "stage": "mul", "S": size, "shape": list(base.shape(size)),
            "elements": size * base.HIDDEN, "role": "independent_holdout", "timing_device": "cuda:0"}


def plan():
    ids = [f"mul/S{s}" for s in SIZES]
    predictions, prediction_id = frozen_predictions()
    return {"scope": SCOPE, "primary_device": "cuda:0", "architecture": "gfx936", "seed": 20261112,
        "dtype": "bf16", "layout": "explicit_contiguous", "hidden": base.HIDDEN, "sizes": list(SIZES),
        "cases": {c: describe(c) for c in ids}, "recipes": {"mul": copy.deepcopy(base.RECIPES["mul"])},
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contract": base.contract(), "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL),
        "load_gate": dict(LOAD_GATE),
        "frozen_predictions": predictions, "frozen_prediction_id": prediction_id,
        "comparison": {"models": ["baseline_linear_us", "piecewise_us"], "metric": "APE and signed error per size",
                       "replacement_threshold": None, "refit": False},
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "existing_measurements_preserved": True, "automatic_admission": False, "model_parameters_updated": False}


def build(torch, frozen, device, stream, output, context):
    generator = torch.Generator(device=device).manual_seed(frozen["seed"])
    built, prepared, warmup = {}, {}, {}
    for case_id, case in frozen["cases"].items():
        gate, up = [torch.randn(case["shape"], device=device, dtype=torch.bfloat16, generator=generator)
                    for _ in range(2)]
        left = torch.empty_like(gate)
        torch.ops.aten.silu.out(gate, out=left)
        inputs = {"gate": gate, "up": up, "left": left}
        snapshots = {k: t.view(torch.uint8).clone() for k, t in inputs.items()}
        context.update(stage="capture_and_warmup", case=case_id, memory=memory_snapshot(torch, device))
        save(output / "last_stage.json", context)
        calls = [base.StageCall("mul", gate, up, left) for _ in range(base.CALLS)]
        refs = base.references(torch, "mul", gate, up, left)
        with torch.cuda.stream(stream):
            for call in calls:
                call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=None):
            for call in calls:
                call()
        with torch.cuda.stream(stream):
            for call in calls:
                call.invalidate()
            for _ in range(base.WARMUP):
                graph.replay()
        torch.cuda.synchronize(device)
        warmup[case_id] = base.check_stage(torch, calls, inputs, snapshots, refs)

        def invalidate(calls=calls):
            with torch.cuda.stream(stream):
                for call in calls:
                    call.invalidate()

        def verify(calls=calls, inputs=inputs, snapshots=snapshots, refs=refs):
            return base.check_stage(torch, calls, inputs, snapshots, refs)

        built[case_id] = {"graph": graph, "stream": stream, "devices": (0,), "timing_device": device,
                          "invalidate": invalidate, "verify": verify}
        prepared[case_id] = calls[0].observation
    return built, prepared, warmup


def run(output, frozen, context):
    gate = wait_for_quiet_host(frozen["load_gate"])
    save(output / "load_gate.json", gate)
    import torch
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 and GPU1 required; no CPU/eager fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with torch.no_grad():
        runtime = base.observe_runtime(torch)
        save(output / "runtime.json", runtime)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        entries, prepared, warmup = build(torch, frozen, device, stream, output, context)
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
                torch.cuda.synchronize(device)
                timing = base.timed(torch, entry)
                checks = entry["verify"]()
                row = {"case": case_id, "round": attempt, "started_monotonic_ns": started,
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks,
                       "host_load_1min": os.getloadavg()[0]}
                records.append(row)
                save(output / f"attempt_{len(records)-1:03d}.json", row)
                print(json.dumps(row), flush=True)
        save(output / "result.json", {"scope": SCOPE, "complete": True, "records": records,
                                      "automatic_admission": False})
        return evaluate(frozen, records)


def compare(frozen, cases):
    rows = []
    for case_id, prediction in frozen["frozen_predictions"].items():
        q = cases[case_id]
        measured = q["statistics"]["median_us"] if q["status"] == "quality_passed" else None
        row = {"case": case_id, "S": frozen["cases"][case_id]["S"], "quality": q["status"], "measured_us": measured}
        for model in frozen["comparison"]["models"]:
            p = prediction[model]
            row[model] = p
            row[model + "_signed_error"] = None if measured is None else (p - measured) / measured
        rows.append(row)
    summary = {}
    for model in frozen["comparison"]["models"]:
        errors = [r[model + "_signed_error"] for r in rows if r[model + "_signed_error"] is not None]
        summary[model] = {"coverage": f"{len(errors)}/{len(rows)}",
                          "mape": statistics.fmean(abs(e) for e in errors) if errors else None,
                          "signed_bias": statistics.fmean(errors) if errors else None,
                          "max_ape": max(abs(e) for e in errors) if errors else None}
    return {"rows": rows, "summary": summary, "prediction_id": frozen["frozen_prediction_id"], "refit": False}


def evaluate(frozen, records):
    result = base.evaluate(frozen, records)
    loads = [r["host_load_1min"] for r in records]
    result.update(scope=SCOPE, data_role="independent holdout for frozen Mul predictions",
                  host_load_1min_during_run={"min": min(loads), "max": max(loads)},
                  holdout_comparison=compare(frozen, result["cases"]))
    return result


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen, current = base.read(directory / "plan.json"), plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan, frozen predictions or source identity changed")
    if base.read(directory / "load_gate.json")["passed"] is not True:
        raise MeasurementContractError("Load gate not passed")
    process, result = base.read(directory / "process.json"), base.read(directory / "result.json")
    if process["python_returncode"] not in (0, 1) or result["scope"] != SCOPE or result["complete"] is not True:
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
    replay = base.identity(frozen, base.read(directory / "preparation.json"), base.read(directory / "runtime.json"))
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "review" and args.input is None:
        parser.error("review requires --input")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "process.log" for p in output.iterdir()):
        parser.error("Fresh output directory required")
    context = {"stage": args.phase, "pool": POOL}
    try:
        if args.phase == "review":
            evaluated = review(args.input)
            save(output / "evaluation.json", evaluated)
        else:
            frozen = plan()
            save(output / "plan.json", frozen)
            if args.phase == "plan":
                print(json.dumps({"phase": "plan", "cases": len(frozen["cases"]),
                                  "prediction_id": frozen["frozen_prediction_id"], "new_gpu_execution": False}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"status": evaluated["status"], "quality_failed": evaluated["quality_failed"]}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
                                       "context": context, "automatic_admission": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
