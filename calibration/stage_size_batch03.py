"""Batch-03 Mul jump check: S=448 plus S=256/384/512 drift repeats.

Same stage recipes and Graph contract as batches 01/02; SiLU is kept as a
drift reference. A host-load gate is frozen in the plan and observed load is
recorded. Diagnostic only: batch-01 fit and holdout results stay unchanged,
and nothing is refitted.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from calibration import stage_size_calibration as base
from calibration import stage_size_followup as b02
from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "gpu0_stage_size_mul_jump_batch03_v1"
SIZES = (256, 384, 448, 512)
NEW_SIZES = (448,)
STAGES = ("silu", "mul")
SOURCES = ("calibration/stage_size_batch03.py", "tools/run_stage_size_batch03.sh", *base.SOURCES)
BATCH02_REVIEW = "data/stage_size_followup_mul_20261106_01/review/evaluation.json"
LOAD_GATE = {"max_load_1min": 4.0, "poll_seconds": 30, "max_wait_seconds": 3600,
             "on_timeout": "fail without GPU execution; no relaxed threshold"}


def case_ids():
    return [f"{stage}/S{size}" for size in SIZES for stage in STAGES]


def describe(case_id):
    stage, size = case_id.split("/")
    size = int(size[1:])
    return {"kind": "stage", "stage": stage, "S": size, "shape": list(base.shape(size)),
            "elements": size * base.HIDDEN, "role": "diagnostic" if size in NEW_SIZES else "drift_repeat",
            "timing_device": "cuda:0"}


def plan():
    ids = case_ids()
    return {"scope": SCOPE, "primary_device": "cuda:0", "architecture": "gfx936", "seed": 20261110,
        "dtype": "bf16", "layout": "explicit_contiguous", "hidden": base.HIDDEN, "sizes": list(SIZES),
        "cases": {c: describe(c) for c in ids},
        "recipes": {s: copy.deepcopy(base.RECIPES[s]) for s in STAGES},
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contract": base.contract(), "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL),
        "load_gate": dict(LOAD_GATE),
        "protocol": {"purpose": "is the Mul S=384->512 jump stable; drift at 256/384/512 across batches",
                     "batch01_line": "frozen fit.json; no refit", "piecewise_fit": False},
        "batch01_fit_sha256": base.digest(ROOT / b02.BATCH01_FIT),
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "existing_measurements_preserved": True, "automatic_admission": False,
        "model_parameters_updated": False}


def wait_for_quiet_host(gate, clock=time.monotonic, sleep=time.sleep, load=None):
    if load is None:
        load = os.getloadavg
    start, polls = clock(), []
    while True:
        value = load()[0]
        polls.append(round(value, 2))
        if value <= gate["max_load_1min"]:
            return {"passed": True, "load_1min_at_start": value, "waited_seconds": clock() - start,
                    "polls": polls[-20:]}
        if clock() - start >= gate["max_wait_seconds"]:
            raise RuntimeError(f"Host load stayed above {gate['max_load_1min']}; no GPU execution")
        sleep(gate["poll_seconds"])


def stage_graphs(torch, frozen, device, stream, output, context):
    generator = torch.Generator(device=device).manual_seed(frozen["seed"])
    built, prepared, warmup = {}, {}, {}
    for size in SIZES:
        dims = base.shape(size)
        gate, up = [torch.randn(dims, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
        left = torch.empty_like(gate)
        torch.ops.aten.silu.out(gate, out=left)
        inputs = {"gate": gate, "up": up, "left": left}
        snapshots = {k: t.view(torch.uint8).clone() for k, t in inputs.items()}
        for stage in STAGES:
            case_id = f"{stage}/S{size}"
            context.update(stage="capture_and_warmup", case=case_id, memory=memory_snapshot(torch, device))
            save(output / "last_stage.json", context)
            calls = [base.StageCall(stage, gate, up, left) for _ in range(base.CALLS)]
            refs = base.references(torch, stage, gate, up, left)
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
        entries, prepared, warmup = stage_graphs(torch, frozen, device, stream, output, context)
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
             "automatic_admission": False, "load_after_1min": os.getloadavg()[0]})
        return evaluate(frozen, records)


def evaluate(frozen, records):
    result = base.evaluate(frozen, records)
    loads = [r["host_load_1min"] for r in records]
    result.update(scope=SCOPE, data_role="diagnostic_only_not_fit_or_holdout",
                  host_load_1min_during_run={"min": min(loads), "max": max(loads)})
    return result


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen, current = base.read(directory / "plan.json"), plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan or source identity changed")
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


def diagnose(evaluation, fit_report, batches):
    """Per-batch residuals vs the frozen batch-01 line; drift at repeated sizes."""
    entries = {e["stage"]: e for e in fit_report["entries"]}
    passed = lambda q: q is not None and q["status"] == "quality_passed"
    out = {}
    for stage in STAGES:
        e = entries[stage]
        line = lambda n: e["a_us_composite_for_prediction"] + e["b_us_per_element"] * n
        rows = []
        for name, cases in (*batches, ("batch03", evaluation["cases"])):
            for case_id, q in cases.items():
                if not case_id.startswith(stage + "/S"):
                    continue
                size = case_id.split("/")[1]
                size = int(size[1:])
                n = size * base.HIDDEN
                m = q["statistics"]["median_us"] if q.get("statistics") else None
                rows.append({"batch": name, "S": size, "status": q["status"], "median_us": m,
                             "iqr_over_median": q["statistics"]["iqr_over_median"] if q.get("statistics") else None,
                             "line_us": line(n), "relative_residual": (m - line(n)) / line(n) if m else None,
                             "us_per_element": m / n if m else None})
        rows.sort(key=lambda r: (r["S"], r["batch"]))
        drift = {}
        for size in SIZES:
            now = evaluation["cases"].get(f"{stage}/S{size}")
            for name, cases in batches:
                then = cases.get(f"{stage}/S{size}")
                if now and then and now.get("statistics") and then.get("statistics"):
                    drift[f"S{size}:{name}->batch03"] = {
                        "relative": now["statistics"]["median_us"] / then["statistics"]["median_us"] - 1,
                        "both_quality_passed": passed(now) and passed(then)}
        out[stage] = {"rows": rows, "drift": drift}
    mul = {r["S"]: r for r in out["mul"]["rows"] if r["batch"] == "batch03"}
    jump = None
    if all(s in mul and mul[s]["median_us"] for s in (384, 448, 512)):
        jump = {"us_per_element": {s: mul[s]["us_per_element"] for s in (384, 448, 512)},
                "time_ratio_512_over_384": mul[512]["median_us"] / mul[384]["median_us"],
                "element_ratio_512_over_384": 512 / 384,
                "all_quality_passed": all(mul[s]["status"] == "quality_passed" for s in (384, 448, 512))}
    return {"scope": "diagnostic; batch-01 fit and holdout validation unchanged",
            "batch01_fit_prediction_id": fit_report["prediction_id"], "stages": out,
            "same_batch_mul_jump": jump, "refit": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run", "review", "diagnose"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.phase in ("review", "diagnose") and args.input is None:
        parser.error(args.phase + " requires --input")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "process.log" for p in output.iterdir()):
        parser.error("Fresh output directory required")
    context = {"stage": args.phase, "pool": POOL}
    try:
        if args.phase == "diagnose":
            batches = (("batch01", base.read(ROOT / b02.BATCH01_REVIEW)["cases"]),
                       ("batch02", base.read(ROOT / BATCH02_REVIEW)["cases"]))
            report = diagnose(base.read(args.input / "evaluation.json"), base.read(ROOT / b02.BATCH01_FIT), batches)
            save(output / "diagnosis.json", report)
            print(json.dumps({"phase": "diagnose", "refit": False}))
            return 0
        if args.phase == "review":
            evaluated = review(args.input)
            save(output / "evaluation.json", evaluated)
        else:
            frozen = plan()
            save(output / "plan.json", frozen)
            if args.phase == "plan":
                print(json.dumps({"phase": "plan", "cases": len(frozen["cases"]),
                                  "attempts": sum(len(o) for o in frozen["order"]),
                                  "load_gate": frozen["load_gate"], "new_gpu_execution": False}))
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
