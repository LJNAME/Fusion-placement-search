"""Batch-02 Mul size diagnostic (S=192, 384, 768 plus an S=256 anchor).

Reuses the batch-01 stage recipes and Graph contract unchanged. The batch-01
collector is not edited, so its committed review stays reproducible. Data
here is diagnostic: it never enters the batch-01 fit set and never rewrites
the completed holdout validation.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from calibration import stage_size_calibration as base
from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "gpu0_stage_size_mul_diagnostic_v1"
DIAG_SIZES = (192, 384, 768)
ANCHOR = 256
SIZES = (192, ANCHOR, 384, 768)
STAGES = ("silu", "mul")
SOURCES = ("calibration/stage_size_followup.py", "tools/run_stage_size_followup.sh", *base.SOURCES)
BATCH01_FIT = "data/stage_size_calibration_20261106_01/fit/fit.json"
BATCH01_REVIEW = "data/stage_size_calibration_20261106_01/review/evaluation.json"
PROTOCOL = {
    "purpose": "test whether batch-01 line residuals vary systematically with S; Mul primary, SiLU comparison",
    "batch01_line": "frozen fit.json; no refit, no addition to its fit set",
    "anchor": "S=256 re-measured to separate cross-batch drift from size effects; not a correction",
    "reported": ["signed residual vs batch-01 line per size", "anchor drift vs batch-01 S=256",
                 "residual sign pattern over all 8 sizes (5 batch-01 + 3 new)"],
    "piecewise_fit": "not performed here; considered only if a clear nonlinear trend is reported",
    "holdout_validation_unchanged": True,
}


def case_ids():
    return [f"{stage}/S{size}" for size in SIZES for stage in STAGES]


def describe(case_id):
    stage, size = case_id.split("/")
    size = int(size[1:])
    return {"kind": "stage", "stage": stage, "S": size, "shape": list(base.shape(size)),
            "elements": size * base.HIDDEN, "role": "anchor" if size == ANCHOR else "diagnostic",
            "timing_device": "cuda:0"}


def plan():
    ids = case_ids()
    return {"scope": SCOPE, "primary_device": "cuda:0", "architecture": "gfx936", "seed": 20261108,
        "dtype": "bf16", "layout": "explicit_contiguous", "hidden": base.HIDDEN, "sizes": list(SIZES),
        "cases": {c: describe(c) for c in ids},
        "recipes": {s: copy.deepcopy(base.RECIPES[s]) for s in STAGES},
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contract": base.contract(), "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL),
        "diagnostic_protocol": copy.deepcopy(PROTOCOL),
        "batch01_fit_sha256": base.digest(ROOT / BATCH01_FIT),
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "existing_measurements_preserved": True, "automatic_admission": False,
        "model_parameters_updated": False}


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
    import torch
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 and GPU1 required; no CPU/eager fallback")
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("Known profiling/debug instrumentation detected")
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
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks}
                records.append(row)
                save(output / f"attempt_{len(records)-1:03d}.json", row)
                print(json.dumps(row), flush=True)
        save(output / "result.json", {"scope": SCOPE, "complete": True, "records": records,
             "automatic_admission": False, "known_instrumentation_guard_passed": True})
        return evaluate(frozen, records)


def evaluate(frozen, records):
    result = base.evaluate(frozen, records)
    result["scope"] = SCOPE
    result["data_role"] = "diagnostic_only_not_fit_or_holdout"
    return result


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen, current = base.read(directory / "plan.json"), plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan or source identity changed")
    process, result = base.read(directory / "process.json"), base.read(directory / "result.json")
    if (process["python_returncode"] not in (0, 1) or process["log_returncode"] != 0
            or result["scope"] != SCOPE or result["complete"] is not True):
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


def diagnose(evaluation, fit_report, batch01):
    """Residuals against the frozen batch-01 line; nothing is refitted."""
    entries = {e["stage"]: e for e in fit_report["entries"]}
    med = lambda cases, c: cases[c]["statistics"]["median_us"] if cases.get(c, {}).get("status") == "quality_passed" else None
    out = {}
    for stage in STAGES:
        e = entries[stage]
        line = lambda n: e["a_us_composite_for_prediction"] + e["b_us_per_element"] * n
        rows = []
        for batch, cases, sizes in (("batch01", batch01["cases"], base.SIZES), ("batch02", evaluation["cases"], SIZES)):
            for size in sizes:
                n, m = size * base.HIDDEN, med(cases, f"{stage}/S{size}")
                role = (("fit" if size in base.FIT_SIZES else "holdout") if batch == "batch01"
                        else ("anchor" if size == ANCHOR else "diagnostic"))
                rows.append({"batch": batch, "S": size, "role": role, "measured_us": m, "line_us": line(n),
                             "residual_us": m - line(n) if m is not None else None,
                             "relative_residual": (m - line(n)) / line(n) if m is not None else None})
        rows.sort(key=lambda r: (r["S"], r["batch"]))
        a01, a02 = med(batch01["cases"], f"{stage}/S{ANCHOR}"), med(evaluation["cases"], f"{stage}/S{ANCHOR}")
        out[stage] = {"rows": rows, "anchor_S256": {"batch01_us": a01, "batch02_us": a02,
                      "drift": (a02 - a01) / a01 if a01 and a02 else None},
                      "residual_signs_by_S": [[r["S"], r["batch"], None if r["residual_us"] is None
                                              else ("+" if r["residual_us"] > 0 else "-")] for r in rows]}
    return {"scope": "diagnostic; batch-01 fit and holdout validation unchanged",
            "batch01_fit_prediction_id": fit_report["prediction_id"], "stages": out, "refit": False}


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
            report = diagnose(base.read(args.input / "evaluation.json"), base.read(ROOT / BATCH01_FIT),
                              base.read(ROOT / BATCH01_REVIEW))
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
                                  "attempts": sum(len(o) for o in frozen["order"]), "new_gpu_execution": False}))
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
