"""GPU0 complete-call collection; separate CPU admission, never profiler timing."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from calibration.lightop_dispatch import SOURCES as DISPATCH_SOURCES
from calibration.lightop_profile_smoke import (POOL, check_outputs, invalidate_buffers,
                                               memory_snapshot, placed_case, save)
from cost_model.fusion_kernel_profile import KernelLatency, key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract, selected_group_profile
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from integration.lightop.silu_mul import SHAPE, adapter_registry
from simulator.placement_artifacts import load_placed_op_graph, save_placed_op_graph

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/lightop_dispatch_20261104_02_review"
SCOPE = "lightop_gpu0_complete_adapter_call_calibration_v1"
SOURCES = (*DISPATCH_SOURCES, "calibration/lightop_profile_calibration.py",
           "tools/run_lightop_profile_calibration.sh")
QUALITY = {"attempts": 7, "extra_attempts": 0, "sample_replacement": False,
           "iqr_method": "inclusive", "max_iqr_over_median": .10,
           "anchor_policy": "no separate anchors; no cross-window drift certification",
           "performance_gain_required": False}
CHECKS = {"outputs_checked": 50, "references_per_output": 2,
          "input_bytes_unchanged": True, "outputs_independent": True}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dispatch_evidence():
    report = read(EVIDENCE / "report.json")
    trace = report["dispatch"]
    if (report["review_passed"] is not True or trace["dispatch_observed"] is not True
            or trace["region"]["roles"] != {"split_input_pack": 50, "lightop_silu_mul": 50}
            or trace["target_correctness"] != CHECKS or not trace["database_csv_consistent"]):
        raise MeasurementContractError("reviewed dispatch evidence is incomplete")
    # Preserve the exact implementation and all seven source files seen by dispatch02.
    expected = report["checks"]["collector_sources_match_git"]
    if set(expected) != set(DISPATCH_SOURCES) or not all(expected.values()):
        raise MeasurementContractError("dispatch source review is incomplete")
    for path in DISPATCH_SOURCES:
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", report["collector_revision"] + ":" + path])
        if hashlib.sha256(raw).hexdigest() != digest(ROOT / path):
            raise MeasurementContractError("frozen dispatch implementation changed: " + path)
    return report


def plan():
    evidence = dispatch_evidence()
    return {"scope": SCOPE, "device": "cuda:0", "seed": 20261104,
            "expected_kernel_key_id": evidence["identity"]["kernel_key_id"],
            "expected_smoke_contract_id": evidence["identity"]["smoke_contract_id"],
            "dispatch_evidence_sha256": {p.name: digest(p) for p in sorted(EVIDENCE.glob("*.json"))},
            "collector": {"git_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {p: digest(ROOT / p) for p in SOURCES}},
            "measurement_template": measurement_contract({}, purpose="calibration"),
            "quality_policy": QUALITY, "pool": POOL,
            "profiling": "separate previous dispatch process; not launched here",
            "automatic_admission": False, "full_ffn_or_model_executed": False}


def quality(records):
    if len(records) != 7 or [r["round"] for r in records] != list(range(7)):
        raise MeasurementContractError("exactly seven chronological attempts required; no replacement")
    reasons, values = [], []
    for row in records:
        if row["checks"] != CHECKS:
            raise MeasurementContractError("correctness/storage checks incomplete")
        boundary = row["timing"]["event_boundary"]
        if (boundary["device"] != "cuda:0" or boundary["start_device"] != "cuda:0"
                or boundary["end_device"] != "cuda:0" or boundary["calls_per_replay"] != 50
                or type(boundary["stream"]) is not int):
            raise MeasurementContractError("Event boundary does not cover the GPU0 complete-call graph")
        elapsed = row["timing"]["elapsed_ms_per_50_calls"]
        if (row["timing"]["error"] is not None or type(elapsed) not in (float, int)
                or not math.isfinite(elapsed) or elapsed <= 0):
            reasons.append("invalid_timing_round_" + str(row["round"]))
        else:
            values.append(elapsed * 1000 / 50)
        if (type(row["started_monotonic_ns"]) is not int or type(row["finished_monotonic_ns"]) is not int
                or row["finished_monotonic_ns"] <= row["started_monotonic_ns"]):
            raise MeasurementContractError("missing round timestamps")
    if len({r["timing"]["event_boundary"]["stream"] for r in records}) != 1:
        raise MeasurementContractError("formal rounds changed stream")
    if any(a["finished_monotonic_ns"] > b["started_monotonic_ns"] for a, b in zip(records, records[1:])):
        raise MeasurementContractError("formal rounds overlap or were reordered")
    stats = None
    if not reasons:
        median = statistics.median(values)
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        stats = {"median_us": median, "min_us": min(values), "max_us": max(values),
                 "iqr_us": q3 - q1, "iqr_over_median": (q3 - q1) / median,
                 "mad_rel": statistics.median(abs(v - median) for v in values) / median,
                 "n_rounds": 7, "n_calls": 350}
        if stats["iqr_over_median"] > QUALITY["max_iqr_over_median"]:
            reasons.append("iqr_over_median_exceeds_10_percent")
    return {"status": "quality_failed" if reasons else "quality_passed",
            "reasons": reasons, "statistics": stats, "samples_us": values}


def timed_replay(torch, graph, stream, device):
    """One attempt, even on Event readout failure; GPU execution errors propagate."""
    elapsed, error, start, end = None, None, None, None
    try:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
    except Exception as exc:
        error = repr(exc)
    graph.replay()
    if error is None:
        try:
            end.record(stream)
            end.synchronize()
        except Exception as exc:
            error = repr(exc)
    torch.cuda.synchronize(device)
    if error is None:
        try:
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("nonpositive or nonfinite Event duration")
        except Exception as exc:
            elapsed, error = None, repr(exc)
    return {"elapsed_ms_per_50_calls": elapsed, "error": error,
            "event_boundary": {"device": str(device), "stream": int(stream.cuda_stream),
                               "start_device": str(start.device) if start and error is None else str(device),
                               "end_device": str(end.device) if end and error is None else str(device),
                               "calls_per_replay": 50}}


def run(output, frozen, context):
    import torch
    import torch.nn.functional as F
    from integration.lightop.profile_identity import observe_runtime
    from integration.lightop.silu_mul import LightopSiluMulAdapter

    context.update(stage="runtime_guard", device="cuda:0", pool=POOL)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU0 required; no CPU/eager/native fallback")
    # This records a specific guard, not a claim to detect every external profiler.
    profiler_environment = {k: v for k, v in os.environ.items() if v and
                            (k in {"ROCP_TOOL_LIB", "ROCPROFILER_TOOL_LIBRARIES", "HSA_TOOLS_LIB"}
                             or (k == "LD_PRELOAD" and any(s in v.lower() for s in ("prof", "trace"))))}
    if profiler_environment or sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("profiling/debug instrumentation detected; preserve this run, do not time it")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = LightopSiluMulAdapter()
    placed, index = placed_case(0)
    bound = adapter.bind(placed, index, {0: device})
    save_placed_op_graph(output / "placed.json", placed)
    actual = observe_runtime(adapter, device)
    save(output / "runtime.json", actual)
    context["stage"] = "input_and_buffer_preparation"
    generator = torch.Generator(device=device).manual_seed(frozen["seed"])
    gate = torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=generator)
    up = torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=generator)
    snapshots = [t.view(torch.uint8).clone() for t in (gate, up)]
    references = (F.silu(gate) * up, F.silu(gate.float()) * up.float())
    calls = [bound.prepare({bound.gate_node: gate, bound.up_node: up}) for _ in range(50)]
    if any(c.observation != calls[0].observation for c in calls):
        raise ValueError("prepared call identities disagree")
    keys = LightopKernelKeys(placed, index, calls[0].observation, actual)
    key, gaps = keys.describe(placed.plan.groups[index])
    smoke_contract = measurement_contract(actual["environment"])
    contract = measurement_contract(actual["environment"], purpose="calibration")
    identity = {"kernel_key": key, "kernel_key_id": key_id(key), "identity_gaps": list(gaps),
                "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract),
                "group_index": index, "smoke_contract_id": measurement_contract_id(smoke_contract)}
    save(output / "identity.json", identity)
    save(output / "preparation.json", {"one_call": calls[0].observation, "call_count": 50})
    if (gaps or key_id(key) != frozen["expected_kernel_key_id"]
            or identity["smoke_contract_id"] != frozen["expected_smoke_contract_id"]):
        raise MeasurementContractError("runtime/key differs from the independently dispatched case")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    context["stage"] = "side_stream_eager_warmup"
    save(output / "last_stage.json", context)
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.synchronize(device)
    context.update(stage="capture", memory_before_capture=memory_snapshot(torch, device))
    save(output / "last_stage.json", context)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=None):
        for call in calls:
            call()
    torch.cuda.synchronize(device)
    context["stage"] = "graph_warmup"
    save(output / "last_stage.json", context)
    with torch.cuda.stream(stream):
        invalidate_buffers(calls)
        for _ in range(10):
            graph.replay()
    torch.cuda.synchronize(device)
    warmup_checks = check_outputs(torch, calls, gate, up, snapshots, references, contract["correctness"])
    records = []
    for attempt in range(7):
        context.update(stage="formal_round", round=attempt)
        save(output / "last_stage.json", context)
        started = time.monotonic_ns()
        with torch.cuda.stream(stream):
            invalidate_buffers(calls)
            torch.cuda.synchronize(device)
            timing = timed_replay(torch, graph, stream, device)
        checks = check_outputs(torch, calls, gate, up, snapshots, references, contract["correctness"])
        record = {"round": attempt, "started_monotonic_ns": started,
                  "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks}
        records.append(record)
        save(output / f"round_{attempt}.json", record)
        print(json.dumps(record), flush=True)
    result = {"scope": SCOPE, "complete": True, "attempts": 7, "capture_completed": True,
              "eager_warmup_calls": 50, "warmup_replays": 10, "warmup_checks": warmup_checks,
              "instrumentation_guard": {"profiler_environment": profiler_environment,
                                        "python_profile_or_trace": False, "profiler_launched": False},
              "rounds": records, "quality": quality(records), "admission": "conditional_dev",
              "collector": frozen["collector"], "automatic_admission": False}
    save(output / "result.json", result)
    return result


def review_receipt(directory):
    """CPU-only: recompute identity, quality, label and an exact cost query."""
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("collection has a failure record")
    frozen, result, actual, prep, identity = [read(directory / p) for p in
        ("plan.json", "result.json", "runtime.json", "preparation.json", "identity.json")]
    current = plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current or result["collector"] != frozen["collector"]:
        raise MeasurementContractError("collection plan/source/evidence no longer matches")
    process = read(directory / "process.json")
    if (process.get("python_returncode") not in (0, 1) or process.get("log_returncode") != 0
            or process.get("automatic_admission") is not False
            or result["scope"] != SCOPE or result["complete"] is not True or result["attempts"] != 7
            or result["capture_completed"] is not True or result["eager_warmup_calls"] != 50
            or result["warmup_replays"] != 10 or result["warmup_checks"] != CHECKS
            or result["automatic_admission"] is not False or result["admission"] != "conditional_dev"
            or result["instrumentation_guard"] != {"profiler_environment": {},
                "python_profile_or_trace": False, "profiler_launched": False}):
        raise MeasurementContractError("collection completion/guard/warmup evidence is incomplete")
    records = [read(directory / f"round_{i}.json") for i in range(7)]
    if sorted(p.name for p in directory.glob("round_*.json")) != [f"round_{i}.json" for i in range(7)]:
        raise MeasurementContractError("extra or missing rounds; no replacement")
    if records != result["rounds"]:
        raise MeasurementContractError("raw round records differ from summary")
    evaluation = quality(records)
    if evaluation != result["quality"]:
        raise MeasurementContractError("quality/statistics mismatch")
    if evaluation["status"] == "quality_passed" and process["python_returncode"] != 0:
        raise MeasurementContractError("a successful admission requires a successful collection process")
    placed = load_placed_op_graph(directory / "placed.json", registry=adapter_registry())
    if prep["call_count"] != 50:
        raise MeasurementContractError("wrong buffer pool size")
    keys = LightopKernelKeys(placed, identity["group_index"], prep["one_call"], actual)
    group = placed.plan.groups[identity["group_index"]]
    key, gaps = keys.describe(group)
    contract = measurement_contract(actual["environment"], purpose="calibration")
    if (gaps or identity["identity_gaps"] or identity["kernel_key"] != key
            or key_id(key) != identity["kernel_key_id"] or key_id(key) != frozen["expected_kernel_key_id"]
            or identity["measurement_contract"] != contract
            or identity["measurement_contract_id"] != measurement_contract_id(contract)
            or identity["smoke_contract_id"] != measurement_contract_id(measurement_contract(actual["environment"]))
            or identity["smoke_contract_id"] != frozen["expected_smoke_contract_id"]):
        raise MeasurementContractError("actual implementation/case/contract identity mismatch")
    raw_hashes = {p.name: digest(p) for p in sorted(directory.glob("*.json"))}
    decision = {"scope": SCOPE, "status": evaluation["status"], "quality_policy": QUALITY,
                "quality": evaluation, "kernel_key_id": key_id(key),
                "measurement_contract_id": measurement_contract_id(contract),
                "dispatch_evidence_sha256": frozen["dispatch_evidence_sha256"],
                "raw_file_sha256": raw_hashes, "independent_end_to_end_validation": False,
                "performance_gain_checked": False, "automatic_admission": False}
    profile = None
    if evaluation["status"] == "quality_passed":
        provenance = "LightOP GPU0 collection " + key_id(raw_hashes)
        entry = KernelLatency(evaluation["statistics"]["median_us"] / 1e6, "measured", provenance,
                              measurement_contract_id(contract), evaluation["statistics"], "accepted")
        profile = selected_group_profile(keys, contract, {key_id(key): entry})
        cost = profile.estimate_group(keys.graph, group)
        decision.update(status="accepted", latency=asdict(entry),
                        cost_query={"duration_s": cost.duration, "source": cost.source,
                                    "launch_time": cost.launch_time, "memory_time": cost.memory_time})
    decision["admission_id"] = key_id(decision)
    return decision, profile


def load_reviewed_profile(directory):
    """No trusting an edited accepted flag: replay CPU admission from raw receipts."""
    decision, profile = review_receipt(directory)
    if profile is None:
        raise MeasurementContractError("formal data failed quality admission: " + repr(decision["quality"]["reasons"]))
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run", "review"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "review" and args.input is None:
        parser.error("review requires --input raw collection directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "process.log" for p in output.iterdir()):
        parser.error("fresh output required; preserve previous receipts")
    context = {"stage": args.phase, "device": "cuda:0", "pool": POOL}
    try:
        if args.phase == "review":
            decision, _ = review_receipt(args.input)
            save(output / "admission.json", decision)
            print(json.dumps({"status": decision["status"], "admission_id": decision["admission_id"]}))
            if decision["status"] != "accepted":
                raise SystemExit(1)
        else:
            frozen = plan()
            save(output / "plan.json", frozen)
            if args.phase == "run":
                result = run(output, frozen, context)
                print(json.dumps({"status": result["quality"]["status"], "admission": "conditional_dev"}))
                if result["quality"]["status"] != "quality_passed":
                    raise SystemExit(1)
    except Exception as error:
        torch = sys.modules.get("torch")
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
            "context": context, "pool": POOL, "memory_at_failure":
            memory_snapshot(torch, context["device"]) if torch and args.phase == "run" else None,
            "automatic_admission": False})
        raise


if __name__ == "__main__":
    main()
