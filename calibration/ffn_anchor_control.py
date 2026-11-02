"""Explicit, crash-isolated anchor pilot. Default phase writes a CPU plan only."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from calibration.ffn_anchor_control_protocol import PROTOCOL, endpoint_summary, frozen_protocol
from calibration.ffn_anchor_report import REPORT_RULES, build_report, render_report, endpoint_readouts
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_profile import digest
from cost_model.measurement_labels import measurement_contract_id

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pilot_plan():
    frozen = read(ROOT / "data/ffn_b4_cpu_preparation_v1/anchor_protocol.json")
    if frozen != frozen_protocol():
        raise ValueError("the frozen pilot protocol or its source has changed")
    payload = {
        "version": "ffn-anchor-collector.v2", "protocol": frozen, "report_rules": REPORT_RULES,
        "collector_implemented": True, "gpu_validated": False, "default_phase": "plan",
        "execution_entry": "explicit run phase only; no automatic GPU execution",
        "scope": "pilot_only_not_profile_admission", "automatic_admission": False,
        "followup_matrix_open": False, "worker_attempts": 1, "worker_retries": 0,
        "pipeline_failure": "stop and block remaining cells; preserve all raw evidence",
        "quality_failure": "retain and continue fixed four-cell order; never retry",
        "telemetry_position": "before pre-anchor and after post-anchor, outside drift window",
        "source_sha256": {n: digest(ROOT / n) for n in (
            "calibration/ffn_anchor_control.py", "calibration/ffn_control_runtime.py",
            "calibration/ffn_b4_runtime.py", "calibration/ffn_anchor_control_protocol.py",
            "integration/flagscale/kerv_profile_runtime.py", "integration/flagscale/kerv_execution.py",
            "calibration/validate_kerv_adapter.py", "calibration/ffn_anchor_report.py",
            "calibration/ffn_mul_metadata_smoke.py", "tools/run_ffn_anchor_control.sh",
            "cost_model/ffn_profile_export.py", "cost_model/measurement_labels.py",
            "data/ffn_measurement_b3/recipe.json")},
    }
    return {**payload, "plan_id": key_id(payload)}


def telemetry(device=0):
    """Optional API observations; unsupported HIP telemetry is explicitly null."""
    import torch
    result = {}
    for field, method, unit in (("clock_mhz", "clock_rate", "MHz"),
                                ("utilization_percent", "utilization", "%"),
                                ("temperature_c", "temperature", "C")):
        item = {"value": None, "unit": unit, "source": "torch.cuda." + method,
                "device": device, "observed_ns": time.perf_counter_ns(), "error": None}
        try:
            value = getattr(torch.cuda, method)(device)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("non-finite or non-numeric telemetry")
            item["value"] = value
        except Exception as error:
            item["error"] = repr(error)
        result[field] = item
    return result


def timed_sample(case):
    """Preserve reset/Event/sync/check order; no disk writes inside Event bounds."""
    import torch
    from calibration.ffn_b4_runtime import sync
    result = {"reset_begin": time.perf_counter_ns()}
    case.reset(0)
    sync(case.devices)
    with torch.cuda.device(case.device), torch.cuda.stream(case.stream):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        result["timing_begin"] = time.perf_counter_ns()
        start.record(case.stream)
        case.graph.replay()
        end.record(case.stream)
        end.synchronize()
        result["event_wait_end"] = time.perf_counter_ns()
    sync(case.devices)
    result["all_device_sync_end"] = time.perf_counter_ns()
    result["batch_ms"] = start.elapsed_time(end)
    checks = case.check(case.outputs, 0)
    result["validation_end"] = time.perf_counter_ns()
    if not checks or any(v is not True for v in checks.values()):
        raise AssertionError("pilot numerical or input invariant failed")
    result.update(numeric_ok=True, calls=50, invariants=checks, device=str(case.device),
                  stream=int(case.stream.cuda_stream), start_device=str(start.device), end_device=str(end.device),
                  path_evidence_state="pending_post_timing_trace")
    # Validate the required timestamps and positive timing without filtering it.
    endpoint_summary([result], 1)
    if result["device"] != "cuda:0" or result["start_device"] != "cuda:0" or result["end_device"] != "cuda:0":
        raise ValueError("pilot Event must remain on GPU0")
    return result


def review_rounds(rounds, repeats):
    if len(rounds) != 7 or [r["attempt"] for r in rounds] != list(range(7)):
        raise ValueError("pilot requires exactly seven chronological attempts")
    reviewed, targets, previous_end = [], [], -1
    for row in rounds:
        sequence = row["pre"] + [row["target"]] + row["post"]
        for sample in sequence:
            endpoint_summary([sample], 1)
            if (any(sample.get(k) != "cuda:0" for k in ("device", "start_device", "end_device")) or
                    not sample.get("invariants") or any(v is not True for v in sample["invariants"].values())):
                raise ValueError("sample device or invariant proof missing")
            if sample["reset_begin"] < previous_end:
                raise ValueError("sample windows overlap or are out of order")
            previous_end = sample["validation_end"]
        pre, post = endpoint_summary(row["pre"], repeats), endpoint_summary(row["post"], repeats)
        drift = abs(post["median_us"] - pre["median_us"]) / pre["median_us"]
        targets.append(row["target"]["batch_ms"] * 1000 / 50)
        reasons = (["pre_endpoint_IQR_above_10pct"] if not pre["endpoint_valid"] else [])
        reasons += ["post_endpoint_IQR_above_10pct"] if not post["endpoint_valid"] else []
        reasons += ["anchor_drift_above_10pct"] if drift > .10 else []
        reviewed.append({"attempt": row["attempt"], "pre": pre, "post": post, "drift": drift,
                         "valid": not reasons, "invalid_reasons": reasons})
    median = statistics.median(targets)
    quartiles = statistics.quantiles(targets, n=4, method="inclusive")
    dispersion = (quartiles[2] - quartiles[0]) / median
    return {"status": "quality_passed" if all(r["valid"] for r in reviewed) and dispersion <= .10 else "quality_failed",
            "rounds": reviewed, "target_median_us": median, "target_iqr_over_median": dispersion,
            "cell_reasons": (["invalid_anchor_rounds"] if any(not r["valid"] for r in reviewed) else []) +
                            (["target_IQR_above_10pct"] if dispersion > .10 else []),
            "extra_attempts": 0, "replacement": False}


def validate_path(path):
    names = path.get("kernel_names", [])
    if path.get("path_trace_written") is not True or path.get("host_staging_detected") is not False or len(names) != 50:
        raise ValueError("pilot requires an observed 50-kernel replay without host staging")
    if any("silu_mul_kernel" in name for name in names):
        raise ValueError("pilot projection/anchor unexpectedly used the fused kernel")


def pilot_contract(cell, identity, protocol_id):
    environment = identity["target"]["devices"]["0"]["environment"]
    return {"executor": "ffn_anchor_control_pilot", "scope": "pilot_only_not_profile_admission",
            "protocol_id": protocol_id, "cell_id": cell["id"], "target": cell["row"],
            "endpoint_replays": cell["endpoint_replays"], "calls_per_replay": 50,
            "attempts": 7, "target_warmup": 10, "anchor_warmup": 10, "extra_attempts": 0,
            "mode": "graph_replay", "unit": "us/call", "allocation_included": False,
            "submission_included": True, "granularity": "50_independent_invocations_per_graph_replay",
            "environment": {**environment, "endpoints": identity["target"]["devices"]},
            "restore_and_validation": "outside Event interval", "trace": "after all timing",
            "runtime_identity_hash": key_id(identity)}


def execute_cell(runtime, cell, output, sample=timed_sample, observe=telemetry, clock=time.perf_counter_ns):
    output = Path(output)
    context = {"anchor_preparation_begin": clock()}
    anchor = runtime.anchor(0)
    runtime.checkpoint("pilot_anchor_capture", memory_before_capture=runtime.snapshots())
    anchor.capture()
    for _ in range(10): anchor.replay(0)
    context["anchor_warmup_end"] = clock()
    context["case_preparation_begin"] = clock()
    case = runtime.make(cell["row"])
    runtime.context["memory_before_capture"] = runtime.snapshots()
    case.capture()
    warmups = [case.replay(0)[0] for _ in range(10)]
    context["case_preparation_end"] = clock()
    identity = {"target": copy.deepcopy(case.identity), "anchor": copy.deepcopy(anchor.identity)}
    contract = pilot_contract(cell, identity, frozen_protocol()["protocol_id"])
    contract_id = measurement_contract_id(contract)
    save(output / "identity.json", {"identity": identity, "identity_hash": key_id(identity),
         "measurement_contract": contract, "measurement_contract_id": contract_id})
    save(output / "preparation.json", {"context": context, "warmup_diagnostic_timings": warmups})
    rounds = []
    for attempt in range(7):
        runtime.checkpoint("pilot_round", attempt=attempt)
        before = observe(0)
        pre = [sample(anchor) for _ in range(cell["endpoint_replays"])]
        target = sample(case)
        post = [sample(anchor) for _ in range(cell["endpoint_replays"])]
        after = observe(0)
        rounds.append({"attempt": attempt, "pre": pre, "target": target, "post": post,
                       "telemetry_before": before, "telemetry_after": after})
        save(output / "samples.json", rounds)
    # Raw samples remain immutable and pending; results carry post-timing proof.
    paths = {"target": case.trace(output / "target_trace.json"),
             "anchor": anchor.trace(output / "anchor_trace.json")}
    for path in paths.values(): validate_path(path)
    quality = review_rounds(rounds, cell["endpoint_replays"])
    result = {"cell": cell, "pipeline_complete": True, "quality": quality,
              "identity_hash": key_id(identity), "context": context, "path_evidence": paths,
              "measurement_contract_id": contract_id,
              "trace_sha256": {name: digest(output / (name + "_trace.json")) for name in paths},
              "samples_sha256": digest(output / "samples.json"),
              "environment_causal_verdict": "inconclusive", "formal_profile": False,
              "automatic_admission": False, "followup_matrix_open": False}
    save(output / "result.json", result)
    return result


def review_cell(directory, cell):
    """CPU recomputation of worker result before the parent opens the next cell."""
    directory = Path(directory)
    identity_record = read(directory / "identity.json")
    identity = identity_record["identity"]
    result = read(directory / "result.json")
    if result["cell"] != cell or result["pipeline_complete"] is not True or result["formal_profile"] or result["automatic_admission"]:
        raise ValueError("worker scope/cell mismatch")
    if result["identity_hash"] != key_id(identity) or identity_record["identity_hash"] != key_id(identity):
        raise ValueError("worker identity hash mismatch")
    expected_contract = pilot_contract(cell, identity, frozen_protocol()["protocol_id"])
    expected_contract_id = measurement_contract_id(expected_contract)
    if (identity_record["measurement_contract"] != expected_contract or
            identity_record["measurement_contract_id"] != expected_contract_id or
            result["measurement_contract_id"] != expected_contract_id):
        raise ValueError("worker measurement contract mismatch")
    signatures = []
    for name in ("target", "anchor"):
        item = identity[name]
        if set(item["devices"]) != {"0"}:
            raise ValueError("unexpected pilot device")
        gpu = item["devices"]["0"]
        if not gpu["observed"].get("architecture") or gpu["observed"].get("uuid") in (None, "unavailable"):
            raise ValueError("missing device identity")
        if gpu["environment"]["device_uuid"] != gpu["observed"]["uuid"]:
            raise ValueError("environment UUID differs from device")
        signatures.append({"device": gpu, "build": item["actual_torch_build"],
                           "loaded_source": item["loaded_kerv_source"]})
        validate_path(result["path_evidence"][name])
        trace_path = directory / (name + "_trace.json")
        events = read(trace_path)["traceEvents"]
        kernels = [e.get("name", "") for e in events if e.get("cat") == "kernel"]
        if digest(trace_path) != result["trace_sha256"][name] or kernels != result["path_evidence"][name]["kernel_names"]:
            raise ValueError("raw trace differs from path proof")
    if signatures[0] != signatures[1]: raise ValueError("target/anchor runtime identity mismatch")
    rounds = read(directory / "samples.json")
    if digest(directory / "samples.json") != result["samples_sha256"] or review_rounds(rounds, cell["endpoint_replays"]) != result["quality"]:
        raise ValueError("worker quality/sample mismatch")
    readouts = endpoint_readouts(rounds, cell["endpoint_replays"])
    if len(readouts) != 14:
        raise ValueError("endpoint report is incomplete")
    context = result["context"]
    stamps = [context[n] for n in ("anchor_preparation_begin", "anchor_warmup_end", "case_preparation_begin", "case_preparation_end")]
    if any(type(s) is not int for s in stamps) or stamps != sorted(stamps) or stamps[-1] > rounds[0]["pre"][0]["reset_begin"]:
        raise ValueError("missing or reordered preparation/warmup timestamps")
    return {"pipeline_complete": True, "quality_status": result["quality"]["status"],
            "endpoint_readouts_complete": True,
            "runtime_signature": key_id(signatures[0]), "identity_hash": result["identity_hash"],
            "measurement_contract_id": expected_contract_id}


def worker_failure(directory, error, returncode=None):
    """CPU parent preserves native-worker diagnostics without touching HIP."""
    directory = Path(directory)
    try:
        last = read(directory / "last_stage.json") if (directory / "last_stage.json").exists() else {}
    except (ValueError, OSError) as read_error:
        last = {"unavailable": repr(read_error)}
    log = (directory / "worker.log").read_text(encoding="utf-8", errors="replace") if (directory / "worker.log").exists() else ""
    payload = {"error": repr(error), "returncode": returncode, "traceback": traceback.format_exc(),
        "worker_fault_traceback_log": log, "runtime_context": last,
        "pool_config": last.get("pool_config", {"selection": "per_arm_default_private_pool", "pool_argument": None, "shared_between_arms": False}),
        "memory": last.get("memory_before_capture", {"unavailable": "worker exited before snapshot"}),
        "memory_scope": "last persisted pre-capture snapshot; not a post-crash GPU observation"}
    save(directory / "parent_failure.json", payload)
    return payload


def write_report(output, rows, *, template=False):
    report = build_report(output, rows)
    prefix = "report_template" if template else "report"
    save(Path(output) / (prefix + ".json"), report)
    with (Path(output) / (prefix + ".md")).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(render_report(report))


def collect(output, flagscale_root, frozen, run_worker=subprocess.run):
    output, rows, signature, stopped = Path(output), {}, None, False
    for cell in PROTOCOL["cells"]:
        if stopped:
            rows[cell["id"]] = {"status": "blocked", "worker_attempts": 0, "reason": "earlier pipeline/identity failure"}
        else:
            directory = output / cell["id"]
            directory.mkdir()
            save(directory / "parent_plan.json", frozen)
            command = [sys.executable, "-X", "faulthandler", "-m", "calibration.ffn_anchor_control", "run",
                       "--output", str(directory), "--flagscale-root", str(flagscale_root), "--worker", cell["id"]]
            record = {"worker_attempts": 1, "retries": 0, "returncode": None}
            print(json.dumps({"pilot_cell_start": cell["id"]}), flush=True)
            try:
                with (directory / "worker.log").open("w", encoding="utf-8") as log:
                    completed = run_worker(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                record["returncode"] = completed.returncode
                if completed.returncode != 0: raise RuntimeError("worker exited unsuccessfully; see worker.log/last_stage.json")
                checked = review_cell(directory, cell)
                if signature is not None and signature != checked["runtime_signature"]:
                    raise ValueError("physical device/build changed across pilot cells")
                signature = checked["runtime_signature"]
                record.update(status=checked["quality_status"], **checked)
            except Exception as error:
                record.update(status="pipeline_failed", error=repr(error))
                worker_failure(directory, error, record["returncode"])
                stopped = True
            rows[cell["id"]] = record
        eligible = len(rows) == 4 and all(r.get("pipeline_complete") is True for r in rows.values()) and all(rows[c]["status"] == "quality_passed" for c in ("up_r3", "down_r3"))
        complete = len(rows) == 4 and all(r.get("pipeline_complete") is True and r.get("endpoint_readouts_complete") is True for r in rows.values())
        save(output / "summary.json", {"cells": rows, "pilot_passed": complete,
             "pilot_gate_passed": eligible, "r3_quality_gate_passed": eligible,
             "pilot_gate_passed_semantics": "legacy quality gate, distinct from pilot_passed completeness",
             "followup_matrix_open": False, "automatic_admission": False, "formal_profile": False,
             "environment_causal_verdict": "inconclusive", "plan_id": frozen["plan_id"]})
        write_report(output, rows)
    return read(output / "summary.json")


def worker(cell_id, output, flagscale_root):
    from calibration.ffn_control_runtime import ControlRuntime
    import torch
    frozen = read(output / "parent_plan.json")
    if pilot_plan() != frozen: raise ValueError("worker source/plan differs from parent")
    runtime = None
    try:
        with torch.no_grad():
            recipe = read(ROOT / "data/ffn_measurement_b3/recipe.json")
            runtime = ControlRuntime(output, recipe, flagscale_root)
            save(output / "runtime_identity.json", runtime.identity_record)
            cell = next(c for c in PROTOCOL["cells"] if c["id"] == cell_id)
            execute_cell(runtime, cell, output)
    except Exception as error:
        import traceback
        from calibration.probe_kerv_graph import POOL
        trace = traceback.format_exc()
        try:
            memory = runtime.snapshots() if runtime else {"unavailable": "runtime initialization did not finish"}
        except Exception as snapshot_error:
            memory = {"unavailable": repr(snapshot_error)}
        save(output / "failure.json", {"error": repr(error), "traceback": trace,
             "pool_config": dict(POOL), "memory": memory, "phase": "anchor_pilot", "cell_id": cell_id,
             "runtime_context": copy.deepcopy(getattr(runtime, "context", {}))})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flagscale-root", type=Path)
    parser.add_argument("--worker", choices=[c["id"] for c in PROTOCOL["cells"]])
    args = parser.parse_args()
    if args.phase == "run" and args.flagscale_root is None:
        parser.error("run requires the pinned --flagscale-root")
    if args.worker:
        if args.phase != "run": parser.error("worker requires run")
        worker(args.worker, args.output, args.flagscale_root)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        frozen = pilot_plan()
        save(args.output / "plan.json", frozen)
        write_report(args.output, {c["id"]: {"status": "not_run"} for c in PROTOCOL["cells"]}, template=True)
        if args.phase == "run":
            result = collect(args.output, args.flagscale_root, frozen)
            if not result["pilot_passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
