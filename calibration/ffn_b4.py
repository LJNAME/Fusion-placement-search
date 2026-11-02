"""B4 plan / fail-fast smoke / fixed-seven formal grid on two HIP GPUs.

Default command is explicit: plan and smoke do not start formal collection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import traceback

from calibration.ffn_b4_policy import POLICY, active_recipe, collect_grid, smoke_pass, formal_quality_review
from calibration.ffn_b4_runtime import Runtime, PathViolation
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from calibration.probe_kerv_graph import POOL
from cost_model.ffn_measurement_b3 import NAMES, verify as verify_b3
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_profile import digest
from cost_model.measurement_labels import measurement_contract_id

ROOT = Path(__file__).resolve().parents[1]
B3 = ROOT / "data/ffn_measurement_b3"


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plan(flagscale_root=DEFAULT_FLAGSCALE):
    bundle = {name: json.loads((B3/name).read_text(encoding="utf-8")) for name in NAMES}
    saved = json.loads((B3/"verification.json").read_text(encoding="utf-8"))
    # Frozen B1 predates the root argument. Supply its lookup root only during
    # this synchronous legacy verification; keep source and receipt hashes intact.
    from cost_model import ffn_placement_b1 as legacy_b1
    previous_root = legacy_b1.DEFAULT_FLAGSCALE
    try:
        legacy_b1.DEFAULT_FLAGSCALE = Path(flagscale_root)
        verified = verify_b3(bundle)
    finally:
        legacy_b1.DEFAULT_FLAGSCALE = previous_root
    if key_id(verified) != key_id(saved):
        raise ValueError("B3 frozen evidence/source identity changed")
    superseded = [{**r, "superseded_by": "c4fc415", "supersession_scope": "new B3/B4 recipe only; historical accepted admission unchanged"}
                  for r in bundle["reuse_audit.json"]["records"]]
    recipe = active_recipe(bundle["recipe.json"])
    matrix = copy.deepcopy(bundle["measurement_matrix.json"])
    matrix["contract_template"].update(recipe_id=recipe["recipe_id"], max_attempts=7, extra_attempts=0)
    for row in matrix["core_rows"] + matrix["diagnostics"]:
        row["base_b3_recipe_id"], row["recipe_id"] = row["recipe_id"], recipe["recipe_id"]
        if row["pair_key"]:
            row["pair_key"]["recipe_id"] = recipe["recipe_id"]
            row["pair_key"]["measurement_contract"]["template_id"] = key_id(matrix["contract_template"])
            row["pair_key_template_id"] = key_id(row["pair_key"])
    return {"policy": {**copy.deepcopy(POLICY), "policy_id": key_id(POLICY)},
            "active_recipe": recipe,
            "b3_source_sha256": {name: digest(B3/name) for name in (*NAMES, "verification.json")},
            "source_sha256": {name: digest(ROOT/name) for name in (
                "calibration/ffn_b4.py", "calibration/ffn_b4_runtime.py", "calibration/ffn_b4_policy.py",
                "calibration/ffn_b4_supervisor.py",
                "execution/ffn_transfer.py", "integration/flagscale/kerv_execution.py")},
            "matrix": matrix, "superseded_profiles": superseded,
            "gpu_validated": False, "smoke_minimum_is_not_full_matrix_certification": True,
            "automatic_admission": False}


def write_cpu_plan(output, frozen=None):
    frozen = frozen or plan()
    output = Path(output)
    save(output/"plan.json", frozen)
    save(output/"superseded_profiles.json", {"records": frozen["superseded_profiles"], "original_sources_unchanged": True})
    save(output/"verification.json", {"scope": "CPU execution-plan verification only", "cpu_plan_ready": True,
         "minimum_smoke_cells": 4, "formal_cells": 19, "formal_attempts": 7, "extra_attempts": 0,
         "superseded_profiles_retained": 2, "gpu_smoke_executed": False, "formal_executed": False,
         "automatic_admission": False, "policy_id": frozen["policy"]["policy_id"], "plan_id": key_id(frozen)})
    return frozen


def failure(error, runtime=None, **context):
    original = traceback.format_exc()
    try:
        if runtime:
            memory = runtime.snapshots()
        else:
            import torch
            from calibration.probe_kerv_graph import memory_snapshot
            memory = {str(g): memory_snapshot(torch, g) for g in range(min(2, torch.cuda.device_count()))} if torch.cuda.is_available() else {"unavailable": "no GPU runtime"}
    except Exception as snapshot_error:
        memory = {"unavailable": repr(snapshot_error)}
    return {"error": repr(error), "traceback": original, "pool_config": dict(POOL), "memory": memory,
            "runtime_context": copy.deepcopy(getattr(runtime, "context", {})), **context}


def run_smoke(row_ids, run_cell, output):
    """Reusable fail-fast controller, tested without GPU fixtures."""
    reports = []
    for row_id in row_ids:
        try:
            report = run_cell(row_id)
            if not smoke_pass(report):
                raise AssertionError("smoke boolean criteria failed: " + row_id)
            reports.append(report)
            save(Path(output)/"partial.json", {"cells": reports})
        except Exception as error:
            saved = failure(error, phase="smoke", row_id=row_id, cells=reports, formal_started=False)
            save(Path(output)/"failure.json", saved)
            raise
    result = {"status": "passed", "scope": "minimum four categories only", "cells": reports,
              "timing_quality_checked": False, "formal_admission": False}
    save(Path(output)/"result.json", result)
    return result


def contract(row, case, frozen):
    # Native/Fused pair members have the same candidate, environment and recipe.
    primary = case.identity["devices"][str(case.device.index)]["environment"]
    return {"executor": "ffn_b4_graph_replay", "submission_included": True, "allocation_included": False,
            "granularity": "50_independent_invocations_per_graph_replay", "unit": "us/call",
            "environment": {"torch": primary["torch"], "hip": primary["hip"], "triton": primary["triton"],
                            "device_model": primary["device_model"], "endpoints": case.identity["devices"]},
            "candidate_id": row["candidate_id"],
            "recipe_id": frozen["active_recipe"]["recipe_id"], "calls_per_round": 50,
            "rounds": 7, "warmup_replays": 10, "extra_attempts": 0,
            "uncaptured_initialization_bodies": 1, "path_trace_replays_before_formal": 1,
            "restore_scope": "outside capture/replay Event interval", "retention": "all 50 outputs"}


def finite_us(timing):
    value = timing["elapsed_ms"]
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("invalid formal timing")
    return value * 1000 / 50


def execute_smoke(runtime, frozen, output):
    rows = {r["row_id"]: r for r in frozen["matrix"]["core_rows"] + frozen["matrix"]["diagnostics"]}
    def run_cell(row_id):
        runtime.release("smoke_activation" if not row_id.startswith("copy/") else "smoke_link")
        directory = Path(output)/row_id.replace("/", "_").replace(">", "to")
        directory.mkdir(parents=True, exist_ok=True)
        try:
            case = runtime.make(row_id)
            case_contract = contract(rows[row_id], case, frozen)
            identity_hash, contract_hash = key_id(case.identity), measurement_contract_id(case_contract)
            save(directory/"identity.json", {"identity": case.identity, "identity_hash": identity_hash,
                 "contract": case_contract, "measurement_contract_id": contract_hash,
                 "source_sha256": frozen["source_sha256"]})
            runtime.context["stage"] = "capture"
            runtime.context["memory_before_capture"] = runtime.snapshots()
            case.capture()
            identity_hash = key_id(case.identity)
            save(directory/"identity.json", {"identity": case.identity, "identity_hash": identity_hash,
                 "contract": case_contract, "measurement_contract_id": contract_hash, "source_sha256": frozen["source_sha256"]})
            diagnostics, invariant_checks = [], []
            for replay in range(3):
                runtime.context.update(stage="smoke_replay", replay=replay)
                timing, checks = case.replay(replay, diagnostic=True)
                diagnostics.append(timing)
                invariant_checks.append(checks)
                print(json.dumps({"phase": "smoke", "row_id": row_id, "replay": replay, "diagnostic_timing": timing}), flush=True)
            # Keep profiler/route observability separate from boolean pipe checks.
            try:
                path = case.trace(directory/"replay_trace.json")
            except PathViolation:
                raise
            except Exception as error:
                path = {"trace_available": False, "error": repr(error), "traceback": traceback.format_exc(),
                        "formal_path_review_required": True}
            result = {"row_id": row_id, "capture_ok": True, "numeric_ok": True, "invariants_ok": True,
                      "identity_written": True, "contract_written": True, "identity_hash": identity_hash,
                      "measurement_contract_id": contract_hash, "invariant_checks": invariant_checks,
                      "diagnostic_timings": diagnostics, "path_evidence": path,
                      "pool_config": dict(POOL), "memory": runtime.snapshots()}
            save(directory/"result.json", result)
            return result
        except Exception as error:
            save(directory/"failure.json", failure(error, runtime, row_id=row_id, phase="smoke"))
            raise
    return run_smoke(POLICY["smoke"]["minimum_rows"], run_cell, output)


def execute_formal(runtime, frozen, output, group_id=None):
    rows = {r["row_id"]: r for r in frozen["matrix"]["core_rows"] + frozen["matrix"]["diagnostics"]}
    groups = [g for g in frozen["matrix"]["round_groups"] if group_id is None or g["group_id"] == group_id]
    if not groups:
        raise ValueError("unknown round group")
    rows = {r: rows[r] for group in groups for r in group["members"]}
    records, anchors = {}, {}
    anchor_error = None
    try:
        for gpu in (0, 1):
            anchors[f"gpu{gpu}"] = runtime.anchor(gpu)
            anchors[f"gpu{gpu}"].capture()
            for _ in range(10): anchors[f"gpu{gpu}"].replay(0)
    except Exception as error:
        anchor_error = failure(error, runtime, stage="anchor_setup")
    for group in groups:
        members = group["members"]
        cases, samples = {}, {row_id: [] for row_id in members}
        try:
            runtime.release(group["group_id"])
        except Exception as error:
            for row_id in members:
                records[row_id] = {"row_id": row_id, "status": "quality_failed", **failure(error, runtime)}
            continue
        for row_id in members:
            directory = Path(output)/row_id.replace("/", "_").replace(">", "to")
            directory.mkdir(parents=True, exist_ok=True)
            try:
                if anchor_error:
                    raise RuntimeError("anchor setup failed: " + repr(anchor_error))
                case = runtime.make(row_id)
                c = contract(rows[row_id], case, frozen)
                record = {"row_id": row_id, "status": "quality_failed", "identity": case.identity,
                          "identity_hash": key_id(case.identity), "measurement_contract": c,
                          "measurement_contract_id": measurement_contract_id(c),
                          "active_recipe_id": frozen["active_recipe"]["recipe_id"], "samples": samples[row_id],
                          "admission": "conditional_dev", "automatic_admission": False}
                if rows[row_id]["pair_key"]:
                    record["pair_key"] = copy.deepcopy(rows[row_id]["pair_key"])
                    record["pair_key"]["measurement_contract"]["runtime_contract_id"] = record["measurement_contract_id"]
                    record["pair_key"]["device_id"] = {"logical": rows[row_id]["device_id"],
                        "uuid": case.identity["devices"][str(case.device.index)]["observed"]["uuid"]}
                    record["pair_key_id"] = key_id(record["pair_key"])
                records[row_id] = record
                save(directory/"identity.json", record)
                runtime.context.update(stage="formal_capture", memory_before_capture=runtime.snapshots())
                case.capture()
                record.update(identity=case.identity, identity_hash=key_id(case.identity))
                save(directory/"identity.json", record)
                # Preserve timing diagnostics before profiling without adding
                # invocations or using warmups as formal samples.
                record["warmup_diagnostic_timings"] = []
                for warmup in range(10):
                    runtime.checkpoint("formal_warmup", warmup_index=warmup)
                    warmup_timing, _ = case.replay(0)
                    record["warmup_diagnostic_timings"].append(warmup_timing)
                save(directory/"preparation.json", record)
                try:
                    record["path_evidence"] = case.trace(directory/"replay_trace.json")
                    record["path_verified"] = not row_id.startswith(("copy/", "diagnostic/")) or record["path_evidence"]["device_copy_trace_observed"]
                except Exception as error:
                    record["path_verified"] = False
                    record["path_evidence"] = {"error": repr(error), "traceback": traceback.format_exc(), "review_required": True}
                cases[row_id] = case
            except Exception as error:
                records[row_id] = {"row_id": row_id, "status": "quality_failed", "samples": samples[row_id],
                                   **failure(error, runtime, phase="formal_capture_or_warmup"), "automatic_admission": False}
                save(directory/"result.json", records[row_id])
        for attempt in range(7):
            pre, post, anchor_failure = {}, {}, None
            try:
                for gpu in group["anchor_devices"]:
                    pre[gpu] = finite_us(anchors[gpu].replay(0)[0])
            except Exception as error:
                anchor_failure = failure(error, runtime, stage="pre_anchor")
            order = group["orders"][attempt % len(group["orders"])]
            role_to_row = dict(zip(group["roles"], members))
            pending = {}
            for role in order:
                row_id = role_to_row[role]
                if row_id not in cases:
                    continue
                try:
                    timing, checks = cases[row_id].replay(0)
                    pending[row_id] = {"duration_us": finite_us(timing), "raw_timing": timing,
                                       "numeric_ok": True, "capture_ok": True,
                                       "path_ok": records[row_id].get("path_verified", False), "invariants": checks}
                except Exception as error:
                    pending[row_id] = {"numeric_ok": False, "capture_ok": True, "path_ok": False,
                                       **failure(error, runtime, phase="formal_replay")}
            try:
                for gpu in group["anchor_devices"]:
                    post[gpu] = finite_us(anchors[gpu].replay(0)[0])
            except Exception as error:
                anchor_failure = failure(error, runtime, stage="post_anchor")
            for row_id, sample in pending.items():
                sample.update(attempt=attempt, anchors={g: {"before_us": pre.get(g), "after_us": post.get(g)} for g in group["anchor_devices"]},
                              anchor_failure=anchor_failure, round_group=group["group_id"], arm_order=order)
                samples[row_id].append(sample)
                save(Path(output)/row_id.replace("/", "_").replace(">", "to")/"samples.json", samples[row_id])
        for row_id, case in cases.items():
            try:
                raw = samples[row_id]
                if len(raw) != 7 or any(s.get("anchor_failure") or not s.get("numeric_ok") for s in raw):
                    raise ValueError("one or more of the fixed seven attempts failed")
                shaped = [{"attempt": s["attempt"], "numeric_ok": True, "capture_ok": True, "path_ok": s["path_ok"],
                           "arm_us": {"body": s["duration_us"]}, "anchors": s["anchors"]} for s in raw]
                quality = formal_quality_review(shaped, ("body",), group["anchor_devices"])
                records[row_id]["quality"] = quality
                records[row_id]["status"] = "quality_passed" if quality["timing_eligible"] else "quality_failed"
            except Exception as error:
                records[row_id].update(status="quality_failed", quality_error=repr(error), quality_traceback=traceback.format_exc())
            save(Path(output)/row_id.replace("/", "_").replace(">", "to")/"result.json", records[row_id])
    # A final fixed grid is mandatory even when preparation failed mid-run.
    grid = collect_grid(rows, lambda row_id: records[row_id])
    result = {"phase": "formal", "cells": grid, "expected_cells": len(rows), "recorded_cells": len(grid),
              "quality_failed": sum(r["status"] == "quality_failed" for r in grid.values()),
              "attempts_per_prepared_cell": 7, "retries": 0, "automatic_admission": False,
              "scope": "raw evidence pending exact Kernel-Key export, route review, pair/chain consistency and admission"}
    save(Path(output)/"grid.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "smoke", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = None
    formal_started = False
    try:
        frozen = plan(args.flagscale_root)
        save(args.output/"plan.json", frozen)
        save(args.output/"superseded_profiles.json", {"records": frozen["superseded_profiles"], "original_sources_unchanged": True})
        if args.phase == "plan":
            write_cpu_plan(args.output, frozen)
            print(json.dumps({"status": "cpu_plan_only", "policy_id": frozen["policy"]["policy_id"], "gpu_executed": False}))
            return
        import torch
        with torch.no_grad():
            runtime = Runtime(args.output, frozen["active_recipe"], args.flagscale_root)
            save(args.output/"runtime_identity.json", runtime.identity_record)
            execute_smoke(runtime, frozen, args.output/"smoke")
            if args.phase == "run":
                formal_started = True
                from calibration.ffn_b4_supervisor import supervise
                supervise(frozen, args.output/"formal", args.flagscale_root)
    except Exception as error:
        save(args.output/"failure.json", failure(error, runtime, phase=args.phase, formal_started=formal_started))
        raise


if __name__ == "__main__":
    main()
