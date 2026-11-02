"""Isolated AB/BA conditioning diagnostic. Default phase is CPU plan only."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from calibration.ffn_anchor_conditioning_plan import proposal
from calibration.ffn_anchor_control import (
    ROOT, read, save, timed_sample, telemetry, review_rounds, validate_path, worker_failure,
)
from calibration.ffn_anchor_control_protocol import endpoint_summary
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_profile import digest
from cost_model.measurement_labels import measurement_contract_id

SCOPE = "conditioning_diagnostic_only_not_profile_admission"


def plan():
    design = read(ROOT / "data/ffn_anchor_conditioning_design_v1/design.json")
    if design != proposal(): raise ValueError("frozen conditioning design/source changed")
    payload = {"version": "ffn-anchor-conditioning-collector.v1", "design": design,
        "scope": SCOPE, "collector_implemented": True, "gpu_validated": False,
        "default_phase": "plan", "execution_entry": "explicit run only",
        "automatic_admission": False, "followup_matrix_open": False,
        "source_sha256": {n: digest(ROOT / n) for n in (
            "calibration/ffn_anchor_conditioning.py", "calibration/ffn_anchor_conditioning_plan.py",
            "calibration/ffn_anchor_control.py", "calibration/ffn_control_runtime.py",
            "calibration/ffn_b4_runtime.py", "calibration/ffn_anchor_control_protocol.py",
            "calibration/validate_kerv_adapter.py", "integration/flagscale/kerv_execution.py",
            "integration/flagscale/kerv_profile_runtime.py", "cost_model/measurement_labels.py",
            "data/ffn_measurement_b3/recipe.json", "tools/run_ffn_anchor_conditioning.sh")}}
    return {**payload, "plan_id": key_id(payload)}


def contract(cell, identity, frozen):
    recipe = {"design_id": frozen["design"]["design_id"], "condition": cell["condition"],
              "conditioning_replays_per_endpoint": cell["conditioning_replays_per_endpoint"],
              "shared_recipe": frozen["design"]["shared_recipe"]}
    return {"executor": "ffn_anchor_conditioning_diagnostic", "scope": SCOPE,
        "mode": "graph_replay", "submission_included": True, "allocation_included": False,
        "granularity": "50_independent_invocations_per_graph_replay", "unit": "us/call",
        "environment": identity["target"]["devices"]["0"]["environment"],
        "recipe": recipe, "recipe_id": key_id(recipe), "cell": cell,
        "collector_plan_id": frozen["plan_id"], "identity_hash": key_id(identity),
        "conditioning_in_primary_statistics": False, "all_three_measured_samples_retained": True}


def runtime_signature(identity):
    signatures = []
    for role in ("target", "anchor"):
        item = identity[role]
        if set(item["devices"]) != {"0"}: raise ValueError("only GPU0 is allowed")
        gpu = item["devices"]["0"]
        obs, env = gpu["observed"], gpu["environment"]
        if obs.get("uuid") in (None, "unavailable") or not obs.get("architecture") or env["device_uuid"] != obs["uuid"]:
            raise ValueError("missing or inconsistent GPU identity")
        signatures.append(key_id({"device": gpu, "build": item["actual_torch_build"],
                                  "loaded_source": item["loaded_kerv_source"]}))
    if signatures[0] != signatures[1]: raise ValueError("anchor/target runtime differs")
    return signatures[0]


def review_recording(rounds, cell):
    conditioning_count = cell["conditioning_replays_per_endpoint"]
    if conditioning_count != (1 if cell["condition"] == "B" else 0):
        raise ValueError("conditioning count differs from condition")
    quality = review_rounds(rounds, 3)  # All measured triples; unchanged 10% rules.
    previous_end, measured, conditioned, endpoints = -1, 0, 0, []
    for row in rounds:
        ordered = []
        for phase in ("pre", "post"):
            conditioning = row[phase + "_conditioning"]
            if len(conditioning) != conditioning_count:
                raise ValueError("missing or extra conditioning replay")
            ordered += [(s, "conditioning") for s in conditioning]
            ordered += [(s, "measurement") for s in row[phase]]
            if phase == "pre": ordered += [(row["target"], "measurement")]
            values = [s["batch_ms"] * 20 for s in row[phase]]
            tail = statistics.median(values[1:])
            endpoints.append({"attempt": row["attempt"], "phase": phase, "raw_us": values,
                "first_minus_next_two_us": values[0] - tail,
                "first_relative_to_next_two": (values[0] - tail) / tail,
                "conditioning_count": len(conditioning),
                "conditioning_to_first_reset_ms": (row[phase][0]["reset_begin"] - conditioning[-1]["validation_end"]) / 1e6 if conditioning else None,
                "conditioning_to_first_timing_ms": (row[phase][0]["timing_begin"] - conditioning[-1]["validation_end"]) / 1e6 if conditioning else None})
        for sample, expected_role in ordered:
            endpoint_summary([sample], 1)
            if sample.get("sample_role") != expected_role or sample.get("primary_measurement") is not (expected_role == "measurement"):
                raise ValueError("conditioning relabeled as a primary measurement")
            if any(sample.get(k) != "cuda:0" for k in ("device", "start_device", "end_device")) or not sample.get("invariants") or any(v is not True for v in sample["invariants"].values()):
                raise ValueError("sample device or invariant proof missing")
            if sample["reset_begin"] < previous_end:
                raise ValueError("conditioning/measurement timestamps overlap or are reordered")
            previous_end = sample["validation_end"]
            measured += expected_role == "measurement"
            conditioned += expected_role == "conditioning"
    return {"quality": quality, "endpoint_readouts": endpoints,
        "primary_pre_contrast_us": statistics.median(r["first_minus_next_two_us"] for r in endpoints if r["phase"] == "pre"),
        "primary_pre_relative_contrast": statistics.median(r["first_relative_to_next_two"] for r in endpoints if r["phase"] == "pre"),
        "measured_replays": measured, "conditioning_replays": conditioned}


def execute_cell(runtime, cell, output, frozen, sample=timed_sample, observe=telemetry, clock=time.perf_counter_ns):
    output = Path(output)
    context = {"anchor_preparation_begin": clock()}
    anchor = runtime.anchor(0)
    runtime.checkpoint("conditioning_anchor_capture", memory_before_capture=runtime.snapshots())
    anchor.capture()
    anchor_warmups = [anchor.replay(0)[0] for _ in range(10)]
    context["anchor_warmup_end"] = clock()
    context["case_preparation_begin"] = clock()
    case = runtime.make(cell["row"])
    runtime.context["memory_before_capture"] = runtime.snapshots()
    case.capture()
    target_warmups = [case.replay(0)[0] for _ in range(10)]
    context["case_preparation_end"] = clock()
    identity = {"target": copy.deepcopy(case.identity), "anchor": copy.deepcopy(anchor.identity)}
    runtime_signature(identity)
    measurement_contract = contract(cell, identity, frozen)
    contract_id = measurement_contract_id(measurement_contract)
    save(output / "identity.json", {"identity": identity, "identity_hash": key_id(identity),
        "measurement_contract": measurement_contract, "measurement_contract_id": contract_id})
    save(output / "preparation.json", {"context": context, "anchor_warmup_diagnostics": anchor_warmups,
        "target_warmup_diagnostics": target_warmups})

    def take(graph_case, role):
        return {**sample(graph_case), "sample_role": role, "primary_measurement": role == "measurement"}

    def endpoint():
        # No checkpoint, disk write, telemetry, or profiler is called between
        # conditioning completion and the first (or any) measured replay.
        conditioning = [take(anchor, "conditioning") for _ in range(cell["conditioning_replays_per_endpoint"])]
        readings = [take(anchor, "measurement") for _ in range(3)]
        return conditioning, readings

    rounds = []
    for attempt in range(7):
        runtime.checkpoint("conditioning_round", attempt=attempt)
        before = observe(0)
        pre_conditioning, pre = endpoint()
        target = take(case, "measurement")
        post_conditioning, post = endpoint()
        after = observe(0)
        rounds.append({"attempt": attempt, "pre_conditioning": pre_conditioning, "pre": pre,
            "target": target, "post_conditioning": post_conditioning, "post": post,
            "telemetry_before": before, "telemetry_after": after})
        save(output / "samples.json", rounds)
    paths = {"target": case.trace(output / "target_trace.json"), "anchor": anchor.trace(output / "anchor_trace.json")}
    for path in paths.values(): validate_path(path)
    analysis = review_recording(rounds, cell)
    result = {"cell": cell, "scope": SCOPE, "pipeline_complete": True, "analysis": analysis,
        "context": context, "identity_hash": key_id(identity), "measurement_contract_id": contract_id,
        "samples_sha256": digest(output / "samples.json"), "path_evidence": paths,
        "trace_sha256": {n: digest(output / (n + "_trace.json")) for n in paths},
        "automatic_admission": False, "formal_profile": False, "followup_matrix_open": False}
    save(output / "result.json", result)
    return result


def review_cell(directory, cell, frozen):
    directory = Path(directory)
    record, result, preparation = [read(directory / n) for n in ("identity.json", "result.json", "preparation.json")]
    identity = record["identity"]
    expected_contract = contract(cell, identity, frozen)
    contract_id = measurement_contract_id(expected_contract)
    if result["cell"] != cell or result["scope"] != SCOPE or result["pipeline_complete"] is not True or result["formal_profile"] or result["automatic_admission"] or result["followup_matrix_open"]:
        raise ValueError("conditioning worker scope/cell mismatch")
    if record["identity_hash"] != key_id(identity) or result["identity_hash"] != key_id(identity):
        raise ValueError("conditioning identity mismatch")
    if record["measurement_contract"] != expected_contract or record["measurement_contract_id"] != contract_id or result["measurement_contract_id"] != contract_id:
        raise ValueError("conditioning contract mismatch")
    rounds = read(directory / "samples.json")
    if digest(directory / "samples.json") != result["samples_sha256"] or review_recording(rounds, cell) != result["analysis"]:
        raise ValueError("conditioning raw sample/analysis mismatch")
    if preparation["context"] != result["context"] or any(len(preparation[n]) != 10 for n in ("anchor_warmup_diagnostics", "target_warmup_diagnostics")):
        raise ValueError("warmup evidence differs from fixed design")
    stamps = [result["context"][k] for k in ("anchor_preparation_begin", "anchor_warmup_end", "case_preparation_begin", "case_preparation_end")]
    first = rounds[0]["pre_conditioning"] + rounds[0]["pre"]
    if any(type(t) is not int or t < 0 for t in stamps) or stamps != sorted(stamps) or stamps[-1] > first[0]["reset_begin"]:
        raise ValueError("conditioning preparation timestamps missing or reordered")
    for name in ("target", "anchor"):
        proof = result["path_evidence"][name]
        validate_path(proof)
        trace_path = directory / (name + "_trace.json")
        names = [e.get("name", "") for e in read(trace_path)["traceEvents"] if e.get("cat") == "kernel"]
        if digest(trace_path) != result["trace_sha256"][name] or names != proof["kernel_names"]:
            raise ValueError("conditioning trace differs from path proof")
    return {"pipeline_complete": True, "quality_status": result["analysis"]["quality"]["status"],
        "target_median_us": result["analysis"]["quality"]["target_median_us"],
        "target_iqr_over_median": result["analysis"]["quality"]["target_iqr_over_median"],
        "invalid_attempts": [r["attempt"] for r in result["analysis"]["quality"]["rounds"] if not r["valid"]],
        "quality_reasons": result["analysis"]["quality"]["cell_reasons"],
        "runtime_signature": runtime_signature(identity), "identity_hash": key_id(identity),
        "measurement_contract_id": contract_id,
        **{k: result["analysis"][k] for k in ("primary_pre_contrast_us", "primary_pre_relative_contrast", "measured_replays", "conditioning_replays")}}


def report(rows):
    contrasts = {}
    for block in ("block1", "block2"):
        a, b = rows.get(block + "_A", {}), rows.get(block + "_B", {})
        if a.get("pipeline_complete") and b.get("pipeline_complete"):
            contrasts[block] = {"B_minus_A_us": b["primary_pre_contrast_us"] - a["primary_pre_contrast_us"],
                "B_minus_A_relative": b["primary_pre_relative_contrast"] - a["primary_pre_relative_contrast"],
                "target_B_minus_A_us": b["target_median_us"] - a["target_median_us"],
                "target_B_minus_A_relative": (b["target_median_us"] - a["target_median_us"]) / a["target_median_us"]}
    signs = [v["B_minus_A_us"] for v in contrasts.values()]
    return {"scope": SCOPE, "primary_readout": "Down0 pre first-minus-median(next two)", "cells": rows,
        "block_contrasts": contrasts,
        "direction": "both_decrease" if len(signs) == 2 and all(v < 0 for v in signs) else
                     "both_increase" if len(signs) == 2 and all(v > 0 for v in signs) else "incomplete_mixed_or_zero_inconclusive",
        "planned_independent_runs_per_condition": 2,
        "completed_runs_per_condition": {condition: sum(
            name.endswith("_" + condition) and r.get("pipeline_complete") is True for name, r in rows.items())
            for condition in ("A", "B")}, "crossing_rate_estimate": None,
        "causal_verdict": "inconclusive", "protocol_adopted": False,
        "automatic_admission": False, "followup_matrix_open": False}


def write_report(output, rows, *, template=False):
    result = report(rows)
    prefix = "report_template" if template else "report"
    save(Path(output) / (prefix + ".json"), result)
    lines = ["PRIMARY READOUT: Down0 pre first-minus-median(next two)",
        "Scope: " + SCOPE + "; MUST NOT enter Profile admission.",
        "Two ordered blocks; no failure-rate, bias, thermal or clock claim. Protocol not adopted; formal matrix closed.", "",
        "| Cell | Quality status / reasons | Pre contrast us | Pre relative contrast | Target median us | Target IQR/median | Scope |", "|---|---|---|---|---|---|---|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row['status']} / {row.get('quality_reasons')} | {row.get('primary_pre_contrast_us')} | {row.get('primary_pre_relative_contrast')} | {row.get('target_median_us')} | {row.get('target_iqr_over_median')} | {SCOPE} |")
    lines += ["", "| Block | B minus A us | B minus A relative | Target B minus A us | Target B minus A relative | Scope |", "|---|---|---|---|---|---|"]
    for name, row in result["block_contrasts"].items():
        lines.append(f"| {name} | {row['B_minus_A_us']} | {row['B_minus_A_relative']} | {row['target_B_minus_A_us']} | {row['target_B_minus_A_relative']} | {SCOPE} |")
    with (Path(output) / (prefix + ".md")).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def collect(output, flagscale_root, frozen, run_worker=subprocess.run):
    output, rows, signature, stopped = Path(output), {}, None, False
    for cell in frozen["design"]["cells"]:
        if stopped:
            rows[cell["id"]] = {"status": "blocked", "worker_attempts": 0}
        else:
            directory = output / cell["id"]
            directory.mkdir()
            save(directory / "parent_plan.json", frozen)
            command = [sys.executable, "-X", "faulthandler", "-m", "calibration.ffn_anchor_conditioning", "run",
                "--output", str(directory), "--flagscale-root", str(flagscale_root), "--worker", cell["id"]]
            row = {"worker_attempts": 1, "retries": 0, "returncode": None}
            print(json.dumps({"conditioning_cell_start": cell["id"]}), flush=True)
            try:
                with (directory / "worker.log").open("w", encoding="utf-8") as log:
                    completed = run_worker(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                row["returncode"] = completed.returncode
                if completed.returncode != 0: raise RuntimeError("conditioning worker failed; see preserved evidence")
                checked = review_cell(directory, cell, frozen)
                if signature is not None and checked["runtime_signature"] != signature:
                    raise ValueError("GPU0/build changed across conditioning cells")
                signature = checked["runtime_signature"]
                row.update(status=checked["quality_status"], **checked)
            except Exception as error:
                row.update(status="pipeline_failed", error=repr(error))
                worker_failure(directory, error, row["returncode"])
                stopped = True
            rows[cell["id"]] = row
        save(output / "summary.json", {"cells": rows, "scope": SCOPE,
            "pipeline_passed": len(rows) == 4 and all(r.get("pipeline_complete") is True for r in rows.values()),
            "protocol_adopted": False, "automatic_admission": False, "followup_matrix_open": False,
            "plan_id": frozen["plan_id"]})
        write_report(output, rows)
    return read(output / "summary.json")


def worker(cell_id, output, flagscale_root):
    import torch
    from calibration.ffn_control_runtime import ControlRuntime
    from calibration.probe_kerv_graph import POOL
    runtime = None
    try:
        frozen = read(output / "parent_plan.json")
        if frozen != plan(): raise ValueError("conditioning worker source/plan differs from parent")
        with torch.no_grad():
            runtime = ControlRuntime(output, read(ROOT / "data/ffn_measurement_b3/recipe.json"), flagscale_root)
            save(output / "runtime_identity.json", runtime.identity_record)
            cell = next(c for c in frozen["design"]["cells"] if c["id"] == cell_id)
            execute_cell(runtime, cell, output, frozen)
    except Exception as error:
        trace = traceback.format_exc()
        try:
            memory = runtime.snapshots() if runtime else {"unavailable": "runtime initialization did not finish"}
        except Exception as snapshot_error:
            memory = {"unavailable": repr(snapshot_error)}
        save(output / "failure.json", {"error": repr(error), "traceback": trace, "pool_config": dict(POOL),
            "memory": memory, "runtime_context": copy.deepcopy(getattr(runtime, "context", {})), "scope": SCOPE})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flagscale-root", type=Path)
    parser.add_argument("--worker", choices=["block1_A", "block1_B", "block2_B", "block2_A"])
    args = parser.parse_args()
    if args.phase == "run" and args.flagscale_root is None: parser.error("run requires pinned --flagscale-root")
    if args.worker:
        if args.phase != "run": parser.error("worker requires run")
        worker(args.worker, args.output, args.flagscale_root)
    else:
        frozen = plan()
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "plan.json", frozen)
        write_report(args.output, {c["id"]: {"status": "not_run"} for c in frozen["design"]["cells"]}, template=True)
        if args.phase == "run" and not collect(args.output, args.flagscale_root, frozen)["pipeline_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
