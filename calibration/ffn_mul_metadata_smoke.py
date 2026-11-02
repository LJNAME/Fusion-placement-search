"""Separate Mul0 metadata smoke; no anchor pilot samples or cost admission."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import traceback

from calibration.ffn_anchor_control import (
    ROOT, PROTOCOL, save, read, pilot_plan, review_cell, worker_failure,
)
from cost_model.ffn_profile_export import tensor_pool, tensor_schema
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import measurement_contract_id

SCOPE = "mul_metadata_smoke_only_not_profile_admission"
REQUIRED_CHECKS = ("all_50_outputs", "up_bytes_unchanged", "mul_left_bytes_unchanged", "gate_restore_exact")


def verify_pilot_for_smoke(root):
    root = Path(root)
    plan, summary = read(root / "plan.json"), read(root / "summary.json")
    if plan != pilot_plan() or summary["plan_id"] != plan["plan_id"] or summary.get("pilot_passed") is not True:
        raise ValueError("requires a complete current pilot; no old or partial run substitution")
    signatures = []
    for cell in PROTOCOL["cells"]:
        record = summary["cells"][cell["id"]]
        checked = review_cell(root / cell["id"], cell)
        if record.get("worker_attempts") != 1 or record.get("returncode") != 0 or any(record.get(k) != v for k, v in checked.items()):
            raise ValueError("pilot summary differs from original four-cell evidence")
        signatures.append(checked["runtime_signature"])
    if len(set(signatures)) != 1:
        raise ValueError("pilot device/build changed")
    return {"plan_id": plan["plan_id"], "runtime_signature": signatures[0],
            "scope": "pilot completeness only; R3 quality remains separate"}


def smoke_plan(pilot_input=None):
    payload = {"version": "ffn-mul-metadata-smoke.v1", "scope": SCOPE,
        "case": "gated_mul@gpu0", "capture_calls": 50, "diagnostic_replays": 3,
        "required_invariants": list(REQUIRED_CHECKS),
        "timing_used_for_pass_fail": False, "worker_attempts": 1, "worker_retries": 0,
        "automatic_admission": False, "followup_matrix_open": False, "default_phase": "plan",
        "pilot_plan_id": pilot_plan()["plan_id"],
        "prerequisite": verify_pilot_for_smoke(pilot_input) if pilot_input else None}
    return {**payload, "plan_id": key_id(payload)}


def metadata_proof(identity):
    inputs = identity["inputs"]
    left, gates, up = inputs["left"], inputs["gates"], inputs["up"]
    descriptor = tensor_pool(left, 0)
    if descriptor["shape"] != [1, 256, 3072] or descriptor["dtype"] != "bf16":
        raise ValueError("Mul actual left differs from the frozen fixture")
    if descriptor != tensor_pool(gates, 0) or descriptor != tensor_schema(up) or up["device"] != "cuda:0":
        raise ValueError("Mul operand layout/device mismatch")
    if {v["data_ptr"] for v in left}.intersection(v["data_ptr"] for v in gates) or up["data_ptr"] in {v["data_ptr"] for v in left}:
        raise ValueError("actual Mul left was replaced with gate/up metadata")
    output = tensor_pool(identity["output_layouts"], 0)
    if output != descriptor:
        raise ValueError("Mul output layout mismatch")
    if {v["data_ptr"] for v in identity["output_layouts"]}.intersection({v["data_ptr"] for v in left + gates} | {up["data_ptr"]}):
        raise ValueError("out-of-place Mul output aliases an input")
    return {"actual_left_metadata_recorded": True, "left_is_not_gate_or_up": True,
            "all_50_operands_and_outputs": True, "runtime_layout_provenance": True}


def execute_smoke(runtime, output, prerequisite):
    output = Path(output)
    runtime.checkpoint("mul_smoke_prepare", memory_before_capture=runtime.snapshots())
    case = runtime.activation(0, mul_smoke_only=True)["gated_mul@gpu0"]
    runtime.checkpoint("mul_smoke_before_capture", memory_before_capture=runtime.snapshots())
    case.progress = runtime.checkpoint
    case.capture()
    proof = metadata_proof(case.identity)
    item = case.identity
    signature = key_id({"device": item["devices"]["0"], "build": item["actual_torch_build"],
                        "loaded_source": item["loaded_kerv_source"]})
    if signature != prerequisite["runtime_signature"]:
        raise ValueError("Mul smoke GPU0/build differs from pilot")
    contract = {"executor": "ffn_mul_metadata_smoke", "scope": SCOPE,
        "submission_included": True, "allocation_included": False,
        "granularity": "50_independent_invocations_per_graph_replay", "unit": "ms/replay_diagnostic_only",
        "environment": item["devices"]["0"]["environment"], "calls_per_replay": 50,
        "diagnostic_replays": 3, "timing_used_for_pass_fail": False,
        "identity_hash": key_id(item), "pilot_prerequisite": prerequisite}
    save(output / "identity.json", {"identity": item, "identity_hash": key_id(item),
        "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract)})
    replays = []
    for index in range(3):
        timing, checks = case.replay(index, diagnostic=True)
        if any(checks.get(name) is not True for name in REQUIRED_CHECKS) or any(v is not True for v in checks.values()):
            raise AssertionError("Mul smoke numerical/input invariants failed")
        replays.append({"index": index, "diagnostic_timing": timing, "invariants": checks})
        save(output / "replays.json", replays)
    result = {"status": "passed", "scope": SCOPE, "metadata": proof,
        "capture_ok": True, "numeric_ok": True, "invariants_ok": True,
        "identity_written": True, "contract_written": True, "metadata_written": True,
        "identity_hash": key_id(item), "measurement_contract_id": measurement_contract_id(contract),
        "timing_used_for_pass_fail": False, "formal_profile": False,
        "automatic_admission": False, "followup_matrix_open": False}
    save(output / "result.json", result)
    return result


def review_smoke(output, prerequisite):
    output = Path(output)
    result, record, replays = read(output / "result.json"), read(output / "identity.json"), read(output / "replays.json")
    if result["scope"] != SCOPE or result["status"] != "passed" or result["timing_used_for_pass_fail"] or result["automatic_admission"] or result["formal_profile"]:
        raise ValueError("smoke scope differs from metadata-only contract")
    if len(replays) != 3 or [r["index"] for r in replays] != [0, 1, 2] or any(
            any(r["invariants"].get(name) is not True for name in REQUIRED_CHECKS) or
            any(v is not True for v in r["invariants"].values()) for r in replays):
        raise ValueError("smoke replay evidence missing or failed")
    if record["identity_hash"] != key_id(record["identity"]) or result["identity_hash"] != record["identity_hash"]:
        raise ValueError("smoke identity hash mismatch")
    if result["metadata"] != metadata_proof(record["identity"]):
        raise ValueError("smoke metadata proof mismatch")
    contract = record["measurement_contract"]
    if contract["scope"] != SCOPE or contract["pilot_prerequisite"] != prerequisite or contract["timing_used_for_pass_fail"]:
        raise ValueError("smoke prerequisite or contract mismatch")
    if record["measurement_contract_id"] != measurement_contract_id(contract) or result["measurement_contract_id"] != record["measurement_contract_id"]:
        raise ValueError("smoke contract hash mismatch")
    return result


def collect(output, pilot_input, flagscale_root, frozen, run_worker=subprocess.run):
    output = Path(output)
    prerequisite = verify_pilot_for_smoke(pilot_input)  # CPU-only, before starting a worker.
    directory = output / "worker"
    directory.mkdir()
    save(directory / "parent_plan.json", frozen)
    command = [sys.executable, "-X", "faulthandler", "-m", "calibration.ffn_mul_metadata_smoke", "run",
        "--output", str(directory), "--pilot-input", str(pilot_input), "--flagscale-root", str(flagscale_root), "--worker"]
    result = {"status": "failed", "scope": SCOPE, "worker_attempts": 1, "worker_retries": 0, "returncode": None,
              "automatic_admission": False, "followup_matrix_open": False}
    try:
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            completed = run_worker(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        result["returncode"] = completed.returncode
        if completed.returncode != 0: raise RuntimeError("Mul smoke worker failed; see preserved evidence")
        review_smoke(directory, prerequisite)
        result["status"] = "passed"
    except Exception as error:
        result["error"] = repr(error)
        worker_failure(directory, error, result["returncode"])
    save(output / "summary.json", result)
    return result


def worker(output, pilot_input, flagscale_root):
    import torch
    from calibration.ffn_control_runtime import ControlRuntime
    from calibration.probe_kerv_graph import POOL
    runtime = None
    try:
        frozen = smoke_plan(pilot_input)
        if frozen != read(output / "parent_plan.json"):
            raise ValueError("smoke worker plan/source differs from parent")
        with torch.no_grad():
            runtime = ControlRuntime(output, read(ROOT / "data/ffn_measurement_b3/recipe.json"), flagscale_root)
            execute_smoke(runtime, output, frozen["prerequisite"])
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
    parser.add_argument("--pilot-input", type=Path)
    parser.add_argument("--flagscale-root", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.phase == "run" and (args.pilot_input is None or args.flagscale_root is None):
        parser.error("run requires --pilot-input and the pinned --flagscale-root")
    if args.worker:
        if args.phase != "run": parser.error("worker requires run")
        worker(args.output, args.pilot_input, args.flagscale_root)
    else:
        frozen = smoke_plan(args.pilot_input)
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "plan.json", frozen)
        if args.phase == "run" and collect(args.output, args.pilot_input, args.flagscale_root, frozen)["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
