"""P2 GPU1 completion: SiLU, Mul, Pack+LightOP complete call, Down GEMM re-check.

Same recipes and contracts as the reviewed GPU0 records so matching stays by
full identity: SiLU/Mul use calibration.native_stage_calibration.StageCall
(the recipe and version the FFN nodes declare), the Fused call uses the
LightOP adapter and its calibration contract, Down uses the P2 GEMM recipe.
Down GPU0 is a same-batch reference only. The current proxy predictions are
frozen into plan.json. No dispatch trace is collected for GPU1 (stated gap).
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from calibration import gemm_projection_calibration as gp
from calibration import native_stage_calibration as native
from calibration import stage_size_calibration as base
from calibration.lightop_profile_smoke import POOL, check_outputs, invalidate_buffers, memory_snapshot, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "gpu1_ffn_cost_completion_v1"
SOURCES = ("calibration/gpu1_cost_completion.py", "tools/run_gpu1_cost_completion.sh",
           "calibration/native_stage_calibration.py", "calibration/gemm_projection_calibration.py",
           "calibration/stage_size_calibration.py", "calibration/lightop_profile_smoke.py",
           "integration/lightop/silu_mul.py", "integration/lightop/profile_identity.py",
           "cost_model/lightop_profile.py", "execution/reference_identity.py",
           "cost_model/fusion_kernel_profile.py", "cost_model/measurement_labels.py")
CASES = {
    "stage/silu/gpu1": {"kind": "stage", "stage": "silu", "nodes": ["L000.mlp.silu"], "device": 1},
    "stage/mul/gpu1": {"kind": "stage", "stage": "mul", "nodes": ["L000.mlp.gated_mul"], "device": 1},
    "fused/silu_mul/gpu1": {"kind": "fused", "nodes": ["L000.mlp.silu", "L000.mlp.gated_mul"], "device": 1},
    "gemm/down/gpu1": {"kind": "gemm", "projection": "down", "nodes": ["L000.mlp.down_proj"], "device": 1},
    "gemm/down/gpu0": {"kind": "gemm", "projection": "down", "nodes": ["L000.mlp.down_proj"], "device": 0,
                       "role": "same_batch_reference_not_a_new_profile"},
}


def describe(case_id):
    case = copy.deepcopy(CASES[case_id])
    case["timing_device"] = f"cuda:{case['device']}"
    case.setdefault("role", "gpu1_cost_candidate")
    if case["kind"] == "gemm":
        case.update({k: v for k, v in gp.describe(f"gemm/down/gpu{case['device']}").items()
                     if k in ("m", "k", "n", "x_shape", "weight_shape", "output_shape", "flops")})
    else:
        case["shape"] = list(native.SHAPE)
    return case


def proxy_predictions():
    from cost_model.ffn_decision_sensitivity import FFNSensitivity
    audit = FFNSensitivity()
    return {c: {"predicted_us": audit.costs[tuple(v["nodes"]), v["device"]].duration * 1e6,
                "source": audit.costs[tuple(v["nodes"]), v["device"]].source} for c, v in CASES.items()}


def contracts():
    stage = {**base.contract(), "correctness": dict(base.TOLERANCE)}
    gemm = {**stage, "correctness": dict(gp.TOLERANCE),
            "buffers": "shared read-only x and weight per case; 50 independent outputs",
            "cache_context": "same weight reused by 50 calls per replay; no cold-weight claim"}
    return {"stage": stage, "gemm": gemm,
            "fused": "cost_model.lightop_profile.measurement_contract(observed environment, purpose='calibration')"}


def plan():
    ids = list(CASES)
    return {"scope": SCOPE, "architecture": "gfx936", "seed": 20261113, "dtype": "bf16",
        "cases": {c: describe(c) for c in ids},
        "recipes": {"stage": copy.deepcopy(native.RECIPES), "gemm": copy.deepcopy(gp.RECIPE),
                    "fused": "integration/lightop/silu_mul.py RECIPE (complete split-input adapter call)"},
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contracts": contracts(), "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL),
        "frozen_proxy_predictions": proxy_predictions(),
        "known_gaps": ["no GPU1 LightOP dispatch trace in this collection; kernel identity rests on loaded source hashes"],
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "automatic_admission": False, "model_parameters_updated": False}


def stage_entry(torch, stage, device, generator):
    import torch.nn.functional as F
    gate, up = [torch.randn(native.SHAPE, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    left = torch.empty_like(gate)
    torch.ops.aten.silu.out(gate, out=left)
    inputs = {"gate": gate, "up": up, "left": left}
    snapshots = {k: t.view(torch.uint8).clone() for k, t in inputs.items()}
    refs = (F.silu(gate), F.silu(gate.float())) if stage == "silu" else (left * up, left.float() * up.float())
    calls = [native.StageCall(stage, gate, up, left) for _ in range(base.CALLS)]

    def invalidate():
        for call in calls:
            call.output.fill_(float("nan"))

    def verify():
        return native.check(torch, calls, inputs, snapshots, refs, base.TOLERANCE)

    return calls, invalidate, verify, calls[0].observation


def gemm_entry(torch, case, device, generator):
    x = torch.randn(case["x_shape"], device=device, dtype=torch.bfloat16, generator=generator)
    weight = (torch.randn(case["weight_shape"], device=device, generator=generator) / math.sqrt(case["k"])).to(torch.bfloat16)
    snapshots = (x.view(torch.uint8).clone(), weight.view(torch.uint8).clone())
    refs = (x @ weight.t(), x.float() @ weight.float().t())
    calls = [gp.GemmCall(x, weight) for _ in range(base.CALLS)]

    def invalidate():
        for call in calls:
            call.output.fill_(float("nan"))

    def verify():
        return gp.check(torch, calls, x, weight, snapshots, refs)

    return calls, invalidate, verify, calls[0].observation


def fused_entry(torch, device, generator, output):
    import torch.nn.functional as F
    from calibration.lightop_profile_smoke import placed_case
    from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
    from cost_model.measurement_labels import measurement_contract_id
    from integration.lightop.profile_identity import observe_runtime
    from integration.lightop.silu_mul import LightopSiluMulAdapter
    from simulator.placement_artifacts import save_placed_op_graph
    adapter = LightopSiluMulAdapter()
    placed, index = placed_case(device.index)
    bound = adapter.bind(placed, index, {device.index: device})
    save_placed_op_graph(output / "fused_placed.json", placed)
    actual = observe_runtime(adapter, device)
    save(output / "fused_runtime.json", actual)
    gate = torch.randn(native.SHAPE, device=device, dtype=torch.bfloat16, generator=generator)
    up = torch.randn(native.SHAPE, device=device, dtype=torch.bfloat16, generator=generator)
    snapshots = [t.view(torch.uint8).clone() for t in (gate, up)]
    refs = (F.silu(gate) * up, F.silu(gate.float()) * up.float())
    calls = [bound.prepare({bound.gate_node: gate, bound.up_node: up}) for _ in range(base.CALLS)]
    if any(c.observation != calls[0].observation for c in calls):
        raise MeasurementContractError("Prepared LightOP call identities disagree")
    keys = LightopKernelKeys(placed, index, calls[0].observation, actual)
    key, gaps = keys.describe(placed.plan.groups[index])
    if gaps:
        raise MeasurementContractError("Incomplete GPU1 LightOP Kernel Key: " + repr(gaps))
    contract = measurement_contract(actual["environment"], purpose="calibration")
    identity = {"kernel_key": key, "kernel_key_id": key_id(key), "identity_gaps": list(gaps),
                "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract),
                "group_index": index}

    def invalidate():
        invalidate_buffers(calls)

    def verify():
        return check_outputs(torch, calls, gate, up, snapshots, refs, contract["correctness"])

    return calls, invalidate, verify, {"one_call": calls[0].observation, "call_count": len(calls)}, identity


def build(torch, frozen, output, context):
    entries, prepared, warmup, fused_identity = {}, {}, {}, None
    for case_id, case in frozen["cases"].items():
        device = torch.device(case["timing_device"])
        context.update(stage="capture_and_warmup", case=case_id, memory=memory_snapshot(torch, device))
        save(output / "last_stage.json", context)
        generator = torch.Generator(device=device).manual_seed(frozen["seed"] + len(entries))
        with torch.cuda.device(device):
            if case["kind"] == "stage":
                calls, invalidate, verify, obs = stage_entry(torch, case["stage"], device, generator)
            elif case["kind"] == "gemm":
                calls, invalidate, verify, obs = gemm_entry(torch, case, device, generator)
            else:
                calls, invalidate, verify, obs, fused_identity = fused_entry(torch, device, generator, output)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for call in calls:
                    call()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=None):
                for call in calls:
                    call()
            with torch.cuda.stream(stream):
                invalidate()
                for _ in range(base.WARMUP):
                    graph.replay()
            torch.cuda.synchronize(device)
        warmup[case_id] = verify()

        def stream_invalidate(invalidate=invalidate, stream=stream, device=device):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                invalidate()

        entries[case_id] = {"graph": graph, "stream": stream, "devices": (case["device"],), "timing_device": device,
                            "invalidate": stream_invalidate, "verify": verify}
        prepared[case_id] = obs
    return entries, prepared, warmup, fused_identity


def identity(frozen, prepared, runtime, fused_identity):
    rt = runtime["runtime"]
    if rt["accelerator"] != "HIP" or any(rt.get(f) is not False for f in
                                          ("grad_enabled", "autocast_enabled", "deterministic_algorithms")):
        raise MeasurementContractError("HIP no_grad BF16 runtime required")
    for field in ("torch_distribution", "torch_git_revision", "torch_extension_sha256",
                  "torch_build_sha256", "hip", "driver"):
        if not rt.get(field):
            raise MeasurementContractError("Missing runtime identity: " + field)
    if set(prepared) != set(frozen["cases"]):
        raise MeasurementContractError("Prepared cases differ from plan")
    keys = {}
    for case_id, case in frozen["cases"].items():
        dev = runtime["devices"][str(case["device"])]
        if dev["ordinal"] != case["device"] or str(dev.get("architecture", "")).split(":")[0] != frozen["architecture"]:
            raise MeasurementContractError("Device identity differs: " + case_id)
        gpu_key = {"device": copy.deepcopy(dev), "runtime": {k: rt[k] for k in (
            "torch_distribution", "torch_git_revision", "torch_build_sha256", "torch_extension_sha256", "hip", "driver")}}
        obs = prepared[case_id]
        if case["kind"] == "fused":
            if fused_identity is None or fused_identity["identity_gaps"]:
                raise MeasurementContractError("Fused Kernel Key missing or incomplete")
            if fused_identity["kernel_key_id"] != key_id(fused_identity["kernel_key"]):
                raise MeasurementContractError("Fused Kernel Key hash differs")
            key = {"kernel_key_id": fused_identity["kernel_key_id"],
                   "measurement_contract_id": fused_identity["measurement_contract_id"], "gpu": gpu_key, "case": case}
            contract_id = fused_identity["measurement_contract_id"]
        else:
            kind_contract = frozen["contracts"][case["kind"]]
            if case["kind"] == "stage":
                if (obs["recipe"] != native.RECIPES[case["stage"]] or obs["schema"] != native.SCHEMAS[case["stage"]]
                        or obs["source_sha256"] != frozen["collector"]["source_sha256"]["calibration/native_stage_calibration.py"]):
                    raise MeasurementContractError("Stage implementation changed: " + case_id)
                shapes = {r: case["shape"] for r in ("gate", "up", "left", "output")}
            else:
                if obs["recipe"] != gp.RECIPE or obs["source_sha256"] != frozen["collector"]["source_sha256"]["calibration/gemm_projection_calibration.py"]:
                    raise MeasurementContractError("GEMM implementation changed: " + case_id)
                shapes = {"x": case["x_shape"], "weight": case["weight_shape"], "output": case["output_shape"]}
            if set(obs["tensors"]) != set(shapes):
                raise MeasurementContractError("Incomplete tensor observation: " + case_id)
            for role, tensor in obs["tensors"].items():
                if (tensor["shape"] != shapes[role] or tensor["dtype"] != "bf16" or tensor["layout"] != "contiguous"
                        or tensor["strides"] != base.strides(shapes[role])
                        or tensor["layout_provenance"]["device"] != case["timing_device"]):
                    raise MeasurementContractError("Actual layout differs: " + case_id)
            key = {"implementation": obs, "gpu": gpu_key, "case": case, "contract": kind_contract, "pool": frozen["pool"]}
            contract_id = key_id(kind_contract)
        keys[case_id] = {"case_identity": key, "case_identity_id": key_id(key),
                         "gpu_key_id": key_id(gpu_key), "contract_id": contract_id}
    return {"scope": SCOPE, "cases": keys, "uuid_is_provenance_only": True,
            "known_gaps": copy.deepcopy(frozen["known_gaps"])}


def run(output, frozen, context):
    import torch
    if not torch.version.hip or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 and GPU1 required; no CPU/eager fallback")
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("Known profiling/debug instrumentation detected")
    torch.cuda.set_device(1)
    with torch.no_grad():
        runtime = base.observe_runtime(torch)
        save(output / "runtime.json", runtime)
        entries, prepared, warmup, fused_identity = build(torch, frozen, output, context)
        save(output / "preparation.json", prepared)
        save(output / "fused_identity.json", fused_identity)
        save(output / "identity.json", identity(frozen, prepared, runtime, fused_identity))
        save(output / "warmup.json", warmup)
        records = []
        for attempt, order in enumerate(frozen["order"]):
            for case_id in order:
                entry = entries[case_id]
                context.update(stage="measurement", case=case_id, round=attempt)
                save(output / "last_stage.json", context)
                started = time.monotonic_ns()
                entry["invalidate"]()
                torch.cuda.synchronize(entry["timing_device"])
                timing = base.timed(torch, entry)
                checks = entry["verify"]()
                row = {"case": case_id, "round": attempt, "started_monotonic_ns": started,
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks}
                records.append(row)
                save(output / f"attempt_{len(records)-1:03d}.json", row)
                print(json.dumps(row), flush=True)
        save(output / "result.json", {"scope": SCOPE, "complete": True, "records": records,
                                      "automatic_admission": False})
        return evaluate(frozen, records)


def evaluate(frozen, records):
    result = base.evaluate(frozen, records)
    result.update(scope=SCOPE, data_role="GPU1 calibration candidates; Down GPU0 same-batch reference")
    result["proxy_error"] = {c: {
        "predicted_us": frozen["frozen_proxy_predictions"][c]["predicted_us"],
        "proxy_source": frozen["frozen_proxy_predictions"][c]["source"],
        "measured_us": q["statistics"]["median_us"] if q["statistics"] else None, "quality": q["status"],
        "signed_error": (frozen["frozen_proxy_predictions"][c]["predicted_us"] - q["statistics"]["median_us"])
                        / q["statistics"]["median_us"] if q["statistics"] else None}
        for c, q in result["cases"].items()}
    return result


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen = base.read(directory / "plan.json")
    current = plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan, frozen predictions or source identity changed")
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
    fused_identity = base.read(directory / "fused_identity.json")
    _replay_fused_key(directory, fused_identity)
    replay = identity(frozen, base.read(directory / "preparation.json"), base.read(directory / "runtime.json"),
                      fused_identity)
    if replay != base.read(directory / "identity.json"):
        raise MeasurementContractError("Case identity or contract differs")
    evaluated = evaluate(frozen, records)
    if process["python_returncode"] != (0 if evaluated["status"] == "calibration_candidates_ready" else 1):
        raise MeasurementContractError("Process result differs from recomputed quality")
    evaluated.update(identity=replay, plan_sha256=base.digest(directory / "plan.json"),
                     raw_sha256={p.name: base.digest(p) for p in sorted(directory.glob("*.json"))})
    return evaluated


def _replay_fused_key(directory, fused_identity):
    """Recompute the GPU1 LightOP Kernel Key from the saved placed graph and runtime."""
    from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
    from cost_model.measurement_labels import measurement_contract_id
    from integration.lightop.silu_mul import adapter_registry
    from simulator.placement_artifacts import load_placed_op_graph
    placed = load_placed_op_graph(directory / "fused_placed.json", registry=adapter_registry())
    actual = base.read(directory / "fused_runtime.json")
    prep = base.read(directory / "preparation.json")["fused/silu_mul/gpu1"]
    if prep["call_count"] != base.CALLS:
        raise MeasurementContractError("Wrong LightOP buffer pool size")
    keys = LightopKernelKeys(placed, fused_identity["group_index"], prep["one_call"], actual)
    key, gaps = keys.describe(placed.plan.groups[fused_identity["group_index"]])
    contract = measurement_contract(actual["environment"], purpose="calibration")
    if (gaps or key != fused_identity["kernel_key"] or key_id(key) != fused_identity["kernel_key_id"]
            or contract != fused_identity["measurement_contract"]
            or measurement_contract_id(contract) != fused_identity["measurement_contract_id"]
            or placed.plan.groups[fused_identity["group_index"]].device != 1):
        raise MeasurementContractError("GPU1 LightOP Kernel Key or contract does not replay")


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
                                  "proxy_us": {c: round(p["predicted_us"], 3)
                                               for c, p in frozen["frozen_proxy_predictions"].items()}}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"status": evaluated["status"], "quality_failed": evaluated["quality_failed"]}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
                                       "context": context})
        raise


if __name__ == "__main__":
    sys.exit(main())
