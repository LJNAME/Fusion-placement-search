"""Freeze two GPU0 predictions, collect a fresh pair, then review on CPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from calibration.lightop_profile_calibration import (
    CHECKS, POOL, QUALITY, ROOT, digest, quality, read, review_receipt, timed_replay)
from calibration.lightop_profile_smoke import (
    check_outputs, invalidate_buffers, memory_snapshot, placed_case, save)
from cost_model.fusion_kernel_profile import key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from execution.reference_identity import tensor_observation
from integration.lightop.silu_mul import RECIPE, SHAPE, adapter_registry
from simulator.hardware import load_hardware
from simulator.placement_artifacts import load_placed_op_graph, save_placed_op_graph
from simulator.shape_model import group_compute_time

SCOPE = "lightop_gpu0_local_pair_reference_not_profile_admission_v1"
RAW = ROOT / "data/lightop_profile_calibration_20261104_01/raw"
HARDWARE = "data/bw1000_2card/bw1000_2card.calibrated.json"
ARMS = ("native", "fused")
NATIVE_RECIPE = {
    "id": "aten.silu_out_mul_out.preallocated_chain.v1",
    "timed_body": ["torch.ops.aten.silu.out(gate, out=intermediate)",
                   "torch.ops.aten.mul.out(intermediate, up, out=output)"],
    "outside_timing": ["validate inputs", "allocate intermediate and output"],
    "input_semantics": "shared read-only gate/up; no restoration between replays",
    "output_semantics": "independent preallocated output",
}
SOURCES = ("calibration/lightop_local_pair.py", "tools/run_lightop_local_pair.sh",
           "simulator/shape_model.py", "simulator/hardware.py", HARDWARE)


def plan():
    admission, profile = review_receipt(RAW)
    if profile is None:
        raise MeasurementContractError("L2 admission replay failed")
    placed, index = placed_case(0)
    group = placed.plan.groups[index]
    hardware = load_hardware(ROOT / HARDWARE)
    device = hardware.devices[0]
    tensor_bytes = 2
    for size in SHAPE:
        tensor_bytes *= size

    def stage(nodes, io_bytes):
        compute = group_compute_time(placed.graph, nodes, device, hardware)
        return {"logical_io_bytes": io_bytes, "compute_s": compute,
                "memory_s": io_bytes / device.mem_bw,
                "launch_proxy_s": hardware.launch_overhead,
                "duration_us": (max(compute, io_bytes / device.mem_bw)
                                + hardware.launch_overhead) * 1e6}

    nodes = group.node_ids
    stages = {"native": [stage(nodes[:1], 2 * tensor_bytes), stage(nodes[1:], 3 * tensor_bytes)],
              "fused": [stage((), 4 * tensor_bytes), stage(nodes, 3 * tensor_bytes)]}
    analytic = {arm: sum(s["duration_us"] for s in stages[arm]) for arm in ARMS}
    calibrated = dict(analytic, fused=admission["latency"]["duration_s"] * 1e6)
    models = {"L1_analytic": analytic, "L2_fused_calibrated": calibrated}
    predictions = {name: {"cost_us": costs, "chosen": min(ARMS, key=lambda a: costs[a])}
                   for name, costs in models.items()}
    contract = measurement_contract({}, purpose="calibration")
    contract.pop("recipe")
    contract.update(purpose="independent_local_reference", executor="two_prepared_calls_gpu0",
                    input_reuse="shared read-only gate/up; 50 independent intermediate/output pairs per arm")
    return {"scope": SCOPE, "seed": 20261105, "device": "cuda:0",
            "shape": list(SHAPE), "dtype": "bf16", "layout": "contiguous",
            "measurement_semantics": contract, "recipes": {"native": NATIVE_RECIPE, "fused": RECIPE},
            "order": [["native", "fused"] if i % 2 == 0 else ["fused", "native"] for i in range(7)],
            "quality_policy": QUALITY, "pool": POOL, "predictions": predictions,
            "analytic_stages": stages, "launch_assumption": "legacy_eager_proxy_for_graph; not measured graph launch",
            "common_projection_cost": "unknown C cancels only under the additive model; not measured FFN",
            "L2": {"admission_id": admission["admission_id"], "kernel_key_id": admission["kernel_key_id"],
                   "smoke_contract_id": read(RAW / "identity.json")["smoke_contract_id"],
                   "raw_sha256": admission["raw_file_sha256"]},
            "collector": {"git_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {p: digest(ROOT / p) for p in SOURCES}},
            "automatic_admission": False, "full_ffn_or_model_executed": False}


class NativeCall:
    def __init__(self, gate, up):
        import torch
        if (tuple(gate.shape) != SHAPE or gate.shape != up.shape or gate.dtype != torch.bfloat16
                or up.dtype != gate.dtype or gate.device != up.device or gate.requires_grad
                or up.requires_grad or not gate.is_contiguous() or not up.is_contiguous()):
            raise ValueError("exact inference BF16 contiguous case required")
        self.gate, self.up = gate, up
        self._packed, self.output = torch.empty_like(gate), torch.empty_like(gate)
        self.observation = {"recipe": NATIVE_RECIPE,
            "schemas": [str(torch.ops.aten.silu.out._schema), str(torch.ops.aten.mul.out._schema)],
            "tensors": {k: tensor_observation(v) for k, v in
                        {"gate": gate, "up": up, "intermediate": self._packed, "output": self.output}.items()},
            "implementation_source_sha256": digest(Path(__file__))}

    def __call__(self):
        import torch
        with torch.no_grad():
            torch.ops.aten.silu.out(self.gate, out=self._packed)
            torch.ops.aten.mul.out(self._packed, self.up, out=self.output)
        return self.output


def evaluate(frozen, prediction, records):
    if (prediction["predictions"] != frozen["predictions"]
            or prediction["prediction_id"] != key_id(frozen["predictions"])
            or type(prediction["frozen_monotonic_ns"]) is not int):
        raise MeasurementContractError("prediction freeze differs from pre-measurement plan")
    expected_order = [(i, arm) for i, order in enumerate(frozen["order"]) for arm in order]
    if [(r["round"], r["arm"]) for r in records] != expected_order:
        raise MeasurementContractError("exactly fourteen ordered attempts required; no retries")
    if (prediction["frozen_monotonic_ns"] >= records[0]["started_monotonic_ns"]
            or any(a["finished_monotonic_ns"] > b["started_monotonic_ns"] for a, b in zip(records, records[1:]))):
        raise MeasurementContractError("prediction was not frozen before ordered measurements")
    qualities = {arm: quality([r for r in records if r["arm"] == arm]) for arm in ARMS}
    result = {"scope": SCOPE, "quality": qualities, "automatic_admission": False,
              "reference_is_complete_ffn": False, "reference_reused_for_selection": False}
    if any(q["status"] != "quality_passed" for q in qualities.values()):
        return dict(result, status="quality_failed", comparisons=None)
    costs = {a: qualities[a]["statistics"]["median_us"] for a in ARMS}
    best = min(ARMS, key=lambda a: costs[a])
    comparisons = {}
    for name, model in prediction["predictions"].items():
        chosen = model["chosen"]
        comparisons[name] = {"frozen_choice": chosen, "reference_choice": best,
                            "same_choice": chosen == best,
                            "local_absolute_regret_us": costs[chosen] - costs[best],
                            "prediction_error_us": {a: model["cost_us"][a] - costs[a] for a in ARMS}}
    return dict(result, status="reference_ready", reference_cost_us=costs, comparisons=comparisons)


def run(output, frozen, context):
    import torch
    import torch.nn.functional as F
    from integration.lightop.profile_identity import observe_runtime
    from integration.lightop.silu_mul import LightopSiluMulAdapter
    instrumentation = {k: v for k, v in os.environ.items() if v and (
        k in {"ROCP_TOOL_LIB", "ROCPROFILER_TOOL_LIBRARIES", "HSA_TOOLS_LIB"}
        or (k == "LD_PRELOAD" and any(s in v.lower() for s in ("prof", "trace"))))}
    if instrumentation or sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("known profiling/debug instrumentation detected")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU0 required; no fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = LightopSiluMulAdapter()
    placed, index = placed_case(0)
    bound = adapter.bind(placed, index, {0: device})
    actual = observe_runtime(adapter, device)
    save(output / "runtime.json", actual)
    save_placed_op_graph(output / "placed.json", placed)
    generator = torch.Generator(device=device).manual_seed(frozen["seed"])
    gate, up = [torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    snapshots = [t.view(torch.uint8).clone() for t in (gate, up)]
    references = (F.silu(gate) * up, F.silu(gate.float()) * up.float())
    calls = {"native": [NativeCall(gate, up) for _ in range(50)],
             "fused": [bound.prepare({bound.gate_node: gate, bound.up_node: up}) for _ in range(50)]}
    prep = {a: calls[a][0].observation for a in ARMS}
    if any(c.observation != prep[a] for a in ARMS for c in calls[a]):
        raise MeasurementContractError("prepared identities disagree")
    save(output / "preparation.json", prep)
    save(output / "identity.json", verify_identity(frozen, placed, index, prep, actual))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graphs, warmup = {}, {}
    tolerance = frozen["measurement_semantics"]["correctness"]
    for arm in ARMS:
        context.update(stage="capture_and_warmup", arm=arm, memory=memory_snapshot(torch, device))
        save(output / "last_stage.json", context)
        with torch.cuda.stream(stream):
            for call in calls[arm]:
                call()
        torch.cuda.synchronize(device)
        graphs[arm] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[arm], stream=stream, pool=None):
            for call in calls[arm]:
                call()
        with torch.cuda.stream(stream):
            invalidate_buffers(calls[arm])
            for _ in range(10):
                graphs[arm].replay()
        torch.cuda.synchronize(device)
        warmup[arm] = check_outputs(torch, calls[arm], gate, up, snapshots, references, tolerance)
    save(output / "warmup.json", warmup)
    records = []
    for attempt, order in enumerate(frozen["order"]):
        for arm in order:
            context.update(stage="measurement", arm=arm, round=attempt)
            save(output / "last_stage.json", context)
            started = time.monotonic_ns()
            with torch.cuda.stream(stream):
                invalidate_buffers(calls[arm])
                torch.cuda.synchronize(device)
                timing = timed_replay(torch, graphs[arm], stream, device)
            checks = check_outputs(torch, calls[arm], gate, up, snapshots, references, tolerance)
            row = {"arm": arm, "round": attempt, "started_monotonic_ns": started,
                   "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks}
            records.append(row)
            save(output / f"attempt_{len(records)-1:02d}.json", row)
            print(json.dumps(row), flush=True)
    result = {"scope": SCOPE, "complete": True, "records": records,
              "automatic_admission": False, "known_instrumentation_guard_passed": True}
    save(output / "result.json", result)


def verify_identity(frozen, placed, index, prep, actual):
    keys = LightopKernelKeys(placed, index, prep["fused"], actual)
    key, gaps = keys.describe(placed.plan.groups[index])
    if (gaps or key_id(key) != frozen["L2"]["kernel_key_id"]
            or measurement_contract_id(measurement_contract(actual["environment"])) != frozen["L2"]["smoke_contract_id"]):
        raise MeasurementContractError("actual fused implementation/case/runtime differs from L2")
    native = prep["native"]
    if (native["recipe"] != NATIVE_RECIPE or native["implementation_source_sha256"] != digest(Path(__file__))
            or native["schemas"] != ["aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
                                      "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"]):
        raise MeasurementContractError("native chain implementation differs")
    expected = copy.deepcopy(prep["fused"]["tensors"]["gate"])
    if any(observed != expected for observed in native["tensors"].values()) or set(native["tensors"]) != {"gate", "up", "intermediate", "output"}:
        raise MeasurementContractError("native observed tensor case/layout differs")
    contract = dict(frozen["measurement_semantics"], environment=actual["environment"])
    return {"fused_kernel_key_id": key_id(key),
            "native_chain_key_id": key_id({"preparation": native, "runtime": actual["runtime"]}),
            "shared_measurement_contract": contract,
            "shared_measurement_contract_id": measurement_contract_id(contract),
            "recipes_sha256": {a: key_id(frozen["recipes"][a]) for a in ARMS},
            "native_dispatch_trace_collected": False}


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("collection failed; preserve diagnostics")
    frozen = read(directory / "plan.json")
    current = plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if current != frozen:
        raise MeasurementContractError("plan or source identity changed")
    process = read(directory / "process.json")
    result = read(directory / "result.json")
    if (process != {"python_returncode": 0, "log_returncode": 0, "automatic_admission": False}
            or result["scope"] != SCOPE or result["complete"] is not True
            or result["automatic_admission"] is not False or result["known_instrumentation_guard_passed"] is not True
            or read(directory / "warmup.json") != {a: CHECKS for a in ARMS}):
        raise MeasurementContractError("incomplete collection/process/warmup")
    files = sorted(directory.glob("attempt_*.json"))
    if [p.name for p in files] != [f"attempt_{i:02d}.json" for i in range(14)]:
        raise MeasurementContractError("extra or missing attempts")
    records = [read(p) for p in files]
    if records != result["records"]:
        raise MeasurementContractError("raw attempts differ from summary")
    placed = load_placed_op_graph(directory / "placed.json", registry=adapter_registry())
    index = next(i for i, g in enumerate(placed.plan.groups) if g.extra.get("pattern_id") == "silu_mul")
    identity = verify_identity(frozen, placed, index, read(directory / "preparation.json"), read(directory / "runtime.json"))
    if identity != read(directory / "identity.json"):
        raise MeasurementContractError("saved identity/contract differs from replay")
    evaluated = evaluate(frozen, read(directory / "prediction.json"), records)
    evaluated["raw_sha256"] = {p.name: digest(p) for p in sorted(directory.glob("*.json"))}
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
        parser.error("fresh output required")
    context = {"stage": args.phase, "device": "cuda:0", "pool": POOL}
    try:
        if args.phase == "review":
            result = review(args.input)
            save(output / "evaluation.json", result)
            print(json.dumps(result), flush=True)
            return 0 if result["status"] == "reference_ready" else 1
        frozen = plan()
        save(output / "plan.json", frozen)
        save(output / "prediction.json", {"predictions": frozen["predictions"],
             "prediction_id": key_id(frozen["predictions"]), "frozen_monotonic_ns": time.monotonic_ns()})
        print(json.dumps({"phase": args.phase, "predictions": frozen["predictions"]}), flush=True)
        if args.phase == "run":
            run(output, frozen, context)
        return 0
    except Exception as error:
        torch = sys.modules.get("torch")
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error),
            "traceback": traceback.format_exc(), "context": context,
            "memory": memory_snapshot(torch, "cuda:0") if torch and args.phase == "run" else None,
            "automatic_admission": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
