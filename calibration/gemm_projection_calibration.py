"""P2: FFN Projection GEMMs on GPU0 and GPU1 under the stage Graph contract.

Cases: gate/up/down x GPU0/GPU1, the exact FFN shapes. The current Roofline
proxy prediction of every case is frozen into plan.json before any GPU work.
Reviewed results are per-GPU-identity calibration candidates; the analytic
Roofline model is not edited.
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

from calibration import stage_size_calibration as base
from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "ffn_projection_gemm_gpu0_gpu1_v1"
TOKENS = 256
PROJECTIONS = {  # FFN node -> (k in, n out)
    "gate": {"node": "L000.mlp.gate_proj", "k": 1024, "n": 3072},
    "up": {"node": "L000.mlp.up_proj", "k": 1024, "n": 3072},
    "down": {"node": "L000.mlp.down_proj", "k": 3072, "n": 1024},
}
DEVICES = (0, 1)
TOLERANCE = {"rtol": .02, "atol": .02}
SOURCES = ("calibration/gemm_projection_calibration.py", "tools/run_gemm_projection_calibration.sh",
           "calibration/stage_size_calibration.py", "calibration/lightop_profile_smoke.py",
           "execution/reference_identity.py", "integration/flagscale/kerv_profile_runtime.py",
           "cost_model/fusion_kernel_profile.py", "cost_model/measurement_labels.py")
RECIPE = {"id": "aten.mm.out.linear_weight_out_in.preallocated.v1",
          "timed_body": "aten.mm.out(x2d, weight.t(), out=output)",
          "weight_layout": "nn.Linear [out, in] contiguous; transposed view, no copy",
          "operands": ["x", "weight"], "bias": None,
          "layout_status": "declared FFN execution contract; not captured from a running FFN"}


def case_ids():
    return [f"gemm/{p}/gpu{d}" for p in PROJECTIONS for d in DEVICES]


def describe(case_id):
    _, proj, dev = case_id.split("/")
    spec, d = PROJECTIONS[proj], int(dev[3:])
    return {"kind": "gemm", "projection": proj, "node": spec["node"], "device": d,
            "m": TOKENS, "k": spec["k"], "n": spec["n"], "x_shape": [TOKENS, spec["k"]],
            "weight_shape": [spec["n"], spec["k"]], "output_shape": [TOKENS, spec["n"]],
            "flops": 2 * TOKENS * spec["k"] * spec["n"], "timing_device": f"cuda:{d}"}


def roofline_predictions():
    """Current FFN proxy costs (baseline scenario), frozen before measurement."""
    from cost_model.ffn_decision_sensitivity import FFNSensitivity
    audit = FFNSensitivity()
    out = {}
    for case_id in case_ids():
        case = describe(case_id)
        cost = audit.costs[(case["node"],), case["device"]]
        out[case_id] = {"predicted_us": cost.duration * 1e6, "source": cost.source,
                        "includes_assumed_graph_overhead_us": audit.hardware.launch_overhead * 1e6,
                        "roofline_core_us": cost.duration * 1e6 - audit.hardware.launch_overhead * 1e6}
    return out


def plan():
    ids = case_ids()
    return {"scope": SCOPE, "architecture": "gfx936", "seed": 20261111, "dtype": "bf16",
        "cases": {c: describe(c) for c in ids}, "recipe": copy.deepcopy(RECIPE),
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contract": {**base.contract(), "correctness": dict(TOLERANCE),
                     "buffers": "shared read-only x and weight per case; 50 independent outputs",
                     "cache_context": "same weight reused by 50 calls per replay; no cold-weight claim"},
        "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL),
        "frozen_predictions": roofline_predictions(),
        "prediction_error_threshold": None,
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "automatic_admission": False, "model_parameters_updated": False}


class GemmCall:
    def __init__(self, x, weight):
        import torch
        from execution.reference_identity import tensor_observation
        if (x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.dim() != 2 or weight.dim() != 2
                or not x.is_contiguous() or not weight.is_contiguous() or x.shape[1] != weight.shape[1]
                or x.device != weight.device):
            raise ValueError("Contiguous BF16 x[m,k] and weight[n,k] on one device required")
        self.x, self.weight = x, weight
        self.output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        self.observation = {"recipe": RECIPE, "source_sha256": base.digest(Path(__file__)),
            "tensors": {"x": tensor_observation(x), "weight": tensor_observation(weight),
                        "output": tensor_observation(self.output)}}

    def __call__(self):
        import torch
        with torch.no_grad():
            torch.ops.aten.mm.out(self.x, self.weight.t(), out=self.output)
        return self.output


def check(torch, calls, x, weight, snapshots, references):
    if not torch.equal(x.view(torch.uint8), snapshots[0]) or not torch.equal(weight.view(torch.uint8), snapshots[1]):
        raise AssertionError("Read-only GEMM input changed")
    pointers = set()
    for call in calls:
        for reference in references:
            torch.testing.assert_close(call.output.float(), reference.float(), **TOLERANCE)
        if torch._C._overlaps(call.output, x) or torch._C._overlaps(call.output, weight):
            raise AssertionError("GEMM output aliases an input")
        pointers.add(call.output.data_ptr())
    if len(pointers) != len(calls):
        raise AssertionError("GEMM calls share output storage")
    return {"outputs_checked": len(calls), "references_per_output": len(references),
            "input_bytes_unchanged": True, "outputs_independent": True}


def build(torch, frozen, output, context):
    built, prepared, warmup = {}, {}, {}
    for case_id, case in frozen["cases"].items():
        device = torch.device(case["timing_device"])
        context.update(stage="capture_and_warmup", case=case_id)
        save(output / "last_stage.json", context)
        generator = torch.Generator(device=device).manual_seed(frozen["seed"] + case["k"] + case["n"])
        x = torch.randn(case["x_shape"], device=device, dtype=torch.bfloat16, generator=generator)
        weight = (torch.randn(case["weight_shape"], device=device, generator=generator)
                  / math.sqrt(case["k"])).to(torch.bfloat16)
        snapshots = (x.view(torch.uint8).clone(), weight.view(torch.uint8).clone())
        references = (x @ weight.t(), x.float() @ weight.float().t())
        calls = [GemmCall(x, weight) for _ in range(base.CALLS)]
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device):
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
                for call in calls:
                    call.output.fill_(float("nan"))
                for _ in range(base.WARMUP):
                    graph.replay()
            torch.cuda.synchronize(device)
        warmup[case_id] = check(torch, calls, x, weight, snapshots, references)

        def invalidate(calls=calls, stream=stream, device=device):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                for call in calls:
                    call.output.fill_(float("nan"))

        def verify(calls=calls, x=x, weight=weight, snapshots=snapshots, references=references):
            return check(torch, calls, x, weight, snapshots, references)

        built[case_id] = {"graph": graph, "stream": stream, "devices": (case["device"],), "timing_device": device,
                          "invalidate": invalidate, "verify": verify}
        prepared[case_id] = calls[0].observation
    return built, prepared, warmup


def identity(frozen, prepared, runtime):
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
    contract_id = key_id(frozen["contract"])
    keys = {}
    for case_id, case in frozen["cases"].items():
        obs, dev = prepared[case_id], runtime["devices"][str(case["device"])]
        if dev["ordinal"] != case["device"] or str(dev.get("architecture", "")).split(":")[0] != frozen["architecture"]:
            raise MeasurementContractError("Device identity differs: " + case_id)
        if obs["recipe"] != RECIPE or obs["source_sha256"] != frozen["collector"]["source_sha256"][SOURCES[0]]:
            raise MeasurementContractError("GEMM implementation changed: " + case_id)
        shapes = {"x": case["x_shape"], "weight": case["weight_shape"], "output": case["output_shape"]}
        for role, tensor in obs["tensors"].items():
            if (tensor["shape"] != shapes[role] or tensor["dtype"] != "bf16" or tensor["layout"] != "contiguous"
                    or tensor["strides"] != base.strides(shapes[role])
                    or tensor["layout_provenance"]["device"] != case["timing_device"]):
                raise MeasurementContractError("Actual GEMM layout differs: " + case_id)
        gpu_key = {"device": copy.deepcopy(dev), "runtime": {k: rt[k] for k in (
            "torch_distribution", "torch_git_revision", "torch_build_sha256", "torch_extension_sha256", "hip", "driver")}}
        key = {"implementation": obs, "gpu": gpu_key, "case": case, "contract": frozen["contract"], "pool": frozen["pool"]}
        keys[case_id] = {"case_identity": key, "case_identity_id": key_id(key),
                         "gpu_key_id": key_id(gpu_key), "contract_id": contract_id}
    return {"scope": SCOPE, "cases": keys, "uuid_is_provenance_only": True}


def run(output, frozen, context):
    import torch
    if not torch.version.hip or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 and GPU1 required; no CPU/eager fallback")
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("Known profiling/debug instrumentation detected")
    torch.cuda.set_device(0)
    with torch.no_grad():
        runtime = base.observe_runtime(torch)
        save(output / "runtime.json", runtime)
        entries, prepared, warmup = build(torch, frozen, output, context)
        save(output / "preparation.json", prepared)
        save(output / "identity.json", identity(frozen, prepared, runtime))
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
    result.update(scope=SCOPE, data_role="per-GPU GEMM calibration candidates")
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
    replay = identity(frozen, base.read(directory / "preparation.json"), base.read(directory / "runtime.json"))
    if replay != base.read(directory / "identity.json"):
        raise MeasurementContractError("Case identity or contract differs")
    evaluated = evaluate(frozen, records)
    if process["python_returncode"] != (0 if evaluated["status"] == "calibration_candidates_ready" else 1):
        raise MeasurementContractError("Process result differs from recomputed quality")
    evaluated.update(identity=replay, plan_sha256=base.digest(directory / "plan.json"),
                     raw_sha256={p.name: base.digest(p) for p in sorted(directory.glob("*.json"))})
    evaluated["prediction_error"] = {c: {
        "predicted_us": frozen["frozen_predictions"][c]["predicted_us"],
        "measured_us": q["statistics"]["median_us"] if q["statistics"] else None,
        "quality": q["status"],
        "signed_error": (frozen["frozen_predictions"][c]["predicted_us"] - q["statistics"]["median_us"])
                        / q["statistics"]["median_us"] if q["statistics"] else None}
        for c, q in evaluated["cases"].items()}
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
                                  "predictions_us": {c: round(p["predicted_us"], 4)
                                                     for c, p in frozen["frozen_predictions"].items()}}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"status": evaluated["status"], "quality_failed": evaluated["quality_failed"]}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        torch = sys.modules.get("torch")
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
            "context": context, "memory": memory_snapshot(torch, "cuda:0") if torch and args.phase == "run" else None})
        raise


if __name__ == "__main__":
    sys.exit(main())
