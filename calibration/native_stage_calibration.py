"""Standalone Native stages: plan on CPU, collect only on HIP, review on CPU.

Records constrain effective stage service. They are not formal singleton
Profiles, a new whole-chain benchmark, or an automatic model parameter fit.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from calibration.lightop_profile_calibration import CHECKS, QUALITY, quality, timed_replay
from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (1, 256, 3072)
STAGES = ("silu", "mul")
STAGE_CHECKS = {"left_matches_bf16_silu_gate": True, "outputs_invalidated_before_replay": True}
SCOPE = "native_stage_effective_cost_calibration_not_profile_admission_v1"
FREEZE = "data/graph_analytic_freeze_v1/freeze.json"
SOURCES = ("calibration/native_stage_calibration.py", "tools/run_native_stage_calibration.sh",
           "calibration/lightop_profile_calibration.py", "calibration/lightop_profile_smoke.py",
           "execution/reference_identity.py", "integration/flagscale/kerv_profile_runtime.py",
           "cost_model/fusion_kernel_profile.py", "cost_model/measurement_labels.py", FREEZE)
SCHEMAS = {
    "silu": "aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
    "mul": "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
}
RECIPES = {
    "silu": {"id": "aten.silu.out.preallocated_stage.v1",
             "timed_body": "aten.silu.out(gate, out=output)", "operands": ["gate"]},
    "mul": {"id": "aten.mul.out.preallocated_derived_left_stage.v1",
            "timed_body": "aten.mul.out(left, up, out=output)", "operands": ["left", "up"],
            "left_origin": "aten.silu.out(gate, out=left), outside capture and timing"},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    frozen = read(ROOT / FREEZE)
    if frozen["case"]["shape"] != list(SHAPE) or frozen["case"]["dtype"] != "bf16":
        raise MeasurementContractError("Stage case differs from frozen model scope")
    # Confirm the model/specification freeze, without touching old GPU labels.
    for path in ("cost_model/graph_analytic.py", "cost_model/GRAPH_ANALYTIC_FROZEN_SPEC.md",
                 "configs/graph_analytic_scenarios.json"):
        if digest(ROOT / path) != frozen["source_sha256"][path]:
            raise MeasurementContractError("Frozen model source changed: " + path)
    return {"scope": SCOPE, "device": "cuda:0", "architecture": "gfx936", "seed": 20261106,
        "shape": list(SHAPE), "dtype": "bf16", "layout": "explicit_contiguous",
        "recipes": copy.deepcopy(RECIPES), "order": [list(STAGES if i % 2 == 0 else STAGES[::-1]) for i in range(7)],
        "contract": {"mode": "graph_replay", "granularity": "isolated_stage_batch_amortized",
            "calls_per_round": 50, "rounds": 7, "extra_attempts": 0,
            "eager_preparation_calls": 50, "warmup_replays": 10, "unit": "us/call",
            "statistic": "median", "correctness": {"rtol": .01, "atol": .001},
            "timing_boundary": "start Event -> one Graph replay -> end Event; same device and stream",
            "synchronization": "device synchronized outside Event interval",
            "excluded": ["allocation", "input generation", "left generation", "validation", "host submission"],
            "buffers": "shared read-only gate/up/derived-left; 50 independent outputs per stage",
            "cache_context": "isolated stages; no claim of Native chain cache equivalence"},
        "quality_policy": copy.deepcopy(QUALITY), "pool": copy.deepcopy(POOL),
        "buffer_plan": {"output_buffers": 100, "output_bytes": 100 * 2 * 256 * 3072,
                        "inputs": ["gate", "up", "left"], "graph_pool_bytes": "runtime observed"},
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: digest(ROOT / p) for p in SOURCES}},
        "existing_measurements_preserved": True, "automatic_admission": False,
        "whole_chain_measured": False, "model_parameters_updated": False,
        "pure_graph_overhead_identified": False}


class StageCall:
    def __init__(self, stage, gate, up, left):
        import torch
        from execution.reference_identity import tensor_observation
        if stage not in STAGES:
            raise ValueError("Unknown stage")
        tensors = {"gate": gate, "up": up, "left": left}
        if any(tuple(t.shape) != SHAPE or t.dtype != torch.bfloat16 or not t.is_contiguous()
               or t.requires_grad or t.device != gate.device for t in tensors.values()):
            raise ValueError("Explicit contiguous BF16 same-device inputs required")
        if any(torch._C._overlaps(a, b) for a, b in ((gate, up), (gate, left), (left, up))):
            raise ValueError("Stage inputs must have independent storage")
        self.stage, self.gate, self.up, self.left = stage, gate, up, left
        self.output = torch.empty_like(gate)
        op = torch.ops.aten.silu.out if stage == "silu" else torch.ops.aten.mul.out
        self.observation = {"recipe": RECIPES[stage], "schema": str(op._schema),
            "tensors": {k: tensor_observation(t) for k, t in {**tensors, "output": self.output}.items()},
            "source_sha256": digest(Path(__file__))}

    def __call__(self):
        import torch
        with torch.no_grad():
            if self.stage == "silu":
                torch.ops.aten.silu.out(self.gate, out=self.output)
            else:
                torch.ops.aten.mul.out(self.left, self.up, out=self.output)
        return self.output


def check(torch, calls, inputs, snapshots, references, tolerance):
    for name, tensor in inputs.items():
        if not torch.equal(tensor.view(torch.uint8), snapshots[name]):
            raise AssertionError("Read-only input changed: " + name)
    # Derivation belongs outside the captured graph and timer.
    expected = torch.nn.functional.silu(inputs["gate"])
    if not torch.equal(inputs["left"].view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError("Mul left is not the BF16 SiLU result")
    pointers = set()
    for call in calls:
        for reference in references:
            torch.testing.assert_close(call.output.float(), reference.float(), **tolerance)
        if any(torch._C._overlaps(call.output, t) for t in inputs.values()):
            raise AssertionError("Stage output aliases a read-only input")
        pointers.add(call.output.data_ptr())
    if len(pointers) != len(calls):
        raise AssertionError("Stage calls share output storage")
    return {"outputs_checked": len(calls), "references_per_output": len(references),
            "input_bytes_unchanged": True, "outputs_independent": True}


def observe_runtime(torch, device):
    from integration.flagscale.kerv_profile_runtime import driver_version
    props = torch.cuda.get_device_properties(device)
    return {"runtime": {"torch_distribution": importlib.metadata.version("torch"),
            "torch_module_version": str(torch.__version__), "torch_git_revision": torch.version.git_version,
            "torch_extension_sha256": digest(Path(torch._C.__file__)),
            "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "accelerator": "HIP", "hip": torch.version.hip, "driver": driver_version(),
            "grad_enabled": torch.is_grad_enabled(), "autocast_enabled": torch.is_autocast_enabled("cuda"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": str(device)},
        "device": {"name": props.name, "architecture": getattr(props, "gcnArchName", None), "ordinal": device.index},
        "provenance": {"uuid": str(getattr(props, "uuid", "unavailable")),
                       "torch_extension_path": str(torch._C.__file__)}}


def identity(frozen, prepared, runtime):
    rt, dev = runtime["runtime"], runtime["device"]
    if (rt["accelerator"] != "HIP" or rt["device"] != "cuda:0" or dev["ordinal"] != 0
            or str(dev.get("architecture", "")).split(":")[0] != frozen["architecture"]):
        raise MeasurementContractError("Runtime must identify HIP GPU0 gfx936")
    if (rt.get("grad_enabled") is not False or rt.get("autocast_enabled") is not False
            or rt.get("deterministic_algorithms") is not False):
        raise MeasurementContractError("Native stage flags differ from no_grad BF16")
    for field in ("torch_distribution", "torch_git_revision", "torch_extension_sha256", "torch_build_sha256", "hip", "driver"):
        if not rt.get(field):
            raise MeasurementContractError("Missing runtime identity: " + field)
    keys = {}
    for stage in STAGES:
        obs = prepared[stage]
        if obs["recipe"] != RECIPES[stage] or obs["schema"] != SCHEMAS[stage] or obs["source_sha256"] != digest(Path(__file__)):
            raise MeasurementContractError("Stage implementation changed")
        if set(obs["tensors"]) != {"gate", "up", "left", "output"}:
            raise MeasurementContractError("Incomplete actual input/output observation")
        for tensor in obs["tensors"].values():
            if (tensor["shape"] != list(SHAPE) or tensor["dtype"] != "bf16" or tensor["layout"] != "contiguous"
                    or tensor["strides"] != [786432, 3072, 1]
                    or tensor["layout_provenance"] != {"kind": "runtime_tensor", "device": "cuda:0",
                        "observer": "Tensor.shape/dtype/stride/is_contiguous"}):
                raise MeasurementContractError("Actual stage layout differs from declared case")
        key = {"implementation": obs, "runtime": rt, "device": dev, "shape": frozen["shape"],
               "contract": frozen["contract"], "pool": frozen["pool"]}
        keys[stage] = {"stage_identity": key, "stage_identity_id": key_id(key),
                       "contract_id": key_id(frozen["contract"])}
    return {"scope": SCOPE, "stages": keys, "uuid_is_provenance_only": True}


def evaluate(frozen, records):
    expected = [(i, stage) for i, order in enumerate(frozen["order"]) for stage in order]
    if [(r["round"], r["stage"]) for r in records] != expected:
        raise MeasurementContractError("Fourteen chronological attempts required, no retries")
    if any(r.get("stage_checks") != STAGE_CHECKS for r in records):
        raise MeasurementContractError("Missing input derivation or output invalidation checks")
    if any(a["finished_monotonic_ns"] > b["started_monotonic_ns"] for a, b in zip(records, records[1:])):
        raise MeasurementContractError("Stage attempts overlap or were reordered")
    if len({r["timing"]["event_boundary"]["stream"] for r in records}) != 1:
        raise MeasurementContractError("Stage stream changed")
    results = {s: quality([r for r in records if r["stage"] == s]) for s in STAGES}
    return {"scope": SCOPE, "stages": results, "automatic_admission": False,
        "status": "calibration_candidates_ready" if all(q["status"] == "quality_passed" for q in results.values()) else "quality_failed",
        "data_role": "calibration_candidates_only", "whole_chain_cost_us": None,
        "pure_graph_overhead_us": None, "model_parameters_updated": False}


def run(output, frozen, context):
    import torch
    import torch.nn.functional as F
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU0 required; no CPU/eager fallback")
    instrumentation = {k: v for k, v in os.environ.items() if v and (
        k in {"ROCP_TOOL_LIB", "ROCPROFILER_TOOL_LIBRARIES", "HSA_TOOLS_LIB"}
        or (k == "LD_PRELOAD" and any(s in v.lower() for s in ("prof", "trace"))))}
    if instrumentation or sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("Known profiling/debug instrumentation detected")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with torch.no_grad():
        runtime = observe_runtime(torch, device)
        save(output / "runtime.json", runtime)
        context["stage"] = "input_preparation"
        generator = torch.Generator(device=device).manual_seed(frozen["seed"])
        gate, up = [torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
        left = torch.empty_like(gate)
        torch.ops.aten.silu.out(gate, out=left)
        inputs = {"gate": gate, "up": up, "left": left}
        snapshots = {k: t.view(torch.uint8).clone() for k, t in inputs.items()}
        refs = {"silu": (F.silu(gate), F.silu(gate.float())),
                "mul": (left * up, left.float() * up.float())}
        calls = {s: [StageCall(s, gate, up, left) for _ in range(50)] for s in STAGES}
        prepared = {s: calls[s][0].observation for s in STAGES}
        if any(c.observation != prepared[s] for s in STAGES for c in calls[s]):
            raise MeasurementContractError("Prepared stage identities disagree")
        save(output / "preparation.json", prepared)
        save(output / "identity.json", identity(frozen, prepared, runtime))
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graphs, warmup = {}, {}
        for stage in STAGES:
            context.update(stage="capture_and_warmup", operator=stage, pool=POOL, memory=memory_snapshot(torch, device))
            save(output / "last_stage.json", context)
            with torch.cuda.stream(stream):
                for call in calls[stage]:
                    call()
            torch.cuda.synchronize(device)
            graphs[stage] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[stage], stream=stream, pool=None):
                for call in calls[stage]:
                    call()
            with torch.cuda.stream(stream):
                for call in calls[stage]:
                    call.output.fill_(float("nan"))
                for _ in range(10):
                    graphs[stage].replay()
            torch.cuda.synchronize(device)
            warmup[stage] = check(torch, calls[stage], inputs, snapshots, refs[stage], frozen["contract"]["correctness"])
        save(output / "warmup.json", warmup)
        records = []
        for attempt, order in enumerate(frozen["order"]):
            for stage in order:
                context.update(stage="measurement", operator=stage, round=attempt)
                save(output / "last_stage.json", context)
                started = time.monotonic_ns()
                with torch.cuda.stream(stream):
                    for call in calls[stage]:
                        call.output.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    timing = timed_replay(torch, graphs[stage], stream, device)
                checks = check(torch, calls[stage], inputs, snapshots, refs[stage], frozen["contract"]["correctness"])
                row = {"stage": stage, "round": attempt, "started_monotonic_ns": started,
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks,
                       "stage_checks": copy.deepcopy(STAGE_CHECKS)}
                records.append(row)
                save(output / f"attempt_{len(records)-1:02d}.json", row)
                print(json.dumps(row), flush=True)
        result = {"scope": SCOPE, "complete": True, "records": records,
                  "automatic_admission": False, "known_instrumentation_guard_passed": True}
        save(output / "result.json", result)
        return evaluate(frozen, records)


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen, current = read(directory / "plan.json"), plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan or source identity changed")
    process, result = read(directory / "process.json"), read(directory / "result.json")
    if (process["python_returncode"] not in (0, 1) or process["log_returncode"] != 0
            or process["automatic_admission"] is not False or result["scope"] != SCOPE
            or result["complete"] is not True or result["automatic_admission"] is not False
            or result["known_instrumentation_guard_passed"] is not True
            or read(directory / "warmup.json") != {s: CHECKS for s in STAGES}):
        raise MeasurementContractError("Incomplete process, result or warmup")
    files = sorted(directory.glob("attempt_*.json"))
    if [p.name for p in files] != [f"attempt_{i:02d}.json" for i in range(14)]:
        raise MeasurementContractError("Missing or extra attempts")
    records = [read(p) for p in files]
    if records != result["records"]:
        raise MeasurementContractError("Raw attempts differ from summary")
    replay = identity(frozen, read(directory / "preparation.json"), read(directory / "runtime.json"))
    if replay != read(directory / "identity.json"):
        raise MeasurementContractError("Stage identity or contract differs")
    evaluated = evaluate(frozen, records)
    expected_return = 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    if process["python_returncode"] != expected_return:
        raise MeasurementContractError("Process result differs from recomputed quality")
    evaluated.update(identity=replay, raw_sha256={p.name: digest(p) for p in sorted(directory.glob("*.json"))})
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
    context = {"stage": args.phase, "device": "cuda:0", "pool": POOL}
    try:
        if args.phase == "review":
            evaluated = review(args.input)
            save(output / "evaluation.json", evaluated)
        else:
            frozen = plan()
            save(output / "plan.json", frozen)
            if args.phase == "plan":
                print(json.dumps({"phase": "plan", "stage_count": 2, "new_gpu_execution": False}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"status": evaluated["status"], "automatic_admission": False}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        torch = sys.modules.get("torch")
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
            "context": context, "pool": POOL, "memory": memory_snapshot(torch, "cuda:0")
            if torch and args.phase == "run" else None, "automatic_admission": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
