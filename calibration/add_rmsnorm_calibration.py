"""GPU0 Add+RMSNorm: LightOP complete fused call vs Native stages, Qwen3 shape [1,256,1024].

Cases (GPU0, BF16):
- fused/add_rmsnorm: x_buf.copy_(x); res_buf.copy_(residual); lightop fused_add_rms_norm_opt
  (the input copies are part of the call).
- stage/add: aten.add.out(x, residual, out=s).
- stage/rmsnorm: aten.rms_norm(s, [1024], weight, eps); s is the real Add output.
  aten.rms_norm has no out= overload, so its output comes from the captured
  graph's private pool.
- chain/add_rmsnorm_native: add.out then rms_norm. A same-batch comparison
  only; never a Profile.

Graph Replay contract as the existing stage records: 50 calls per replay,
10 warm-up replays, 7 rounds, median, IQR/Median <= 10%, no extra rounds.
Analytic predictions are frozen in plan.json before GPU work, and the same
load gate as batch-03 applies. No Profile is admitted automatically.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from calibration import stage_size_calibration as base
from calibration.stage_size_batch03 import LOAD_GATE, wait_for_quiet_host
from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = base.ROOT
SCOPE = "gpu0_add_rmsnorm_fused_vs_native_v1"
SHAPE = (1, 256, 1024)
EPS = 1e-6
TOLERANCE = {"add": {"rtol": .01, "atol": .01}, "rmsnorm": {"rtol": .02, "atol": .02}}
SOURCES = ("calibration/add_rmsnorm_calibration.py", "tools/run_add_rmsnorm_calibration.sh",
           "integration/lightop/add_rmsnorm.py", "integration/lightop/silu_mul.py", "integration/lightop/catalog.py",
           "calibration/stage_size_calibration.py", "calibration/stage_size_batch03.py",
           "calibration/lightop_profile_smoke.py", "execution/reference_identity.py",
           "integration/flagscale/kerv_profile_runtime.py", "cost_model/fusion_kernel_profile.py",
           "cost_model/measurement_labels.py")
RECIPES = {
    "add": {"id": "aten.add.out.preallocated_stage.v1", "timed_body": "aten.add.out(x, residual, out=s)",
            "operands": ["x", "residual"]},
    "rmsnorm": {"id": "aten.rms_norm.functional_graph_pool_stage.v1",
                "timed_body": "aten.rms_norm(s, [1024], weight, eps)", "operands": ["s", "weight"],
                "output": "allocated from the captured graph's private pool (no out= overload)",
                "eps": EPS},
    "chain": {"id": "aten.add_rmsnorm.native_chain.v1",
              "timed_body": "aten.add.out(x, residual, out=s); aten.rms_norm(s, [1024], weight, eps)"},
}
CASES = ("fused/add_rmsnorm/gpu0", "stage/add/gpu0", "stage/rmsnorm/gpu0", "chain/add_rmsnorm_native/gpu0")


def describe(case_id):
    kind, name, _ = case_id.split("/")
    return {"kind": kind, "name": name, "device": 0, "timing_device": "cuda:0", "shape": list(SHAPE), "dtype": "bf16",
            "role": "comparison_only_not_a_profile" if kind == "chain" else "profile_candidate"}


def analytic_predictions():
    """Current analytic costs for the model-graph groups these cases correspond to."""
    from cost_model.hardware_profile import AnalyticHardwareProfile
    from cost_model.search_evaluation import build_evaluation_graph
    from cost_model.structural import StructuralCostModel
    from integration.lightop.add_rmsnorm import executable_registry
    from simulator import FusionGroup, Plan
    from simulator.hardware import load_hardware
    reg = executable_registry()
    graph = build_evaluation_graph("layer")
    profile = AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))
    add, norm = "L000.residual_add_1", "L000.pre_mlp_norm"
    out = {}
    for fused in (False, True):
        groups = []
        for n in graph.topological_order():
            if fused and n == add:
                continue
            groups.append(FusionGroup([add, norm], 0, {"pattern_id": "add_rmsnorm"}) if fused and n == norm
                          else FusionGroup([n], 0, {"pattern_id": "unfused"}))
        prepared = StructuralCostModel(registry=reg).prepare(graph, Plan(groups=groups), 2)
        for g in prepared.structure.groups:
            if g.node_ids == (add, norm):
                out["fused/add_rmsnorm/gpu0"] = profile.estimate_group(graph, g).duration * 1e6
            elif not fused and g.node_ids == (add,):
                out["stage/add/gpu0"] = profile.estimate_group(graph, g).duration * 1e6
            elif not fused and g.node_ids == (norm,):
                out["stage/rmsnorm/gpu0"] = profile.estimate_group(graph, g).duration * 1e6
    out["chain/add_rmsnorm_native/gpu0"] = out["stage/add/gpu0"] + out["stage/rmsnorm/gpu0"]
    return {c: {"predicted_us": v, "source": "analytic_roofline (AnalyticHardwareProfile)"} for c, v in out.items()}


def plan():
    ids = list(CASES)
    from integration.lightop.add_rmsnorm import RECIPE as FUSED_RECIPE
    return {"scope": SCOPE, "architecture": "gfx936", "seed": 20261115, "dtype": "bf16", "shape": list(SHAPE),
        "cases": {c: describe(c) for c in ids}, "recipes": {**copy.deepcopy(RECIPES), "fused": copy.deepcopy(FUSED_RECIPE)},
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(base.ROUNDS)],
        "contract": {**base.contract(), "correctness": copy.deepcopy(TOLERANCE),
                     "buffers": "shared read-only x/residual/weight; per-call outputs (fused: per-call x/res copies)"},
        "quality_policy": copy.deepcopy(base.QUALITY), "pool": copy.deepcopy(POOL), "load_gate": dict(LOAD_GATE),
        "frozen_analytic_predictions": analytic_predictions(),
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: base.digest(ROOT / p) for p in SOURCES}},
        "automatic_admission": False, "model_parameters_updated": False}


def build(torch, frozen, output, context):
    import torch.nn.functional as F
    from execution.reference_identity import tensor_observation
    from integration.lightop.add_rmsnorm import PreparedAddRmsnormCall, declared_binding
    device = torch.device("cuda:0")
    g = torch.Generator(device=device).manual_seed(frozen["seed"])
    x, residual = [torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=g) for _ in range(2)]
    weight = (1 + .1 * torch.randn(SHAPE[-1], device=device, generator=g)).to(torch.bfloat16)
    s_ref = torch.empty_like(x)
    torch.ops.aten.add.out(x, residual, out=s_ref)
    inputs = {"x": x, "residual": residual, "weight": weight, "s": s_ref}
    snapshots = {k: t.clone() for k, t in inputs.items()}
    y32 = s_ref.float() * torch.rsqrt(s_ref.float().pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    s32 = x.float() + residual.float()
    entries, prepared, warmup = {}, {}, {}

    def inputs_unchanged():
        return all(torch.equal(inputs[k], snapshots[k]) for k in inputs)

    for case_id in frozen["cases"]:
        kind, name, _ = case_id.split("/")
        context.update(stage="capture_and_warmup", case=case_id, memory=memory_snapshot(torch, device))
        save(output / "last_stage.json", context)
        if kind == "fused":
            calls = [PreparedAddRmsnormCall(x, residual, weight, EPS) for _ in range(base.CALLS)]
            body = lambda calls=calls: [c() for c in calls]
            outs = lambda calls=calls: [(c.res_buf, c.x_buf) for c in calls]
            obs = {"implementation": declared_binding(), "tensors": calls[0].observation["tensors"]}
        elif name == "add":
            bufs = [torch.empty_like(x) for _ in range(base.CALLS)]
            body = lambda bufs=bufs: [torch.ops.aten.add.out(x, residual, out=b) for b in bufs]
            outs = lambda bufs=bufs: [(b, None) for b in bufs]
            obs = {"implementation": {"recipe": RECIPES["add"], "source_sha256": base.digest(Path(__file__))},
                   "tensors": {"x": tensor_observation(x), "residual": tensor_observation(residual),
                               "output": tensor_observation(bufs[0])}}
        elif name == "rmsnorm":
            holder = []
            body = lambda holder=holder: holder.__setitem__(slice(None), [
                torch.ops.aten.rms_norm(s_ref, [SHAPE[-1]], weight, EPS) for _ in range(base.CALLS)])
            outs = lambda holder=holder: [(None, h) for h in holder]
            obs = {"implementation": {"recipe": RECIPES["rmsnorm"], "source_sha256": base.digest(Path(__file__))},
                   "tensors": {"s": tensor_observation(s_ref), "weight": tensor_observation(weight)}}
        else:
            bufs, holder = [torch.empty_like(x) for _ in range(base.CALLS)], []
            def body(bufs=bufs, holder=holder):
                holder[:] = []
                for b in bufs:
                    torch.ops.aten.add.out(x, residual, out=b)
                    holder.append(torch.ops.aten.rms_norm(b, [SHAPE[-1]], weight, EPS))
            outs = lambda bufs=bufs, holder=holder: list(zip(bufs, holder))
            obs = {"implementation": {"recipe": RECIPES["chain"], "source_sha256": base.digest(Path(__file__))},
                   "tensors": {"x": tensor_observation(x), "residual": tensor_observation(residual)}}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            body()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=None):
            body()

        def invalidate(outs=outs, stream=stream):
            with torch.cuda.stream(stream):
                for a, b in outs():
                    for t in (a, b):
                        if t is not None:
                            t.fill_(float("nan"))

        def verify(outs=outs, kind=kind, name=name):
            pairs = outs()
            if not inputs_unchanged():
                raise AssertionError("read-only input changed")
            for a, b in pairs:
                if a is not None:
                    torch.testing.assert_close(a.float(), s32, **TOLERANCE["add"])
                if b is not None:
                    torch.testing.assert_close(b.float(), y32, **TOLERANCE["rmsnorm"])
            pointers = {t.data_ptr() for p in pairs for t in p if t is not None}
            if len(pointers) != sum(t is not None for p in pairs for t in p):
                raise AssertionError("calls share output storage")
            return {"outputs_checked": len(pairs), "references_per_output": 1,
                    "input_bytes_unchanged": True, "outputs_independent": True}

        invalidate()
        with torch.cuda.stream(stream):
            for _ in range(base.WARMUP):
                graph.replay()
        torch.cuda.synchronize(device)
        warmup[case_id] = verify()
        entries[case_id] = {"graph": graph, "stream": stream, "devices": (0,), "timing_device": device,
                            "invalidate": invalidate, "verify": verify}
        prepared[case_id] = obs
    return entries, prepared, warmup


def expected_checks(case):
    return {"outputs_checked": base.CALLS, "references_per_output": 1,
            "input_bytes_unchanged": True, "outputs_independent": True}


def identity(frozen, prepared, runtime):
    rt, dev = runtime["runtime"], runtime["devices"]["0"]
    if rt["accelerator"] != "HIP" or str(dev.get("architecture", "")).split(":")[0] != frozen["architecture"]:
        raise MeasurementContractError("HIP gfx936 GPU0 required")
    if any(rt.get(f) is not False for f in ("grad_enabled", "autocast_enabled", "deterministic_algorithms")):
        raise MeasurementContractError("no_grad BF16 runtime required")
    gpu_key = {"device": copy.deepcopy(dev), "runtime": {k: rt[k] for k in (
        "torch_distribution", "torch_git_revision", "torch_build_sha256", "torch_extension_sha256", "hip", "driver")}}
    contract_id = key_id(frozen["contract"])
    keys = {}
    for case_id, case in frozen["cases"].items():
        obs = prepared[case_id]
        for tensor in obs["tensors"].values():
            if tensor["dtype"] != "bf16" or tensor["layout"] != "contiguous" or tensor["layout_provenance"]["device"] != "cuda:0":
                raise MeasurementContractError("actual tensor outside contract: " + case_id)
        key = {"implementation": obs, "gpu": gpu_key, "case": case, "contract": frozen["contract"], "pool": frozen["pool"]}
        keys[case_id] = {"case_identity": key, "case_identity_id": key_id(key), "gpu_key_id": key_id(gpu_key),
                         "contract_id": contract_id}
    return {"scope": SCOPE, "cases": keys}


def run(output, frozen, context):
    save(output / "load_gate.json", wait_for_quiet_host(frozen["load_gate"]))
    import torch
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 required; no CPU/eager fallback")
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
                torch.cuda.synchronize(0)
                timing = base.timed(torch, entry)
                checks = entry["verify"]()
                row = {"case": case_id, "round": attempt, "started_monotonic_ns": started,
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks,
                       "host_load_1min": os.getloadavg()[0]}
                records.append(row)
                save(output / f"attempt_{len(records)-1:03d}.json", row)
                print(json.dumps(row), flush=True)
        save(output / "result.json", {"scope": SCOPE, "complete": True, "records": records, "automatic_admission": False})
        return evaluate(frozen, records)


def evaluate(frozen, records):
    original = base.expected_checks
    base.expected_checks = expected_checks  # one reference per output in this batch
    try:
        result = base.evaluate(frozen, records)
    finally:
        base.expected_checks = original
    result.update(scope=SCOPE, data_role="GPU0 Add+RMSNorm calibration candidates; chain is comparison only")
    med = lambda c: result["cases"][c]["statistics"]["median_us"] if result["cases"][c]["statistics"] else None
    result["analytic_error"] = {c: {"predicted_us": p["predicted_us"], "measured_us": med(c),
        "signed_error": (p["predicted_us"] - med(c)) / med(c) if med(c) else None}
        for c, p in frozen["frozen_analytic_predictions"].items()}
    f, a, n, ch = (med(c) for c in CASES)
    result["comparison"] = {"fused_us": f, "native_add_plus_rmsnorm_stages_us": a + n if a and n else None,
                            "native_chain_us": ch, "fused_minus_native_chain_us": f - ch if f and ch else None,
                            "fused_over_native_chain": f / ch if f and ch else None}
    return result


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen, current = base.read(directory / "plan.json"), plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan, frozen predictions or source identity changed")
    if base.read(directory / "load_gate.json")["passed"] is not True:
        raise MeasurementContractError("Load gate not passed")
    process, result = base.read(directory / "process.json"), base.read(directory / "result.json")
    if process["python_returncode"] not in (0, 1) or result["scope"] != SCOPE or result["complete"] is not True:
        raise MeasurementContractError("Incomplete process or result")
    if base.read(directory / "warmup.json") != {c: expected_checks(k) for c, k in frozen["cases"].items()}:
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
                                  "analytic_us": {c: round(p["predicted_us"], 3) for c, p in frozen["frozen_analytic_predictions"].items()}}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"status": evaluated["status"], "quality_failed": evaluated["quality_failed"],
                          "comparison": evaluated["comparison"]}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
                                       "context": context})
        raise


if __name__ == "__main__":
    sys.exit(main())
