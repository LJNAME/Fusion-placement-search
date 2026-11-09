"""Three source-pinned LightOP cases, one GPU, no automatic search admission."""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from calibration.lightop_profile_calibration import timed_replay
from cost_model.fusion_kernel_profile import key_id
from cost_model.search_evaluation import build_evaluation_graph
from integration.lightop.catalog import CATALOG, catalog, check_installed, resolve

ROOT = Path(__file__).resolve().parents[1]
CALLS, ROUNDS, WARMUP, EPS = 50, 7, 10, 1e-6
CASES = {
    "silu_mul_3b": {"capability": "op.silu_and_mul_opt", "model": "model_3b",
        "nodes": ["L000.mlp.silu", "L000.mlp.gated_mul"], "shape": [1, 256, 11008],
        "recipe": "split_inputs_cat_out_then_lightop.v1", "rtol": .01, "atol": .001,
        "timed_body": ["cat(gate, up, out=packed)", "op.silu_and_mul_opt(out, packed)"]},
    "add_rmsnorm_3b": {"capability": "fused_add_rms_norm", "model": "model_3b",
        "nodes": ["L000.residual_add_1", "L000.pre_mlp_norm"], "shape": [1, 256, 2048],
        "recipe": "copy_two_inputs_then_lightop_dual_output.v1", "rtol": .02, "atol": .02,
        "timed_body": ["x_buf.copy_(x)", "res_buf.copy_(residual)", "lightop.fused_add_rms_norm(x_buf,res_buf,w,eps)"]},
    "rmsnorm_rope_qwen3": {"capability": "rms_rotary_embedding_fuse", "model": "model",
        "nodes": ["L000.self_attention.q_norm", "L000.self_attention.q_rope",
                  "L000.self_attention.k_norm", "L000.self_attention.k_rope"],
        "shape": [1, 256, 16, 128], "key_shape": [1, 256, 8, 128],
        "recipe": "copy_qk_then_joint_rmsnorm_rope.v1", "rtol": .02, "atol": .02,
        "timed_body": ["q_buf.copy_(q)", "k_buf.copy_(k)", "lightop.rms_rotary_embedding_fuse(...)"]},
}
SOURCES = ("calibration/fusion_timing_expansion.py", "tools/run_fusion_timing_expansion.sh",
    "calibration/lightop_profile_calibration.py", "calibration/lightop_profile_smoke.py",
    "integration/lightop/catalog.py", "execution/reference_identity.py",
    "cost_model/search_evaluation.py", "simulator/model_graphs/dense_layer.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return {"mode": "graph_replay", "calls_per_replay": CALLS, "rounds": ROUNDS,
        "warmup_replays": WARMUP, "statistic": "median", "unit": "us/call",
        "boundary": "same-device same-stream Events around one replay",
        "included": "complete declared call, including input packing or protective copies",
        "excluded": ["allocation", "input generation", "numerical validation", "host submission"],
        "input_reuse": "shared read-only sources; 50 independent mutable buffers and outputs",
        "quality": {"max_iqr_over_median": .10, "attempts": 7, "extra_attempts": 0,
                    "sample_replacement": False, "gain_required": False},
        "scope": "isolated fixed-case amortized device work; not whole-plan time"}


def plan():
    data = catalog()
    inventory = {c["id"]: c for c in data["capabilities"]}
    cases = copy.deepcopy(CASES)
    reviews = [ROOT / "data" / n / "capability_status.json" for n in
               ("lightop_vendor_suite01_review", "lightop_vendor_20261104_02_review")]
    vendor = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in reviews]
    for case in cases.values():
        graph = build_evaluation_graph(case["model"])
        shapes = {n: graph.nodes[n].extra["output_shape"] for n in case["nodes"]}
        for n, shape in shapes.items():
            expected = case.get("key_shape") if ".k_" in n else case["shape"]
            if list(shape) != expected or graph.nodes[n].extra["dtype"] != "bf16":
                raise ValueError(f"graph shape/dtype changed: {n}")
        evidence = [p for p, v in vendor if v["capabilities"][case["capability"]]["gpu_timing_and_checks_available"]]
        if not evidence:
            raise ValueError("selected vendor capability has no previously passed test")
        case["prior_vendor_evidence"] = {p.relative_to(ROOT).as_posix(): digest(p) for p in evidence}
        record = inventory[case["capability"]]
        case.update(dtype="bf16", device="cuda:0", graph_shapes=shapes,
            implementation={"id": record["binding"], "version": data["version"],
                "native_sha256": data["op_so_sha256"], "wrapper_source": record["source"],
                "wrapper_sha256": record["source_sha256"]},
            source_of_shape="current static graph, not model runtime observation",
            prior_evidence_scope="vendor family evidence at its own shapes/dtypes; new target still unverified",
            graph_adapter_verified_for_this_case=False, automatic_search_registration=False)
    cases["rmsnorm_rope_qwen3"]["rotary"] = {"theta": 1e6, "rotary_dim": 128, "is_neox": True,
        "source": "explicit full-head NeoX experiment declaration; graph does not encode rotary convention",
        "graph_semantic_equivalence_verified": False, "residual_q": None, "residual_k": None}
    cases["silu_mul_3b"]["owned_buffer_bytes"] = CALLS * math.prod(CASES["silu_mul_3b"]["shape"]) * 2 * 3
    cases["add_rmsnorm_3b"]["owned_buffer_bytes"] = CALLS * math.prod(CASES["add_rmsnorm_3b"]["shape"]) * 2 * 2
    cases["rmsnorm_rope_qwen3"]["owned_buffer_bytes"] = CALLS * (math.prod(CASES["rmsnorm_rope_qwen3"]["shape"]) + math.prod(CASES["rmsnorm_rope_qwen3"]["key_shape"])) * 2
    return {"scope": "three_lightop_complete_calls_expansion_v1", "cases": cases,
        "contract": contract(), "pool": POOL, "seed": 20261109, "eps": EPS,
        "source_sha256": {p: digest(ROOT / p) for p in SOURCES},
        "catalog_sha256": digest(CATALOG), "automatic_admission": False,
        "old_native_and_kerv_remeasurement": False,
        "source_binding": "source_sha256 binds actual working-tree code; git revision is provenance only",
        "collector_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "gpu_executed": False, "physical_kernel_count_certified": False}


def rotate(torch, value, cache, neox=True):
    half = value.shape[-1] // 2
    cos = cache[:, :half].float()[None, :, None, :]
    sin = cache[:, half:].float()[None, :, None, :]
    if neox:
        a, b = value.float()[..., :half], value.float()[..., half:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
    a, b = value.float()[..., 0::2], value.float()[..., 1::2]
    return torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2)


def references(torch, case_id, inputs, eps=EPS):
    if case_id == "silu_mul_3b":
        gate, up = inputs["gate"], inputs["up"]
        return [(torch.nn.functional.silu(gate) * up,),
                (torch.nn.functional.silu(gate.float()) * up.float(),)]
    if case_id == "add_rmsnorm_3b":
        x, res, w = (inputs[n] for n in ("x", "residual", "weight"))
        sums = (x + res, x.float() + res.float())
        return [(s, s.float() * torch.rsqrt(s.float().square().mean(-1, keepdim=True) + eps) * w.float()) for s in sums]
    results = []
    for round_norm in (True, False):
        outputs = []
        for name in ("q", "k"):
            v, w = inputs[name].float(), inputs["weight_" + name].float()
            norm = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * w
            if round_norm:
                norm = norm.to(inputs[name].dtype)
            outputs.append(rotate(torch, norm, inputs["cache"], True))
        results.append(tuple(outputs))
    return results


class PreparedCall:
    """One private call; sources are read-only even for in-place vendor ops."""
    def __init__(self, torch, case_id, inputs, operator):
        self.torch, self.case_id, self.inputs, self.operator = torch, case_id, inputs, operator
        if case_id == "silu_mul_3b":
            shape = inputs["gate"].shape
            self.packed = torch.empty((*shape[:-1], shape[-1] * 2), dtype=inputs["gate"].dtype, device=inputs["gate"].device)
            self.buffers = (self.packed, torch.empty_like(inputs["gate"]))
            self.outputs = self.buffers[1:]
        elif case_id == "add_rmsnorm_3b":
            self.buffers = tuple(torch.empty_like(inputs[n]) for n in ("x", "residual"))
            self.outputs = self.buffers[::-1]  # Add, Norm
        else:
            self.buffers = tuple(torch.empty_like(inputs[n]) for n in ("q", "k"))
            self.outputs = self.buffers

    def __call__(self):
        t, x = self.torch, self.inputs
        if self.case_id == "silu_mul_3b":
            t.cat((x["gate"], x["up"]), dim=-1, out=self.packed)
            self.operator(self.outputs[0].view(-1, x["gate"].shape[-1]), self.packed.view(-1, self.packed.shape[-1]))
        elif self.case_id == "add_rmsnorm_3b":
            a, b = self.buffers
            a.copy_(x["x"]); b.copy_(x["residual"])
            self.operator(a.view(-1, a.shape[-1]), b.view(-1, b.shape[-1]), x["weight"], EPS)
        else:
            q, k = self.buffers
            q.copy_(x["q"]); k.copy_(x["k"])
            self.operator(x["positions"], q, k, q.shape[-1], x["cache"], True,
                          x["weight_q"], x["weight_k"], None, None, EPS)
        return self.outputs

    def invalidate(self):
        for tensor in self.buffers:
            tensor.fill_(float("nan"))


def build_inputs(torch, case_id, device, seed):
    spec = CASES[case_id]
    gen = torch.Generator(device=device).manual_seed(seed)
    def rand(shape):
        return torch.randn(tuple(shape), device=device, dtype=torch.bfloat16, generator=gen)
    if case_id == "silu_mul_3b":
        return {"gate": rand(spec["shape"]), "up": rand(spec["shape"])}
    if case_id == "add_rmsnorm_3b":
        return {"x": rand(spec["shape"]), "residual": rand(spec["shape"]),
                "weight": (1 + .1 * rand((spec["shape"][-1],))).to(torch.bfloat16)}
    seq, head = spec["shape"][1], spec["shape"][-1]
    pos = torch.arange(seq, device=device, dtype=torch.int64)
    freq = 1. / (1e6 ** (torch.arange(0, head, 2, device=device, dtype=torch.float32) / head))
    angles = pos.float()[:, None] * freq[None, :]
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(torch.bfloat16).contiguous()
    return {"q": rand(spec["shape"]), "k": rand(spec["key_shape"]), "positions": pos[None, :].contiguous(),
            "weight_q": (1 + .1 * rand((head,))).to(torch.bfloat16),
            "weight_k": (1 + .1 * rand((head,))).to(torch.bfloat16), "cache": cache}


def verify_outputs(torch, case, calls, inputs, snapshots, refs):
    for name, tensor in inputs.items():
        if not torch.equal(tensor.view(torch.uint8), snapshots[name]):
            raise AssertionError("read-only source changed: " + name)
    pointers = set()
    for call in calls:
        for output, expected in zip(call.outputs, refs[0]):
            if output.shape != expected.shape:
                raise AssertionError("output shape mismatch")
            if not torch.isfinite(output).all():
                raise AssertionError("nonfinite or unwritten output")
        for ref in refs:
            for output, expected in zip(call.outputs, ref):
                torch.testing.assert_close(output.float(), expected.float(), rtol=case["rtol"], atol=case["atol"])
        for buf in call.buffers:
            if any(torch._C._overlaps(buf, source) for source in inputs.values()):
                raise AssertionError("private buffer aliases shared input")
            if buf.data_ptr() in pointers:
                raise AssertionError("calls share mutable buffers")
            pointers.add(buf.data_ptr())
    return {"calls_checked": len(calls), "outputs_per_call": len(calls[0].outputs),
            "references_per_output": len(refs), "input_bytes_unchanged": True,
            "private_buffers_independent": True, "all_outputs_finite": True}


def summarize(rows, outputs_per_call=None):
    if len(rows) != ROUNDS or [r["round"] for r in rows] != list(range(ROUNDS)):
        raise ValueError("exactly seven chronological rounds required")
    values, errors, streams = [], [], set()
    for row in rows:
        timing, checks = row["timing"], row["checks"]
        if checks["calls_checked"] != CALLS or checks["references_per_output"] != 2 or not all(
                checks[k] for k in ("input_bytes_unchanged", "private_buffers_independent", "all_outputs_finite")):
            raise ValueError("incomplete correctness checks")
        if outputs_per_call is not None and checks["outputs_per_call"] != outputs_per_call:
            raise ValueError("missing group output")
        b = timing["event_boundary"]
        if any(b[k] != "cuda:0" for k in ("device", "start_device", "end_device")) or b["calls_per_replay"] != CALLS:
            raise ValueError("incorrect Event boundary")
        streams.add(b["stream"])
        value = timing["elapsed_ms_per_50_calls"]
        if timing["error"] is not None or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            errors.append("invalid_timing_round_" + str(row["round"]))
        else:
            values.append(value * 1000 / CALLS)
    if len(streams) != 1:
        raise ValueError("stream changed")
    stats = None
    if not errors:
        med = statistics.median(values)
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        stats = {"median_us": med, "min_us": min(values), "max_us": max(values),
                 "iqr_over_median": (q3 - q1) / med, "calls": CALLS * ROUNDS, "rounds": ROUNDS}
        if stats["iqr_over_median"] > .1:
            errors.append("iqr_over_median_exceeds_10_percent")
    return {"status": "quality_failed" if errors else "quality_passed", "statistics": stats,
            "samples_us": values, "reasons": errors}


def observe_runtime(torch, case_id, operator, device):
    from integration.flagscale.kerv_profile_runtime import driver_version
    expected, dist = catalog(), importlib.metadata.distribution("lightop")
    module = importlib.import_module("lightop")
    native = Path(module.op.__file__).resolve()
    if native != Path(dist.locate_file(expected["op_so_path"])).resolve() or digest(native) != expected["op_so_sha256"]:
        raise ValueError("loaded native extension differs from pinned package")
    if case_id == "silu_mul_3b":
        if operator is not module.op.silu_and_mul_opt:
            raise ValueError("wrong native callable")
    else:
        source = Path(inspect.getsourcefile(operator)).resolve()
        record = next(c for c in expected["capabilities"] if c["id"] == CASES[case_id]["capability"])
        if source != Path(dist.locate_file(record["source"])).resolve() or digest(source) != record["source_sha256"]:
            raise ValueError("loaded wrapper differs from pinned source")
    props = torch.cuda.get_device_properties(device)
    arch = str(getattr(props, "gcnArchName", ""))
    if not arch.startswith("gfx936"):
        raise ValueError("this expansion targets BW1000 gfx936")
    return {"torch": str(torch.__version__), "torch_git_revision": torch.version.git_version,
        "torch_extension_sha256": digest(Path(torch._C.__file__)),
        "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "hip": torch.version.hip, "driver": driver_version(),
        "device": {"name": props.name, "architecture": arch, "total_memory": props.total_memory,
                   "logical_device": str(device)},
        "lightop_version": dist.version, "native_sha256": digest(native),
        "uuid_provenance_only": str(getattr(props, "uuid", "unavailable")),
        "native_kernel_count_certified": False, "dispatch_trace_collected": False}


def observe_tensor(tensor):
    from execution.reference_identity import tensor_observation
    if str(tensor.dtype) == "torch.int64":
        return {"shape": list(tensor.shape), "dtype": "int64", "strides": list(tensor.stride()),
                "layout": "contiguous" if tensor.is_contiguous() else "strided",
                "layout_provenance": {"kind": "runtime_tensor", "device": str(tensor.device),
                                      "observer": "Tensor.shape/dtype/stride/is_contiguous"}}
    return tensor_observation(tensor)


def run_case(case_id, output, frozen):
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU required; no CPU/eager/native fallback")
    if any(os.environ.get(k) for k in ("ROCP_TOOL_LIB", "ROCPROFILER_TOOL_LIBRARIES", "HSA_TOOLS_LIB")) or any(
            s in os.environ.get("LD_PRELOAD", "").lower() for s in ("prof", "trace")):
        raise RuntimeError("disable profiling instrumentation during timing")
    check_installed()
    output.mkdir(parents=True, exist_ok=True)
    context = {"stage": "runtime", "case": case_id, "pool": POOL}
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    operator = resolve(CASES[case_id]["capability"])
    runtime = observe_runtime(torch, case_id, operator, device)
    save(output / "runtime.json", runtime)
    stream = torch.cuda.Stream(device=device)
    try:
        with torch.no_grad(), torch.cuda.stream(stream):
            inputs = build_inputs(torch, case_id, device, frozen["seed"])
            snapshots = {n: t.view(torch.uint8).clone() for n, t in inputs.items()}
            refs = references(torch, case_id, inputs)
            calls = [PreparedCall(torch, case_id, inputs, operator) for _ in range(CALLS)]
            identity = {"format": "fusion_timing_expansion.local_call.v1", "case": frozen["cases"][case_id],
                "runtime": {k: v for k, v in runtime.items() if k != "uuid_provenance_only"},
                "contract": frozen["contract"], "collector_sources": frozen["source_sha256"],
                "inputs": {n: observe_tensor(t) for n, t in inputs.items()},
                "outputs": [observe_tensor(t) for t in calls[0].outputs]}
            # Tensor observations include local storage addresses only in provenance, if present.
            save(output / "identity.json", {"execution_key": identity, "execution_key_id": key_id(identity),
                 "not_an_accepted_search_profile": True})
            def body():
                for call in calls:
                    call()
            def invalidate():
                for call in calls:
                    call.invalidate()
            def verify():
                return verify_outputs(torch, frozen["cases"][case_id], calls, inputs, snapshots, refs)
            context.update(stage="eager_preparation", memory=memory_snapshot(torch, device))
            save(output / "progress.json", context)
            body(); torch.cuda.synchronize(device); verify()
            graph = torch.cuda.CUDAGraph()
            context.update(stage="capture", memory=memory_snapshot(torch, device))
            save(output / "progress.json", context)
            with torch.cuda.graph(graph, stream=stream, pool=None):
                body()
            context["stage"] = "capture_smoke"
            save(output / "progress.json", context)
            smoke = []
            for i in range(3):
                invalidate(); torch.cuda.synchronize(device)
                timing = timed_replay(torch, graph, stream, device)
                checks = verify()
                smoke.append({"replay": i, "checks": checks, "diagnostic_timing": {**timing, "used_for_pass_fail": False}})
            save(output / "smoke.json", {"passed": True, "records": smoke})
            for _ in range(WARMUP):
                graph.replay()
            torch.cuda.synchronize(device); verify()
            rows = []
            for i in range(ROUNDS):
                context.update(stage="formal", round=i, memory=memory_snapshot(torch, device))
                save(output / "progress.json", context)
                invalidate(); torch.cuda.synchronize(device)
                timing = timed_replay(torch, graph, stream, device)
                row = {"round": i, "timing": timing, "checks": verify()}
                save(output / f"round_{i}.json", row); rows.append(row)
            quality = summarize(rows, len(calls[0].outputs))
            result = {"case": case_id, "gpu_executed": True, "quality": quality,
                "execution_key_id": key_id(identity), "identity": identity, "contract": frozen["contract"],
                "automatic_admission": False, "search_rule_changed": False,
                "physical_single_kernel_verified": False, "status": quality["status"]}
            save(output / "result.json", result)
            return result
    except BaseException:
        context.update(memory=memory_snapshot(torch, device))
        save(output / "failure.json", {"context": context, "traceback": traceback.format_exc(), "status": "runtime_failed"})
        raise


def run(output, frozen):
    results = {}
    for case_id in frozen["cases"]:
        folder = output / case_id; folder.mkdir()
        with (folder / "process.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-X", "faulthandler", "-u", "-m", "calibration.fusion_timing_expansion",
                "_case", "--output", str(folder), "--case", case_id, "--plan", str(output / "plan.json")],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        path = folder / "result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        status = result.get("status", "runtime_failed") if proc.returncode == 0 else "runtime_failed"
        results[case_id] = {"returncode": proc.returncode, "status": status,
                           "statistics": result.get("quality", {}).get("statistics")}
        save(output / "summary.json", {"cases": results, "automatic_admission": False, "gpu_requested": True})
        print(json.dumps({"case": case_id, **results[case_id]}), flush=True)
    return all(r["status"] == "quality_passed" for r in results.values())


def review(directory, output):
    frozen = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    parent = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for path, expected in frozen["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("collector source changed: " + path)
    results = {}
    for case_id in CASES:
        item = parent["cases"].get(case_id, {})
        folder = directory / case_id
        if item.get("returncode") != 0 or not (folder / "result.json").exists():
            results[case_id] = {"status": "runtime_failed", "accepted": False}
            continue
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        rows = [json.loads((folder / f"round_{i}.json").read_text(encoding="utf-8")) for i in range(ROUNDS)]
        quality = summarize(rows, 1 if case_id == "silu_mul_3b" else 2)
        if result["identity"]["case"] != frozen["cases"][case_id] or key_id(result["identity"]) != result["execution_key_id"]:
            raise ValueError("identity mismatch")
        if result["contract"] != frozen["contract"] or quality != result["quality"] or not result["gpu_executed"]:
            raise ValueError("result differs from raw rounds")
        if result["status"] != parent["cases"][case_id]["status"] or result["status"] != quality["status"]:
            raise ValueError("parent status disagrees with case result")
        smoke = json.loads((folder / "smoke.json").read_text(encoding="utf-8"))
        if not smoke["passed"] or len(smoke["records"]) != 3:
            raise ValueError("capture smoke evidence missing")
        runtime = result["identity"]["runtime"]
        implementation = frozen["cases"][case_id]["implementation"]
        if (runtime["native_sha256"] != implementation["native_sha256"]
                or runtime["lightop_version"] != implementation["version"]
                or not runtime["hip"] or not runtime["device"]["architecture"].startswith("gfx936")):
            raise ValueError("provider or runtime differs from frozen identity")
        results[case_id] = {"status": quality["status"], "statistics": quality["statistics"],
                           "execution_key_id": result["execution_key_id"], "accepted": False}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "review.json", {"cases": results, "automatic_admission": False,
        "search_rule_changed": False, "scope": "local complete-call timing, not deployment or global cost coverage"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "review", "_case"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.action == "_case":
        frozen = json.loads(args.plan.read_text(encoding="utf-8"))
        run_case(args.case, args.output, frozen)
        return
    if args.action == "review":
        print(json.dumps(review(args.input, args.output), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = plan(); save(args.output / "plan.json", frozen)
    if args.action == "plan":
        print(json.dumps({"cases": list(frozen["cases"]), "gpu_executed": False}))
    elif not run(args.output.resolve(), frozen):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
