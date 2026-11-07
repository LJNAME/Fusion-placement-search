"""One source-pinned Add+RMSNorm correctness probe; no timing or admission."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("flagscale/models/kerv/ops/KERVRuntimeOptimization/adaptive_rms_norm.py")
SOURCE_SHA256 = "f9b3bafdd1a1c303627cf2dbe2561879155088f763c40c341cc997beed107f88"
SHAPE = (1, 256, 2048)
EPS = 1e-6
TOLERANCE = {"rtol": .02, "atol": .02}
KERNEL = "_add_rms_norm_inference_kernel"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_path(flagscale_root):
    path = Path(flagscale_root).resolve() / SOURCE
    if sha(path) != SOURCE_SHA256:
        raise ValueError("adaptive RMSNorm source differs from the audited snapshot")
    return path


def graph_contract(graph):
    """Inspect actual group liveness; unknown physical layouts remain unknown."""
    exported = set(graph.extra.get("output_nodes", []))
    if graph.extra.get("output_node"):
        exported.add(graph.extra["output_node"])
    rows = []
    for layer in range(36):
        prefix = f"L{layer:03d}."
        x, residual = prefix + "self_attention.out_proj", prefix + "input"
        add, norm = prefix + "residual_add_1", prefix + "pre_mlp_norm"
        nodes = [graph.nodes[n] for n in (x, residual, add, norm)]
        outside = sorted(set(graph.successors(x)) - {add})
        valid = (not outside and x not in exported
                 and set(graph.predecessors(add)) == {x, residual}
                 and graph.predecessors(norm) == [add]
                 and [n.op_type for n in nodes[-2:]] == ["add", "rmsnorm"]
                 and all(tuple(n.extra.get("output_shape", ())) == SHAPE
                         and n.extra.get("dtype") == "bf16" for n in nodes)
                 and nodes[-1].param_bytes == 2048 * 2)
        rows.append({"group": [add, norm], "x": x, "residual": residual,
                     "x_outside_consumers": outside, "x_is_exported": x in exported,
                     "add_outside_consumers": sorted(set(graph.successors(add)) - {norm}),
                     "layouts": {n: graph.nodes[n].extra.get("layout", "unknown") for n in (x, residual)},
                     "graph_overwrite_precondition": valid})
    return {"groups": rows, "eligible_graph_positions": sum(r["graph_overwrite_precondition"] for r in rows),
            "runtime_aliases_verified": False, "runtime_producer_layouts_verified": False}


def references(x, residual, weight):
    """Return the two BF16-boundary semantics, not an ideal unrounded sum."""
    import torch
    import torch.nn.functional as F
    summed = x + residual
    normalized_fp32 = summed.float() * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + EPS)
    handwritten = normalized_fp32.to(x.dtype) * weight
    aten = F.rms_norm(summed, (SHAPE[-1],), weight, eps=EPS)
    return summed, handwritten, aten


def synthetic_inputs(device="cpu"):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(20261107)
    x, residual = [torch.randn(SHAPE, dtype=torch.bfloat16, generator=generator).to(device) for _ in range(2)]
    weight = (1 + .1 * torch.randn(SHAPE[-1], generator=generator)).to(dtype=torch.bfloat16, device=device)
    return x, residual, weight


def plan(flagscale_root):
    from cost_model.search_evaluation import qwen25_3b_spec
    from simulator.model_graphs import build_model_graph
    from simulator.placement_artifacts import graph_fingerprint
    path = source_path(flagscale_root)
    graph = build_model_graph(qwen25_3b_spec())
    contract = graph_contract(graph)
    x, residual, weight = synthetic_inputs()
    _, handwritten, aten = references(x, residual, weight)
    return {"scope": "kerv_add_rmsnorm_2048_correctness_only_not_profile_admission",
            "source": {"relative_path": SOURCE.as_posix(), "sha256": sha(path),
                       "revision": subprocess.check_output(["git", "-C", str(flagscale_root), "rev-parse", "HEAD"], text=True).strip(),
                       "entrypoint": "adaptive_rms_norm.add_rms_norm_inference", "kernel": KERNEL},
            "collector_sha256": sha(__file__), "reference_executor_sha256": sha(ROOT / "execution/reference.py"),
            "graph_fingerprint": graph_fingerprint(graph), "graph_contract": contract,
            "case": {"shape": list(SHAPE), "dtype": "bf16", "weight_shape": [2048],
                     "required_layout": "contiguous", "layout_source": "required, not observed in the model graph",
                     "eps": EPS, "eps_source": "explicit declaration matching execution/reference.py:_rmsnorm; not graph metadata",
                     "seed": 20261107, "input_origin": "synthetic values shaped from the existing graph"},
            "recipe": {"mode": "eager_correctness_only", "repetitions": 3,
                       "restore": "exact x.copy_(original) before each invocation",
                       "outputs": ["normalized (new allocation)", "sum (aliases consumed x)"],
                       "mutation": "x is disposable; original residual and weight are read-only",
                       "allocation": "vendor wrapper empty_like; no allocation-free claim",
                       "timing": "none", "graph_capture": "not tested"},
            "cpu_semantics_diagnostic": {"device": "cpu", "not_GPU_equivalence_evidence": True,
                "different_elements_handwritten_vs_aten": int((handwritten != aten).sum()),
                "max_abs_difference": float((handwritten.float() - aten.float()).abs().max())},
            "tolerance": TOLERANCE, "automatic_admission": False, "search_rules_changed": False,
            "gpu_executed": False}


def load_module(flagscale_root):
    spec = importlib.util.spec_from_file_location("_fusion_audited_adaptive_rmsnorm", source_path(flagscale_root))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module._TRITON_AVAILABLE:
        raise RuntimeError("Triton unavailable; no native fallback allowed")
    return module


def dispatch_once(module, x, residual, weight):
    """Observe the selected JIT entry independently of device profiler evidence."""
    original = getattr(module, KERNEL)
    launches = []

    class Probe:
        def __getitem__(self, grid):
            launch = original[grid]

            def invoke(*args, **kwargs):
                launches.append({"kernel": KERNEL, "grid": list(grid), "num_warps": kwargs["num_warps"]})
                return launch(*args, **kwargs)
            return invoke

    setattr(module, KERNEL, Probe())
    try:
        result = module.add_rms_norm_inference(x, residual, (2048,), weight, EPS)
    finally:
        setattr(module, KERNEL, original)
    if len(launches) != 1:
        raise RuntimeError("expected exactly one AddRMS JIT dispatch")
    return result, launches


def verify_trace(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if len(kernels) != 1 or KERNEL not in kernels[0].get("name", ""):
        raise RuntimeError("device trace must contain exactly one target AddRMS kernel")
    return {"kernel_count": 1, "kernel_name": kernels[0]["name"]}


def bitwise_equal(a, b):
    import torch
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def run(flagscale_root, output, device, frozen):
    import torch
    from execution.reference_identity import tensor_observation
    target = torch.device(device)
    if target.type != "cuda" or target.index is None or not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("an explicit HIP GPU is required; no CPU fallback")
    if frozen["graph_contract"]["eligible_graph_positions"] != 36:
        raise ValueError("graph overwrite preconditions failed")
    torch.cuda.set_device(target)
    module = load_module(flagscale_root)
    original, residual, weight = synthetic_inputs(target)
    x, residual_before, weight_before = original.clone(), residual.clone(), weight.clone()
    tensors = {"x": x, "residual": residual, "weight": weight}
    if len({t.untyped_storage().data_ptr() for t in (original, *tensors.values())}) != 4:
        raise ValueError("synthetic input buffers must not alias")
    observations = {k: tensor_observation(v) for k, v in tensors.items()}
    save(output / "runtime_context.json", {"device": str(target), "torch": str(torch.__version__),
         "hip": torch.version.hip, "inputs": observations, "execution_mode": "eager",
         "pool": "default Torch allocator; no graph/private pool",
         "memory_allocated": torch.cuda.memory_allocated(target),
         "memory_reserved": torch.cuda.memory_reserved(target)})
    summed, handwritten, aten = references(original, residual, weight)
    rows = []
    with torch.no_grad():
        for replay in range(3):
            x.copy_(original)
            restored = bitwise_equal(x, original)
            (normalized, result_sum), dispatch = dispatch_once(module, x, residual, weight)
            torch.cuda.synchronize(target)
            checks = {"x_restored_exact": restored, "sum_aliases_x": result_sum.data_ptr() == x.data_ptr(),
                      "normalized_has_own_storage": normalized.untyped_storage().data_ptr() not in
                          {t.untyped_storage().data_ptr() for t in (original, x, residual, weight)},
                      "residual_unchanged": bitwise_equal(residual, residual_before),
                      "weight_unchanged": bitwise_equal(weight, weight_before),
                      "sum_exact_vs_bf16_add": bitwise_equal(result_sum, summed),
                      "finite": bool(torch.isfinite(normalized).all())}
            numerical = {}
            for name, reference in (("handwritten", handwritten), ("project_aten", aten)):
                numerical[name] = {"max_abs_error": float((normalized.float() - reference.float()).abs().max()),
                                   "close": bool(torch.allclose(normalized, reference, **TOLERANCE))}
            rows.append({"invocation": replay, "checks": checks, "numerics": numerical, "jit_dispatch": dispatch})
            if not all(checks.values()) or not all(r["close"] for r in numerical.values()):
                save(output / "checks.json", rows)
                raise RuntimeError("correctness/in-place contract failed; see checks.json")
        save(output / "checks.json", rows)
        # Profiling is isolated from all restoration/reference operations. There is no timed phase.
        x.copy_(original)
        torch.cuda.synchronize(target)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            outputs, dispatch = dispatch_once(module, x, residual, weight)
            torch.cuda.synchronize(target)
        profiler.export_chrome_trace(str(output / "device_trace.json"))
        trace = verify_trace(output / "device_trace.json")
    props = torch.cuda.get_device_properties(target)
    return {"scope": frozen["scope"], "pipeline_passed": True, "gpu_executed": True,
            "source": frozen["source"], "collector_sha256": frozen["collector_sha256"],
            "runtime": {"torch": str(torch.__version__), "hip": torch.version.hip,
                        "triton": str(module.triton.__version__), "device": str(target), "name": props.name,
                        "architecture": str(getattr(props, "gcnArchName", "unknown")),
                        "uuid_diagnostic_only": str(getattr(props, "uuid", "unavailable"))},
            "input_observations": observations, "checks": rows, "device_trace": trace,
            "automatic_admission": False, "profile_created": False, "search_rules_changed": False,
            "actual_model_producer_layout_verified": False, "graph_capture_verified": False,
            "latency_measured": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--flagscale-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    fault_log = (args.output / "fatal_traceback.log").open("w", encoding="utf-8")
    faulthandler.enable(file=fault_log, all_threads=True)
    try:
        frozen = plan(args.flagscale_root)
        save(args.output / "plan.json", frozen)
        if args.action == "run":
            result = run(args.flagscale_root, args.output, args.device, frozen)
            save(args.output / "result.json", result)
        print(json.dumps({"action": args.action, "graph_positions": frozen["graph_contract"]["eligible_graph_positions"],
                          "gpu_executed": args.action == "run", "profile_created": False}), flush=True)
    except Exception as error:
        save(args.output / "failure.json", {"pipeline_passed": False, "error": repr(error),
             "traceback": traceback.format_exc(), "profile_created": False, "automatic_admission": False})
        raise
    finally:
        faulthandler.disable()
        fault_log.close()


if __name__ == "__main__":
    main()
