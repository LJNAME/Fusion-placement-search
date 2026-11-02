"""Two-device, paired Native/Fused graph measurements through the placed adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics

from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from calibration.probe_kerv_graph import restore
from cost_model.measurement_labels import measurement_contract_id
from integration.flagscale.kerv_execution import KervSiluMulAdapter
from integration.flagscale.kerv_profile_runtime import loaded_source, native_binding, observe_environment
from simulator.placement_artifacts import load_placed_op_graph, save_placed_op_graph


ROOT = Path(__file__).resolve().parents[1]
RECIPE = {"rounds": 7, "calls_per_round": 50, "warmup_replays": 10, "seed": 1234,
          "device_order": [0, 1], "arm_order": "fused/native on even rounds; native/fused on odd rounds",
          "preference": "all seven paired signs agree and median relative saving magnitude >= 1%; otherwise inconclusive",
          "hardware_profile_sharing": False, "equivalence_claim": False,
          "automatic_admission": False, "metric": "graph_batch_amortized_seconds_per_call"}


def save(path, data):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sources():
    names = ("calibration/measure_kerv_profile_pair.py", "calibration/validate_kerv_adapter.py",
             "calibration/probe_kerv_silu_mul.py", "calibration/probe_kerv_graph.py",
             "calibration/kerv_silu_mul.py", "integration/flagscale/kerv_execution.py",
             "integration/flagscale/kerv_profile_runtime.py", "simulator/fusion_rules.py",
             "simulator/placement_artifacts.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def capture(torch, function, gates, original, up):
    views = list(gates.unbind(0))
    current = torch.cuda.current_stream(gates.device)
    stream = torch.cuda.Stream(device=gates.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for gate in views:
            function(gate, up)
    current.wait_stream(stream)
    restore(torch, gates, original)
    stream.wait_stream(current)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=None, stream=stream):
        # Retain every output in both arms, not only the last Native result.
        outputs = [function(gate, up) for gate in views]
    current.wait_stream(stream)
    torch.cuda.synchronize(gates.device)
    return graph, outputs


def inspect_trace(path, arm):
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    operators = [e["name"] for e in events if e.get("cat") == "cpu_op"]
    expected = 1 if arm == "fused" else 2
    if len(kernels) != expected:
        raise ValueError(f"{arm} trace has {len(kernels)} device kernels, expected {expected}")
    if arm == "fused":
        if kernels[0]["name"] != "_silu_mul_kernel" or operators.count("flagos_embodied::kerv_silu_mul") != 1:
            raise ValueError("fused trace does not prove the public KERV path")
    elif operators.count("aten::silu") != 1 or operators.count("aten::mul") != 1 or any("flagos" in n for n in operators):
        raise ValueError("Native trace does not prove separate SiLU and Mul")
    launches = [e for e in events if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name", "")]
    for kernel in kernels:
        matches = [e for e in launches if e.get("args", {}).get("correlation") == kernel["args"].get("correlation")]
        if len(matches) != 1:
            raise ValueError("device Kernel lacks a unique correlated HIP launch")
    return {"device_kernel_count": len(kernels), "device_kernel_names": [k["name"] for k in kernels],
            "observed_jit_launches": 1 if arm == "fused" else 0,
            "synchronized_successfully": True, "hip_driver_version": trace.get("hip_driver_version")}


def device_case(torch, F, adapter, placed_source, index, physical, output):
    device = torch.device(f"cuda:{physical}")
    torch.cuda.set_device(device)
    placed = copy.deepcopy(placed_source)
    if physical >= placed.hardware.n_devs:
        raise ValueError("hardware snapshot lacks the selected logical device")
    placed.plan.groups[index].device = physical
    bound = adapter.bind(placed, index, {physical: device})
    save_placed_op_graph(output / "placed.json", placed)
    source = loaded_source(adapter.module, adapter.flagscale_root)
    if source["module_sha256"] != adapter.manifest["source_sha256"][
            "flagscale/models/kerv/ops/KERVRuntimeOptimization/embodied_ops/kerv_ops.py"]:
        raise ValueError("loaded module bytes differ from the inspected capability")
    observed, environment = observe_environment(adapter.module, device)
    meta = placed.graph.nodes[bound.gate_node].extra
    if list(meta["output_shape"]) != [1, 256, 3072] or meta["dtype"] != "bf16":
        raise ValueError("first paired profile experiment is fixed to BF16 [1,256,3072]")
    generator = torch.Generator(device="cpu").manual_seed(RECIPE["seed"])
    original = torch.randn(meta["output_shape"], dtype=torch.bfloat16, generator=generator).to(device)
    up = torch.randn(meta["output_shape"], dtype=torch.bfloat16, generator=generator).to(device)
    up_original = up.clone()
    native = F.silu(original) * up
    fp32 = (F.silu(original.float()) * up.float()).to(original.dtype)
    numeric = adapter.manifest["numeric_contract"]
    rtol, atol = numeric["bf16_fp16_rtol"], numeric["bf16_fp16_atol"]

    def fused(gate, other, observe=False):
        live = {bound.gate_node: gate, bound.up_node: other}
        call = bound(live, observe_dispatch=observe)
        if bound.gate_node in live or live[call.output_node] is not call.output:
            raise RuntimeError("gate invalidation or output publication failed")
        return call.output

    functions = {"fused": fused, "native": lambda gate, other: F.silu(gate) * other}
    traces, errors = {}, {}
    for arm, function in functions.items():
        arm_dir = output / arm
        arm_dir.mkdir()
        gate = original.clone()
        result = fused(gate, up, True) if arm == "fused" else function(gate, up)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result, native, rtol=rtol, atol=atol)
        torch.testing.assert_close(result, fp32, rtol=rtol, atol=atol)
        torch.testing.assert_close(up, up_original, rtol=0, atol=0)
        if arm == "native":
            torch.testing.assert_close(gate, original, rtol=0, atol=0)
            if result.data_ptr() == gate.data_ptr():
                raise RuntimeError("Native output unexpectedly aliases gate")
        elif result.data_ptr() != gate.data_ptr():
            raise RuntimeError("Fused output must alias gate")
        errors[arm] = {"max_abs_error_native": (result.float()-native.float()).abs().max().item(),
                       "max_abs_error_fp32": (result.float()-fp32.float()).abs().max().item()}
        gate.copy_(original)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            function(gate, up)
            torch.cuda.synchronize(device)
        path = arm_dir / "dispatch_trace.json"
        prof.export_chrome_trace(str(path))
        traces[arm] = inspect_trace(path, arm)
    # The driver value in the device trace is an independent runtime observation.
    traced = traces["fused"]["hip_driver_version"]
    if traced is not None:
        if environment["driver_version"] not in (None, str(traced)):
            raise ValueError("HIP driver observations disagree")
        environment["driver_version"] = str(traced)
    contract = {"executor": "placed_kerv_native_fused_graph_replay", "submission_included": True,
        "allocation_included": False, "granularity": "independent_gate_batch_graph_replay",
        "unit": "us/call", "environment": environment, "calls_per_round": 50, "rounds": 7,
        "timing_boundary": "one Event pair around one graph replay, divided by 50",
        "synchronization": "device synchronize after restore; end Event synchronize after replay",
        "warmup": {"replays_per_arm": 10, "compilation": "before capture and timing"},
        "statistic": "median of seven batch-amortized per-call durations",
        "working_set": "same gate pool and read-only up; restore original gates before every replay",
        "retention": "all 50 final outputs retained for both arms; Native outputs do not alias gate",
        "arm_order": RECIPE["arm_order"],
        "allocation_scope": "capture allocation excluded; graph device writes included",
        "restore_scope": "outside Event interval", "seed": 1234}
    gates = original.unsqueeze(0).repeat((50, 1, 1, 1))
    captures = {arm: capture(torch, fn, gates, original, up) for arm, fn in functions.items()}
    for _ in range(10):
        for replay, _outputs in captures.values():
            restore(torch, gates, original)
            replay.replay()
            torch.cuda.synchronize(device)
    samples = {arm: [] for arm in functions}
    for round_id in range(7):
        for arm_order_index, arm in enumerate(("fused", "native") if round_id % 2 == 0 else ("native", "fused")):
            replay, outputs = captures[arm]
            restore(torch, gates, original)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            replay.replay()
            end.record()
            end.synchronize()
            total = start.elapsed_time(end) / 1000
            if not math.isfinite(total) or total <= 0:
                raise ValueError("invalid Event timing")
            for tensor in outputs:
                torch.testing.assert_close(tensor, native, rtol=rtol, atol=atol)
                torch.testing.assert_close(tensor, fp32, rtol=rtol, atol=atol)
            torch.testing.assert_close(up, up_original, rtol=0, atol=0)
            if arm == "native":
                torch.testing.assert_close(gates, original.unsqueeze(0).expand_as(gates), rtol=0, atol=0)
            samples[arm].append({"round": round_id, "graph_calls": 50, "graph_event_total_s": total,
                                 "amortized_per_call_s": total/50, "arm_order_index": arm_order_index})
            save(output / arm / "samples.json", samples[arm])
    from cost_model.kerv_profile import digest
    for arm in functions:
        arm_source = source
        binding = adapter.manifest["binding"] if arm == "fused" else native_binding()
        if arm == "native":
            path = Path(inspect.getsourcefile(F.silu)).resolve()
            arm_source = {"module_path": str(path), "module_sha256": digest(path),
                          "git_revision": getattr(torch.version, "git_version", None)}
        report = {"format": "fusion-placement-search.kerv-profile-measurement", "version": "1.0",
            "arm": arm, "graph_hash": placed.graph_hash, "hardware_hash": placed.hardware_hash,
            "candidate_id": placed.candidate_id, "placed_artifact_sha256": digest(output / "placed.json"),
            "group_index": index, "node_ids": list(bound.node_ids), "gate_node": bound.gate_node,
            "up_node": bound.up_node, "logical_device": physical, "runtime_device": str(device),
            "observed_device": observed, "capability_id": adapter.manifest["capability_id"],
            "binding": binding, "loaded_source": arm_source, "source_sha256": sources(),
            "kernel_source_sha256": adapter.manifest["source_sha256"],
            "input": {"shape": list(original.shape), "dtype": "bf16", "layout": "contiguous",
                      "gate_strides": list(original.stride()), "up_strides": list(up.stride()), "seed": 1234,
                      "origin": "synthetic boundary tensors with selected reference graph shape"},
            "execution": traces[arm], "correctness": {"rtol": rtol, "atol": atol, **errors[arm],
                "gate_alias": arm == "fused", "gate_invalidated": arm == "fused",
                "gate_preserved": arm == "native", "up_preserved": True, "all_replay_outputs_checked": 350},
            "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract),
            "rows": samples[arm], "warmup": 10,
            "graph_batch_amortized_per_call_s": statistics.median(r["amortized_per_call_s"] for r in samples[arm]),
            "admission": "conditional_dev", "downstream_deployable": False}
        save(output / arm / "result.json", report)
    saving = [(n["amortized_per_call_s"]-f["amortized_per_call_s"])/n["amortized_per_call_s"]
              for n, f in zip(samples["native"], samples["fused"])]
    preference = "inconclusive"
    if all(x > 0 for x in saving) and statistics.median(saving) >= .01:
        preference = "fused_preferred"
    elif all(x < 0 for x in saving) and statistics.median(saving) <= -.01:
        preference = "native_preferred"
    summary = {"logical_device": physical, "paired_relative_savings": saving,
        "median_relative_saving": statistics.median(saving), "preference": preference,
        "median_us": {arm: statistics.median(r["amortized_per_call_s"] for r in samples[arm])*1e6 for arm in functions},
        "relative_saving_of_medians": (statistics.median(r["amortized_per_call_s"] for r in samples["native"])
            - statistics.median(r["amortized_per_call_s"] for r in samples["fused"]))
            / statistics.median(r["amortized_per_call_s"] for r in samples["native"]),
        "statistical_significance_claim": False, "hardware_profile_sharing": False}
    save(output / "comparison.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placed", type=Path, required=True)
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--capability", type=Path, default=DEFAULT_BUNDLE / "capability.json")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save(args.out_dir / "plan.json", {"recipe": RECIPE, "source_sha256": sources()})
    try:
        import torch
        import torch.nn.functional as F
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("two GPUs required; CPU fallback is forbidden")
        adapter = KervSiluMulAdapter(args.capability, args.flagscale_root)
        adapter.flagscale_root = args.flagscale_root
        placed = load_placed_op_graph(args.placed, registry=adapter.registry)
        selected = args.group_index
        if selected is None:
            indices = [i for i, g in enumerate(placed.plan.groups) if g.extra["pattern_id"] == "silu_mul"]
            if len(indices) != 1:
                raise ValueError("select --group-index when multiple fused groups exist")
            selected = indices[0]
        results = []
        with torch.no_grad():
            for index in (0, 1):
                target = args.out_dir / f"gpu{index}"
                target.mkdir()
                results.append(device_case(torch, F, adapter, placed, selected, index, target))
        save(args.out_dir / "summary.json", {"complete": True, "cases": results, "profile_sharing": False,
                                            "automatic_admission": False})
    except Exception as error:
        save(args.out_dir / "failure.json", {"status": "failed_or_not_run", "error": repr(error),
                                              "profile_accepted": False})
        raise


if __name__ == "__main__":
    main()
