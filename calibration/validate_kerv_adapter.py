"""Prepare one Qwen3 group and collect GPU evidence through the placed adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from calibration.probe_kerv_graph import capture_arm, restore
from cost_model.measurement_labels import measurement_contract_id
from cost_model.search_evaluation import build_evaluation_graph
from integration.flagscale.kerv_execution import KervSiluMulAdapter, load_kerv_registry
from simulator import enumerate_fusion_candidates, load_hardware, make_reference_plan
from simulator.placement_artifacts import PlacedOpGraph, load_placed_op_graph, save_placed_op_graph


ROOT = Path(__file__).resolve().parents[1]


def _save(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "integration/flagscale/kerv_execution.py", "calibration/validate_kerv_adapter.py",
        "calibration/kerv_silu_mul.py", "calibration/probe_kerv_silu_mul.py",
        "calibration/probe_kerv_graph.py", "simulator/fusion_rules.py",
        "simulator/placement_artifacts.py")}


def prepare(output, capability, flagscale_root, hardware, layer=0, device=0):
    registry, manifest = load_kerv_registry(capability, flagscale_root)
    graph = build_evaluation_graph("model")
    hw = load_hardware(hardware)
    if not 0 <= device < hw.n_devs:
        raise ValueError("device is outside the hardware snapshot")
    candidates = [c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                  if c.pattern_id == "silu_mul"]
    matches = [c for c in candidates if graph.nodes[c.node_ids[0]].extra["layer"] == layer]
    if len(matches) != 1:
        raise ValueError("expected one SiLU-Mul candidate in the selected layer")
    candidate = matches[0]
    plan = make_reference_plan(graph, device=device, registry=registry)
    # Move the internal SiLU to Mul's position. All boundary producers precede
    # Mul, and the pattern guard forbids any external SiLU consumer.
    groups = []
    for group in plan.groups:
        if group.node_ids == [candidate.node_ids[0]]:
            continue
        if group.node_ids == [candidate.node_ids[-1]]:
            group = candidate.to_group(device)
            group.extra["implementation"] = copy.deepcopy(manifest["binding"])
        groups.append(group)
    plan.groups = groups
    placed = PlacedOpGraph(graph, hw, plan, extra={
        "scope": "one_selected_group_execution_only", "deployable": False,
        "capability_id": manifest["capability_id"],
        "input_values": "synthetic_boundary_tensors_with_reference_graph_shape",
    })
    group_index = next(i for i, g in enumerate(groups) if g.extra["pattern_id"] == "silu_mul")
    save_placed_op_graph(output / "placed.json", placed)
    _save(output / "prepare.json", {
        "nodes": len(graph.nodes), "layers": 28, "executable_silu_mul_candidates": len(candidates),
        "selected_node_ids": list(candidate.node_ids), "group_index": group_index,
        "logical_device": device, "capability_id": manifest["capability_id"],
        "gpu_executed": False, "downstream_deployable": False,
        "source_sha256": source_hashes(),
    })
    return group_index


def verify_device_trace(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if len(kernels) != 1 or "silu_mul_kernel" not in kernels[0].get("name", ""):
        raise RuntimeError("GPU trace must contain exactly one KERV silu_mul_kernel")
    return {"device_kernel_count": 1, "device_kernel_name": kernels[0]["name"]}


def run(output, placed_path, capability, flagscale_root, group_index, runtime_device,
        *, rounds=7, calls=50, warmup=10):
    import torch
    import torch.nn.functional as F

    if rounds < 1 or calls < 1 or warmup < 0:
        raise ValueError("invalid sampling counts")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution unavailable; no CPU/eager fallback")
    adapter = KervSiluMulAdapter(capability, flagscale_root)
    placed = load_placed_op_graph(placed_path, registry=adapter.registry)
    if group_index is None:
        indices = [i for i, g in enumerate(placed.plan.groups) if g.extra["pattern_id"] == "silu_mul"]
        if len(indices) != 1:
            raise ValueError("select --group-index when the plan has multiple fused groups")
        group_index = indices[0]
    logical = placed.plan.groups[group_index].device
    bound = adapter.bind(placed, group_index, {logical: runtime_device})
    device = bound.target
    torch.cuda.set_device(device)
    meta = placed.graph.nodes[bound.gate_node].extra
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[meta["dtype"]]
    generator = torch.Generator(device="cpu").manual_seed(1234)
    shape = tuple(meta["output_shape"])
    original = torch.randn(shape, dtype=dtype, generator=generator).to(device)
    up = torch.randn(shape, dtype=dtype, generator=generator).to(device)
    original_up = up.clone()
    contract = adapter.manifest["numeric_contract"]
    rtol = contract["fp32_rtol" if dtype == torch.float32 else "bf16_fp16_rtol"]
    atol = contract["fp32_atol" if dtype == torch.float32 else "bf16_fp16_atol"]
    native = F.silu(original) * up
    fp32 = (F.silu(original.float()) * up.float()).to(dtype)

    def invoke(gate, other, observe=False):
        values = {bound.gate_node: gate, bound.up_node: other}
        result = bound(values, observe_dispatch=observe)
        if bound.gate_node in values or values[result.output_node] is not result.output:
            raise RuntimeError("adapter failed to invalidate gate or publish the Mul output")
        return result.output

    with torch.no_grad():
        gate = original.clone()
        result = invoke(gate, up, observe=True)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result, native, rtol=rtol, atol=atol)
        torch.testing.assert_close(result, fp32, rtol=rtol, atol=atol)
        torch.testing.assert_close(up, original_up, rtol=0, atol=0)
        native_error = (result.float() - native.float()).abs().max().item()
        fp32_error = (result.float() - fp32.float()).abs().max().item()
        # Compile/warm up before profiling; input restore is outside the trace.
        gate.copy_(original)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            invoke(gate, up)
            torch.cuda.synchronize(device)
        trace_path = output / "dispatch_trace.json"
        prof.export_chrome_trace(str(trace_path))
        trace = verify_device_trace(trace_path)
        gates = original.unsqueeze(0).repeat((calls,) + (1,) * original.ndim)
        context = {"device": str(device)}
        replay, _result, capture = capture_arm(torch, invoke, gates, original, up, context)
        for _ in range(warmup):
            restore(torch, gates, original)
            replay.replay()
            torch.cuda.synchronize(device)
        rows = []
        for index in range(rounds):
            restore(torch, gates, original)
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            replay.replay()
            end.record()
            end.synchronize()
            total = begin.elapsed_time(end) / 1000
            if not math.isfinite(total) or total <= 0:
                raise RuntimeError("nonpositive or nonfinite GPU Event duration")
            torch.testing.assert_close(gates, native.unsqueeze(0).expand_as(gates), rtol=rtol, atol=atol)
            torch.testing.assert_close(gates, fp32.unsqueeze(0).expand_as(gates), rtol=rtol, atol=atol)
            torch.testing.assert_close(up, original_up, rtol=0, atol=0)
            rows.append({"round": index, "graph_event_total_s": total,
                         "graph_calls": calls, "amortized_per_call_s": total / calls})
            _save(output / "samples.json", rows)
    measurement = {
        "executor": "placed_kerv_adapter_graph_replay", "submission_included": True,
        "allocation_included": False, "granularity": "independent_gate_batch_graph_replay",
        "unit": "seconds_per_call",
        "environment": {"torch": str(torch.__version__), "hip": torch.version.hip,
                        "cuda": torch.version.cuda, "triton": str(adapter.module.triton.__version__),
                        "device_model": torch.cuda.get_device_name(device)},
        "timing": "Event interval includes one graph replay submission boundary; amortized across calls",
        "restore": "outside Event interval; each call has a fresh gate slice",
    }
    props = torch.cuda.get_device_properties(device)
    report = {
        "format": "fusion-placement-search.placed-kerv-validation", "version": "1.0",
        "graph_hash": placed.graph_hash, "hardware_hash": placed.hardware_hash,
        "candidate_id": placed.candidate_id, "group_index": group_index,
        "node_ids": list(bound.node_ids), "gate_node": bound.gate_node, "up_node": bound.up_node,
        "logical_device": logical, "runtime_device": str(device),
        "observed_device": {"name": props.name, "total_memory": props.total_memory,
                            "uuid": str(getattr(props, "uuid", "unavailable"))},
        "capability_id": adapter.manifest["capability_id"], "binding": adapter.manifest["binding"],
        "kernel_source_sha256": adapter.manifest["source_sha256"], "source_sha256": source_hashes(),
        "placed_artifact_sha256": hashlib.sha256(Path(placed_path).read_bytes()).hexdigest(),
        "input": {"shape": list(shape), "dtype": meta["dtype"], "gate_strides": list(original.stride()),
                  "up_strides": list(up.stride()), "layout": "contiguous", "seed": 1234,
                  "origin": "synthetic values; shape and node IDs from the placed reference graph"},
        "execution": {"observed_jit_launches": 1, "synchronized_successfully": True, **trace},
        "correctness": {"rtol": rtol, "atol": atol, "max_abs_error_native": native_error,
                        "max_abs_error_fp32": fp32_error, "gate_alias": True, "up_preserved": True,
                        "gate_invalidated": True, "all_replay_outputs_checked": rounds * calls},
        "measurement_contract": measurement, "measurement_contract_id": measurement_contract_id(measurement),
        "capture": capture, "warmup": warmup, "rows": rows,
        "graph_batch_amortized_per_call_s": statistics.median(r["amortized_per_call_s"] for r in rows),
        "admission": "conditional_dev", "canonical_profile_exported": False,
        "downstream_deployable": False,
    }
    _save(output / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--capability", type=Path, default=DEFAULT_BUNDLE / "capability.json")
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--hardware", type=Path, default=ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--logical-device", type=int, default=0)
    parser.add_argument("--placed", type=Path)
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    if args.action == "run" and args.placed is None:
        parser.error("run requires --placed")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.action == "prepare":
            group_index = prepare(args.out_dir, args.capability, args.flagscale_root,
                                  args.hardware, args.layer, args.logical_device)
            print("Prepared one KERV group; group_index=", group_index)
        else:
            run(args.out_dir, args.placed, args.capability, args.flagscale_root, args.group_index,
                args.device, rounds=args.rounds, calls=args.calls, warmup=args.warmup)
            print("Placed KERV GPU validation: PASS; profile admission remains conditional_dev")
    except Exception as error:
        _save(args.out_dir / "failure.json", {"status": "failed_or_not_run", "error": repr(error),
              "gpu_validation_passed": False, "profile_accepted": False, "source_sha256": source_hashes()})
        raise


if __name__ == "__main__":
    main()
