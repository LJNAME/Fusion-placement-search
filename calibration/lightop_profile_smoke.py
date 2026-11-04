"""One selected Qwen3 group: capture/numerics smoke, never latency admission."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

from cost_model.fusion_kernel_profile import key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
from cost_model.measurement_labels import measurement_contract_id
from cost_model.search_evaluation import build_evaluation_graph
from integration.lightop.silu_mul import RECIPE, SHAPE, adapter_registry, declared_binding
from simulator import enumerate_fusion_candidates, load_hardware, make_reference_plan
from simulator.placement_artifacts import PlacedOpGraph, graph_fingerprint, save_placed_op_graph

ROOT = Path(__file__).resolve().parents[1]
POOL = {"pool_argument": None, "selection": "default_private_pool", "shared_with_other_graphs": False}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plan(device=0):
    if device not in (0, 1):
        raise ValueError("this smoke supports one of the two explicit GPU ordinals")
    return {"scope": "lightop_selected_group_smoke_not_profile_admission", "device": f"cuda:{device}",
            "graph_source": "logical Qwen3-0.6B reference builder, not an upstream captured graph",
            "shape": list(SHAPE), "dtype": "bf16", "implementation": declared_binding(), "recipe": RECIPE,
            "calls_per_replay": 50, "eager_warmup_calls": 50,
            "warmup_replays": 10, "diagnostic_replays": 3, "attempts": 1,
            "pool": POOL, "buffers": "shared read-only inputs; 50 independent packed/output pairs",
            "replay_validation": "invalidate packed/output with NaN outside timing before each checked replay",
            "pass_conditions": ["Graph capture and all replays complete", "all output numerics pass",
                                "gate/up bytewise unchanged", "outputs independent of inputs and each other",
                                "runtime identity and measurement contract hashes saved"],
            "timing_used_for_pass_fail": False, "automatic_admission": False,
            "latency_labels_written": False, "full_graph_executed": False,
            "dispatch_trace_certified": False}


def placed_case(device):
    graph, registry = build_evaluation_graph("model"), adapter_registry()
    candidate = next(c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                     if c.pattern_id == "silu_mul")
    reference = make_reference_plan(graph, registry=registry)
    group = candidate.to_group(device)
    group.extra["implementation"] = declared_binding()
    groups = []
    for singleton in reference.groups:
        if singleton.node_ids[0] == candidate.node_ids[-1]:
            groups.append(group)
        elif singleton.node_ids[0] not in candidate.node_ids:
            groups.append(singleton)
    reference.groups = groups
    hardware = load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    placed = PlacedOpGraph(graph, hardware, reference, extra={
        "scope": "one_selected_group_with_synthetic_boundary_inputs", "deployable": False})
    return placed, groups.index(group)


def memory_snapshot(torch, device):
    try:
        return {name: int(getattr(torch.cuda, name)(device)) for name in
                ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")}
    except Exception as error:
        return {"unavailable": repr(error)}


def check_outputs(torch, calls, gate, up, snapshots, references, tolerance):
    for tensor, before in zip((gate, up), snapshots):
        if not torch.equal(tensor.view(torch.uint8), before):
            raise AssertionError("shared read-only input bytes changed")
    output_pointers, packed_pointers = set(), set()
    for call in calls:
        for reference in references:
            torch.testing.assert_close(call.output.float(), reference.float(),
                                       rtol=tolerance["rtol"], atol=tolerance["atol"])
        if any(torch._C._overlaps(call.output, source) for source in (gate, up, call._packed)):
            raise AssertionError("output aliases an input or packed buffer")
        output_pointers.add(call.output.data_ptr())
        packed_pointers.add(call._packed.data_ptr())
    if len(output_pointers) != len(calls) or len(packed_pointers) != len(calls):
        raise AssertionError("prepared calls share mutable buffers")
    return {"outputs_checked": len(calls), "references_per_output": len(references),
            "input_bytes_unchanged": True, "outputs_independent": True}


def invalidate_buffers(calls):
    # A no-op capture must not pass using values left by eager warmup.
    for call in calls:
        call._packed.fill_(float("nan"))
        call.output.fill_(float("nan"))


def diagnostic_replay(torch, graph, stream, device):
    # Even Event creation/readout failures must not skip or repeat the replay.
    start, end, error, elapsed = None, None, None, None
    try:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
    except Exception as exc:
        error = repr(exc)
    graph.replay()
    if error is None:
        try:
            end.record(stream)
        except Exception as exc:
            error = repr(exc)
    torch.cuda.synchronize(device)  # Execution errors remain hard Smoke failures.
    if error is None:
        try:
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed):
                raise ValueError("nonfinite diagnostic timing")
        except Exception as exc:
            elapsed, error = None, repr(exc)
    return {"elapsed_ms_per_50_calls": elapsed, "error": error, "used_for_pass_fail": False}


def run(output, device_index, context):
    import torch
    import torch.nn.functional as F
    from integration.lightop.profile_identity import observe_runtime
    from integration.lightop.silu_mul import LightopSiluMulAdapter

    context.update(stage="runtime_guard", device=f"cuda:{device_index}", pool=POOL)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch with an available GPU is required; no CPU/native fallback")
    device = torch.device(context["device"])
    torch.cuda.set_device(device)
    adapter = LightopSiluMulAdapter()
    placed, index = placed_case(device_index)
    bound = adapter.bind(placed, index, {device_index: device})
    save_placed_op_graph(output / "placed.json", placed)
    actual = observe_runtime(adapter, device)
    save(output / "runtime.json", actual)
    context["stage"] = "input_and_buffer_preparation"
    generator = torch.Generator(device=device).manual_seed(20261104)
    gate = torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=generator)
    up = torch.randn(SHAPE, device=device, dtype=torch.bfloat16, generator=generator)
    snapshots = [tensor.view(torch.uint8).clone() for tensor in (gate, up)]
    references = (F.silu(gate) * up, F.silu(gate.float()) * up.float())
    calls = [bound.prepare({bound.gate_node: gate, bound.up_node: up}) for _ in range(50)]
    if any(call.observation != calls[0].observation for call in calls):
        raise ValueError("prepared calls do not share one execution identity")
    keys = LightopKernelKeys(placed, index, calls[0].observation, actual)
    key, gaps = keys.describe(placed.plan.groups[index])
    contract = measurement_contract(actual["environment"])
    identity = {"kernel_key": key, "kernel_key_id": key_id(key), "identity_gaps": gaps,
                "contract": contract, "measurement_contract_id": measurement_contract_id(contract),
                "graph_fingerprint": graph_fingerprint(placed.graph), "group_index": index,
                "node_ids": list(bound.node_ids), "layout_source": "runtime Tensor observations",
                "uuid_policy": "run provenance only; device ordinal is retained in the key"}
    save(output / "identity.json", identity)
    save(output / "preparation.json", {"one_call": calls[0].observation, "call_count": len(calls),
                                      "explicit_mutable_buffer_bytes": sum(c.observation["explicit_buffer_bytes"] for c in calls),
                                      "excludes": "input/reference/snapshot tensors, runtime workspace and Graph pool"})
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    context["stage"] = "side_stream_eager_warmup"
    save(output / "last_stage.json", context)
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.synchronize(device)
    context.update(stage="capture", memory_before_capture=memory_snapshot(torch, device))
    save(output / "last_stage.json", context)
    print(json.dumps(context), flush=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=None):
        for call in calls:
            call()
    torch.cuda.synchronize(device)
    context["stage"] = "graph_warmup"
    save(output / "last_stage.json", context)
    with torch.cuda.stream(stream):
        invalidate_buffers(calls)
        for _ in range(contract["warmup_replays"]):
            graph.replay()
    torch.cuda.synchronize(device)
    check_outputs(torch, calls, gate, up, snapshots, references, contract["correctness"])
    records = []
    for replay in range(contract["rounds"]):
        context.update(stage="smoke_replay", replay=replay)
        save(output / "last_stage.json", context)
        with torch.cuda.stream(stream):
            invalidate_buffers(calls)
            timing = diagnostic_replay(torch, graph, stream, device)
        checks = check_outputs(torch, calls, gate, up, snapshots, references, contract["correctness"])
        record = {"replay": replay, "diagnostic_timing": timing, "checks": checks}
        records.append(record)
        save(output / f"replay_{replay}.json", record)
        print(json.dumps(record), flush=True)
    result = {"scope": plan(device_index)["scope"], "capture_numeric_passed": True,
              "identity_complete": not gaps, "identity_gaps": gaps,
              "kernel_key_id": identity["kernel_key_id"], "measurement_contract_id": identity["measurement_contract_id"],
              "replays": records, "automatic_admission": False, "latency_labels_written": False,
              "dispatch_trace_certified": False, "full_graph_executed": False,
              "memory_after_smoke": memory_snapshot(torch, device)}
    save(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    # The shell may already have opened process.log. Never overwrite an experiment.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(path.name != "process.log" for path in args.output.iterdir()):
        parser.error("output must be fresh (only the wrapper process.log is allowed)")
    context = {"stage": "plan", "device": f"cuda:{args.device}", "pool": POOL, "started_unix_s": time.time()}
    try:
        save(args.output / "plan.json", plan(args.device))
        if args.phase == "run":
            result = run(args.output, args.device, context)
            print(json.dumps({k: result[k] for k in ("scope", "capture_numeric_passed", "identity_complete")}), flush=True)
    except BaseException as error:
        trace = traceback.format_exc()
        import sys
        torch = sys.modules.get("torch")
        save(args.output / "failure.json", {"error": repr(error), "traceback": trace, "context": context,
                                           "pool": POOL, "memory_at_failure": memory_snapshot(torch, context["device"]) if torch else None,
                                           "automatic_admission": False, "latency_labels_written": False})
        raise


if __name__ == "__main__":
    main()
