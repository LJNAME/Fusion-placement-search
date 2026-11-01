"""Four-arm graph additivity diagnostic; never exports accepted Kernel labels."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import time

from .e2_manifest import content_id
from .kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE, ROOT
from .probe_kerv_graph import POOL, capture_arm, memory_snapshot, restore, save, write_failure
from .probe_kerv_silu_mul import checked_launch_probe, load_kernel_module
from cost_model.measurement_labels import measurement_contract_id

ARMS = ("silu", "mul", "chain", "fused")
# Each position occurs twice per arm; every ordered within-round neighbor pair
# occurs twice. These orders are fixed before any GPU measurements.
ORDERS = (("silu", "mul", "fused", "chain"), ("mul", "chain", "silu", "fused"),
          ("chain", "fused", "mul", "silu"), ("fused", "silu", "chain", "mul")) * 2
CASES = ((0, 256), (0, 1), (1, 1), (1, 256))
CALLS = 50
SOURCES = ("calibration/probe_kerv_additivity.py", "calibration/probe_kerv_graph.py",
    "calibration/probe_kerv_silu_mul.py", "calibration/kerv_silu_mul.py",
    "cost_model/measurement_labels.py", "calibration/e2_manifest.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    value = {"format": "fusion-placement-search.graph-additivity", "version": "1.0",
        "case_order": [f"gpu{d}_seq{s}" for d, s in CASES], "arms": list(ARMS),
        "capture_order": list(ARMS), "round_orders": [list(order) for order in ORDERS],
        "calls_per_graph": CALLS, "warmup_replays_per_arm": 10,
        "timed_body_calls": 6400, "timed_event_intervals": 128,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "attempted_pool_config": dict(POOL), "admission": "conditional_dev",
        "input_contract": "50 independent contiguous gate views, same shared up; restore outside each replay Event",
        "mul_input_contract": "GPU F.silu(original), prepared outside capture/replay timing",
        "output_lifetime": "last final output retained per arm; intermediates not accumulated",
        "metric": "graph_batch_amortized_per_call_us", "serial_plan_only": True,
        "scope": "single-stream diagnostic bodies only; no model plan-domain gate or downstream-runtime claim",
        "residual": "silu + mul - chain; positive overstates unfused cost and fusion savings",
        "uncertainty": "all eight quadruples; median/range/inclusive IQR/unscaled MAD; no significance claim",
        "no_outlier_trimming_or_automatic_rerun": True, "automatic_model_writeback": False}
    return {**value, "protocol_id": content_id("graph-additivity-protocol", value)}


def spread(values):
    q = st.quantiles(values, n=4, method="inclusive")
    m = st.median(values)
    return {"median": m, "min": min(values), "max": max(values),
        "iqr": q[2] - q[0], "mad": st.median(abs(v - m) for v in values)}


def summarize(rows):
    rounds = []
    for index in range(8):
        values = {r["method"]: r["graph_batch_amortized_per_call_us"]
                  for r in rows if r["round"] == index}
        if set(values) != set(ARMS):
            raise ValueError("round does not contain all four arms")
        delta = values["silu"] + values["mul"] - values["chain"]
        predicted = values["silu"] + values["mul"] - values["fused"]
        observed = values["chain"] - values["fused"]
        rounds.append({"round": index, "additivity_residual_us": delta,
            "additivity_residual_relative": delta / values["chain"],
            "predicted_fusion_saving_us": predicted, "chain_fusion_saving_us": observed,
            "fusion_saving_prediction_error_us": predicted - observed,
            "decision_disagrees": predicted * observed < 0,
            "either_comparison_tied": predicted == 0 or observed == 0})
    return {"graph_batch_amortized_per_call_us": {arm: spread([
        r["graph_batch_amortized_per_call_us"] for r in rows if r["method"] == arm]) for arm in ARMS},
        "round_results": rounds,
        **{field: spread([r[field] for r in rounds]) for field in (
            "additivity_residual_us", "additivity_residual_relative", "predicted_fusion_saving_us",
            "chain_fusion_saving_us", "fusion_saving_prediction_error_us")},
        "decision_disagreement_rounds": sum(r["decision_disagrees"] for r in rounds),
        "comparison_tie_rounds": sum(r["either_comparison_tied"] for r in rounds),
        "statistical_significance_claim": False, "cache_causality_claim": False,
        "automatic_model_writeback": False}


def prepare_case(torch, F, module, device, seq):
    shape = (1, seq, 3072)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    gate_cpu = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    up_cpu = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    original, up = gate_cpu.to(device), up_cpu.to(device)
    silu_original = F.silu(original)  # Same backend/value as the chain intermediate.
    gates = original.unsqueeze(0).repeat((CALLS, 1, 1, 1))
    functions = {"silu": lambda gate, up: F.silu(gate), "mul": lambda gate, up: gate * up,
                 "chain": lambda gate, up: F.silu(gate) * up, "fused": module.kerv_silu_mul}
    sources = {arm: silu_original if arm == "mul" else original for arm in ARMS}
    fused_ref = (gate_cpu.float() * torch.sigmoid(gate_cpu.float()) * up_cpu.float()).to(torch.bfloat16).to(device)
    refs = {"silu": silu_original, "mul": silu_original * up,
            "chain": silu_original * up, "fused": fused_ref}
    return gates, original, up, up_cpu, functions, sources, refs


def sample_replay(torch, graph, device):
    torch.cuda.reset_peak_memory_stats(device)
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    started = time.perf_counter()
    begin.record()
    graph.replay()
    end.record()
    submitted = time.perf_counter() - started
    torch.cuda.synchronize()
    completed = time.perf_counter() - started
    total = begin.elapsed_time(end) / 1000
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid graph Event duration")
    return {"graph_event_total_s": total, "graph_batch_amortized_per_call_us": total * 1e6 / CALLS,
        "graph_submission_wall_s": submitted, "graph_completion_wall_s": completed,
        "memory_before_numeric_checks": memory_snapshot(torch, device)}


def run_case(torch, F, module, device, seq, output, context, plan):
    context.clear()
    context.update(case_id=f"gpu{device.index}_seq{seq}", device=str(device),
                   stage="input_setup", capture_attempted=False)
    torch.cuda.set_device(device)
    gates, original, up, up_cpu, functions, sources, refs = prepare_case(torch, F, module, device, seq)
    torch.testing.assert_close(checked_launch_probe(module, gates[0], up), refs["fused"], rtol=.01, atol=.001)
    contract = {"executor": "graph_replay", "submission_included": True, "allocation_included": False,
        "granularity": "graph_replay", "unit": "throughput_per_call", "serial_plan_only": True,
        "environment": {"torch": str(torch.__version__), "hip": torch.version.hip,
            "triton": str(module.triton.__version__), "device_model": torch.cuda.get_device_name(device)}}
    report = {"case_id": context["case_id"], "shape": [1, seq, 3072], "dtype": "bf16",
        "gate_pool_bytes": gates.numel() * gates.element_size(), "protocol_id": plan["protocol_id"],
        "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract),
        "admission": "conditional_dev", "numeric_tolerance": {"rtol": .01, "atol": .001},
        "captures": {}, "rows": []}
    captures = {}
    for arm in ARMS:
        context["method"] = arm
        graph, result, record = capture_arm(torch, functions[arm], gates, sources[arm], up, context)
        captures[arm] = (graph, result)
        report["captures"][arm] = record
        save(output / "partial.json", report)
    context["stage"] = "replay_warmup"
    for arm, (graph, result) in captures.items():
        context["method"] = arm
        for _ in range(10):
            restore(torch, gates, sources[arm])
            graph.replay()
            torch.cuda.synchronize()
    for index, order in enumerate(ORDERS):
        for arm in order:
            context.update(stage="timed_replay", method=arm, round=index)
            graph, result = captures[arm]
            restore(torch, gates, sources[arm])
            row = {"round": index, "method": arm, "graph_calls": CALLS,
                   **sample_replay(torch, graph, device), "numeric_passed": None}
            report["rows"].append(row)
            save(output / "partial.json", report)
            context["stage"] = "numeric_check"
            if arm == "fused":
                torch.testing.assert_close(gates, refs[arm].unsqueeze(0).expand_as(gates), rtol=.01, atol=.001)
            else:
                torch.testing.assert_close(result, refs[arm], rtol=.01, atol=.001)
            torch.testing.assert_close(up, up_cpu.to(device), rtol=0, atol=0)
            row.update(numeric_passed=True, numeric_coverage="all_50_gate_outputs" if arm == "fused" else "last_final_output_only")
            save(output / "partial.json", report)
    report["summary"] = summarize(report["rows"])
    save(output / "result.json", report)
    print(report["case_id"], json.dumps(report["summary"]))


def verify_bundle(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if plan != protocol():
        raise ValueError("protocol/source mismatch; use the pinned collector version")
    if (directory / "failure.json").exists():
        return {"complete": False, "failure": json.loads((directory / "failure.json").read_text(encoding="utf-8")),
                "completed_case_files": [str(p.relative_to(directory)) for p in directory.glob("*/result.json")]}
    env = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    expected_cap = json.loads((DEFAULT_BUNDLE / "capability.json").read_text(encoding="utf-8"))
    if env["capability"] != expected_cap:
        raise ValueError("capability mismatch")
    reports, count = [], 0
    if len(list(directory.glob("*/result.json"))) != 4:
        raise ValueError("missing or unexpected case")
    for device, seq in CASES:
        case_id = f"gpu{device}_seq{seq}"
        path = directory / case_id / "result.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["case_id"] == case_id and d["shape"] == [1, seq, 3072] and d["dtype"] == "bf16"
        assert d["gate_pool_bytes"] == CALLS * seq * 3072 * 2
        assert d["protocol_id"] == plan["protocol_id"] and d["admission"] == "conditional_dev"
        contract = d["measurement_contract"]
        assert contract == {"executor": "graph_replay", "submission_included": True, "allocation_included": False,
            "granularity": "graph_replay", "unit": "throughput_per_call", "serial_plan_only": True,
            "environment": {"torch": env["torch"], "hip": env["hip"], "triton": env["triton"],
                            "device_model": env["devices"][str(device)]}}
        assert d["measurement_contract_id"] == measurement_contract_id(contract)
        assert d["numeric_tolerance"] == {"rtol": .01, "atol": .001}
        assert list(d["captures"]) == list(ARMS)
        for capture in d["captures"].values():
            assert capture["single_stream"] and capture["retained_outputs"] == "last_final_output_only"
            assert capture["attempted_pool_config"] == POOL
        partial = json.loads(path.with_name("partial.json").read_text(encoding="utf-8"))
        assert all(d[k] == v for k, v in partial.items())
        expected = [(r, arm) for r, order in enumerate(ORDERS) for arm in order]
        assert [(r["round"], r["method"]) for r in d["rows"]] == expected
        for row in d["rows"]:
            assert row["graph_calls"] == CALLS and row["numeric_passed"] is True
            assert row["numeric_coverage"] == ("all_50_gate_outputs" if row["method"] == "fused" else "last_final_output_only")
            assert math.isfinite(row["graph_event_total_s"]) and row["graph_event_total_s"] > 0
            assert 0 < row["graph_submission_wall_s"] <= row["graph_completion_wall_s"]
            assert math.isclose(row["graph_batch_amortized_per_call_us"], row["graph_event_total_s"] * 1e6 / CALLS, rel_tol=1e-12)
            count += CALLS
        assert d["summary"] == summarize(d["rows"])
        reports.append({"case_id": case_id, "measurement_contract_id": d["measurement_contract_id"], "summary": d["summary"]})
    assert count == plan["timed_body_calls"]
    return {"complete": True, "timed_body_calls": count, "event_intervals": 128,
            "admission": "conditional_dev", "automatic_model_writeback": False, "cases": reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out-dir", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    args = parser.parse_args(argv)
    if args.verify:
        if args.plan_only:
            parser.error("--plan-only requires --out-dir")
        result = verify_bundle(args.verify)
        print(json.dumps(result, indent=2))
        if not result["complete"]:
            raise SystemExit(1)
        return
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = protocol()
    save(args.out_dir / "plan.json", plan)
    if args.plan_only:
        print("Protocol only; no GPU operations:", plan["protocol_id"])
        return
    torch = None
    context = {"stage": "environment", "capture_attempted": False}
    try:
        import torch
        import torch.nn.functional as F
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("two GPUs required; no CPU/eager fallback")
        module, cap = load_kernel_module(args.flagscale_root, DEFAULT_BUNDLE / "capability.json")
        module.configure_recording(None, False)
        module.configure_backend("triton")
        module.configure_enabled_operators({"kerv_silu_mul"})
        save(args.out_dir / "environment.json", {"capability": cap, "torch": str(torch.__version__),
            "hip": torch.version.hip, "triton": str(module.triton.__version__),
            "devices": {str(d): torch.cuda.get_device_name(d) for d in (0, 1)},
            "allocator_config": {n: os.environ.get(n) for n in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}})
        with torch.no_grad():
            for device, seq in CASES:
                output = args.out_dir / f"gpu{device}_seq{seq}"
                output.mkdir()
                run_case(torch, F, module, torch.device(f"cuda:{device}"), seq, output, context, plan)
    except Exception as error:
        write_failure(args.out_dir / "failure.json", error, context, torch)
        raise


if __name__ == "__main__":
    main()
