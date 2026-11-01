"""Six-arm ATen graph composition diagnostic; conditional development only."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from .e2_manifest import content_id
from .kerv_silu_mul import ROOT
from .probe_kerv_additivity import sha, spread
from .probe_kerv_graph import POOL, memory_snapshot, restore, save, write_failure
from cost_model.measurement_labels import measurement_contract_id

ARMS = ("silu", "mul", "add", "silu_mul", "mul_add", "silu_mul_add")
BASE = (0, 1, 5, 2, 4, 3)
ORDERS = tuple(tuple(ARMS[(i + j) % 6] for i in BASE) for j in range(6)) * 2
BATCHES = (50, 100)
BUFFERS = 50
SOURCES = ("calibration/probe_sequence_composition.py", "calibration/probe_kerv_additivity.py",
    "calibration/probe_kerv_graph.py", "calibration/probe_kerv_silu_mul.py",
    "calibration/kerv_silu_mul.py", "calibration/e2_manifest.py", "cost_model/measurement_labels.py")


def schedule():
    return [(r, n, arm) for r, order in enumerate(ORDERS)
            for n in (BATCHES if r % 2 == 0 else BATCHES[::-1]) for arm in order]


def protocol():
    value = {"format": "fusion-placement-search.sequence-composition", "version": "1.0",
        "case_id": "gpu0_seq256", "shape": [1, 256, 3072], "dtype": "bf16", "seed": 1234,
        "arms": list(ARMS), "batches": list(BATCHES), "input_buffers": BUFFERS,
        "round_orders": [list(o) for o in ORDERS], "batch_order": "50,100 on even rounds; 100,50 on odd rounds",
        "capture_order": [[n, a] for n in BATCHES for a in ARMS],
        "warmup_replays_per_capture": 10, "side_stream_warmup": "one full body batch before capture",
        "timed_body_calls": 10800, "timed_event_intervals": 144,
        "input_contract": "50 contiguous views; N100 rereads them twice; shared up and add operand; restore outside Event",
        "intermediate_contract": "Mul/MA input GPU SiLU(original); Add input GPU SiLU(original)*up; prepared outside timing",
        "implementation": "out-of-place eager ATen bodies captured into single-stream graph; no KERV/custom op",
        "output_lifetime": "last final output retained per capture; all 12 graphs stay alive",
        "numeric_tolerance": {"rtol": .01, "atol": .001},
        "numeric_coverage": "last_final_output_and_all_input_buffers_and_operands_unchanged",
        "attempted_pool_config": dict(POOL), "metric": "graph_batch_amortized_per_call_us",
        "admission": "conditional_dev", "serial_plan_only": True,
        "source_sha256": {s: sha(ROOT / s) for s in SOURCES},
        "no_outlier_trimming_or_automatic_rerun": True, "automatic_model_writeback": False,
        "interpretation": "N scaling is diagnostic, not causal identification or an adjacency fit"}
    return {**value, "protocol_id": content_id("sequence-composition-protocol", value)}


def contract_for(env, calls):
    return {"executor": "graph_replay", "submission_included": True, "allocation_included": False,
        "granularity": "graph_replay", "unit": "throughput_per_call", "serial_plan_only": True,
        "calls_per_graph": calls, "input_buffers": BUFFERS,
        "input_revisit": "cyclic_read_only", "implementation": "aten_out_of_place_sequence",
        "environment": {k: env[k] for k in ("torch", "hip", "triton", "device_model")}}


def prepare_case(torch, F, device, seq=256):
    gen = torch.Generator(device="cpu").manual_seed(1234)
    original, up, addend = [torch.randn((1, seq, 3072), dtype=torch.bfloat16, generator=gen).to(device)
                            for _ in range(3)]
    si = F.silu(original)
    mu = si * up
    gates = original.unsqueeze(0).repeat((BUFFERS, 1, 1, 1))
    functions = {"silu": lambda x, u: F.silu(x), "mul": lambda x, u: x * u,
        "add": lambda x, u: x + addend, "silu_mul": lambda x, u: F.silu(x) * u,
        "mul_add": lambda x, u: x * u + addend, "silu_mul_add": lambda x, u: F.silu(x) * u + addend}
    sources = {a: si if a in ("mul", "mul_add") else mu if a == "add" else original for a in ARMS}
    refs = dict(silu=si, mul=mu, add=mu + addend, silu_mul=mu,
                mul_add=mu + addend, silu_mul_add=mu + addend)
    return gates, up, addend, functions, sources, refs


def capture(torch, function, gates, source, up, calls, context):
    device = gates.device
    views = list(gates.unbind(0)) * (calls // BUFFERS)
    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    context.update(stage="side_stream_warmup", capture_attempted=False)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        gates.copy_(source.unsqueeze(0).expand_as(gates))
        for gate in views:
            result = function(gate, up)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    del result
    restore(torch, gates, source)
    torch.cuda.reset_peak_memory_stats(device)
    context.update(stage="capture", capture_attempted=True, memory_before_capture=memory_snapshot(torch, device))
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    started = time.perf_counter()
    with torch.cuda.graph(graph, pool=None, stream=stream):
        for gate in views:
            result = function(gate, up)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    return graph, result, {"capture_wall_s": time.perf_counter() - started,
        "memory_before_capture": context["memory_before_capture"],
        "memory_after_capture": memory_snapshot(torch, device), "attempted_pool_config": dict(POOL),
        "single_stream": True, "retained_outputs": "last_final_output_only", "graph_calls": calls}


def sample(torch, graph, device, calls):
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
    return {"graph_event_total_s": total, "graph_batch_amortized_per_call_us": total * 1e6 / calls,
        "graph_submission_wall_s": submitted, "graph_completion_wall_s": completed,
        "memory_before_numeric_checks": memory_snapshot(torch, device)}


RESIDUALS = ("delta_sm_us", "delta_ma_us", "delta_sma_us", "interaction_us")


def summarize(rows):
    if [(r["round"], r["graph_calls"], r["method"]) for r in rows] != schedule():
        raise ValueError("incomplete, duplicate, or reordered observations")
    batches = {}
    for n in BATCHES:
        rounds = []
        for index in range(12):
            v = {r["method"]: r["graph_batch_amortized_per_call_us"]
                 for r in rows if r["round"] == index and r["graph_calls"] == n}
            sm = v["silu"] + v["mul"] - v["silu_mul"]
            ma = v["mul"] + v["add"] - v["mul_add"]
            sma = v["silu"] + v["mul"] + v["add"] - v["silu_mul_add"]
            rounds.append(dict(round=index, delta_sm_us=sm, delta_ma_us=ma,
                               delta_sma_us=sma, interaction_us=sma - sm - ma))
        batches[str(n)] = {"round_results": rounds,
            **{f: spread([r[f] for r in rounds]) for f in RESIDUALS},
            "graph_batch_amortized_per_call_us": {a: spread([r["graph_batch_amortized_per_call_us"]
                for r in rows if r["graph_calls"] == n and r["method"] == a]) for a in ARMS}}
    contrasts = [{"round": i, **{f: batches["100"]["round_results"][i][f]
                  - batches["50"]["round_results"][i][f] for f in RESIDUALS}} for i in range(12)]
    return {"by_calls_per_graph": batches, "n100_minus_n50_round_results": contrasts,
        "n100_minus_n50": {f: spread([r[f] for r in contrasts]) for f in RESIDUALS},
        "statistical_significance_claim": False, "mechanism_identified": False,
        "adjacency_parameter_fitted": False, "automatic_model_writeback": False}


def run(torch, F, env, output, context, plan):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    context.update(stage="input_setup", device=str(device))
    gates, up, addend, functions, sources, refs = prepare_case(torch, F, device)
    saved_up, saved_addend = up.clone(), addend.clone()
    contracts = {str(n): contract_for(env, n) for n in BATCHES}
    ids = {n: measurement_contract_id(c) for n, c in contracts.items()}
    report = {"case_id": plan["case_id"], "shape": plan["shape"], "dtype": "bf16",
        "input_pool_bytes": gates.numel() * gates.element_size(), "protocol_id": plan["protocol_id"],
        "measurement_contracts": contracts, "measurement_contract_ids": ids,
        "admission": "conditional_dev", "captures": {}, "rows": []}
    captures = {}
    for n in BATCHES:
        for arm in ARMS:
            key = f"n{n}_{arm}"
            context.update(method=arm, graph_calls=n)
            graph, result, record = capture(torch, functions[arm], gates, sources[arm], up, n, context)
            captures[n, arm] = (graph, result)
            report["captures"][key] = record
            save(output / "partial.json", report)
    for (n, arm), (graph, _) in captures.items():
        context.update(stage="replay_warmup", method=arm, graph_calls=n)
        for _ in range(10):
            restore(torch, gates, sources[arm])
            graph.replay()
            torch.cuda.synchronize()
    for index, n, arm in schedule():
        context.update(stage="timed_replay", method=arm, graph_calls=n, round=index)
        graph, result = captures[n, arm]
        restore(torch, gates, sources[arm])
        row = {"round": index, "graph_calls": n, "method": arm,
            "measurement_contract_id": ids[str(n)], **sample(torch, graph, device, n), "numeric_passed": None}
        report["rows"].append(row)
        save(output / "partial.json", report)
        context["stage"] = "numeric_check"
        torch.testing.assert_close(result, refs[arm], rtol=.01, atol=.001)
        torch.testing.assert_close(gates, sources[arm].unsqueeze(0).expand_as(gates), rtol=0, atol=0)
        torch.testing.assert_close(up, saved_up, rtol=0, atol=0)
        torch.testing.assert_close(addend, saved_addend, rtol=0, atol=0)
        row.update(numeric_passed=True, numeric_coverage=plan["numeric_coverage"])
        save(output / "partial.json", report)
    report["summary"] = summarize(report["rows"])
    save(output / "result.json", report)
    print(json.dumps(report["summary"], indent=2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_bundle(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    plan = read("plan.json")
    require(plan == protocol(), "protocol/source mismatch; use pinned source")
    if (directory / "failure.json").exists():
        return {"complete": False, "failure": read("failure.json")}
    env, report = read("environment.json"), read("result.json")
    require(report["case_id"] == plan["case_id"] and report["shape"] == plan["shape"]
        and report["dtype"] == "bf16" and report["input_pool_bytes"] == BUFFERS * 256 * 3072 * 2,
        "case/input mismatch")
    require(report["protocol_id"] == plan["protocol_id"] and report["admission"] == "conditional_dev", "admission mismatch")
    expected_contracts = {str(n): contract_for(env, n) for n in BATCHES}
    require(report["measurement_contracts"] == expected_contracts, "contract mismatch")
    ids = {n: measurement_contract_id(c) for n, c in expected_contracts.items()}
    require(report["measurement_contract_ids"] == ids, "contract identity mismatch")
    require(all(report.get(k) == v for k, v in read("partial.json").items()), "partial/result mismatch")
    require(list(report["captures"]) == [f"n{n}_{a}" for n in BATCHES for a in ARMS], "capture order mismatch")
    for n in BATCHES:
        for a in ARMS:
            c = report["captures"][f"n{n}_{a}"]
            require(c["single_stream"] is True and c["graph_calls"] == n
                and c["retained_outputs"] == "last_final_output_only"
                and c["attempted_pool_config"] == POOL, "capture contract mismatch")
    for r in report["rows"]:
        n = r["graph_calls"]
        require(n in BATCHES and r["measurement_contract_id"] == ids[str(n)], "row contract mismatch")
        require(r["numeric_passed"] is True and r["numeric_coverage"] == plan["numeric_coverage"], "numeric check incomplete")
        require(math.isfinite(r["graph_event_total_s"]) and r["graph_event_total_s"] > 0
            and math.isclose(r["graph_batch_amortized_per_call_us"], r["graph_event_total_s"] * 1e6 / n, rel_tol=1e-12),
            "Event/amortization mismatch")
        require(0 < r["graph_submission_wall_s"] <= r["graph_completion_wall_s"], "invalid wall time")
    summary = summarize(report["rows"])
    require(summary == report["summary"], "summary mismatch")
    require(sum(r["graph_calls"] for r in report["rows"]) == plan["timed_body_calls"], "call count mismatch")
    return {"complete": True, "timed_body_calls": plan["timed_body_calls"],
        "event_intervals": len(report["rows"]), "admission": "conditional_dev",
        "measurement_contract_ids": ids, "summary": summary, "automatic_model_writeback": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out-dir", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--plan-only", action="store_true")
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
        if not torch.cuda.is_available():
            raise RuntimeError("GPU0 required; no CPU/eager fallback")
        env = {"torch": str(torch.__version__), "hip": torch.version.hip,
            "triton": importlib.metadata.version("triton"), "device_model": torch.cuda.get_device_name(0),
            "allocator_config": {n: os.environ.get(n) for n in
                ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}}
        save(args.out_dir / "environment.json", env)
        with torch.no_grad():
            run(torch, F, env, args.out_dir, context, plan)
    except Exception as error:
        write_failure(args.out_dir / "failure.json", error, context, torch)
        raise


if __name__ == "__main__":
    main()
