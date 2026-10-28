"""Reproduce light GPU diagnostics and a non-fitted serial-work ablation."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import load_freeze
from execution.audit_gpu_holdout import timing_statistics, write_json
from execution.diagnose_gpu_holdout import expected_transfers, prediction_breakdown
from execution.light_gpu_trace import analyze_trace_frames, assess_trace_timing
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def main():
    base = Path(__file__).resolve().parent
    freeze, _ = load_freeze(base.parent.parent)
    report = json.loads((base / "diagnostic_report.json").read_text())
    environment = json.loads((base / "diagnostic_environment.json").read_text())
    predictions = json.loads((base / "prediction_breakdown.json").read_text())
    records = load_jsonl_records(base / "baseline_samples.jsonl")
    assert report["freeze_id"] == freeze["freeze_id"] and report["trace_mode"] == "light"
    assert environment == report["environment"]
    assert report["environment_id"] == content_id("gpu-three-plan-diagnostic-environment", environment)
    assert predictions == report["predictions"] == json.loads(json.dumps(prediction_breakdown(freeze, report["selection"])))
    selected = {row["role"]: row for row in report["selection"]["plans"]}
    expected_slots = {(role, phase, repeat) for role in selected for phase, repeats in (("before_trace", range(3)), ("after_trace", (3,))) for repeat in repeats}
    assert len(records) == len(expected_slots) == 12
    assert {(row["role"], row["phase"], row["repeat"]) for row in records} == expected_slots
    assert len({row["record_id"] for row in records}) == 12
    for row in records:
        assert row["record_id"] == content_id("gpu-three-plan-diagnostic-baseline", {key: value for key, value in row.items() if key != "record_id"})
        assert row["freeze_id"] == freeze["freeze_id"] and row["environment_id"] == report["environment_id"]
        assert row["model_id"] == report["model_id"] and row["plan_id"] == selected[row["role"]]["plan_id"]
        assert len(row["samples_s"]) == 20 and all(math.isfinite(value) and value > 0 for value in row["samples_s"])
        assert row["median_s"] == statistics.median(row["samples_s"]) and row["correctness"]["pass"]
        assert row["correctness"]["max_abs_error"] == 0
    before = {row["role"]: row for row in report["baselines"]}
    after = {row["role"]: row for row in report["post_trace_baselines"]}
    for role in selected:
        entries = [row for row in records if row["role"] == role and row["phase"] == "before_trace"]
        assert timing_statistics([value for row in entries for value in row["samples_s"]]) == before[role]["pooled"]
        assert [{"repeat": row["repeat"], **timing_statistics(row["samples_s"])} for row in entries] == before[role]["rounds"]
        entry = next(row for row in records if row["role"] == role and row["phase"] == "after_trace")
        assert all(after[role][key] == value for key, value in timing_statistics(entry["samples_s"]).items())
    workload = next(row for row in freeze["workloads"] if row["model_id"] == report["model_id"])
    graph, hardware = graph_from_dict(workload["graph"]), hardware_from_dict(freeze["hardware"])
    trace_summaries = []
    for trace in report["traces"]:
        role = trace["role"]
        assert trace["correctness"]["pass"] and trace["correctness"]["max_abs_error"] == 0
        raw = json.loads((base / trace["trace_file"]).read_text())
        parsed = json.loads(json.dumps(analyze_trace_frames(raw["traceEvents"])))
        assert all(trace[key] == value for key, value in parsed.items())
        comparison = assess_trace_timing(trace, before[role]["pooled"]["median_ms"], after[role]["median_ms"])
        assert comparison == trace["timing_comparison"] and comparison["timing_target_pass"]
        assert len(parsed["frames"]) == 3 and parsed["gpu_event_attribution_complete"]
        assert all(trace["activity_checks"].values())
        placed = PlacedOpGraph(graph, hardware, plan_from_dict(workload["candidate_plans"][selected[role]["plan_id"]]))
        expected = expected_transfers(placed)
        pair = "k_rope_q_proj" if role == "intermediate" else "gate_up"
        frames = []
        for frame in parsed["frames"]:
            assert frame["physical_kernels"] == 37
            assert frame["gpu_memcpy_events"] == len(expected)
            assert frame["gpu_memcpy_bytes"] == sum(row["payload_bytes"] for row in expected)
            copies = [row for row in frame["gpu_events"] if row["kind"] == "gpu_memcpy"]
            assert Counter(row["bytes"] for row in copies) == Counter(row["payload_bytes"] for row in expected)
            nodes = frame["node_gpu_intervals"]
            first, second = (("L000.self_attention.k_rope", "L000.self_attention.q_proj") if pair == "k_rope_q_proj" else
                ("L000.mlp.gate_proj", "L000.mlp.up_proj"))
            frames.append({"label": frame["label"], "physical_kernels": frame["physical_kernels"],
                "copy_calls": len(copies), "copy_bytes": frame["gpu_memcpy_bytes"],
                "gpu_dma_duration_sum_ms": frame["gpu_memcpy_duration_sum_ms"],
                "gpu0_gpu1_kernel_overlap_ms": frame["gpu0_gpu1_kernel_overlap_ms"],
                "relevant_pair": pair, "branch_overlap": frame["branch_overlap"][pair],
                "first_branch_end_ms": nodes[first]["end_ms"], "second_branch_start_ms": nodes[second]["start_ms"],
                "second_starts_after_first": nodes[second]["start_ms"] >= nodes[first]["end_ms"],
                "silu_end_ms": nodes["L000.mlp.silu"]["end_ms"],
                "node_gpu_intervals": nodes, "dma_events": copies})
        trace_summaries.append({"role": role, "timing_comparison": comparison,
            "traced_samples_ms": [value * 1000 for value in trace["traced_samples_s"]],
            "frames": frames})
    assert report["all_light_timing_targets_pass"] and report["all_light_activity_checks_pass"] and report["status"] == "complete"
    ref_before, ref_after = before["single_gpu_reference"]["pooled"]["median_ms"], after["single_gpu_reference"]["median_ms"]
    baselines = [{"role": role, "before": before[role]["pooled"], "after": after[role],
        "before_relative_to_single_gpu_pct": (before[role]["pooled"]["median_ms"] / ref_before - 1) * 100,
        "after_relative_to_single_gpu_pct": (after[role]["median_ms"] / ref_after - 1) * 100} for role in selected]
    order = ["single_gpu_reference", "intermediate", "predicted_top1"]
    stable_order = all(sorted(selected, key=lambda role: next(row["median_s"] for row in records if row["role"] == role and row["repeat"] == repeat)) == order for repeat in range(4))
    diagnostic = []
    for row in predictions["plans"]:
        cost = row["frozen_prediction"]
        serial_sum_ms = (cost["computation_time"] + cost["communication_time"]) * 1000
        measured = before[row["role"]]["pooled"]["median_ms"]
        diagnostic.append({"role": row["role"], "original_dag_ms": cost["makespan"] * 1000,
            "serial_work_sum_ms": serial_sum_ms, "measured_before_ms": measured,
            "original_dag_ape": abs(cost["makespan"] * 1000 - measured) / measured,
            "serial_work_sum_ape": abs(serial_sum_ms - measured) / measured})
    output = {"freeze_id": freeze["freeze_id"], "checks_pass": True, "baseline_records": len(records),
        "before_samples": 180, "after_samples": 60, "captured_forwards": sum(len(row["frames"]) for row in trace_summaries),
        "role_order_same_in_all_four_uninstrumented_blocks": stable_order,
        "all_predeclared_timing_targets_pass": True, "all_activity_checks_pass": True,
        "baseline_samples_above_twice_candidate_phase_median": sum(row["before"]["samples_above_twice_median"] + row["after"]["samples_above_twice_median"] for row in baselines),
        "baselines": baselines, "traces": trace_summaries,
        "serial_work_sum_diagnostic": {"rows": diagnostic,
            "original_dag_order": sorted(selected, key=lambda role: next(row["original_dag_ms"] for row in diagnostic if row["role"] == role)),
            "serial_work_sum_order": sorted(selected, key=lambda role: next(row["serial_work_sum_ms"] for row in diagnostic if row["role"] == role)),
            "scope": "Post-hoc arithmetic ablation only: sum the original computation and transfer service costs, no fitting. Not a new validated execution model or a bound on actual latency; host overhead, stream dependencies and partial overlap are not modeled."},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(base.iterdir()) if path.name in
            {"diagnostic_report.json", "diagnostic_environment.json", "prediction_breakdown.json", "baseline_samples.jsonl", *(row["trace_file"] for row in report["traces"])}},
        "limitations": ["One already-exposed workload, three post-hoc selected plans; not independent milestone-A validation or SROP benefit.",
            "15% median perturbation and 10% baseline-shift checks were frozen before collection; passing them does not prove zero observer effect. The first Top1 trace sample exceeds 15% and is retained.",
            "ROCtracer duplicate-flow warning persists. Raw physical events are matched without flow arrows or GPU annotation spans; causal fidelity and completeness cannot be guaranteed merely by status=complete.",
            "No exact causal decomposition of baseline latency is inferred by adding nested scopes or subtracting instrumented times."]}
    write_json(base / "analysis.json", output)
    print(f"checks_pass=True baseline_records=12 samples=240 captured_forwards=9 stable_order={stable_order}")
    for row in baselines:
        t = next(trace for trace in trace_summaries if trace["role"] == row["role"])["timing_comparison"]
        print(row["role"], "before", row["before"]["median_ms"], "after", row["after"]["median_ms"],
            "regret_before_pct", row["before_relative_to_single_gpu_pct"], "trace_ratio", t["trace_over_before_ratio"],
            "baseline_shift_pct", t["after_over_before_change_fraction"] * 100)
    print(json.dumps(output["serial_work_sum_diagnostic"]))


if __name__ == "__main__":
    main()
