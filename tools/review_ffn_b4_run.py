"""CPU review of a returned B4 bundle; never promotes measurements to accepted."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from calibration.ffn_b4_policy import formal_quality_review, smoke_pass
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import measurement_contract_id

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sample_binding(record, samples, identity):
    assert key_id(record["identity"]) == record["identity_hash"] == identity["identity_hash"]
    assert measurement_contract_id(record["measurement_contract"]) == record["measurement_contract_id"]
    assert key_id(record["samples"]) == key_id(samples)
    for index, sample in enumerate(samples):
        assert sample["attempt"] == index and sample["numeric_ok"] and sample["capture_ok"]
        assert all(value is True for value in sample["invariants"].values())
        if "raw_timing" in sample:
            raw = sample["raw_timing"]
            assert math.isclose(sample["duration_us"], raw["elapsed_ms"]*1000/50, rel_tol=1e-12)
            boundary = raw["event_boundary"]
            assert boundary["device"] == boundary["start_device"] == boundary["end_device"]
            assert boundary["calls_per_replay"] == 50
        if record.get("path_evidence_phase") == "after_all_group_timing":
            assert sample["path_evidence_state"] == "pending_post_timing_trace"
            assert record["path_verified"]


def review(root, source_ref):
    root = Path(root)
    archive = root.parent/(root.name+".tar.gz")
    frozen = read(root/"plan.json")
    source_checks = {}
    for name, expected in frozen["source_sha256"].items():
        blob = subprocess.run(["git", "-C", str(ROOT), "show", source_ref+":"+name], capture_output=True, check=True).stdout
        source_checks[name] = hashlib.sha256(blob).hexdigest() == expected
    if not all(source_checks.values()):
        raise ValueError("returned collector differs from declared source revision")
    smoke = read(root/"smoke/result.json")
    smoke_rows = []
    for row in smoke["cells"]:
        directory = root/"smoke"/row["row_id"].replace("/", "_").replace(">", "to")
        identity = read(directory/"identity.json")
        assert key_id(identity["identity"]) == row["identity_hash"] == identity["identity_hash"]
        assert measurement_contract_id(identity["contract"]) == row["measurement_contract_id"]
        assert smoke_pass(row)
        smoke_rows.append({"row_id": row["row_id"], "passed": True,
                           "invariants": row["invariant_checks"],
                           "profiled_kernel_count": len(row["path_evidence"].get("kernel_names", [])),
                           "copy_trace_observed": row["path_evidence"].get("device_copy_trace_observed")})
    master = read(root/"formal/grid.json") if (root/"formal/grid.json").exists() else None
    rows = []
    row_defs = frozen["matrix"]["core_rows"] + frozen["matrix"]["diagnostics"]
    for row in row_defs:
        master_row = master["cells"][row["row_id"]] if master else {}
        base = root/"formal"
        if master_row.get("worker_evidence"):
            base = base/master_row["worker_evidence"]/"formal"
        directory = base/row["row_id"].replace("/", "_").replace(">", "to")
        result_path, sample_path = directory/"result.json", directory/"samples.json"
        item = {"row_id": row["row_id"], "identity_written": (directory/"identity.json").exists(),
                "result_written": result_path.exists(), "attempts": 0, "admitted": False,
                "grid_status": master_row.get("status"), "execution_status": master_row.get("execution_status")}
        if sample_path.exists():
            samples = read(sample_path)
            item["attempts"] = len(samples)
            if len(samples) == 7:
                result = read(result_path)
                verify_sample_binding(result, samples, read(directory/"identity.json"))
                if master:
                    assert key_id(master_row["samples"]) == key_id(samples)
                shaped = [{"attempt": s["attempt"], "numeric_ok": s["numeric_ok"], "capture_ok": s["capture_ok"],
                           "path_ok": s.get("path_ok", read(result_path).get("path_verified", False)),
                           "arm_us": {"body": s["duration_us"]}, "anchors": s["anchors"]}
                          for s in samples]
                checked = formal_quality_review(shaped, ("body",), tuple(samples[0]["anchors"]))
                original = read(result_path)["quality"]
                assert key_id(checked) == key_id(original)
                item.update(recorded_status=read(result_path)["status"], quality_recomputed=True,
                            sample_binding_verified=True,
                            quality_status=checked["status"], invalid_attempts=checked.get("invalid_attempts", []),
                            median_event_us_per_call=statistics.median(s["duration_us"] for s in samples))
                item["invalid_anchor_details"] = [{"attempt": index, "anchors": {
                    gpu: {**a, "relative_drift": abs(a["after_us"]-a["before_us"])/a["before_us"]}
                    for gpu,a in samples[index]["anchors"].items()}} for index in checked.get("invalid_attempts", [])]
                if master:
                    assert master_row["status"] == item["recorded_status"]
                warm = read(result_path).get("warmup_diagnostic_timings", [])
                if warm:
                    warm_ms = statistics.median(w["elapsed_ms"] for w in warm)
                    formal_ms = statistics.median(s["raw_timing"]["elapsed_ms"] for s in samples)
                    if read(result_path).get("path_evidence_phase") == "after_all_group_timing":
                        item.update(warmup_median_ms=warm_ms, formal_median_ms=formal_ms,
                                    warmup_over_formal_event_ratio=warm_ms/formal_ms, trace_phase="after_all_group_timing")
                    else:
                        item.update(pre_profiler_warmup_median_ms=warm_ms, post_profiler_formal_median_ms=formal_ms,
                                    pre_over_post_event_ratio=warm_ms/formal_ms)
                    item["raw_event_devices"] = sorted({s["raw_timing"]["event_boundary"]["device"] for s in samples})
                events = read(directory/"replay_trace.json")["traceEvents"]
                kernels = [e for e in events if e.get("cat") == "kernel"]
                if kernels:
                    span = max(e["ts"]+e["dur"] for e in kernels)-min(e["ts"] for e in kernels)
                    item.update(separate_profiled_replay_kernel_count=len(kernels),
                                separate_profiled_replay_span_us=span,
                                event_median_batch_us=item["median_event_us_per_call"]*50,
                                trace_span_over_event_batch=span/(item["median_event_us_per_call"]*50),
                                timing_boundary_review_required=True)
        rows.append(item)
    completed = sum(r["attempts"] == 7 and r["result_written"] for r in rows)
    groups = []
    if master:
        assert set(master["cells"]) == {r["row_id"] for r in row_defs} and master["recorded_cells"] == len(row_defs)
        for group in master["groups"]:
            item = {k:v for k,v in group.items() if k != "command"}
            if group.get("returncode"):
                directory = root/"formal/groups"/group["group_id"].replace("/", "_").replace(">", "to")
                item["last_stage"] = read(directory/"last_stage.json")
                log = (directory/"worker.log").read_text(encoding="utf-8")
                item["python_fault_stack"] = log.split("Extension modules:")[0].strip()
            groups.append(item)
    crashes = [g for g in groups if g.get("returncode")]
    post_trace = [r for r in rows if r.get("trace_phase") == "after_all_group_timing"]
    qualified = [r for r in rows if r.get("recorded_status") == "quality_passed"]
    eligible_ids = {r["row_id"] for r in qualified}
    b1_root = ROOT/"data/ffn_placement_b1"
    t1, t2 = read(b1_root/"T1_legal_placements.json"), read(b1_root/"T2_cross_device_edges.json")
    hypothetical = []
    for placement in t1["plans"]:
        required = set()
        for group in placement["groups"]:
            name = "silu_mul/fused" if len(group["node_ids"]) == 2 else group["node_ids"][0].rsplit(".", 1)[1]
            required.add(f"{name}@gpu{group['device']}")
        for edge in t2["rows"]:
            if edge["plan_id"] == placement["plan_id"] and edge["cross_device"]:
                required.add(f"copy/{edge['tensor_bytes']}/{edge['source_gpu']}->{edge['destination_gpu']}")
        hypothetical.append({"plan_id": placement["plan_id"], "missing_quality_passed_rows": sorted(required-eligible_ids)})
    comparison = None
    if master and all(master["cells"].get(n, {}).get("status") == "quality_passed" for n in
                      ("silu@gpu1", "gated_mul@gpu1", "silu_mul/native@gpu1", "silu_mul/fused@gpu1")):
        silu, mul, native, fused = [master["cells"][n] for n in
                                   ("silu@gpu1", "gated_mul@gpu1", "silu_mul/native@gpu1", "silu_mul/fused@gpu1")]
        assert native["pair_key_id"] == fused["pair_key_id"]
        assert native["measurement_contract_id"] == fused["measurement_contract_id"]
        n, f = (statistics.median(s["duration_us"] for s in r["samples"]) for r in (native, fused))
        delta = [s["duration_us"]+m["duration_us"]-chain["duration_us"] for s,m,chain in
                 zip(silu["samples"],mul["samples"],native["samples"])]
        comparison = {"scope": "GPU1 local paired evidence only; not formal admission or FFN/link validation",
                      "native_median_us": n, "fused_median_us": f, "saving_fraction": (n-f)/n,
                      "delta_add_definition": "SiLU + Mul - Native chain, matched rounds",
                      "delta_add_rounds_us": delta, "delta_add_median_us": statistics.median(delta)}
    return {"scope": "returned B4 audit; separate profiler replay is not the same timing sample",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None,
            "source_ref": source_ref, "source_checks": source_checks,
            "smoke_status": smoke["status"], "smoke_passed_cells": len(smoke_rows), "smoke": smoke_rows,
            "formal_expected_cells": len(rows), "formal_completed_cells": completed,
            "formal_prepared_without_samples": sum(r["identity_written"] and not r["attempts"] for r in rows),
            "formal_unstarted_cells": sum(not r["identity_written"] for r in rows),
            "grid_written": master is not None, "grid_quality_passed": sum(r.get("grid_status")=="quality_passed" for r in rows),
            "grid_quality_failed": sum(r.get("grid_status")=="quality_failed" for r in rows),
            "blocked_cells": sum(r.get("execution_status")=="blocked_backend_capture" for r in rows),
            "anchor_failed_cells": sum(r.get("quality_status")=="quality_failed_invalid_round" for r in rows),
            "worker_crashes": len(crashes),
            "groups": groups, "formal_rows": rows,
            "last_published_case": crashes[-1]["last_stage"]["row_id"] if crashes else None,
            "crash_localization": "see failed worker stack" if crashes else (None if master else "no grid; crash stage not instrumented"),
            "warmup_over_formal_range": [min(r["warmup_over_formal_event_ratio"] for r in post_trace),
                                          max(r["warmup_over_formal_event_ratio"] for r in post_trace)] if post_trace else None,
            "timing_interpretation": "post-timing Trace; warmup/formal consistency checked, no automatic admission" if post_trace else "pre-timing profiling state requires separate review",
            "gpu1_local_comparison": comparison,
            "hypothetical_quality_only_plan_availability": {
                "scope": "availability only; not admission, layout/route proof or chain-additivity validation",
                "legal_plans": len(hypothetical), "complete_plans": sum(not p["missing_quality_passed_rows"] for p in hypothetical),
                "b1_source_sha256": {n: hashlib.sha256((b1_root/n).read_bytes()).hexdigest() for n in
                                      ("T1_legal_placements.json", "T2_cross_device_edges.json")}, "plans": hypothetical},
            "cpu_admission_candidates": [{"row_id": r["row_id"], "quality_passed": True,
                                          "canonical_kernel_key_exported": False,
                                          "strict_query_available": False, "admitted": False} for r in qualified],
            "new_accepted_profiles": 0, "search_cost_items_added": 0,
            "raw_file_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.rglob("*")) if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-ref", default="86f56ba")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.source_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k,v in result.items() if k not in ("smoke", "formal_rows", "raw_file_sha256",
                                                               "hypothetical_quality_only_plan_availability")}))


if __name__ == "__main__":
    main()
