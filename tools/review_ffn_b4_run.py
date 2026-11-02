"""CPU review of a returned B4 bundle; never promotes measurements to accepted."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from calibration.ffn_b4_policy import formal_quality_review, smoke_pass
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import measurement_contract_id

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
                shaped = [{"attempt": s["attempt"], "numeric_ok": s["numeric_ok"], "capture_ok": s["capture_ok"],
                           "path_ok": s.get("path_ok", read(result_path).get("path_verified", False)),
                           "arm_us": {"body": s["duration_us"]}, "anchors": s["anchors"]}
                          for s in samples]
                checked = formal_quality_review(shaped, ("body",), tuple(samples[0]["anchors"]))
                original = read(result_path)["quality"]
                assert key_id(checked) == key_id(original)
                item.update(recorded_status=read(result_path)["status"], quality_recomputed=True,
                            quality_status=checked["status"], invalid_attempts=checked.get("invalid_attempts", []),
                            median_event_us_per_call=statistics.median(s["duration_us"] for s in samples))
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
    return {"scope": "returned B4 audit; separate profiler replay is not the same timing sample",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None,
            "source_ref": source_ref, "source_checks": source_checks,
            "smoke_status": smoke["status"], "smoke_passed_cells": len(smoke_rows), "smoke": smoke_rows,
            "formal_expected_cells": len(rows), "formal_completed_cells": completed,
            "formal_prepared_without_samples": sum(r["identity_written"] and not r["attempts"] for r in rows),
            "formal_unstarted_cells": sum(not r["identity_written"] for r in rows),
            "grid_written": master is not None, "grid_quality_passed": sum(r.get("grid_status")=="quality_passed" for r in rows),
            "grid_quality_failed": sum(r.get("grid_status")=="quality_failed" for r in rows),
            "groups": groups, "formal_rows": rows,
            "last_published_case": "diagnostic/down0-copy524288-gate1",
            "crash_localization": "capture_body_returned -> torch.cuda.CUDAGraph.capture_end SIGSEGV" if master else "last identity precedes capture; native stack unavailable",
            "timing_interpretation": "large event/trace discrepancy requires investigation; profiler overhead and boundary cause unverified",
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
    print(json.dumps({k: v for k,v in result.items() if k not in ("smoke", "formal_rows", "raw_file_sha256")}))


if __name__ == "__main__":
    main()
