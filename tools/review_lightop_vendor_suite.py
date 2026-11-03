"""CPU-only review of vendor outputs; never import or execute received code."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/lightop_fusion_catalog_v1/capabilities.json"
MATRIX = ROOT / "data/lightop_fusion_catalog_v1/vendor_jobs.json"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ASSERTION_ONLY = {"test_blockwise_scaled_1d_cast_transpose", "test_fuse_rmsnorm_rope_quant_qkv"}
CORRECTNESS_ONLY = {"test_fuse_rmsnorm_rope_quant_qkv", "test_gemma_fused_add_rmsnorm",
                    "test_scaled_masked_softmax", "test_scaled_upper_traing_masked_softmax",
                    "test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant"}
PRIMARY = {
    "test_fuse_rmsnorm_rope_qkv_no_quant": [],  # Missing API is not in the catalog.
    "test_fused_trans_bmm_trans_benchmark": ["fused_bmm"],
    "test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant": ["split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant"],
    "test_scaled_masked_softmax_dropout": ["FusedScaleMaskSoftmaxDropout"],
    "test_scaled_upper_traing_masked_softmax_dropout": ["FusedScaleMaskSoftmaxDropout"],
}


def digest(path):
    return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def classify(job_id, returncode, stdout, stderr):
    text = ANSI.sub("", stdout)
    missing = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)", stderr)
    false_lines = [i for i, s in enumerate(text.splitlines(), 1) if s.strip() == "False" or re.search(r"\bFAIL(?:ED)?\b", s) or re.search(r"allclose\([^\)]*\):\s*False\s*$", s)]
    if missing:
        return "missing_dependency", missing[1], false_lines
    if "ImportError: cannot import name" in stderr:
        return "vendor_api_mismatch", stderr.splitlines()[-1], false_lines
    if "Fusion mismatch exceeds tolerance" in stderr:
        return "correctness_failed", stderr.splitlines()[-1], false_lines
    if returncode != 0:
        return "execution_failed", stderr.splitlines()[-1] if stderr else "nonzero exit", false_lines
    if job_id == "test_gemma_fused_add_rmsnorm":
        checks = re.findall(r"allclose\(atol=([^,]+), rtol=([^\)]+)\): (True|False)", text)
        loose = [passed for atol, rtol, passed in checks if atol == rtol == "1e-2"]
        strict_failed = any(passed == "False" for atol, rtol, passed in checks if atol == rtol and atol in ("1e-3", "1e-4"))
        if loose and all(passed == "True" for passed in loose) and strict_failed:
            return "tolerance_dependent", "1e-2 allclose passes; 1e-3/1e-4 output checks fail; no timing", false_lines
        if any(passed == "False" for _, _, passed in checks):
            return "vendor_checks_failed", "Gemma allclose check failed", false_lines
    if false_lines:
        return "vendor_checks_failed", "printed False/FAIL despite zero exit", false_lines
    checks = bool(re.search(r"\b(True|PASS|passed|Success)\b", text)) or job_id in ASSERTION_ONLY
    # Quantization script reports zero differences instead of a Boolean assertion.
    if job_id == "test_quantization_cuda":
        differences = re.findall(r"\u5dee\u5f02:\s*(" + NUMBER + r")", text)
        if len(differences) == 8 and all(float(x) == 0 for x in differences):
            return "timed_zero_difference_reported", "eight zero differences printed; no substitute admission threshold", false_lines
        return "requires_review", "quantization difference report is incomplete or nonzero", false_lines
    if not checks:
        return "requires_review", "no explicit validation evidence recognized", false_lines
    return ("vendor_checks_passed_correctness_only" if job_id in CORRECTNESS_ONLY else "vendor_checks_passed_timed"), "vendor checks only; not Canonical Profile admission", false_lines


def timings(job_id, stdout):
    """Keep the unit and context of each vendor line; never pool contracts."""
    records, context = [], None
    for line_no, raw in enumerate(stdout.splitlines(), 1):
        line = ANSI.sub("", raw)
        hidden = re.search(r"hidden_size = (\d+)", line)
        if hidden:
            context = {"hidden_size": int(hidden[1])}
        record = {"job_id": job_id, "stdout_line": line_no, "raw": line}
        scalar = re.search(r"(" + NUMBER + r")\s*(ms|us)\b", line)
        table = re.match(r"^\s*(\([^)]*\))\s+(?:(True|False)\s+)?(" + NUMBER + r")\s+(" + NUMBER + r")\s+\d+\s*$", line)
        rms = re.match(r"^(\d+)\s*\|\s*(\d+)\s*\|\s*(FP16|BF16)\s*\|\s*PASS\s*\|\s*(" + NUMBER + r")", line)
        if line.startswith("===fuse:") or line.startswith("===src:"):
            record.update(kind="loop_wall_seconds", values=[float(line.split(":", 1)[1])], unit="s/990_iterations",
                          boundary="original 1000-iteration loop resets timer at iteration 10; includes forward/backward and module construction")
        elif scalar:
            record.update(kind="vendor_printed_timing", values=[float(m[0]) for m in re.findall(r"(" + NUMBER + r")\s*(ms|us)\b", line)], unit=scalar[2], case_context=context)
        elif table:
            record.update(kind="vendor_mean_std_table", shape=table[1], interleaved=table[2], values=[float(table[3]), float(table[4])], unit="ms", columns=["mean", "std"])
        elif rms:
            record.update(kind="vendor_case_table", case={"tokens": int(rms[1]), "hidden": int(rms[2]), "dtype": rms[3]}, values=[float(rms[4])], unit="us")
        else:
            continue
        records.append(record)
    return records


def selected_jobs(plan, matrix, rows):
    jobs = {j["id"]: j for j in matrix["jobs"]}
    selected = plan["selected_jobs"]
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(jobs):
        raise ValueError("Invalid selected vendor jobs")
    if selected != [name for name in jobs if name in selected] or [r["id"] for r in rows] != selected:
        raise ValueError("Results omit, duplicate or reorder selected jobs")
    return jobs


def review(received, output, archive=None, report_key="lightop_vendor_suite01_review"):
    received, output = Path(received).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Review output must be empty: " + str(output))
    catalog, matrix = read(CATALOG), read(MATRIX)
    plan, identity = read(received / "plan.json"), read(received / "identity.json")
    assert plan["catalog_sha256"] == digest(CATALOG)
    assert plan["matrix_sha256"] == digest(MATRIX)
    assert plan["wheel_sha256"] == catalog["wheel_sha256"]
    assert plan["jobs"] == matrix["jobs"]
    assert identity["version"] == catalog["version"] and identity["op_so_sha256"] == catalog["op_so_sha256"]
    rows = read(received / "results.json")
    jobs = selected_jobs(plan, matrix, rows)
    records, all_timings, verified_sources = [], [], 0
    for tree in [received / "vendor_original", *(received / "jobs" / r["id"] / "vendor" for r in rows)]:
        for relative, expected in matrix["test_tree_sha256"].items():
            assert digest(tree / relative) == expected, str(tree / relative)
            verified_sources += 1
    for row in rows:
        job = jobs[row["id"]]
        directory = received / "jobs" / row["id"]
        assert row == read(directory / "result.json")
        assert row["attempts"] == 1 and row["script_sha256"] == job["source_sha256"]
        stdout = (directory / "stdout.log").read_text(encoding="utf-8", errors="replace")
        stderr = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
        status, reason, false_lines = classify(row["id"], row["returncode"], stdout, stderr)
        measurement = timings(row["id"], stdout)
        record = {"job_id": row["id"], "returncode": row["returncode"], "review_status": status, "reason": reason,
            "primary_capabilities": PRIMARY.get(row["id"], job["capability_references"]),
            "static_source_references": job["capability_references"], "timing_rows": len(measurement),
            "false_check_count": len(false_lines), "false_check_stdout_lines": false_lines,
            "source": job["script"], "source_sha256": job["source_sha256"],
            "stdout_sha256": digest(directory / "stdout.log"), "stderr_sha256": digest(directory / "stderr.log"),
            "elapsed_host_seconds": row["finished_unix"] - row["started_unix"],
            "automatic_admission": False, "kernel_dispatch_certified": False}
        records.append(record)
        all_timings.extend(measurement)
    summary = read(received / "summary.json")
    assert summary["attempted_jobs"] == len(records)
    assert summary["process_completed"] == sum(r["returncode"] == 0 for r in records)
    report = {"scope": "vendor_original_tests_cpu_review_not_profile_admission", "archive_sha256": digest(archive) if archive else None,
        "identity": identity, "jobs": len(records), "process_completed": summary["process_completed"],
        "process_failed": len(records) - summary["process_completed"],
        "review_counts": dict(sorted(Counter(r["review_status"] for r in records).items())),
        "missing_dependency_counts": dict(sorted(Counter(r["reason"] for r in records if r["review_status"] == "missing_dependency").items())),
        "jobs_with_printed_timing": sum(r["timing_rows"] > 0 for r in records), "timing_rows": len(all_timings),
        "verified_original_source_copies": verified_sources, "runtime_collector_source_captured": False,
        "new_accepted_profiles": 0, "priceable_ffn_plans": "0/48", "records": records}
    if len(rows) != len(jobs):
        report.update(selection_scope="selected_subset_followup", baseline_total_jobs=len(jobs))
    overlay = {"source_report": "data/" + report_key + "/report.json", "scope": report["scope"], "capabilities": {}}
    for c in catalog["capabilities"]:
        observations = [{k: r[k] for k in ("job_id", "review_status", "timing_rows", "false_check_count")} for r in records if c["id"] in r["primary_capabilities"]]
        overlay["capabilities"][c["id"]] = {"observations": observations, "in_this_batch": bool(observations),
            "gpu_timing_and_checks_available": any(r["review_status"] in ("vendor_checks_passed_timed", "timed_zero_difference_reported") for r in observations),
            "graph_adapter": False, "automatic_admission": False}
    save(output / "report.json", report)
    save(output / "capability_status.json", overlay)
    with (output / "timings.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in all_timings:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
    with (output / "jobs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["job_id", "returncode", "review_status", "reason", "timing_rows", "false_check_count"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in records)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("received", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--report-key", default="lightop_vendor_suite01_review")
    args = parser.parse_args()
    report = review(args.received, args.output, args.archive, args.report_key)
    print(json.dumps({k: report[k] for k in ("jobs", "process_completed", "process_failed", "review_counts", "missing_dependency_counts", "jobs_with_printed_timing", "timing_rows", "verified_original_source_copies")}))


if __name__ == "__main__":
    main()
