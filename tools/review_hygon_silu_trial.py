"""CPU review of the exploratory LightOP receipt; never execute returned code."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import zipfile

from calibration.hygon_silu_trial import ROOT, ARMS, SCOPE, save


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def kernel_role(name):
    if "act_and_mul_kernel_opt1" in name:
        return "lightop_fused"
    if "elementwise_kernel_manual_unroll" in name:
        if "silu_kernel" in name:
            return "native_silu"
        if "BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16" in name:
            return "native_mul"
    return "fixture_or_validation"


def profile_evidence(directory, arm):
    directory = Path(directory)
    databases, summaries = list(directory.glob("*.db")), list(directory.glob("*.hipkernel.csv"))
    if len(databases) != 1 or len(summaries) != 1:
        raise ValueError("expected an actual hipprof database and kernel summary")
    with sqlite3.connect("file:" + databases[0].as_posix() + "?mode=ro", uri=True) as db:
        table, = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'HIPOPS_%'")]
        names = {(pid, str(sid)): name for pid, sid, name in db.execute("SELECT PID,STR_ID,STR_NAME FROM STR_TABLE WHERE TYPE=6")}
        quoted = '"' + table.replace('"', '""') + '"'
        rows = db.execute("SELECT BeginNs,EndNs,dev_id,queue_id,Name,pid,_Index,DurationNs FROM " + quoted + " ORDER BY BeginNs,_Index").fetchall()
    timeline, counts, durations = [], Counter(), defaultdict(int)
    for i, (begin, end, device, queue, sid, pid, correlation, duration) in enumerate(rows):
        if device != 0 or duration <= 0 or end - begin != duration:
            raise ValueError("invalid device or raw kernel duration")
        name = names[(pid, str(sid))]
        counts[name] += 1; durations[name] += duration
        timeline.append({"ordinal": i, "begin_ns": begin, "end_ns": end, "duration_ns": duration,
            "device": device, "queue": str(queue), "correlation_index": correlation,
            "role": kernel_role(name), "kernel_name": name})
    with summaries[0].open(encoding="utf-8-sig", newline="") as stream:
        aggregates = [r for r in csv.DictReader(stream) if r["Name"] != "Total"]
    if {r["Name"] for r in aggregates} != set(counts) or any(
            counts[r["Name"]] != int(r["Calls"]) or durations[r["Name"]] != int(r["TotalDurationNs"]) for r in aggregates):
        raise ValueError("CSV differs from raw database kernel events")
    roles = Counter(r["role"] for r in timeline if r["role"] != "fixture_or_validation")
    expected = {"lightop_fused": 11} if arm == "lightop_packed" else {"native_silu": 11, "native_mul": 11}
    if dict(roles) != expected:
        raise ValueError("profiled optimized/native dispatch counts differ from eleven operator calls")
    if arm == "native_packed" and any(
            i + 1 >= len(timeline) or timeline[i + 1]["role"] != "native_mul"
            for i, row in enumerate(timeline) if row["role"] == "native_silu"):
        raise ValueError("Native SiLU/Mul pairs are not adjacent")
    return {"all_process_kernel_events": len(timeline), "target_counts": dict(roles),
        "database_csv_counts_and_durations_match": True,
        "evidence_scope": "separate profiling process, not the hundred timed invocations"}, timeline


def benchmark_evidence(value):
    if value["scope"] != SCOPE or value["complete"] is not True or value["profiler_active"] or value["automatic_admission"] or value["independent_batches_per_arm"] != 1 or set(value["arms"]) != set(ARMS):
        raise ValueError("benchmark completeness or scope mismatch")
    for row in value["arms"].values():
        if row["calls"] != 100 or row["warmup_calls"] != 10 or not row["numeric_ok"] or not row["input_bytes_unchanged"]:
            raise ValueError("wrong call counts or failed correctness/input checks")
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("gpu_batch_ms", "gpu_us_per_call", "synchronized_wall_us_per_call")) or row["gpu_us_per_call"] != row["gpu_batch_ms"] * 1000 / 100:
            raise ValueError("benchmark timing unit conversion mismatch")
    comparisons = {}
    for boundary, native, fused in (("packed", "native_packed", "lightop_packed"), ("split", "native_split", "lightop_split_with_pack")):
        a, b = value["arms"][native]["gpu_us_per_call"], value["arms"][fused]["gpu_us_per_call"]
        comparisons[boundary] = {"native_us": a, "lightop_path_us": b,
            "lightop_minus_native_us": b - a, "relative_duration_change_percent": 100 * (b - a) / a,
            "native_over_lightop": a / b, "descriptive_single_batch_only": True}
    return comparisons


def review(root, wheel):
    root, wheel = Path(root), Path(wheel)
    originals = inventory(root)
    frozen, summary, inspection = [read(root / n) for n in ("plan.json", "summary.json", "inspection/inspection.json")]
    if frozen != read(ROOT / "data/hygon_silu_trial_v2/plan.json"):
        raise ValueError("receipt differs from frozen v2 plan")
    for name, expected in frozen["source_sha256"].items():
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", "228ee20:" + name])
        if hashlib.sha256(data).hexdigest() != expected: raise ValueError("collector source mismatch")
    manifest = frozen["wheel_manifest"]
    if sha(wheel) != manifest["sha256"]: raise ValueError("wheel digest mismatch")
    with zipfile.ZipFile(wheel) as archive:
        for item in inspection["package_files"]:
            relative = item["path"].split("/site-packages/", 1)[1]
            if hashlib.sha256(archive.read(relative)).hexdigest() != item["sha256"]:
                raise ValueError("recorded installed file differs from pinned wheel")
    if inspection["packages"]["lightop"] != manifest["version"] or not inspection["lightop"]["required_export_callable"]:
        raise ValueError("wrong package version or export")
    expected_jobs = {"inspection", "benchmark", "hipprof_native_packed", "hipprof_lightop_packed"}
    if set(summary["jobs"]) != expected_jobs or not summary["pipeline_passed"] or summary["automatic_admission"] or summary["followup_matrix_open"] or any(j["returncode"] != 0 or j["attempts"] != 1 for j in summary["jobs"].values()):
        raise ValueError("wrong pipeline scope or process attempts")
    benchmark = read(root / "benchmark/benchmark.json")
    comparisons = benchmark_evidence(benchmark)
    logged = [json.loads(line) for line in (root / "benchmark/worker.log").read_text(encoding="utf-8").splitlines() if line.startswith('{"arm":')]
    if [r["arm"] for r in logged] != list(ARMS) or any(
            {k: v for k, v in row.items() if k != "arm"} != benchmark["arms"][row["arm"]] for row in logged):
        raise ValueError("benchmark JSON differs from its chronological worker log")
    identity = read(root / "benchmark/runtime_identity.json")
    if (identity["binding"] != frozen["binding"].split("(", 1)[0] or identity["device_index"] != 0
            or identity["torch_distribution"] != inspection["packages"]["torch"]
            or identity["lightop_distribution"] != manifest["version"] or identity["hip"] != inspection["torch"]["hip"]):
        raise ValueError("benchmark runtime differs from inspected implementation/environment")
    profiles, timeline = {}, []
    for arm in ("native_packed", "lightop_packed"):
        directory = root / ("hipprof_" + arm)
        if read(directory / "runtime_identity.json") != identity: raise ValueError("runtime changed within the trial")
        worker = read(directory / "profile_worker.json")
        if worker["scope"] != SCOPE or worker["arm"] != arm or worker["complete"] is not True or worker["operator_calls"] != 11 or worker["timing_labels_emitted"]:
            raise ValueError("profile worker semantics changed")
        profiles[arm], rows = profile_evidence(directory, arm)
        timeline.extend({"arm": arm, **r} for r in rows)
    if originals != inventory(root): raise ValueError("review changed original files")
    archive = root.parent / (root.name + ".tar.gz")
    report = {"scope": "CPU review of exploratory LightOP evidence, not Profile admission",
        "source_ref": "228ee20", "archive_sha256": sha(archive), "source_checks": len(frozen["source_sha256"]),
        "recorded_package_hashes_matching_wheel": len(inspection["package_files"]),
        "identity": identity, "benchmark": benchmark, "comparisons": comparisons, "profiles": profiles,
        "all_processes_completed_once": True, "installation_report_supplied": False,
        "profile_dispatch_observed": True, "timed_invocations_individually_traced": False,
        "pftrace_preserved_not_decoded": True, "independent_batches_per_arm": 1,
        "new_accepted_profiles": 0, "priceable_plans": 0, "legal_plans": 48,
        "automatic_followup": False, "originals_unchanged": True, "raw_file_sha256": originals}
    return report, timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, timeline = review(args.root, args.wheel)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "report.json", report)
    with (args.output / "profile_kernel_events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timeline[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(timeline)
    print(json.dumps(report["comparisons"]))


if __name__ == "__main__":
    main()
