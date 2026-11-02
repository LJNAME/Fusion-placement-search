"""CPU review of archived pilot evidence; never imports returned code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from calibration.ffn_anchor_control import ROOT, read, save, review_cell, pilot_plan, PROTOCOL
from calibration.ffn_anchor_report import build_report, render_report
from cost_model.fusion_kernel_profile import key_id


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_cell(rounds, preparation):
    """Describe position and gaps, without discarding samples or inferring rates."""
    rows, pre_first_max, telemetry, windows = [], [], [], []
    previous_end = preparation["anchor_warmup_end"]
    for attempt in rounds:
        pre = attempt["pre"]
        if len(pre) == 3:
            values = [s["batch_ms"] * 20 for s in pre]
            pre_first_max.append(values[0] > max(values[1:]))
        windows.append((attempt["post"][-1]["validation_end"] - pre[0]["reset_begin"]) / 1e6)
        for label in ("telemetry_before", "telemetry_after"):
            telemetry.extend(attempt[label].values())
        for phase in ("pre", "target", "post"):
            samples = [attempt["target"]] if phase == "target" else attempt[phase]
            for index, sample in enumerate(samples):
                rows.append({"attempt": attempt["attempt"], "phase": phase, "repeat": index,
                    "event_us_per_call": sample["batch_ms"] * 20,
                    "gap_from_previous_validation_or_anchor_warmup_ms": (sample["timing_begin"] - previous_end) / 1e6,
                    **{k: sample[k] for k in ("reset_begin", "timing_begin", "event_wait_end", "all_device_sync_end", "validation_end")}})
                previous_end = sample["validation_end"]
    return {"samples": rows, "r3_pre_first_strict_max": {"observed": sum(pre_first_max), "endpoints": len(pre_first_max)},
            "target_preparation_s": (preparation["case_preparation_end"] - preparation["case_preparation_begin"]) / 1e9,
            "window_median_ms": statistics.median(windows),
            "telemetry": {"observations": len(telemetry), "available": sum(t["value"] is not None for t in telemetry),
                          "errors": sorted({t["error"] for t in telemetry if t["error"]})}}


def review(root, source_ref="06af153"):
    root = Path(root)
    originals = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    plan, summary = read(root / "plan.json"), read(root / "summary.json")
    checks = {}
    for name, expected in plan["source_sha256"].items():
        blob = subprocess.run(["git", "-C", str(ROOT), "show", source_ref + ":" + name],
                              check=True, capture_output=True).stdout
        checks[name] = hashlib.sha256(blob).hexdigest() == expected
    if not all(checks.values()) or plan != pilot_plan() or key_id({k:v for k,v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("archive source/plan differs from reviewed collector")
    if summary["plan_id"] != plan["plan_id"] or set(summary["cells"]) != {c["id"] for c in PROTOCOL["cells"]}:
        raise ValueError("pilot grid differs from fixed plan")
    cells, csv_rows, signatures, observations, all_times = {}, [], [], [], []
    for cell in PROTOCOL["cells"]:
        name, directory = cell["id"], root / cell["id"]
        if read(directory / "parent_plan.json") != plan: raise ValueError("worker source plan mismatch")
        checked = review_cell(directory, cell)
        record = summary["cells"][name]
        if any(record.get(k) != v for k,v in checked.items()) or record["status"] != checked["quality_status"]:
            raise ValueError("worker summary differs from original evidence")
        if record["worker_attempts"] != 1 or record["retries"] != 0 or record["returncode"] != 0:
            raise ValueError("pilot did not complete once per cell")
        signatures.append(checked["runtime_signature"])
        result, samples = read(directory / "result.json"), read(directory / "samples.json")
        preparation = read(directory / "preparation.json")
        if preparation["context"] != result["context"]: raise ValueError("preparation timestamps changed")
        analysis = analyze_cell(samples, result["context"])
        csv_rows.extend({"cell": name, **r} for r in analysis.pop("samples"))
        identity = read(directory / "identity.json")["identity"]["target"]
        observations.append(identity["devices"]["0"])
        warmup = preparation["warmup_diagnostic_timings"]
        if len(warmup) != 10: raise ValueError("target warmup count changed")
        cells[name] = {"pipeline_complete": True, "quality": result["quality"], **analysis,
            "warmup_over_target_median": statistics.median(s["elapsed_ms"]*20 for s in warmup)/result["quality"]["target_median_us"],
            "trace_kernel_counts": {k: len(v["kernel_names"]) for k,v in result["path_evidence"].items()},
            "identity_hash": result["identity_hash"], "measurement_contract_id": result["measurement_contract_id"]}
        all_times.extend([result["context"]["anchor_preparation_begin"], samples[-1]["post"][-1]["validation_end"]])
    if len(set(signatures)) != 1 or all_times != sorted(all_times): raise ValueError("runtime or chronological case order differs")
    if summary["pilot_passed"] is not True or summary["r3_quality_gate_passed"] != all(cells[c]["quality"]["status"] == "quality_passed" for c in ("up_r3", "down_r3")):
        raise ValueError("pilot completeness/quality result mismatch")
    if summary["followup_matrix_open"] or summary["automatic_admission"] or summary["formal_profile"]:
        raise ValueError("pilot escaped its diagnostic scope")
    rebuilt = build_report(root, summary["cells"])
    if rebuilt != read(root / "report.json") or render_report(rebuilt) != (root / "report.md").read_text(encoding="utf-8"):
        raise ValueError("report differs from raw evidence")
    if originals != {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}:
        raise ValueError("review changed original evidence")
    anchor_rows = [r for r in csv_rows if r["phase"] != "target"]
    comparisons = rebuilt["comparisons"]
    for pair in comparisons.values():
        pair["target_R3_minus_R1_percent"] = 100 * pair["target_R3_minus_R1_us"] / pair["target_R1_median_us"]
        for endpoint in pair["anchor_endpoints"].values():
            first, third = endpoint["R1_median_of_seven_single_readings_us"], endpoint["R3_median_of_seven_endpoint_medians_us"]
            endpoint["R3_minus_R1_percent"] = 100 * (third - first) / first
    archive = root.parent / (root.name + ".tar.gz")
    old = read(ROOT / "data/ffn_b4_cpu_preparation_v1/profiles.json")
    old_key = next(r["kernel_key"] for r in old["records"] if r["row_id"] == "gate_proj@gpu0")
    # The original export carries the UUID in its canonical hardware source.
    old_uuid = old_key["hardware"]["source"]["uuid"]
    current_uuid = observations[0]["observed"]["uuid"]
    report = {"scope": "CPU review of pilot_only_not_profile_admission evidence", "source_ref": source_ref,
        "archive_sha256": sha(archive), "source_checks": checks, "report_recomputed": True,
        "primary_readout": rebuilt["primary_readout"], "pilot_passed": summary["pilot_passed"],
        "r3_quality_gate_passed": summary["r3_quality_gate_passed"], "cells": cells,
        "comparisons": comparisons, "gpu0": observations[0],
        "run05_gpu0_uuid": old_uuid, "same_physical_gpu0_as_run05": old_uuid == current_uuid,
        "timed_target_replays": len(csv_rows) - len(anchor_rows), "timed_anchor_replays": len(anchor_rows),
        "timed_function_invocations": 50*len(csv_rows),
        "r3_pre_first_strict_max": {"observed": sum(c["r3_pre_first_strict_max"]["observed"] for c in cells.values()),
                                     "endpoints": sum(c["r3_pre_first_strict_max"]["endpoints"] for c in cells.values())},
        "telemetry": {"observations": sum(c["telemetry"]["observations"] for c in cells.values()),
                      "available": sum(c["telemetry"]["available"] for c in cells.values())},
        "independent_runs_per_workload_strategy": 1, "crossing_rate_estimate": None,
        "environment_causal_verdict": "inconclusive", "protocol_changed": False, "r3_adopted": False,
        "mul_metadata_smoke_executed": False, "new_accepted_profiles": 0, "priceable_plans": 0,
        "legal_plans": 48, "followup_matrix_open": False, "raw_file_sha256": originals}
    return report, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, rows = review(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "report.json", report)
    with (args.output / "replay_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:report[k] for k in ("pilot_passed", "r3_quality_gate_passed", "r3_pre_first_strict_max", "same_physical_gpu0_as_run05")}))


if __name__ == "__main__":
    main()
