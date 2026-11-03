"""CPU-only audit of returned conditioning evidence; never run attached code."""
import argparse
import csv
import hashlib
from pathlib import Path
import statistics
import subprocess
import tempfile

from calibration import ffn_anchor_conditioning as collector
from calibration.ffn_anchor_control import ROOT, read, save
from cost_model.fusion_kernel_profile import key_id


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def describe(rounds, context):
    timeline, endpoints, telemetry = [], [], []
    previous_end = context["anchor_warmup_end"]
    for row in rounds:
        for phase in ("pre", "post"):
            readings = row[phase]
            values = [s["batch_ms"] * 20 for s in readings]
            conditions = row[phase + "_conditioning"]
            endpoints.append({"attempt": row["attempt"], "phase": phase,
                "first_us": values[0], "second_us": values[1], "third_us": values[2],
                "first_strict_max": values[0] > max(values[1:]),
                "first_minus_next_two_us": values[0] - statistics.median(values[1:]),
                "conditioning_us": conditions[0]["batch_ms"] * 20 if conditions else None,
                "conditioning_to_first_timing_ms": (readings[0]["timing_begin"] - conditions[-1]["validation_end"]) / 1e6 if conditions else None})
        for phase in ("pre_conditioning", "pre", "target", "post_conditioning", "post"):
            group = [row[phase]] if phase == "target" else row[phase]
            for index, sample in enumerate(group):
                timeline.append({"attempt": row["attempt"], "phase": phase, "repeat": index,
                    "sample_role": sample["sample_role"], "primary_measurement": sample["primary_measurement"],
                    "event_us_per_call": sample["batch_ms"] * 20,
                    "host_gap_from_previous_validation_or_anchor_warmup_ms": (sample["timing_begin"] - previous_end) / 1e6,
                    "gap_contains_target_preparation": row["attempt"] == 0 and phase in ("pre", "pre_conditioning") and index == 0 and previous_end == context["anchor_warmup_end"],
                    **{k: sample[k] for k in ("reset_begin", "timing_begin", "event_wait_end", "all_device_sync_end", "validation_end")}})
                previous_end = sample["validation_end"]
        for name in ("telemetry_before", "telemetry_after"):
            telemetry.extend(row[name].values())
    return timeline, endpoints, telemetry


def review(root, source_ref="3adcddc"):
    root = Path(root)
    original = inventory(root)
    frozen, summary = read(root / "plan.json"), read(root / "summary.json")
    source_checks = {}
    for name, expected in frozen["source_sha256"].items():
        blob = subprocess.run(["git", "-C", str(ROOT), "show", source_ref + ":" + name],
                              check=True, capture_output=True).stdout
        source_checks[name] = hashlib.sha256(blob).hexdigest() == expected
    if not all(source_checks.values()) or frozen != collector.plan() or key_id({k: v for k, v in frozen.items() if k != "plan_id"}) != frozen["plan_id"]:
        raise ValueError("archive source/plan differs from reviewed collector")
    cells = frozen["design"]["cells"]
    if summary["plan_id"] != frozen["plan_id"] or set(summary["cells"]) != {c["id"] for c in cells}:
        raise ValueError("conditioning grid differs from fixed plan")
    checked_cells, times, signatures, timeline, endpoints, telemetry = {}, [], [], [], [], []
    for cell in cells:
        name, directory = cell["id"], root / cell["id"]
        if read(directory / "parent_plan.json") != frozen:
            raise ValueError("worker plan mismatch")
        checked = collector.review_cell(directory, cell, frozen)
        saved = summary["cells"][name]
        if any(saved.get(k) != v for k, v in checked.items()) or saved["status"] != checked["quality_status"]:
            raise ValueError("summary differs from raw worker evidence")
        if saved["worker_attempts"] != 1 or saved["retries"] != 0 or saved["returncode"] != 0:
            raise ValueError("expected one completed process per fixed cell, without retries")
        identity = read(directory / "identity.json")["identity"]
        runtime = read(directory / "runtime_identity.json")
        for role in ("target", "anchor"):
            if any(runtime[k] != identity[role][k] for k in ("devices", "actual_torch_build", "loaded_kerv_source", "peer_capabilities")):
                raise ValueError("runtime evidence differs from target/anchor identity")
        if runtime["peer_capabilities"]:
            raise ValueError("single-device diagnostic contains peer capabilities")
        signatures.append(checked["runtime_signature"])
        result, rounds = read(directory / "result.json"), read(directory / "samples.json")
        context = result["context"]
        times += [context["anchor_preparation_begin"], rounds[-1]["post"][-1]["validation_end"]]
        replay_rows, endpoint_rows, observations = describe(rounds, context)
        timeline.extend({"cell": name, **r} for r in replay_rows)
        endpoints.extend({"cell": name, **r} for r in endpoint_rows)
        telemetry.extend(observations)
        quality = result["analysis"]["quality"]
        checked_cells[name] = {**checked, "quality": quality,
            "max_pre_endpoint_iqr_over_median": max(r["pre"]["iqr_over_median"] for r in quality["rounds"]),
            "max_post_endpoint_iqr_over_median": max(r["post"]["iqr_over_median"] for r in quality["rounds"]),
            "max_anchor_drift": max(r["drift"] for r in quality["rounds"]),
            "pre_first_strict_max": sum(r["first_strict_max"] for r in endpoint_rows if r["phase"] == "pre"),
            "target_preparation_s": (context["case_preparation_end"] - context["case_preparation_begin"]) / 1e9,
            "trace_kernel_counts": {k: len(v["kernel_names"]) for k, v in result["path_evidence"].items()}}
    if len(set(signatures)) != 1 or times != sorted(times):
        raise ValueError("physical GPU/build or ABBA chronological order differs")
    if summary["scope"] != collector.SCOPE or summary["pipeline_passed"] is not True or any(summary[k] for k in ("protocol_adopted", "automatic_admission", "followup_matrix_open")):
        raise ValueError("diagnostic escaped scope or did not complete")
    with tempfile.TemporaryDirectory() as temporary:
        collector.write_report(temporary, summary["cells"])
        if read(Path(temporary) / "report.json") != read(root / "report.json") or (Path(temporary) / "report.md").read_bytes() != (root / "report.md").read_bytes():
            raise ValueError("returned report differs from rebuilt evidence")
        collector.write_report(temporary, {c["id"]: {"status": "not_run"} for c in cells}, template=True)
        if any((Path(temporary) / name).read_bytes() != (root / name).read_bytes() for name in ("report_template.json", "report_template.md")):
            raise ValueError("returned preregistered template differs")
    if original != inventory(root):
        raise ValueError("review changed original evidence")
    rebuilt = collector.report(summary["cells"])
    current_gpu = read(root / cells[0]["id"] / "identity.json")["identity"]["target"]["devices"]["0"]
    old_gpu = read(ROOT / "data/ffn_anchor_pilot01_review/report.json")["gpu0"]
    archive = root.parent / (root.name + ".tar.gz")
    result = {"scope": "CPU review of " + collector.SCOPE, "source_ref": source_ref,
        "archive_sha256": sha(archive) if archive.is_file() else None,
        "plan_id": frozen["plan_id"], "source_checks": source_checks,
        "pipeline_passed": True, "cells": checked_cells,
        "quality_passed_cells": sum(c["quality_status"] == "quality_passed" for c in checked_cells.values()),
        "block_contrasts": rebuilt["block_contrasts"], "direction": rebuilt["direction"],
        "completed_runs_per_condition": rebuilt["completed_runs_per_condition"],
        "gpu0": current_gpu, "pilot01_gpu0_uuid": old_gpu["observed"]["uuid"],
        "same_physical_gpu0_as_pilot01": current_gpu["observed"]["uuid"] == old_gpu["observed"]["uuid"],
        "measured_replays": sum(c["measured_replays"] for c in checked_cells.values()),
        "conditioning_replays": sum(c["conditioning_replays"] for c in checked_cells.values()),
        "primary_timed_function_invocations": sum(c["measured_replays"] for c in checked_cells.values()) * 50,
        "conditioning_timed_function_invocations": sum(c["conditioning_replays"] for c in checked_cells.values()) * 50,
        "initialization_capture_warmup_trace_included_in_counts": False,
        "telemetry": {"observations": len(telemetry), "available": sum(t["value"] is not None for t in telemetry),
                      "errors": sorted({t["error"] for t in telemetry if t["error"]})},
        "report_and_template_recomputed": True, "raw_files_unchanged": True,
        "crossing_rate_estimate": None, "causal_verdict": "inconclusive",
        "protocol_adopted": False, "old_quality_changed": False, "samples_removed": 0,
        "new_accepted_profiles": 0, "priceable_plans": 0, "legal_plans": 48,
        "followup_matrix_open": False, "raw_file_sha256": original}
    return result, timeline, endpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, timeline, endpoints = review(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "report.json", result)
    for name, rows in (("replay_timeline.csv", timeline), ("endpoint_positions.csv", endpoints)):
        with (args.output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print({k: result[k] for k in ("pipeline_passed", "quality_passed_cells", "direction", "measured_replays", "conditioning_replays", "same_physical_gpu0_as_pilot01")})


if __name__ == "__main__":
    main()
