"""Offline attribution of run05 anchor failures; never rewrites admission."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics as st
import tarfile

from calibration.ffn_b4 import save
from cost_model.fusion_kernel_profile import key_id

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values):
    q1, median, q3 = st.quantiles(values, n=4, method="inclusive")
    return {"n": len(values), "min": min(values), "q1": q1, "median": median, "q3": q3, "max": max(values),
            "iqr_over_median": (q3-q1)/median, "range_over_median": (max(values)-min(values))/median}


def windows_and_operators(grid, plan):
    windows, operators, groups = [], [], []
    limit = plan["active_recipe"]["anchor_drift_limit"]
    for group_index, group in enumerate(plan["matrix"]["round_groups"]):
        active = [row for row in group["members"] if grid["cells"][row].get("samples")]
        if not active:
            continue
        reference = grid["cells"][active[0]]["samples"]
        assert len(reference) == 7
        for row in active:
            assert key_id([s["anchors"] for s in grid["cells"][row]["samples"]]) == key_id([s["anchors"] for s in reference])
        bad_attempts = []
        for index, sample in enumerate(reference):
            assert index == sample["attempt"]
            bad = False
            for gpu, values in sample["anchors"].items():
                before, after = values["before_us"], values["after_us"]
                drift = abs(after-before)/before
                rejected = drift > limit
                bad |= rejected
                windows.append({"group_index": group_index, "group_id": group["group_id"], "attempt": index,
                                "device": gpu, "before_us": before, "after_us": after, "relative_drift": drift,
                                "rejected": rejected, "shared_operator_cells": len(active)})
            if bad:
                bad_attempts.append(index)
        groups.append({"group_index": group_index, "group_id": group["group_id"], "active_cells": active,
                       "rejected_attempts": bad_attempts, "failed_cell_count": len(active) if bad_attempts else 0})
        for row in active:
            record = grid["cells"][row]
            samples = record["samples"]
            stats = describe([s["duration_us"] for s in samples])
            for sample in samples:
                operators.append({"group_index": group_index, "group_id": group["group_id"], "row_id": row,
                                  "attempt": sample["attempt"], "duration_us": sample["duration_us"],
                                  "deviation_from_seven_round_median": sample["duration_us"]/stats["median"]-1,
                                  "rejected_anchor_window": sample["attempt"] in bad_attempts,
                                  "original_status": record["status"]})
    return windows, operators, groups


def coverage_diagnostic(availability):
    plans = availability["plans"]
    down = {"down_proj@gpu0", "down_proj@gpu1"}
    assert all(down.intersection(p["missing_quality_passed_rows"]) for p in plans)
    return {"priceable_plans": sum(not p["missing_quality_passed_rows"] for p in plans), "legal_plans": len(plans),
            "basis": "quality-only hypothetical availability, not admitted or end-to-end validated cost",
            "strict_priceable_plans": 0,
            "missing_for_all_plans": sorted(down),
            "missing_for_all_plans_semantics": "every plan lacks its chosen Down Projection; each uses one device alternative, not both",
            "exact_missing_term_intersection": sorted(set.intersection(*(set(p["missing_quality_passed_rows"]) for p in plans))),
            "chosen_down_alternative_missing_plan_counts": {d: sum(d in p["missing_quality_passed_rows"] for p in plans) for d in sorted(down)},
            "counterfactual_only_both_down_available": sum(not (set(p["missing_quality_passed_rows"])-down) for p in plans),
            "counterfactual_does_not_change_any_status": True}


def analyze(root, archive, review):
    root, archive = Path(root), Path(archive)
    assert review["search_cost_items_added"] == 0 and review["new_accepted_profiles"] == 0
    grid, plan = read(root/"formal/grid.json"), read(root/"plan.json")
    for name, sha in review["raw_file_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == review["archive_sha256"]
    windows, operators, groups = windows_and_operators(grid, plan)
    device_rejected = [w for w in windows if w["rejected"]]
    bad_windows = {(w["group_id"], w["attempt"]) for w in device_rejected}
    all_windows = {(w["group_id"], w["attempt"]) for w in windows}
    endpoints = {gpu: {side: describe([w[side+"_us"] for w in windows if w["device"] == gpu])
                       for side in ("before", "after")} for gpu in sorted({w["device"] for w in windows})}
    operator_stats = {}
    for row, record in grid["cells"].items():
        if not record.get("samples"):
            continue
        samples = record["samples"]
        statistics = describe([s["duration_us"] for s in samples])
        peak = max(samples, key=lambda s: s["duration_us"])
        operator_stats[row] = {"original_status": record["status"], **statistics,
                               "max_attempt": peak["attempt"],
                               "max_attempt_anchor_rejected": any(o["rejected_anchor_window"] for o in operators if o["row_id"] == row and o["attempt"] == peak["attempt"]),
                               "all_original_attempts_retained": len(samples) == 7}
    with tarfile.open(archive) as tar:
        mtimes = {m.name:m.mtime for m in tar.getmembers()}
    local_tz = dt.timezone(dt.timedelta(hours=8))
    for group in groups:
        prefix = root.name+"/formal/groups/"+group["group_id"].replace("/", "_").replace(">", "to")+"/"
        start, end = mtimes[prefix+"parent_plan.json"], mtimes[prefix+"worker_status.json"]
        group.update(worker_file_mtime_start=dt.datetime.fromtimestamp(start, local_tz).isoformat(),
                     worker_file_mtime_end=dt.datetime.fromtimestamp(end, local_tz).isoformat(),
                     worker_envelope_seconds=end-start)
    # Target kernel names are observations, not inferred fallback or library causes.
    down_paths = {}
    for row in ("down_proj@gpu0", "down_proj@gpu1"):
        record = grid["cells"][row]
        down_paths[row] = sorted(set(record["path_evidence"]["kernel_names"]))
    gemm_details = {}
    for row in ("down_proj@gpu0", "down_proj@gpu1", "gate_proj@gpu0", "gate_proj@gpu1", "up_proj@gpu0", "up_proj@gpu1"):
        record = grid["cells"][row]
        gemm_details[row] = {"input_shape": record["identity"]["inputs"][0]["shape"],
                             "weight_shape": record["identity"]["weight"]["shape"],
                             "kernel_names": sorted(set(record["path_evidence"]["kernel_names"]))}
    report = {"scope": "run05 CPU attribution only; original fixed-seven admission unchanged",
              "archive_sha256": review["archive_sha256"], "source_ref": review["source_ref"],
              "pricing": coverage_diagnostic(review["hypothetical_quality_only_plan_availability"]),
              "counts": {"operator_cells": len(operator_stats), "operator_round_rows": len(operators),
                         "group_windows": len(all_windows), "rejected_group_windows": len(bad_windows),
                         "device_window_pairs": len(windows), "rejected_device_window_pairs": len(device_rejected),
                         "endpoint_replay_measurements": 2*len(windows),
                         "failed_groups": sum(bool(g["rejected_attempts"]) for g in groups),
                         "failed_cells_from_shared_anchor_rule": sum(g["failed_cell_count"] for g in groups)},
              "failed_group_windows_by_attempt": {str(i):sum(a==i for _,a in bad_windows) for i in range(7)},
              "early_late_windows": {"attempts_0_2": {"total": sum(a<3 for _,a in all_windows), "rejected": sum(a<3 for _,a in bad_windows)},
                                     "attempts_3_6": {"total": sum(a>=3 for _,a in all_windows), "rejected": sum(a>=3 for _,a in bad_windows)}},
              "endpoint_distributions": endpoints, "failed_device_pairs": device_rejected,
              "group_timeline": groups, "operator_statistics": operator_stats,
              "target_iqr_within_ten_percent_diagnostic_only": sum(s["iqr_over_median"] <= .1 for s in operator_stats.values()),
              "down_projection_paths": {"same_observed_kernel_symbol_set": down_paths["down_proj@gpu0"]==down_paths["down_proj@gpu1"],
                                        "kernel_names": down_paths,
                                        "down_differs_from_gate_up_symbols": all(gemm_details[row]["kernel_names"] != down_paths["down_proj@gpu0"]
                                                                                 for row in gemm_details if not row.startswith("down")),
                                        "gemm_details": gemm_details,
                                        "correctness_or_path_failures": sum(not r.get("path_verified") or any(not s.get("numeric_ok") for s in r["samples"])
                                                                             for row,r in grid["cells"].items() if row.startswith("down_proj@"))},
              "observability": {"independent_anchor_replays_per_endpoint": 1, "kernels_in_that_replay": 50,
                                "anchor_warmups_once_before_case_preparation": 10,
                                "anchor_endpoint_absolute_timestamps_recorded": False,
                                "validation_host_duration_recorded": False, "per_round_clock_utilization_telemetry_recorded": False,
                                "filesystem_mtimes_are_worker_envelopes_not_anchor_timestamps": True},
              "conclusions": {"median_ratio_is_not_pointwise_stability": True,
                              "shared_window_rule_amplifies_cell_failure_fraction": True,
                              "early_round_concentration_observed_not_causal_proof": True,
                              "pure_protocol_or_environment_cause_identified": False,
                              "down_specific_runtime_failure_demonstrated": False,
                              "retroactive_reclassification_allowed": False},
              "new_gpu_measurements": 0, "new_accepted_profiles": 0,
              "source_sha256": {"tools/analyze_ffn_b4_anchor_failures.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "data/ffn_b4_run05_review/report.json": hashlib.sha256((ROOT/"data/ffn_b4_run05_review/report.json").read_bytes()).hexdigest()}}
    assert report["counts"]["failed_cells_from_shared_anchor_rule"] == review["anchor_failed_cells"]
    return report, windows, operators


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = read(ROOT/"data/ffn_b4_run05_review/report.json")
    report, windows, operators = analyze(args.root, args.archive, review)
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output/"report.json", report)
    write_csv(args.output/"T1_anchor_device_windows.csv", windows)
    write_csv(args.output/"T2_operator_rounds.csv", operators)
    write_csv(args.output/"T3_worker_timeline.csv", [{k:v for k,v in g.items() if k not in ("active_cells", "rejected_attempts")}
                                                 for g in report["group_timeline"]])
    print(json.dumps({"pricing": report["pricing"], "counts": report["counts"], "early_late_windows": report["early_late_windows"]}))


if __name__ == "__main__":
    main()
