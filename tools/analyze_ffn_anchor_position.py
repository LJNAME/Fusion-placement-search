"""Post-hoc CPU diagnostics of endpoint position; never reclassify old quality."""
import argparse
import csv
import hashlib
from pathlib import Path
import statistics

from calibration.ffn_anchor_control import ROOT, read, save
from tools.review_ffn_anchor_pilot import review


def span(values):
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def describe(rounds, context):
    if len(rounds) != 7 or [r["attempt"] for r in rounds] != list(range(7)):
        raise ValueError("position analysis requires all seven original attempts")
    rows = []
    for attempt in rounds:
        for phase in ("pre", "post"):
            samples = attempt[phase]
            if len(samples) != 3:
                raise ValueError("position contrast requires all three endpoint samples")
            values = [s["batch_ms"] * 20 for s in samples]
            following = statistics.median(values[1:])
            first_pre = phase == "pre" and attempt["attempt"] == 0
            if first_pre:
                previous = context["anchor_warmup_end"]
                reference = "anchor_warmup_end_before_target_preparation"
            elif phase == "pre":
                previous = rounds[attempt["attempt"] - 1]["post"][-1]["validation_end"]
                reference = "previous_round_post_last_validation_end"
            else:
                previous = attempt["target"]["validation_end"]
                reference = "same_round_target_validation_end"
            gaps = [(samples[0]["timing_begin"] - previous) / 1e6] + [
                (samples[i]["timing_begin"] - samples[i-1]["validation_end"]) / 1e6 for i in (1, 2)]
            if any(g < 0 for g in gaps):
                raise ValueError("negative or reordered host timestamp gap")
            rows.append({"attempt": attempt["attempt"], "phase": phase,
                "first_us": values[0], "second_us": values[1], "third_us": values[2],
                "next_two_median_us": following,
                "first_minus_next_two_us": values[0] - following,
                "first_relative_to_next_two": (values[0] - following) / following,
                "first_strict_max": values[0] > max(values[1:]),
                "first_host_gap_ms": gaps[0], "second_host_gap_ms": gaps[1], "third_host_gap_ms": gaps[2],
                "first_gap_reference": reference, "first_gap_contains_target_preparation": first_pre,
                "gpu_idle_or_clock_observed": False})
    summaries = {}
    for phase in ("pre", "post"):
        group = [r for r in rows if r["phase"] == phase]
        # Only the host-gap summary separates the differently defined first
        # pre reference. All original endpoint values remain in every metric.
        ordinary_gap_rows = [r for r in group if not r["first_gap_contains_target_preparation"]]
        summaries[phase] = {
            "endpoints": len(group), "first_strict_max": sum(r["first_strict_max"] for r in group),
            "first_minus_next_two_us": span([r["first_minus_next_two_us"] for r in group]),
            "first_relative_to_next_two": span([r["first_relative_to_next_two"] for r in group]),
            "first_host_gap_ms_excluding_different_preparation_reference": span([r["first_host_gap_ms"] for r in ordinary_gap_rows]),
            "within_endpoint_host_gap_ms": span([r[k] for r in group for k in ("second_host_gap_ms", "third_host_gap_ms")]),
            "gap_is_gpu_idle_proxy": False,
        }
    return rows, summaries


def analyze(root):
    root = Path(root)
    verified, _ = review(root)
    all_rows, summaries = [], {}
    for cell in ("down_r3", "up_r3"):
        rows, summary = describe(read(root / cell / "samples.json"), read(root / cell / "result.json")["context"])
        all_rows.extend({"cell": cell, **row} for row in rows)
        summaries[cell] = summary
    largest = max(all_rows, key=lambda r: r["first_us"])
    return {"scope": "posthoc_endpoint_position_diagnostic_not_profile_admission",
        "archive_sha256": verified["archive_sha256"], "review_source_ref": "06af153",
        "input_review_sha256": hashlib.sha256((ROOT / "data/ffn_anchor_pilot01_review/report.json").read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summaries": summaries, "largest_first_reading": largest,
        "independent_runs_per_workload_strategy": 1, "crossing_rate_estimate": None,
        "descriptive_only": True, "old_samples_removed": 0, "old_quality_changed": False,
        "causal_verdict": "inconclusive", "r3_adopted": False,
        "gpu_measurements_added": 0, "new_accepted_profiles": 0,
        "priceable_plans": 0, "legal_plans": 48, "followup_matrix_open": False}, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, rows = analyze(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "report.json", report)
    with (args.output / "endpoint_positions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
