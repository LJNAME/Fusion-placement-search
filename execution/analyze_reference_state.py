"""Summarize the bounded state diagnostic without fitting or promoting calibration."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from calibration.e2_records import load_jsonl_records
from execution.audit_gpu_holdout import write_json
from execution.diagnose_reference_state import audit


def analyze(directory):
    directory = Path(directory)
    checked = audit(directory)
    report = json.loads((directory / "diagnostic_report.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    records = load_jsonl_records(directory / "blocks.jsonl")
    grouped = defaultdict(list)
    for row in report["triplets"]:
        grouped[row["plan_name"], row["lifecycle"], row["middle_kind"]].append(row)
    cells = []
    for (plan, life, middle), rows in grouped.items():
        cells.append({"plan_name": plan, "lifecycle": life, "middle_kind": middle,
            "rounds": len(rows), "endpoint_change_pct_by_round": [x["after_over_before_change_pct"] for x in rows],
            "max_absolute_endpoint_change_pct": max(abs(x["after_over_before_change_pct"]) for x in rows),
            "max_absolute_middle_change_pct": max(abs(x[key]) for x in rows for key in
                ("middle_over_before_change_pct", "middle_over_after_change_pct")),
            "all_endpoints_within_reference_10pct": all(x["baseline_within_reference_10pct"] for x in rows)})
    baselines = defaultdict(list)
    for index, triplet in enumerate(report["triplets"]):
        for row in records[index * 3:index * 3 + 3]:
            if row["phase"] in ("before", "after"):
                baselines[triplet["plan_name"]].extend(row["samples_s"])
    pooled = {plan: statistics.median(values) * 1000 for plan, values in baselines.items()}
    memories = [item for row in records for key in ("memory_before", "memory_after") for item in row["context"][key]]
    loads = [row["context"]["load_average"][0] for row in records if row["context"]["load_average"] is not None]
    worst = max(report["triplets"], key=lambda row: abs(row["after_over_before_change_pct"]))
    sham_peak = max((x for x in report["triplets"] if x["middle_kind"] == "baseline_only"),
                    key=lambda row: max(abs(row[k]) for k in ("middle_over_before_change_pct", "middle_over_after_change_pct")))
    return {"audit": checked,
        "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in
            ("protocol.json", "environment.json", "blocks.jsonl", "diagnostic_report.json", "collection_status.json")},
        "environment": environment,
        "summary": {"correctness_blocks_pass": sum(row["correctness"]["pass"] for row in records),
            "triplets": len(report["triplets"]),
            "endpoint_reference_10pct_pass": sum(row["baseline_within_reference_10pct"] for row in report["triplets"]),
            "middle_reference_15pct_pass": sum(row["middle_within_reference_15pct"] for row in report["triplets"]),
            "samples_above_twice_block_median": sum(stats["samples_above_twice_median"]
                for row in report["triplets"] for stats in row["statistics"].values()),
            "no_earlier_calibration_promoted": True},
        "cells": cells, "largest_endpoint_change": worst, "largest_sham_middle_change": sham_peak,
        "pooled_before_after_medians_ms": pooled,
        "gpu0_over_gpu1_baseline_pct": (pooled["uniform0"] / pooled["uniform1"] - 1) * 100,
        "state_observations": {"load_average_1m_range": [min(loads), max(loads)] if loads else None,
            "allocator_retry_values": sorted({x["allocation_retries"] for x in memories if x["allocation_retries"] is not None}),
            "reserved_bytes_by_device": {device: sorted({x["reserved"] for x in memories if x["device"] == device})
                for device in sorted({x["device"] for x in memories})}},
        "limitations": ["A new host and diagnostic protocol: passing reference targets cannot retroactively accept earlier calibrations.",
            "The larger endpoint change occurred with reused tensors; this does not establish rebuilding as the earlier drift cause.",
            "Three rounds of one training workload are not independent tasks, a significance test or a model generalization result.",
            "Sham middle blocks also vary, but all stay within reference bounds; that does not reproduce the earlier >10% endpoint failure.",
            "Pooled GPU0/GPU1 differences describe this runtime state, not intrinsic chip-speed differences or a calibrated hardware asymmetry.",
            "No allocation retries does not rule out allocator/cache effects. Host load does not identify a causal CPU-frequency/driver mechanism.",
            "No new cost fitting, host-profile export, default-search change, SROP benefit or independent labels."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    write_json(args.directory / "analysis_report.json", result)
    print(result["summary"])
    for row in result["cells"]:
        print(row["plan_name"], row["lifecycle"], row["middle_kind"], row["max_absolute_endpoint_change_pct"])
    print("pooled baselines ms:", result["pooled_before_after_medians_ms"])


if __name__ == "__main__":
    main()
