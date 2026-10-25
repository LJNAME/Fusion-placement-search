"""Validate E2 JSONL measurements and report coverage and sample stability."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .e2_manifest import validate_shape_manifest
from .e2_records import (
    load_environment_registry,
    load_jsonl_records,
    validate_raw_measurement_record,
)


def validate_measurement_set(
    manifest: Mapping[str, Any],
    environments: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_replicas: int = 1,
) -> dict[str, Any]:
    if expected_replicas <= 0:
        raise ValueError("expected_replicas must be positive")
    validate_shape_manifest(manifest)
    contract = manifest["environment_contract"]
    for environment in environments.values():
        hardware = environment["hardware"]
        software = environment["software"]
        if hardware["class"] != contract["hardware_class"]:
            raise ValueError("environment hardware class does not match manifest")
        if software["framework_version"] != contract["framework_version"]:
            raise ValueError("environment framework version does not match manifest")
        if software["accelerator_version"] != contract["accelerator_version"]:
            raise ValueError("environment accelerator version does not match manifest")

    for record in records:
        validate_raw_measurement_record(
            record,
            manifest=manifest,
            environments=environments,
        )

    objects = {
        item["object_id"]: item
        for item in manifest["objects"]
        if bool(item["measurement"]["enabled"])
    }
    successful = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] == "error"]
    successful_counts = Counter(record["object_id"] for record in successful)
    missing = sorted(set(objects) - set(successful_counts))
    under_replicated = sorted(
        object_id
        for object_id, count in successful_counts.items()
        if count < expected_replicas
    )
    over_replicated = sorted(
        object_id
        for object_id, count in successful_counts.items()
        if count > expected_replicas
    )

    by_operator: dict[str, list[float]] = defaultdict(list)
    sample_cv: list[tuple[float, str]] = []
    for record in successful:
        object_id = record["object_id"]
        operator = objects[object_id]["kernel_key"]["operator"]
        summary = record["timing"]["summary"]
        median = float(summary["median_s"])
        mean = float(summary["mean_s"])
        cv = float(summary["std_s"]) / mean if mean > 0.0 else 0.0
        by_operator[operator].append(median)
        sample_cv.append((cv, object_id))

    operator_summary = {
        operator: {
            "records": len(values),
            "min_s": min(values),
            "median_s": statistics.median(values),
            "max_s": max(values),
        }
        for operator, values in sorted(by_operator.items())
    }
    max_cv, max_cv_object_id = max(sample_cv, default=(0.0, None))
    return {
        "enabled_objects": len(objects),
        "records": len(records),
        "successful_records": len(successful),
        "failed_records": len(failed),
        "unique_successful_objects": len(successful_counts),
        "device_records": dict(sorted(Counter(
            int(record["physical_device_index"]) for record in successful
        ).items())),
        "missing_object_ids": missing,
        "under_replicated_object_ids": under_replicated,
        "over_replicated_object_ids": over_replicated,
        "max_sample_cv": max_cv,
        "max_sample_cv_object_id": max_cv_object_id,
        "operator_summary": operator_summary,
        "complete": not missing and not under_replicated,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/bw1000_e2/shape_manifest.json")
    )
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--expected-replicas", type=int, default=1)
    parser.add_argument("--expect-complete", action="store_true")
    parser.add_argument("--max-sample-cv", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    environments = load_environment_registry(args.environments)
    records = load_jsonl_records(args.raw)
    report = validate_measurement_set(
        manifest,
        environments,
        records,
        expected_replicas=args.expected_replicas,
    )
    print(f"enabled_objects: {report['enabled_objects']}")
    print(f"records: {report['records']}")
    print(f"successful_records: {report['successful_records']}")
    print(f"failed_records: {report['failed_records']}")
    print(f"unique_successful_objects: {report['unique_successful_objects']}")
    print(f"device_records: {report['device_records']}")
    print(f"missing_objects: {len(report['missing_object_ids'])}")
    print(f"under_replicated: {len(report['under_replicated_object_ids'])}")
    print(f"over_replicated: {len(report['over_replicated_object_ids'])}")
    print(f"max_sample_cv: {report['max_sample_cv']:.6f}")
    for operator, summary in report["operator_summary"].items():
        print(
            f"{operator}: records={summary['records']} "
            f"min/median/max_us={summary['min_s'] * 1e6:.3f}/"
            f"{summary['median_s'] * 1e6:.3f}/{summary['max_s'] * 1e6:.3f}"
        )
    if args.expect_complete and not report["complete"]:
        raise SystemExit("measurement set is incomplete")
    if args.max_sample_cv is not None and report["max_sample_cv"] > args.max_sample_cv:
        raise SystemExit(
            f"maximum sample CV {report['max_sample_cv']:.6f} exceeds "
            f"{args.max_sample_cv:.6f}"
        )
    print("Fusion E2 measurement validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
