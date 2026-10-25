"""Offline verification for the frozen E2a/E2b data boundary."""

from __future__ import annotations

from .e2_contract import LabelArtifact, LabelRegistry, build_split_contract
from .e2_manifest import build_shape_manifest, validate_shape_manifest
from .e2_records import (
    build_environment_record,
    build_raw_measurement_record,
    pending_measurement_object_ids,
    validate_raw_measurement_record,
)
from .profile_e2_bw1000 import plan_measurements


def main() -> int:
    manifest = build_shape_manifest()
    validate_shape_manifest(manifest)
    contract = build_split_contract(manifest)

    enabled = [item for item in manifest["objects"] if item["measurement"]["enabled"]]
    disabled = [item for item in manifest["objects"] if not item["measurement"]["enabled"]]
    print(f"models: {len(manifest['models'])}")
    print(f"measurement_objects: {len(manifest['objects'])}")
    print(f"enabled: {len(enabled)}")
    print(f"disabled: {len(disabled)}")
    environment = build_environment_record(
        hardware_class="BW1000",
        device_name="BW",
        total_memory_bytes=68_702_699_520,
        framework_version="2.9.0",
        accelerator_version="6.3.26093",
        python_version="3.11.9",
        platform="linux-x86_64",
        source_revision="offline-interface-verification",
    )
    raw_record = build_raw_measurement_record(
        object_id=enabled[0]["object_id"],
        environment_id=environment["environment_id"],
        physical_device_index=0,
        samples_s=(1.0e-5, 1.1e-5, 1.2e-5),
        warmup=5,
        iterations_per_sample=20,
        run_id="offline-interface-verification",
        measured_at="2026-10-25T00:00:00+08:00",
    )
    validate_raw_measurement_record(
        raw_record,
        manifest=manifest,
        environments={environment["environment_id"]: environment},
    )
    pending = pending_measurement_object_ids(
        manifest, (raw_record,), environment_id=environment["environment_id"]
    )
    if len(pending) != len(enabled) - 1:
        raise AssertionError("resumable measurement accounting failed")
    print(f"raw_record_schema: OK (pending_after_one={len(pending)})")
    shard = plan_measurements(
        manifest,
        (),
        environment_by_device={0: "environment", 1: "environment"},
        assignment="shard",
        resume=True,
    )
    replicate = plan_measurements(
        manifest,
        (),
        environment_by_device={0: "environment", 1: "environment"},
        assignment="replicate",
        resume=True,
    )
    if len(shard) != len(enabled) or len(replicate) != 2 * len(enabled):
        raise AssertionError("device assignment planning failed")
    print(f"device_assignment: shard={len(shard)} replicate={len(replicate)}")
    for split in contract["splits"]:
        train = split["train_object_ids"]
        hidden = split["hidden_object_ids"]
        shared = split["shared_seen_object_ids"]
        registry = LabelRegistry(
            [
                LabelArtifact("train", "raw_measurement", train[0]),
                LabelArtifact("hidden", "raw_measurement", hidden[0]),
                LabelArtifact("derived", "aggregate", parent_record_ids=("hidden",)),
            ]
        )
        predictor = registry.visible_record_ids(
            contract,
            split_id=split["split_id"],
            mode="zero_shot",
            role="predictor",
        )
        oracle = registry.visible_record_ids(
            contract,
            split_id=split["split_id"],
            mode="zero_shot",
            role="oracle_evaluator",
        )
        if predictor != ("train",) or set(oracle) != {"train", "hidden", "derived"}:
            raise AssertionError(f"label-lineage isolation failed: {split['split_id']}")
        print(
            f"{split['split_id']}: train={len(train)} hidden={len(hidden)} "
            f"shared_seen={len(shared)} zero_shot_visible={len(predictor)}"
        )
    print("label_lineage_isolation: OK")
    print("Fusion E2a/E2b interface verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
