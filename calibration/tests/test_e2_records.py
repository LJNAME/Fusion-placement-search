from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from calibration.e2_manifest import build_shape_manifest
from calibration.e2_records import (
    append_jsonl_record,
    build_environment_record,
    build_raw_measurement_record,
    load_jsonl_records,
    pending_measurement_object_ids,
    validate_environment_record,
    validate_raw_measurement_record,
)


class E2RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_shape_manifest()
        cls.object_id = next(
            item["object_id"]
            for item in cls.manifest["objects"]
            if item["measurement"]["enabled"]
        )
        cls.environment = build_environment_record(
            hardware_class="BW1000",
            device_name="BW",
            total_memory_bytes=68_702_699_520,
            framework_version="2.9.0",
            accelerator_version="6.3.26093",
            python_version="3.11.9",
            platform="linux-x86_64",
            source_revision="test-revision",
        )

    def _record(self):
        return build_raw_measurement_record(
            object_id=self.object_id,
            environment_id=self.environment["environment_id"],
            physical_device_index=0,
            samples_s=(1.0e-5, 1.2e-5, 1.1e-5),
            warmup=5,
            iterations_per_sample=20,
            run_id="test-run",
            measured_at="2026-10-25T12:00:00+08:00",
        )

    def test_environment_and_raw_record_validate(self) -> None:
        validate_environment_record(self.environment)
        record = self._record()
        validate_raw_measurement_record(
            record,
            manifest=self.manifest,
            environments={self.environment["environment_id"]: self.environment},
        )
        self.assertEqual(record["timing"]["summary"]["median_s"], 1.1e-5)

    def test_tampered_summary_is_rejected(self) -> None:
        record = copy.deepcopy(self._record())
        record["timing"]["summary"]["median_s"] = 99.0
        with self.assertRaisesRegex(ValueError, "summary mismatch"):
            validate_raw_measurement_record(
                record,
                manifest=self.manifest,
                environments={self.environment["environment_id"]: self.environment},
            )

    def test_jsonl_append_is_idempotent(self) -> None:
        record = self._record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_measurements.jsonl"
            self.assertTrue(append_jsonl_record(path, record))
            self.assertFalse(append_jsonl_record(path, record))
            self.assertEqual(load_jsonl_records(path), [record])

    def test_pending_objects_are_environment_specific(self) -> None:
        record = self._record()
        before = pending_measurement_object_ids(
            self.manifest, (), environment_id=self.environment["environment_id"]
        )
        after = pending_measurement_object_ids(
            self.manifest, (record,), environment_id=self.environment["environment_id"]
        )
        other = pending_measurement_object_ids(
            self.manifest, (record,), environment_id="sha256:another-environment"
        )
        self.assertIn(self.object_id, before)
        self.assertNotIn(self.object_id, after)
        self.assertIn(self.object_id, other)
        self.assertEqual(len(before) - len(after), 1)


if __name__ == "__main__":
    unittest.main()
