from __future__ import annotations

import unittest

from calibration.e2_manifest import build_shape_manifest
from calibration.e2_records import (
    build_environment_record,
    build_failed_measurement_record,
    build_raw_measurement_record,
)
from calibration.verify_e2_measurements import validate_measurement_set


class E2MeasurementValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_shape_manifest()
        cls.enabled = [
            item for item in cls.manifest["objects"] if item["measurement"]["enabled"]
        ]
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

    def _success(self, object_id: str, device: int = 0):
        return build_raw_measurement_record(
            object_id=object_id,
            environment_id=self.environment["environment_id"],
            physical_device_index=device,
            samples_s=(1.0e-5, 1.1e-5, 1.2e-5),
            warmup=2,
            iterations_per_sample=5,
            run_id="test-run",
            measured_at="2026-10-25T12:00:00+08:00",
            profiling_wall_time_s=0.01,
        )

    def test_complete_single_replica_set(self) -> None:
        records = [
            self._success(item["object_id"], index % 2)
            for index, item in enumerate(self.enabled)
        ]
        report = validate_measurement_set(
            self.manifest,
            {self.environment["environment_id"]: self.environment},
            records,
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["successful_records"], len(self.enabled))
        self.assertEqual(report["device_records"], {0: 56, 1: 56})
        self.assertFalse(report["missing_object_ids"])

    def test_missing_and_under_replicated_are_separate(self) -> None:
        records = [self._success(self.enabled[0]["object_id"])]
        report = validate_measurement_set(
            self.manifest,
            {self.environment["environment_id"]: self.environment},
            records,
            expected_replicas=2,
        )
        self.assertFalse(report["complete"])
        self.assertEqual(report["under_replicated_object_ids"], [self.enabled[0]["object_id"]])
        self.assertEqual(len(report["missing_object_ids"]), len(self.enabled) - 1)

    def test_historical_failure_does_not_invalidate_later_success(self) -> None:
        object_id = self.enabled[0]["object_id"]
        failure = build_failed_measurement_record(
            object_id=object_id,
            environment_id=self.environment["environment_id"],
            physical_device_index=0,
            run_id="failed-run",
            measured_at="2026-10-25T11:00:00+08:00",
            error_type="RuntimeError",
            error_message="synthetic",
            profiling_wall_time_s=0.01,
        )
        records = [failure]
        records.extend(self._success(item["object_id"]) for item in self.enabled)
        report = validate_measurement_set(
            self.manifest,
            {self.environment["environment_id"]: self.environment},
            records,
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["failed_records"], 1)


if __name__ == "__main__":
    unittest.main()
