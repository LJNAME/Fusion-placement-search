from __future__ import annotations

import unittest

from calibration.e2_contract import build_split_contract
from calibration.e2_dataset import (
    analytic_kernel_estimate,
    build_profiling_dataset,
    make_dataset_view,
)
from calibration.e2_manifest import build_shape_manifest
from calibration.e2_records import build_environment_record, build_raw_measurement_record
from simulator.hardware import DeviceSpec, HardwareSpec


class E2DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_shape_manifest()
        cls.contract = build_split_contract(cls.manifest)
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
        cls.hardware = HardwareSpec(
            devices=[DeviceSpec(90e12, 1.3e12, 68e9, name="BW")],
            launch_overhead=8e-6,
            name="bw-test",
        )

    def _raw(self, object_id: str, duration_s: float, device: int = 0):
        return build_raw_measurement_record(
            object_id=object_id,
            environment_id=self.environment["environment_id"],
            physical_device_index=device,
            samples_s=(duration_s * 0.9, duration_s, duration_s * 1.1),
            warmup=5,
            iterations_per_sample=20,
            run_id=f"test-{device}",
            measured_at="2026-10-25T12:00:00+08:00",
        )

    def _objects(self):
        split = self.contract["splits"][0]
        return split, split["train_object_ids"][0], split["hidden_object_ids"][0]

    def test_analytic_estimate_has_positive_components(self) -> None:
        item = next(item for item in self.manifest["objects"] if item["measurement"]["enabled"])
        estimate = analytic_kernel_estimate(item, self.hardware)
        self.assertGreater(estimate["estimate_s"], 0.0)
        self.assertGreater(estimate["memory_bytes"], 0.0)
        self.assertGreaterEqual(estimate["estimate_s"], estimate["launch_s"])

    def test_dataset_aggregates_replicas_and_builds_log_residual(self) -> None:
        _, train_id, hidden_id = self._objects()
        raw = [
            self._raw(train_id, 10e-6, 0),
            self._raw(train_id, 14e-6, 1),
            self._raw(hidden_id, 20e-6, 0),
        ]
        metadata, records = build_profiling_dataset(
            self.manifest,
            self.contract,
            {self.environment["environment_id"]: self.environment},
            raw,
            self.hardware,
            require_complete=False,
        )
        self.assertEqual(metadata["record_count"], 2)
        train = next(record for record in records if record["object_id"] == train_id)
        self.assertAlmostEqual(train["measurement"]["duration_s"], 12e-6)
        self.assertEqual(train["measurement"]["replicas"], 2)
        self.assertAlmostEqual(
            train["target"]["log_residual"],
            __import__("math").log(12e-6 / train["analytic"]["estimate_s"]),
        )

    def test_dataset_view_masks_zero_shot_hidden_labels(self) -> None:
        split, train_id, hidden_id = self._objects()
        _, records = build_profiling_dataset(
            self.manifest,
            self.contract,
            {self.environment["environment_id"]: self.environment},
            [self._raw(train_id, 10e-6), self._raw(hidden_id, 20e-6)],
            self.hardware,
            require_complete=False,
        )
        view = make_dataset_view(
            records,
            self.contract,
            split_id=split["split_id"],
            mode="zero_shot",
            role="predictor",
        )
        train = next(record for record in view if record["object_id"] == train_id)
        hidden = next(record for record in view if record["object_id"] == hidden_id)
        self.assertIn("target", train)
        self.assertNotIn("target", hidden)
        self.assertNotIn("measurement", hidden)
        self.assertTrue(train["label_access"]["visible"])
        self.assertFalse(hidden["label_access"]["visible"])

    def test_active_reveal_and_oracle_access_are_separate(self) -> None:
        split, train_id, hidden_id = self._objects()
        _, records = build_profiling_dataset(
            self.manifest,
            self.contract,
            {self.environment["environment_id"]: self.environment},
            [self._raw(train_id, 10e-6), self._raw(hidden_id, 20e-6)],
            self.hardware,
            require_complete=False,
        )
        active = make_dataset_view(
            records,
            self.contract,
            split_id=split["split_id"],
            mode="active_calibration",
            role="sampler",
            revealed_object_ids=(hidden_id,),
        )
        oracle = make_dataset_view(
            records,
            self.contract,
            split_id=split["split_id"],
            mode="zero_shot",
            role="oracle_evaluator",
        )
        self.assertTrue(all("target" in record for record in active))
        self.assertTrue(all("target" in record for record in oracle))
        with self.assertRaisesRegex(ValueError, "zero-shot"):
            make_dataset_view(
                records,
                self.contract,
                split_id=split["split_id"],
                mode="zero_shot",
                role="predictor",
                revealed_object_ids=(hidden_id,),
            )

    def test_complete_build_rejects_missing_objects(self) -> None:
        _, train_id, _ = self._objects()
        with self.assertRaisesRegex(ValueError, "missing"):
            build_profiling_dataset(
                self.manifest,
                self.contract,
                {self.environment["environment_id"]: self.environment},
                [self._raw(train_id, 10e-6)],
                self.hardware,
            )


if __name__ == "__main__":
    unittest.main()
