from __future__ import annotations

import unittest

from calibration.e2_manifest import build_shape_manifest
from calibration.profile_e2_bw1000 import plan_measurements


class E2ProfilingPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_shape_manifest()
        cls.enabled = [
            item for item in cls.manifest["objects"] if item["measurement"]["enabled"]
        ]
        cls.environments = {0: "environment-a", 1: "environment-a"}

    def test_shard_assigns_every_object_once_and_balances_devices(self) -> None:
        plan = plan_measurements(
            self.manifest,
            (),
            environment_by_device=self.environments,
            assignment="shard",
            resume=True,
        )
        self.assertEqual(len(plan), len(self.enabled))
        self.assertEqual(len({item["object_id"] for _, item in plan}), len(self.enabled))
        counts = {device: sum(planned == device for planned, _ in plan) for device in (0, 1)}
        self.assertLessEqual(abs(counts[0] - counts[1]), 1)

    def test_replicate_assigns_every_object_to_every_device(self) -> None:
        plan = plan_measurements(
            self.manifest,
            (),
            environment_by_device=self.environments,
            assignment="replicate",
            resume=True,
        )
        self.assertEqual(len(plan), len(self.enabled) * 2)

    def test_resume_skips_success_but_retries_failure(self) -> None:
        object_id = self.enabled[0]["object_id"]
        success = {
            "object_id": object_id,
            "environment_id": "environment-a",
            "physical_device_index": 0,
            "status": "ok",
        }
        failure = {
            "object_id": self.enabled[1]["object_id"],
            "environment_id": "environment-a",
            "physical_device_index": 1,
            "status": "error",
        }
        shard = plan_measurements(
            self.manifest,
            (success, failure),
            environment_by_device=self.environments,
            assignment="shard",
            resume=True,
        )
        shard_ids = {item["object_id"] for _, item in shard}
        self.assertNotIn(object_id, shard_ids)
        self.assertIn(failure["object_id"], shard_ids)
        shard_on_other_device = plan_measurements(
            self.manifest,
            (success,),
            environment_by_device={1: "environment-a"},
            assignment="shard",
            resume=True,
        )
        self.assertNotIn(
            object_id, {item["object_id"] for _, item in shard_on_other_device}
        )

        replicate = plan_measurements(
            self.manifest,
            (success,),
            environment_by_device=self.environments,
            assignment="replicate",
            resume=True,
        )
        measured_pairs = {(device, item["object_id"]) for device, item in replicate}
        self.assertNotIn((0, object_id), measured_pairs)
        self.assertIn((1, object_id), measured_pairs)

    def test_op_filter_and_limit_apply_before_sharding(self) -> None:
        plan = plan_measurements(
            self.manifest,
            (),
            environment_by_device=self.environments,
            assignment="shard",
            resume=True,
            include_ops={"rope"},
            limit=3,
        )
        self.assertEqual(len(plan), 3)
        self.assertTrue(all(item["kernel_key"]["operator"] == "rope" for _, item in plan))


if __name__ == "__main__":
    unittest.main()
