"""B3 CPU protocol fixtures; no samples below are GPU measurements."""
import copy
import unittest

from cost_model.ffn_measurement_b3 import build, link_residual, quality_review, verify


def sample(index, *, drift=False, native=10., fused=12.):
    return {"attempt": index, "numeric_ok": True, "capture_ok": True, "path_ok": True,
            "arm_us": {"native": native, "fused": fused},
            "anchors": {"gpu0": {"before_us": 100., "after_us": 111. if drift else 100.}}}


def review(rows):
    return quality_review(rows, ("native", "fused"), ("gpu0",))


class TestFfnMeasurementB3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build()

    def test_18_plus_1_matrix_and_unique_round_groups(self):
        result = verify(self.bundle)
        self.assertEqual((result["core_rows"], result["diagnostic_rows"], result["total_evidence_slots"]), (18, 1, 19))
        self.assertEqual((result["singleton_rows"], result["native_rows"], result["fused_rows"], result["communication_rows"]), (10, 2, 2, 4))
        self.assertEqual(result["round_groups"], 10)
        self.assertFalse(result["formal_gpu_timing_allowed"])

    def test_pair_keys_equal_within_device_and_distinct_across_devices(self):
        rows = self.bundle["measurement_matrix.json"]["core_rows"]
        pairs = [[r for r in rows if r["row_id"] in (f"silu_mul/native@gpu{gpu}", f"silu_mul/fused@gpu{gpu}")] for gpu in (0, 1)]
        for pair in pairs:
            self.assertEqual(pair[0]["pair_key"], pair[1]["pair_key"])
            self.assertIsNone(pair[0]["measurement_contract_id"])
        self.assertNotEqual(pairs[0][0]["pair_key_template_id"], pairs[1][0]["pair_key_template_id"])

    def test_old_fused_warmup_is_recorded_but_anchor_recipe_blocks_reuse(self):
        records = self.bundle["reuse_audit.json"]["records"]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["recorded_warmup"] == 10 and r["restore_outside_timing_recorded"] for r in records))
        self.assertTrue(all(not r["anchor_observations_present"] and not r["eligible_for_b3_reuse"] for r in records))
        self.assertTrue(all(r["original_admission"] == "accepted" for r in records))
        self.assertEqual(self.bundle["measurement_matrix.json"]["summary"]["new_measurement_tasks"], 19)

    def test_buffers_preserve_gate_alias_and_shared_readonly_up(self):
        audit = self.bundle["buffer_audit.json"]
        buffers = {b["name"]: b for b in audit["activation_group"]}
        self.assertEqual(buffers["gate_pool"]["count"], 50)
        self.assertEqual(buffers["up_shared_readonly"]["count"], 1)
        self.assertEqual(buffers["fused_outputs"]["alias_of"], "gate_pool")
        self.assertEqual(buffers["fused_outputs"]["allocated_bytes"], 0)
        self.assertEqual(audit["activation_controlled_bytes"] / 2**20, 381.)
        self.assertEqual(audit["anchor_controlled_bytes_per_gpu"] / 2**20, 76.5)
        self.assertEqual(audit["link_quartet_controlled_bytes"] / 2**20, 389.)
        self.assertFalse(audit["gpu_buffer_alias_and_lifetime_verified"])

    def test_slower_fused_is_quality_eligible_without_automatic_admission(self):
        result = review([sample(i) for i in range(7)])
        self.assertTrue(result["timing_eligible"])
        self.assertEqual(result["primary_attempts"], list(range(7)))
        self.assertFalse(result["automatic_admission"])

    def test_anchor_invalidates_whole_pair_and_uses_first_seven_valid(self):
        rows = [sample(i, drift=i in (1, 4)) for i in range(9)]
        result = review(rows)
        self.assertEqual(result["invalid_attempts"], [1, 4])
        self.assertEqual(result["primary_attempts"], [0, 2, 3, 5, 6, 7, 8])
        self.assertTrue(result["timing_eligible"])

    def test_anchor_at_threshold_allowed_and_budget_exhaustion_stops(self):
        rows = [sample(i) for i in range(7)]
        rows[0]["anchors"]["gpu0"]["after_us"] = 110.
        self.assertTrue(review(rows)["timing_eligible"])
        self.assertEqual(review([sample(i, drift=i < 3) for i in range(9)])["status"], "stop_insufficient_valid_rounds")
        with self.assertRaises(ValueError):
            review([sample(i) for i in range(10)])

    def test_excess_iqr_not_rescued_by_extra_rounds_or_trimming(self):
        rows = [sample(i, fused=(10. if i % 2 == 0 else 20.)) for i in range(7)]
        before = review(rows)
        after = review(rows + [sample(7, fused=10.), sample(8, fused=10.)])
        self.assertEqual(before["status"], "stop_dispersion")
        self.assertEqual(after["status"], "stop_dispersion")
        self.assertEqual(before["statistics"], after["statistics"])
        self.assertEqual(after["primary_attempts"], list(range(7)))

    def test_fatal_error_rejects_even_an_anchor_invalid_round(self):
        for field in ("numeric_ok", "capture_ok", "path_ok"):
            rows = [sample(i) for i in range(7)]
            rows[2] = sample(2, drift=True)
            rows[2][field] = False
            self.assertEqual(review(rows)["status"], "stop_fatal")

    def test_round_or_device_mismatch_cannot_silently_pair(self):
        for change in ("attempt", "arm", "device"):
            rows = [sample(i) for i in range(7)]
            if change == "attempt":
                rows[1]["attempt"] = 2
            elif change == "arm":
                rows[1]["arm_us"].pop("native")
            else:
                rows[1]["anchors"] = {"gpu1": rows[1]["anchors"]["gpu0"]}
            with self.assertRaises(ValueError):
                review(rows)

    def test_link_residual_sign_relative_value_and_inclusive_half_us(self):
        self.assertEqual(link_residual(9., 3., 3., 4.)["R_link_us"], -1.)
        boundary = link_residual(10.5, 3., 3., 4.)
        self.assertEqual(boundary["R_link_us"], .5)
        self.assertAlmostEqual(boundary["R_link_over_chain"], .5 / 10.5)
        self.assertEqual(boundary["engineering_action"], "inspect_dependencies_and_composition")
        self.assertEqual(link_residual(10.4, 3., 3., 4.)["engineering_action"], "locally_allow_additivity")

    def test_checklist_does_not_claim_gpu_smoke_or_global_zero_count(self):
        checklist = self.bundle["freeze_checklist.json"]
        self.assertEqual((checklist["cpu_checks_complete"], checklist["gpu_smoke_pending"]), (7, 1))
        self.assertFalse(checklist["all_eight_passed"])
        self.assertEqual(checklist["b4_smoke_plan"]["replays_per_body"], 3)
        self.assertFalse(checklist["b4_smoke_plan"]["verified_on_gpu"])
        sanity = self.bundle["identity.json"]["boundary_sanity"]
        self.assertEqual((sanity["input_gpu"], sanity["output_gpu"]), (0, 0))
        self.assertEqual(len(sanity["zero_communication_plan_ids"]), 2)

    def test_changed_freeze_or_reuse_cannot_pass(self):
        for filename, change in (("recipe.json", "warmup_replays_per_arm"), ("reuse_audit.json", "reuse"), ("freeze_checklist.json", "all_eight_passed")):
            bundle = copy.deepcopy(self.bundle)
            if change == "reuse":
                bundle[filename]["records"][0]["eligible_for_b3_reuse"] = True
            else:
                bundle[filename][change] = 99
            with self.assertRaises(ValueError):
                verify(bundle)


if __name__ == "__main__":
    unittest.main()
