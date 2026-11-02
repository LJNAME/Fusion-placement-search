"""Structural exhaustive coverage and edge-union evidence; no cost scoring."""
import copy
import unittest

from cost_model.ffn_placement_b1 import COMPUTE, FILES, FUSED, derive_t3, generate, verify


class TestFfnPlacementB1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate()

    def test_complete_product_and_reproducibility(self):
        result = verify(self.bundle)
        self.assertEqual((result["native_plans"], result["fused_plans"]), (32, 16))
        self.assertEqual(result["legal_labeled_plans"], 48)
        self.assertEqual(result["compute_structure_classes"], 24)
        self.assertEqual(result["labeled_plans_to_score"], 48)
        self.assertTrue(result["repeat_byte_equivalent"])

    def test_no_cost_symmetry_merge_and_pinned_boundaries(self):
        t1 = self.bundle[FILES[1]]
        symmetry = t1["symmetry"]
        self.assertTrue(symmetry["compute_only_swap_preserves_declared_constraints"])
        self.assertFalse(symmetry["global_swap_preserves_boundary_constraints"])
        self.assertFalse(symmetry["cost_invariance_proven"])
        self.assertFalse(symmetry["known_fused_terms_equal"])
        self.assertFalse(symmetry["merge_for_scoring"])
        self.assertEqual(sorted(p["labeled_placements"] for p in t1["fusion_partitions"]), [16, 32])
        self.assertTrue(all(p["compute_swap_partner_id"] for p in t1["plans"]))

    def test_every_original_edge_recorded_including_fused_internal(self):
        result = verify(self.bundle)
        self.assertEqual(result["t2_rows"], 48 * 7)
        self.assertEqual(result["cross_device_edge_occurrences"], 160)
        rows = [r for r in self.bundle[FILES[2]]["rows"] if (r["producer"], r["consumer"]) == FUSED]
        eliminated = [r for r in rows if r["internal_to_fused_group"]]
        self.assertEqual(len(eliminated), 16)
        self.assertTrue(all(not r["cross_device"] and not r["materialized_tensor"] and r["transfer_bytes"] == 0 for r in eliminated))
        for src, dst in ((0, 1), (1, 0)):
            self.assertEqual(sum(r["cross_device"] and r["source_gpu"] == src and r["destination_gpu"] == dst for r in rows), 8)

    def test_four_requirements_are_derived_with_complete_witnesses(self):
        rows = self.bundle[FILES[2]]["rows"]
        union = self.bundle[FILES[3]]["requirements"]
        self.assertEqual(derive_t3(rows), union)
        self.assertEqual({(u["tensor_bytes"], u["source_gpu"], u["destination_gpu"]) for u in union},
                         {(size, src, dst) for size in (524288, 1572864) for src, dst in ((0, 1), (1, 0))})
        self.assertTrue(all(u["witnesses"] and not u["measurement_ready"] for u in union))

    def test_transfer_semantics_are_not_lost_in_deduplication(self):
        row = copy.deepcopy(next(r for r in self.bundle[FILES[2]]["rows"] if r["cross_device"]))
        other = {**row, "synchronization": "different-dependency-boundary"}
        self.assertEqual(len(derive_t3([row, other])), 2)
        other = {**row, "transfer_mechanism": "different-transport"}
        self.assertEqual(len(derive_t3([row, other])), 2)

    def test_node_constraints_reduce_actual_count_not_theory(self):
        bundle = generate(restrictions={COMPUTE[0]: [0]})
        result = verify(bundle)
        self.assertEqual((result["native_plans"], result["fused_plans"]), (16, 8))
        self.assertEqual(result["legal_labeled_plans"], 24)
        self.assertEqual(bundle[FILES[1]]["summary"]["unconstrained_theory"]["total"], 48)
        self.assertFalse(bundle[FILES[1]]["symmetry"]["compute_only_swap_preserves_declared_constraints"])

    def test_incompatible_devices_remove_fusion_not_native(self):
        bundle = generate(restrictions={FUSED[0]: [0], FUSED[1]: [1]})
        result = verify(bundle)
        self.assertEqual((result["native_plans"], result["fused_plans"]), (8, 0))
        self.assertEqual(result["native_silu_mul_directions"], [(0, 1)])

    def test_boundary_changes_update_edge_directions_without_changing_internal_space(self):
        bundle = generate(input_device=1, output_device=1)
        result = verify(bundle)
        self.assertEqual(result["legal_labeled_plans"], 48)
        expected = sorted((u["tensor_bytes"], 1-u["source_gpu"], 1-u["destination_gpu"], len(u["witnesses"]))
                          for u in self.bundle[FILES[3]]["requirements"])
        actual = sorted((u["tensor_bytes"], u["source_gpu"], u["destination_gpu"], len(u["witnesses"]))
                        for u in bundle[FILES[3]]["requirements"])
        self.assertEqual(expected, actual)

    def test_invalid_restrictions_and_boundary_rejected(self):
        for restrictions in ({"unknown": [0]}, {COMPUTE[0]: [2]}, {COMPUTE[0]: [True]}):
            with self.assertRaises(ValueError):
                generate(restrictions=restrictions)
        with self.assertRaises(ValueError):
            generate(input_device=2)

    def test_tampered_missing_duplicate_or_wrong_edge_records_rejected(self):
        for mutation in ("missing_plan", "duplicate_plan", "edge", "union"):
            with self.subTest(mutation=mutation):
                bundle = copy.deepcopy(self.bundle)
                if mutation == "missing_plan":
                    bundle[FILES[1]]["plans"].pop()
                elif mutation == "duplicate_plan":
                    bundle[FILES[1]]["plans"].append(bundle[FILES[1]]["plans"][0])
                elif mutation == "edge":
                    bundle[FILES[2]]["rows"][0]["source_gpu"] = 1
                else:
                    bundle[FILES[3]]["requirements"].pop()
                with self.assertRaises(AssertionError):
                    verify(bundle)


if __name__ == "__main__":
    unittest.main()
