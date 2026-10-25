from __future__ import annotations

import copy
import unittest

from calibration.e2_manifest import (
    build_shape_manifest,
    canonical_json,
    measurement_object_id,
    validate_shape_manifest,
)


class E2ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_shape_manifest()

    def test_complete_kernel_key_hash_is_order_independent(self) -> None:
        key = {"shape": {"m": 4, "n": 8}, "dtype": "bf16"}
        reordered = {"dtype": "bf16", "shape": {"n": 8, "m": 4}}
        self.assertEqual(canonical_json(key), canonical_json(reordered))
        self.assertEqual(measurement_object_id(key), measurement_object_id(reordered))

    def test_default_manifest_is_deterministic_and_valid(self) -> None:
        validate_shape_manifest(self.manifest)
        rebuilt = build_shape_manifest()
        self.assertEqual(self.manifest["manifest_id"], rebuilt["manifest_id"])
        self.assertEqual(self.manifest, rebuilt)
        self.assertEqual(len(self.manifest["models"]), 8)

    def test_objects_are_deduplicated_but_keep_all_usages(self) -> None:
        candidates = []
        for item in self.manifest["objects"]:
            names = {usage["local_name"] for usage in item["usages"]}
            if {
                "self_attention.k_proj",
                "self_attention.v_proj",
            } <= names:
                candidates.append(item)
        self.assertTrue(candidates)
        for item in candidates:
            names = [usage["local_name"] for usage in item["usages"]]
            self.assertIn("self_attention.k_proj", names)
            self.assertIn("self_attention.v_proj", names)

    def test_bmm_identity_is_not_flattened_into_mm(self) -> None:
        implementations = {
            item["kernel_key"]["implementation"]["id"]
            for item in self.manifest["objects"]
            if item["measurement"]["enabled"]
        }
        self.assertIn("torch.mm.reference", implementations)
        self.assertIn("torch.bmm.reference", implementations)
        bmm_objects = [
            item
            for item in self.manifest["objects"]
            if item["kernel_key"]["implementation"]["id"] == "torch.bmm.reference"
        ]
        self.assertTrue(all("batch" in item["kernel_key"]["shape"] for item in bmm_objects))

    def test_rope_key_freezes_layout_and_runtime_definition(self) -> None:
        rope = next(
            item
            for item in self.manifest["objects"]
            if item["kernel_key"]["operator"] == "rope"
        )
        key = rope["kernel_key"]
        self.assertEqual(
            key["layout"]["logical_axes"],
            ["batch", "sequence", "heads", "head_dim"],
        )
        self.assertEqual(key["runtime_flags"]["rotary_base"], 1_000_000)

    def test_enabled_objects_have_a_replayable_measurement_recipe(self) -> None:
        enabled = [
            item for item in self.manifest["objects"] if item["measurement"]["enabled"]
        ]
        self.assertTrue(enabled)
        for item in enabled:
            recipe = item["measurement"]["recipe"]
            self.assertGreater(recipe["warmup"], 0)
            self.assertGreater(recipe["iterations_per_sample"], 0)
            self.assertGreater(recipe["samples"], 0)
            self.assertEqual(
                recipe["implementation_id"],
                item["kernel_key"]["implementation"]["id"],
            )

    def test_tampered_key_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["objects"][0]["kernel_key"]["runtime_flags"]["phase"] = "backward"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_shape_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
