from __future__ import annotations

import unittest

from calibration.e2_contract import (
    LabelArtifact,
    LabelRegistry,
    build_split_contract,
    validate_split_contract,
)
from calibration.e2_manifest import build_shape_manifest


class E2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_shape_manifest()
        cls.contract = build_split_contract(cls.manifest)

    def test_contract_has_strict_model_and_sequence_splits(self) -> None:
        validate_split_contract(self.contract, self.manifest)
        self.assertEqual(
            {split["kind"] for split in self.contract["splits"]},
            {"model_holdout", "sequence_holdout"},
        )
        for split in self.contract["splits"]:
            train = set(split["train_object_ids"])
            hidden = set(split["hidden_object_ids"])
            shared = set(split["shared_seen_object_ids"])
            self.assertFalse(train & hidden)
            self.assertFalse(hidden & shared)
            self.assertTrue(hidden)
            self.assertTrue(shared <= train)

    def test_shared_test_keys_are_not_claimed_as_unseen(self) -> None:
        model_split = next(
            split for split in self.contract["splits"] if split["kind"] == "model_holdout"
        )
        self.assertTrue(model_split["shared_seen_object_ids"])
        self.assertFalse(
            set(model_split["shared_seen_object_ids"])
            & set(model_split["hidden_object_ids"])
        )

    def _registry_and_ids(self):
        split = self.contract["splits"][0]
        train_id = split["train_object_ids"][0]
        hidden_id = split["hidden_object_ids"][0]
        registry = LabelRegistry(
            [
                LabelArtifact("raw-train", "raw_measurement", train_id),
                LabelArtifact("raw-hidden", "raw_measurement", hidden_id),
                LabelArtifact(
                    "derived-train", "family_median", parent_record_ids=("raw-train",)
                ),
                LabelArtifact(
                    "derived-hidden", "family_median", parent_record_ids=("raw-hidden",)
                ),
                LabelArtifact(
                    "derived-mixed",
                    "normalized_statistic",
                    parent_record_ids=("raw-train", "raw-hidden"),
                ),
            ]
        )
        return split, registry, train_id, hidden_id

    def test_zero_shot_hides_raw_and_derived_hidden_labels(self) -> None:
        split, registry, _, _ = self._registry_and_ids()
        visible = set(
            registry.visible_record_ids(
                self.contract,
                split_id=split["split_id"],
                mode="zero_shot",
                role="predictor",
            )
        )
        self.assertEqual(visible, {"raw-train", "derived-train"})

    def test_active_calibration_reveals_selected_lineage_only(self) -> None:
        split, registry, _, hidden_id = self._registry_and_ids()
        visible = set(
            registry.visible_record_ids(
                self.contract,
                split_id=split["split_id"],
                mode="active_calibration",
                role="sampler",
                revealed_object_ids=(hidden_id,),
            )
        )
        self.assertEqual(
            visible,
            {
                "raw-train",
                "raw-hidden",
                "derived-train",
                "derived-hidden",
                "derived-mixed",
            },
        )

    def test_zero_shot_cannot_reveal_and_active_cannot_reveal_train(self) -> None:
        split, registry, train_id, hidden_id = self._registry_and_ids()
        with self.assertRaisesRegex(ValueError, "zero-shot"):
            registry.visible_record_ids(
                self.contract,
                split_id=split["split_id"],
                mode="zero_shot",
                role="predictor",
                revealed_object_ids=(hidden_id,),
            )
        with self.assertRaisesRegex(ValueError, "hidden objects only"):
            registry.visible_record_ids(
                self.contract,
                split_id=split["split_id"],
                mode="active_calibration",
                role="predictor",
                revealed_object_ids=(train_id,),
            )

    def test_oracle_evaluator_does_not_change_predictor_visibility(self) -> None:
        split, registry, _, _ = self._registry_and_ids()
        oracle = set(
            registry.visible_record_ids(
                self.contract,
                split_id=split["split_id"],
                mode="zero_shot",
                role="oracle_evaluator",
            )
        )
        predictor = set(
            registry.visible_record_ids(
                self.contract,
                split_id=split["split_id"],
                mode="zero_shot",
                role="predictor",
            )
        )
        self.assertEqual(len(oracle), 5)
        self.assertEqual(predictor, {"raw-train", "derived-train"})


if __name__ == "__main__":
    unittest.main()
