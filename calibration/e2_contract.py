"""Strict E2 split contracts and label-lineage access control."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .e2_manifest import MANIFEST_FORMAT, content_id, validate_shape_manifest


CONTRACT_FORMAT = "fusion-placement-search.e2-split-contract"
CONTRACT_VERSION = "1.0"

E2_NUMERIC_FEATURES = (
    "batch",
    "m",
    "n",
    "k",
    "normalized_size",
    "numel",
    "flops",
    "memory_bytes",
    "arithmetic_intensity",
    "compute_demand_s",
    "memory_demand_s",
    "launch_s",
)
E2_CATEGORICAL_FEATURES = ("operator", "implementation_id")
E2_RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def _enabled_objects(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        item["object_id"]: item
        for item in manifest["objects"]
        if bool(item["measurement"]["enabled"])
    }


def _objects_used_by(
    objects: Mapping[str, Mapping[str, Any]],
    *,
    model_family: str | None = None,
    sequence_lengths: set[int] | None = None,
) -> set[str]:
    selected: set[str] = set()
    for object_id, item in objects.items():
        for usage in item["usages"]:
            if model_family is not None and usage["model_family"] != model_family:
                continue
            if sequence_lengths is not None and int(usage["sequence_length"]) not in sequence_lengths:
                continue
            selected.add(object_id)
            break
    return selected


def _split_record(
    split_id: str,
    kind: str,
    train: Iterable[str],
    test: Iterable[str],
    train_filter: Mapping[str, Any],
    test_filter: Mapping[str, Any],
) -> dict[str, Any]:
    train_ids = set(train)
    test_ids = set(test)
    return {
        "split_id": split_id,
        "kind": kind,
        "train_object_ids": sorted(train_ids),
        "hidden_object_ids": sorted(test_ids - train_ids),
        "shared_seen_object_ids": sorted(test_ids & train_ids),
        "train_usage_filter": dict(train_filter),
        "test_usage_filter": dict(test_filter),
    }


def build_split_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create model- and sequence-holdout splits without kernel-key leakage."""

    validate_shape_manifest(manifest)
    objects = _enabled_objects(manifest)

    qwen06 = _objects_used_by(objects, model_family="qwen3-0.6b")
    qwen3b = _objects_used_by(objects, model_family="qwen2.5-3b")
    short_sequences = {128, 256, 512}
    long_sequences = {1024}
    short = _objects_used_by(objects, sequence_lengths=short_sequences)
    long = _objects_used_by(objects, sequence_lengths=long_sequences)

    splits = [
        _split_record(
            "model-qwen06-to-qwen25-3b",
            "model_holdout",
            qwen06,
            qwen3b,
            {"model_family": ["qwen3-0.6b"]},
            {"model_family": ["qwen2.5-3b"]},
        ),
        _split_record(
            "sequence-short-to-1024",
            "sequence_holdout",
            short,
            long,
            {"sequence_lengths": sorted(short_sequences)},
            {"sequence_lengths": sorted(long_sequences)},
        ),
    ]
    experiments: list[dict[str, Any]] = []
    for split in splits:
        experiments.extend(
            [
                {
                    "experiment_id": f"{split['split_id']}:zero-shot",
                    "split_id": split["split_id"],
                    "mode": "zero_shot",
                    "reveal_policy": "forbidden",
                },
                {
                    "experiment_id": f"{split['split_id']}:active-calibration",
                    "split_id": split["split_id"],
                    "mode": "active_calibration",
                    "reveal_policy": "selected_hidden_objects_only",
                },
            ]
        )
    contract = {
        "format": CONTRACT_FORMAT,
        "version": CONTRACT_VERSION,
        "manifest_id": manifest["manifest_id"],
        "split_identity": "complete measurement_object_id",
        "splits": splits,
        "experiments": experiments,
        "roles": {
            "predictor": "train labels plus explicitly revealed active-calibration labels",
            "sampler": "same visibility as predictor",
            "oracle_evaluator": "all labels; never available to candidate generation or sampling",
        },
        "prediction_protocol": {
            "target": "log(measured_duration_s / analytic_estimate_s)",
            "baselines": [
                "analytic_only",
                "family_median",
                "nearest_shape",
                "shared_ridge",
            ],
            "numeric_features": list(E2_NUMERIC_FEATURES),
            "categorical_features": list(E2_CATEGORICAL_FEATURES),
            "numeric_transform": "log1p_then_train_fold_standardize",
            "ridge_alphas": list(E2_RIDGE_ALPHAS),
            "ridge_cv_folds": 5,
            "family_uncertainty_min_samples": 5,
            "uncertainty_interval_coverage": 0.90,
            "model_selection_scope": "train_fold_only",
        },
    }
    contract["contract_id"] = content_id("split-contract", contract)
    validate_split_contract(contract, manifest)
    return contract


def _split_by_id(contract: Mapping[str, Any], split_id: str) -> Mapping[str, Any]:
    matches = [split for split in contract["splits"] if split["split_id"] == split_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate split_id: {split_id}")
    return matches[0]


def validate_split_contract(contract: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if contract.get("format") != CONTRACT_FORMAT:
        raise ValueError("unexpected E2 split contract format")
    if contract.get("manifest_id") != manifest.get("manifest_id"):
        raise ValueError("split contract points to a different manifest")
    unhashed = dict(contract)
    claimed_contract_id = unhashed.pop("contract_id", None)
    if claimed_contract_id != content_id("split-contract", unhashed):
        raise ValueError("split contract content hash mismatch")
    known = set(_enabled_objects(manifest))
    split_ids: set[str] = set()
    for split in contract.get("splits", ()):
        split_id = split["split_id"]
        if split_id in split_ids:
            raise ValueError(f"duplicate split_id: {split_id}")
        split_ids.add(split_id)
        train = set(split["train_object_ids"])
        hidden = set(split["hidden_object_ids"])
        shared = set(split["shared_seen_object_ids"])
        if train & hidden:
            raise ValueError(f"train/hidden leakage in split: {split_id}")
        if hidden & shared:
            raise ValueError(f"hidden/shared overlap in split: {split_id}")
        if not hidden:
            raise ValueError(f"split has no genuinely unseen kernel keys: {split_id}")
        if not train:
            raise ValueError(f"split has no training kernel keys: {split_id}")
        unknown = (train | hidden | shared) - known
        if unknown:
            raise ValueError(f"split references unknown or disabled objects: {sorted(unknown)}")
        if not shared <= train:
            raise ValueError(f"shared-seen objects must also be train-visible: {split_id}")


@dataclass(frozen=True)
class LabelArtifact:
    """A raw label or a statistic derived from one or more label artifacts."""

    record_id: str
    kind: str
    source_object_id: str | None = None
    parent_record_ids: tuple[str, ...] = ()


class LabelRegistry:
    """Track transitive label lineage and enforce experiment visibility."""

    def __init__(self, artifacts: Iterable[LabelArtifact] = ()) -> None:
        self._artifacts: dict[str, LabelArtifact] = {}
        for artifact in artifacts:
            self.add(artifact)

    def add(self, artifact: LabelArtifact) -> None:
        if artifact.record_id in self._artifacts:
            raise ValueError(f"duplicate label artifact: {artifact.record_id}")
        if artifact.source_object_id is None and not artifact.parent_record_ids:
            raise ValueError("derived label artifacts require parents")
        missing = set(artifact.parent_record_ids) - set(self._artifacts)
        if missing:
            raise ValueError(f"label artifact has unknown parents: {sorted(missing)}")
        self._artifacts[artifact.record_id] = artifact

    def source_objects(self, record_id: str) -> frozenset[str]:
        if record_id not in self._artifacts:
            raise ValueError(f"unknown label artifact: {record_id}")
        artifact = self._artifacts[record_id]
        sources = set()
        if artifact.source_object_id is not None:
            sources.add(artifact.source_object_id)
        for parent in artifact.parent_record_ids:
            sources.update(self.source_objects(parent))
        return frozenset(sources)

    def visible_record_ids(
        self,
        contract: Mapping[str, Any],
        *,
        split_id: str,
        mode: str,
        role: str,
        revealed_object_ids: Iterable[str] = (),
    ) -> tuple[str, ...]:
        split = _split_by_id(contract, split_id)
        train = set(split["train_object_ids"])
        hidden = set(split["hidden_object_ids"])
        revealed = set(revealed_object_ids)
        if mode not in {"zero_shot", "active_calibration"}:
            raise ValueError(f"unknown E2 mode: {mode}")
        if role not in {"predictor", "sampler", "oracle_evaluator"}:
            raise ValueError(f"unknown E2 role: {role}")
        if mode == "zero_shot" and revealed:
            raise ValueError("zero-shot mode forbids revealing hidden labels")
        if not revealed <= hidden:
            raise ValueError("active calibration can reveal hidden objects only")
        if role == "oracle_evaluator":
            return tuple(sorted(self._artifacts))
        allowed_sources = train | (revealed if mode == "active_calibration" else set())
        visible = [
            record_id
            for record_id in self._artifacts
            if self.source_objects(record_id) <= allowed_sources
        ]
        return tuple(sorted(visible))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = build_split_contract(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    for split in contract["splits"]:
        print(
            f"{split['split_id']}: train={len(split['train_object_ids'])} "
            f"hidden={len(split['hidden_object_ids'])} "
            f"shared_seen={len(split['shared_seen_object_ids'])}"
        )
    print(f"contract_id: {contract['contract_id']}")
    print(f"output: {args.output}")
    print("Fusion E2b strict split contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
