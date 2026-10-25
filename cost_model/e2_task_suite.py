"""Freeze label-independent E2 diagnostic tasks before running expanded studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calibration.e2_manifest import content_id


def build_task_suite(manifest, contract):
    tasks = []
    for split in contract["splits"]:
        hidden = set(split["hidden_object_ids"])
        train = sorted(split["train_object_ids"])
        models = sorted({
            usage["model_id"]
            for item in manifest["objects"] if item["object_id"] in hidden
            for usage in item["usages"]
        })
        for model_id in models:
            for scale in ("layer", "model"):
                for budget, seeds in ((None, (None,)), (16, (0, 1, 7)), (32, (0, 1, 7))):
                    for seed in seeds:
                        ranked = sorted(train, key=lambda item: hashlib.sha256(
                            f"{seed}:{item}".encode("utf-8")).hexdigest())
                        selected = sorted(train if budget is None else ranked[:budget])
                        task = {
                            "split_id": split["split_id"], "model_id": model_id,
                            "scale": scale, "initial_train_count": len(selected),
                            "subset_seed": seed, "train_object_ids": selected,
                            "hidden_object_ids": sorted(hidden),
                            "status": "planned",
                            "constraint_config": "existing_homogeneous_two_card_no_extra_constraint",
                        }
                        task["task_id"] = content_id("e2-task", task)
                        tasks.append(task)
    suite = {
        "format": "fusion-placement-search.e2-task-suite", "version": "1.0",
        "manifest_id": manifest["manifest_id"], "contract_id": contract["contract_id"],
        "evidence_scope": "development diagnostics; existing labels already inspected",
        "subset_rule": "sha256(seed:object_id); labels never read",
        "report_rule": "report all tasks including zero-initial-regret tasks",
        "independent_workload_rule": "group statistical analysis by model_id and scale, not subset/MCTS seed",
        "run_status": "planned_not_executed",
        "tasks": tasks,
    }
    suite["suite_id"] = content_id("e2-task-suite", suite)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = build_task_suite(json.loads(args.manifest.read_text(encoding="utf-8")),
                             json.loads(args.contract.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(suite, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"tasks: {len(suite['tasks'])}")
    print(f"suite_id: {suite['suite_id']}")
    print("status: planned_not_executed")


if __name__ == "__main__":
    main()
