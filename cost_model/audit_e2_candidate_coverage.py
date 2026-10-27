"""Inspect frozen diagnostic candidate placement coverage without changing pools."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re


def audit_coverage(report, checkpoint_directory):
    checkpoint_root = checkpoint_directory / report["experiment_id"].split(":")[1][:16]
    rows = []
    for task in report["tasks"]:
        path = checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            stored = json.load(handle)
        zero = task["zero"]
        varied_nodes = set()
        varied_layers = set()
        graph_layers = set()
        last_group = -1
        maximum_minority = 0
        for plan in stored["candidate_plans"]:
            groups = plan["groups"]
            for group in groups:
                for node_id in group["node_ids"]:
                    match = re.match(r"L(\d+)\.", node_id)
                    if match:
                        graph_layers.add(int(match.group(1)))
            majority_device = Counter(group["device"] for group in groups).most_common(1)[0][0]
            minority = [index for index, group in enumerate(groups) if group["device"] != majority_device]
            last_group = max(last_group, max(minority, default=-1))
            maximum_minority = max(maximum_minority, len(minority))
            for index in minority:
                for node_id in groups[index]["node_ids"]:
                    varied_nodes.add(node_id)
                    match = re.match(r"L(\d+)\.", node_id)
                    if match:
                        varied_layers.add(int(match.group(1)))
        rows.append({
            "task_id": task["task_id"], "model_id": zero["model_id"],
            "split_id": zero["split_id"], "scale": zero["scale"],
            "initial_train_count": zero["initial_train_count"], "subset_seed": zero["subset_seed"],
            "nodes": zero["nodes"], "candidate_plans": len(stored["candidate_plans"]),
            "initial_top1_regret_ms": zero["top1_regret_ms"],
            "varied_nodes": sorted(varied_nodes), "varied_layer_ids": sorted(varied_layers),
            "graph_layer_ids": sorted(graph_layers),
            "varied_layer_fraction": len(varied_layers) / len(graph_layers) if graph_layers else None,
            "varied_node_count": len(varied_nodes),
            "varied_node_fraction": len(varied_nodes) / zero["nodes"],
            "maximum_minority_groups_in_any_plan": maximum_minority,
            "last_minority_group_index": last_group,
        })
    models = [row for row in rows if row["scale"] == "model"]
    return {
        "format": "fusion-placement-search.e2-candidate-coverage", "version": "1.1",
        "experiment_id": report["experiment_id"],
        "definition": "nodes placed away from each plan's majority device; invariant to global device exchange",
        "scope": "post-hoc diagnostic of existing pools; not a new search or task-selection rule",
        "summary": {
            "configs": len(rows), "model_configs": len(models),
            "model_configs_with_variations_after_first_layer": sum(any(layer > 0 for layer in row["varied_layer_ids"]) for row in models),
            "model_configs_with_variations_in_all_layers": sum(row["varied_layer_fraction"] == 1.0 for row in models),
            "model_max_last_minority_group_index": max((row["last_minority_group_index"] for row in models), default=-1),
            "model_max_varied_nodes": max((row["varied_node_count"] for row in models), default=0),
        },
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/bw1000_e2/diagnostics_checkpoints"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    audit = audit_coverage(report, args.checkpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(audit["summary"], indent=2))


if __name__ == "__main__":
    main()
