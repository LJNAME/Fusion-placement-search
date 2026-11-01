"""CPU artifact roundtrip and local logical-partition consumer conformance."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.probe_kerv_graph import save
from cost_model.structure import analyze_plan_structure
from integration.partition_contract import (
    balanced_group_assignment, equal_layer_assignment, stage_metrics, validate_stage_assignment,
)
from simulator.placement_artifacts import load_placed_op_graph, placed_op_graph_artifact, save_placed_op_graph
from simulator.schema import edge_bytes
from tools.evaluate_mcts_mainline import file_sha


def split_groups(placed, assignment):
    return [f"fg{i:04d}" for i, group in enumerate(placed.plan.groups)
            if len({assignment[n] for n in group.node_ids}) > 1]


def internal_cut_bytes(placed, assignment):
    groups = {n: i for i, group in enumerate(placed.plan.groups) for n in group.node_ids}
    return sum(edge_bytes(e, placed.graph) for e in placed.graph.edges
               if groups[e.src] == groups[e.dst] and assignment[e.src] != assignment[e.dst])


def inspect_proposal(placed, assignment, count):
    try:
        validate_stage_assignment(placed, assignment, count)
    except ValueError as error:
        return dict(status="rejected", reason=str(error), split_groups=split_groups(placed, assignment),
            would_cut_internal_bytes=internal_cut_bytes(placed, assignment), metrics=None, stage_by_node=assignment)
    return dict(status="accepted_development_proposal", reason=None, split_groups=[],
                metrics=stage_metrics(placed, assignment, count), stage_by_node=assignment)


def audit(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    paths = sorted([*(ROOT / "data/mcts_mainline_v1/best_plans").glob("*.json"),
                    *(ROOT / "data/mcts_medium_capacity_v2/best_plans").glob("*.json")])
    sources = ("tools/audit_output_partition_contract.py", "integration/partition_contract.py",
        "simulator/placement_artifacts.py", "simulator/validate.py", "simulator/fusion_rules.py",
        "simulator/schema.py", "cost_model/structure.py", "cost_model/hardware_profile.py")
    hashes = {str(p.relative_to(ROOT)): file_sha(p) for p in paths}
    code = {p: file_sha(ROOT / p) for p in sources}
    save(output / "plan.json", dict(inputs=hashes, source_sha256=code,
        comparisons=["A respects atomic groups", "B equal layer count", "C operator-count negative control"],
        stage_counts=[1, 2], all_operator_cuts=True,
        scope="local consumer diagnostic, not a downstream PP implementation; original devices preserved",
        formal_capability_available=False, measurement_contract="unconfirmed", label_promotion=False,
        new_search_or_gpu=False))
    records, summary = [], []
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        for number, path in enumerate(paths):
            placed = load_placed_op_graph(path)
            canonical = placed_op_graph_artifact(placed)
            restored_path = output / "roundtrip" / f"{number:02d}.json"
            save_placed_op_graph(restored_path, placed)
            restored_path.write_text(restored_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            restored = load_placed_op_graph(restored_path)
            if placed_op_graph_artifact(restored) != canonical:
                raise ValueError("artifact roundtrip changed graph, groups or metadata")
            layers = sorted({node.extra.get("layer") for node in placed.graph.nodes.values()})
            result = dict(artifact_id=number, source=str(path.relative_to(ROOT)), graph_hash=placed.graph_hash,
                candidate_id=placed.candidate_id, nodes=len(placed.graph.nodes), edges=len(placed.graph.edges),
                groups=len(placed.plan.groups), fused_groups=sum(len(g.node_ids) > 1 for g in placed.plan.groups),
                layers=layers, roundtrip_pass=True, artifact_deployable=bool(placed.extra.get("deployable", False)),
                formal_capability_verified=False, comparisons={})
            for count in (1, 2):
                group_map = balanced_group_assignment(placed, count)
                result["comparisons"][f"A{count}"] = inspect_proposal(placed, group_map, count)
                try:
                    layer_map = equal_layer_assignment(placed, count)
                    result["comparisons"][f"B{count}"] = inspect_proposal(placed, layer_map, count)
                except ValueError as error:
                    if not str(error).startswith("not_applicable:"):
                        raise
                    result["comparisons"][f"B{count}"] = dict(status="not_applicable", reason=str(error), metrics=None)
            order = placed.graph.topological_order()
            operator_map = {n: int(i >= len(order)//2) for i, n in enumerate(order)}
            result["comparisons"]["C2_operator_count_negative_control"] = inspect_proposal(placed, operator_map, 2)
            bad_cuts = []
            for cut in range(1, len(order)):
                assignment = {n: int(i >= cut) for i, n in enumerate(order)}
                split = split_groups(placed, assignment)
                if split:
                    try:
                        validate_stage_assignment(placed, assignment, 2)
                    except ValueError:
                        bad_cuts.append(dict(cut_after_nodes=cut, split_groups=split,
                                             would_cut_internal_bytes=internal_cut_bytes(placed, assignment)))
                    else:
                        raise ValueError("consumer admitted a split atomic group")
            result["all_operator_cuts"] = dict(total=len(order)-1, rejected_atomicity_cases=bad_cuts)
            try:
                validate_stage_assignment(placed, {n: 0 for n in placed.graph.nodes}, 1,
                                          require_registry_executable=True)
            except ValueError as error:
                result["registry_executable_check"] = dict(pass_check=False, reason=str(error))
            else:
                result["registry_executable_check"] = dict(pass_check=True, reason=None)
            failures = []
            for mutation in ("node_id", "dependency", "duplicate_ownership", "device_annotation", "backend_annotation"):
                tampered = copy.deepcopy(canonical)
                if mutation == "node_id":
                    tampered["graph"]["nodes"][0]["id"] = "changed-node-id"
                elif mutation == "dependency":
                    tampered["graph"]["edges"].pop()
                elif mutation == "duplicate_ownership":
                    tampered["fusion_groups"][1]["node_ids"].append(tampered["fusion_groups"][0]["node_ids"][0])
                elif mutation == "device_annotation":
                    tampered["graph"]["nodes"][0]["device_id"] = 1-tampered["graph"]["nodes"][0]["device_id"]
                else:
                    tampered["fusion_groups"][0]["backend"] = "wrong-backend"
                test_path = Path(temporary) / "mutated.json"
                save(test_path, tampered)
                try:
                    load_placed_op_graph(test_path)
                except ValueError as error:
                    failures.append(dict(mutation=mutation, rejected=True, reason=str(error)))
                else:
                    raise ValueError("artifact loader admitted mutation: " + mutation)
            result["rejected_artifact_mutations"] = failures
            if placed_op_graph_artifact(placed) != canonical:
                raise ValueError("consumer mutated source graph")
            records.append(result)
            a = result["comparisons"]["A2"]; b = result["comparisons"]["B2"]
            c = result["comparisons"]["C2_operator_count_negative_control"]
            summary.append(dict(artifact_id=number, source=result["source"], nodes=result["nodes"],
                layers=len(layers), fused_groups=result["fused_groups"],
                A2_status=a["status"], A2_imbalance=a["metrics"]["imbalance_ratio"],
                A2_interstage_bytes=a["metrics"]["interstage_dependency_bytes"],
                B2_status=b["status"], B2_imbalance=None, B2_interstage_bytes=None,
                C2_status=c["status"], C2_imbalance=c["metrics"]["imbalance_ratio"] if c["metrics"] else None,
                C2_interstage_bytes=c["metrics"]["interstage_dependency_bytes"] if c["metrics"] else None,
                operator_cuts=len(order)-1, invalid_operator_cuts=len(bad_cuts),
                registry_executable=result["registry_executable_check"]["pass_check"], deployable=False))
    with (output / "artifacts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False)+"\n")
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    if any(file_sha(ROOT / p) != h for p, h in hashes.items()) or any(file_sha(ROOT / p) != h for p, h in code.items()):
        raise ValueError("input artifacts or source changed")
    report = dict(pass_checks=True, artifacts=len(records), fused_artifacts=sum(r["fused_groups"] > 0 for r in records),
        roundtrip_pass=len(records), artifact_mutations_rejected=sum(len(r["rejected_artifact_mutations"]) for r in records),
        all_single_layer=all(len(r["layers"]) == 1 for r in records),
        actual_equal_layer_B2_applicable=sum(r["B2_status"] != "not_applicable" for r in summary),
        actual_equal_layer_B1_violations=sum(r["comparisons"]["B1"]["status"] != "accepted_development_proposal" for r in records),
        constructed_C2_violations=sum(r["C2_status"] == "rejected" for r in summary),
        operator_cut_proposals=sum(r["operator_cuts"] for r in summary),
        constructed_atomicity_violations_rejected=sum(r["invalid_operator_cuts"] for r in summary),
        planned_registry_rejections=sum(not r["registry_executable_check"]["pass_check"] for r in records),
        formal_capability_verified=False, downstream_schema_confirmed=False, source_artifacts_unchanged=True,
        new_search_runs=0, new_gpu_measurements=0,
        conclusion="local conformance only; no observed equal-layer split violation, PP runtime or speedup claim")
    save(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))


if __name__ == "__main__":
    main()
