"""Export verified selected-group evidence and demonstrate exact cost lookup."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from calibration.measure_kerv_profile_pair import inspect_trace
from calibration.validate_kerv_adapter import verify_device_trace
from cost_model.fusion_kernel_profile import GroupKernelKeys, key_id
from cost_model.kerv_profile import FORMAT, KervMeasuredProfile, KervProfileStore, digest, make_record, normalize_identity
from cost_model.structural import StructuralCostModel
from integration.flagscale.kerv_execution import load_kerv_registry
from simulator import OpGraph, Plan, default_fusion_registry
from simulator.placement_artifacts import load_placed_op_graph


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_measurement(report_path, placed_path, capability_path, flagscale_root):
    report_path, placed_path = Path(report_path), Path(placed_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry, cap = load_kerv_registry(capability_path, flagscale_root)
    placed = load_placed_op_graph(placed_path, registry=registry)
    if (report["graph_hash"], report["hardware_hash"], report["candidate_id"]) != (
            placed.graph_hash, placed.hardware_hash, placed.candidate_id):
        raise ValueError("report and placed snapshot identities disagree")
    if digest(placed_path) != report["placed_artifact_sha256"]:
        raise ValueError("placed artifact hash mismatch")
    if report["kernel_source_sha256"] != cap["source_sha256"] or report["capability_id"] != cap["capability_id"]:
        raise ValueError("KERV source/capability mismatch")
    for name, expected in report["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("collector source changed; verify with the recorded revision: " + name)
    arm = report.get("arm", "fused")
    if arm == "fused" and report["binding"] != cap["binding"]:
        raise ValueError("fused binding differs from the loaded capability")
    if "loaded_source" in report and arm == "fused":
        proof = report["loaded_source"]
        core = "flagscale/models/kerv/ops/KERVRuntimeOptimization/embodied_ops/kerv_ops.py"
        if proof["module_sha256"] != cap["source_sha256"][core] or proof["git_revision"] != cap["flagscale_revision"]:
            raise ValueError("loaded KERV source proof differs from actual inspected checkout")
    if "loaded_source" in report and arm == "native":
        if report["binding"].get("functional_source_sha256") != report["loaded_source"]["module_sha256"]:
            raise ValueError("loaded Native functional source disagrees with the binding")
    trace_path = report_path.with_name("dispatch_trace.json")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    physical = int(report["runtime_device"].split(":")[-1])
    if any(e["args"].get("device") != physical for e in trace["traceEvents"] if e.get("cat") == "kernel"):
        raise ValueError("device Kernel trace is from another physical GPU")
    if "arm" in report:
        observed = inspect_trace(trace_path, arm)
        if observed != report["execution"]:
            raise ValueError("dispatch report differs from raw trace")
    else:
        observed = verify_device_trace(trace_path)
        if any(report["execution"][k] != v for k, v in observed.items()):
            raise ValueError("legacy dispatch report differs from raw trace")
    samples = json.loads(report_path.with_name("samples.json").read_text(encoding="utf-8"))
    if samples != report["rows"]:
        raise ValueError("raw timing rows differ from report")
    correct = report["correctness"]
    fields = ("up_preserved", "gate_alias", "gate_invalidated") if arm == "fused" else ("up_preserved", "gate_preserved")
    if not all(correct[k] is True for k in fields) or correct["all_replay_outputs_checked"] != sum(r["graph_calls"] for r in samples):
        raise ValueError("numeric/alias checks incomplete")
    if correct["rtol"] != cap["numeric_contract"]["bf16_fp16_rtol"] or correct["atol"] != cap["numeric_contract"]["bf16_fp16_atol"]:
        raise ValueError("numeric tolerance changed")
    dev = next(d for d in trace["deviceProperties"] if d["id"] == int(report["runtime_device"].split(":")[-1]))
    if dev["name"] != report["observed_device"]["name"] or dev["totalGlobalMem"] != report["observed_device"]["total_memory"]:
        raise ValueError("device observation differs from trace")
    return placed, report


def export(source, output, capability=DEFAULT_BUNDLE / "capability.json", flagscale_root=DEFAULT_FLAGSCALE):
    source, output = Path(source), Path(output)
    if list(source.rglob("failure.json")):
        raise ValueError("source contains a failed measurement")
    # Validate every source before writing a fresh output bundle.
    if (source / "gpu/result.json").exists():
        tasks = [("gpu0", "fused", source / "gpu/result.json", source / "prepare/placed.json")]
    else:
        tasks = [(f"gpu{d}", arm, source / f"gpu{d}/{arm}/result.json", source / f"gpu{d}/placed.json")
                 for d in (0, 1) for arm in ("fused", "native")]
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        if not summary["complete"]:
            raise ValueError("paired measurement did not complete both devices")
    prepared = [(case, arm, path, placed_path, *verify_measurement(path, placed_path, capability, flagscale_root))
                for case, arm, path, placed_path in tasks]
    if len(prepared) == 4:
        for case in ("gpu0", "gpu1"):
            pair = {row[1]: row[-1] for row in prepared if row[0] == case}
            for field in ("measurement_contract", "input", "candidate_id", "observed_device"):
                if pair["native"][field] != pair["fused"][field]:
                    raise ValueError("Native/Fused pair differs in " + field)
            for arm in ("native", "fused"):
                for row in pair[arm]["rows"]:
                    expected = int((arm == "native") == (row["round"] % 2 == 0))
                    if row.get("arm_order_index") != expected:
                        raise ValueError("paired arm execution order differs from the recipe")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, arm, path, placed_path, placed, report in prepared:
        if report.get("arm", "fused") != arm:
            raise ValueError("report arm differs from the requested measurement variant")
        destination = output / "inputs" / case / arm
        destination.mkdir(parents=True)
        evidence = {}
        for file in (path, path.with_name("samples.json"), path.with_name("dispatch_trace.json"), placed_path):
            target = destination / file.name
            shutil.copyfile(file, target)
            evidence[target.relative_to(output).as_posix()] = digest(target)
        role = "fused_kernel" if arm == "fused" else "native_chain"
        record, _, _ = make_record(placed, report, role, evidence, case_id=case)
        records.append(record)
    save(output / "profile.json", {"format": FORMAT, "version": "1.0", "records": records,
        "hardware_profile_sharing": False, "automatic_admission": False})
    KervProfileStore.load(output / "profile.json")
    return records


def demonstrate(profile_path, *, allow_conditional_dev=False, admission_path=None):
    profile_path = Path(profile_path)
    if admission_path is not None:
        if allow_conditional_dev:
            raise ValueError("admitted queries must not enable conditional diagnostic bypass")
        from cost_model.kerv_admission import USE, load_admitted_store
        store = load_admitted_store(profile_path, admission_path, usage_scope=USE)
    else:
        store = KervProfileStore.load(profile_path)
    results = []
    for record in store.records.values():
        if record["role"] != "fused_kernel":
            continue
        # Query the record through the shared admission gate first.
        matched = store.query(record["kernel_key"], record["measurement_contract_id"],
                              allow_conditional_dev=allow_conditional_dev)
        report_file = next(p for p in record["evidence"] if p.endswith("/result.json"))
        placed_file = next(p for p in record["evidence"] if p.endswith("/placed.json"))
        report = json.loads((profile_path.parent / report_file).read_text(encoding="utf-8"))
        placed = load_placed_op_graph(profile_path.parent / placed_file)
        original_keys, selected, _, _ = normalize_identity(placed, report)
        # Only the selected group and its two boundary inputs are scored.
        graph = OpGraph(name="selected-kerv-profile-cost-check")
        boundary = (report["gate_node"], report["up_node"])
        for name in (*boundary, *selected.node_ids):
            node = copy.deepcopy(original_keys.graph.nodes[name])
            if name in boundary:
                node.op_type, node.flops, node.param_bytes = "input", 0., 0.
            graph.add_node(node)
        for edge in original_keys.graph.edges:
            if edge.src in graph.nodes and edge.dst in graph.nodes:
                graph.add_edge(copy.deepcopy(edge))
        fallback = default_fusion_registry().get("unfused")
        from simulator import FusionGroup
        inputs = [FusionGroup([name], selected.device, {"pattern_id": fallback.pattern_id, "backend": fallback.backend})
                  for name in boundary]
        plan = Plan([*inputs, selected])
        keys = GroupKernelKeys(graph, original_keys.hardware, original_keys.bindings, original_keys.runtime)
        if key_id(keys.describe(selected)[0]) != record["kernel_key_id"]:
            raise ValueError("boundary extraction changed the Canonical Kernel Key")
        profile = KervMeasuredProfile(keys, store, {selected.device: record["measurement_contract_id"]},
                                      allow_conditional_dev=allow_conditional_dev)
        estimate = StructuralCostModel().predict(graph, plan, profile)
        if not math.isclose(estimate.makespan, matched["statistics"]["median_us"] / 1e6, rel_tol=1e-12):
            raise ValueError("measured duration was not consumed exactly once")
        results.append({"case_id": record["case_id"], "makespan_s": estimate.makespan,
                        "profile_usage": profile.usage, "deployable": False})
        if admission_path is not None:
            results[-1].update(out_of_distribution=estimate.out_of_distribution,
                               admission_decision_id=store.admission_id)
    result = {"scope": "selected fused candidate only; no Native cost decomposition or whole-model search",
              "measured_candidates": len(results), "results": results, "profile_sharing": False}
    if admission_path is not None:
        result.update(admission_use=store.admission_use, admission_decision_id=store.admission_id)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "query"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--capability", type=Path, default=DEFAULT_BUNDLE / "capability.json")
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--allow-conditional-dev", action="store_true")
    parser.add_argument("--admission", type=Path, help="explicit scoped admission decision for strict query")
    args = parser.parse_args(argv)
    if args.action == "export":
        if args.source is None or args.output is None:
            parser.error("export requires --source and --output")
        records = export(args.source, args.output, args.capability, args.flagscale_root)
        print(json.dumps({"records": len(records), "admission": "conditional_dev",
                          "identity_gaps": [r["identity_gaps"] for r in records]}, indent=2))
    else:
        if args.profile is None:
            parser.error("query requires --profile")
        result = demonstrate(args.profile, allow_conditional_dev=args.allow_conditional_dev, admission_path=args.admission)
        if args.output:
            save(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
