"""Fixed-contract SiLU-Mul cost validation using an existing four-arm graph batch.

This is a conditional local fixture, not production Kernel-Key label ingestion.
No profiling, contract promotion, SROP or global-search benefit is performed.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import statistics as st

from calibration.kerv_silu_mul import development_registry
from calibration.probe_kerv_additivity import ARMS, CASES, verify_bundle
from calibration.probe_kerv_graph import save
from cost_model.hardware_profile import AnalyticHardwareProfile, GroupCost
from cost_model.measurement_labels import (
    MeasurementContractError, MeasurementLabel, MeasurementLabelLookup, measurement_contract_id,
)
from cost_model.search_evaluation import _CachedScorer, _Mode, _plan_key, _run_search
from simulator import OpGraph, OpNode, TensorEdge
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import PlacedOpGraph, save_placed_op_graph
from tools.evaluate_mcts_mainline import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"


def paired_preference(paired):
    """Reuse the earlier 1% preference band; this is not a significance test."""
    relative = [(v["fused"]-v["chain"])/v["chain"] for v in paired.values()]
    if abs(st.median(relative)) < .01:
        return "inconclusive_or_small_difference"
    if all(v < 0 for v in relative):
        return "fused_preferred"
    if all(v > 0 for v in relative):
        return "unfused_preferred"
    return "inconclusive_or_small_difference"


def build_fixture_graph(shape, dtype="bf16"):
    if dtype != "bf16" or len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("this source-bound template requires a positive rank-3 bf16 shape")
    count = math.prod(shape); nbytes = count*2
    graph = OpGraph(name="conditional-silu-mul-fixture", extra={"output_node": "mul"})
    for name, op, flops in (("gate", "input", 0.), ("up", "input", 0.),
                             ("silu", "silu", 4.*count), ("mul", "mul", float(count))):
        graph.add_node(OpNode(name, op, flops, nbytes, 0., dict(output_shape=list(shape), dtype=dtype,
            layout="contiguous", phase="forward", layer=0, shape={"n": count})))
    for src, dst in (("gate", "silu"), ("silu", "mul"), ("up", "mul")):
        graph.add_edge(TensorEdge(src, dst, nbytes))
    return graph


def summarize_case(raw):
    contract = raw["measurement_contract"]
    if raw["measurement_contract_id"] != measurement_contract_id(contract):
        raise MeasurementContractError("case contract content hash mismatch")
    if contract["executor"] != "graph_replay" or contract["unit"] != "throughput_per_call":
        raise MeasurementContractError("template requires graph replay amortized-throughput units")
    if raw["admission"] != "conditional_dev":
        raise MeasurementContractError("this fixture cannot change source admission")
    values = {arm: [r["graph_batch_amortized_per_call_us"] for r in raw["rows"] if r["method"] == arm] for arm in ARMS}
    if any(len(v) != 8 for v in values.values()):
        raise ValueError("expected eight complete same-batch four-arm rounds")
    stats = {}
    for arm, v in values.items():
        median = st.median(v)
        stats[arm] = dict(median_us=median, mad_rel=st.median(abs(x-median) for x in v)/median,
                          n_rounds=8, n_calls=400)
    paired = {i: {r["method"]: r["graph_batch_amortized_per_call_us"]
                  for r in raw["rows"] if r["round"] == i} for i in range(8)}
    return dict(case_id=raw["case_id"], shape=raw["shape"], dtype=raw["dtype"],
        device=int(raw["case_id"].split("_")[0][3:]), admission=raw["admission"],
        measurement_contract=copy.deepcopy(contract), measurement_contract_id=raw["measurement_contract_id"],
        statistics=stats, paired=paired)


class ConditionalGraphProfile(AnalyticHardwareProfile):
    """Exact fixture-arm substitution through the shared label-admission gate.

    Arm IDs are explicitly not canonical production Kernel Keys. Only the three
    scalar kernel arms are stored; the joint-chain reference is not searchable.
    """
    def __init__(self, hardware, graph, case, *, expected_contract_id, allow_conditional_dev=False):
        super().__init__(hardware)
        if case["admission"] != "conditional_dev":
            raise MeasurementContractError("fixture permits conditional_dev only; no accepted ingestion")
        if case["measurement_contract"]["executor"] != "graph_replay" or \
           case["measurement_contract"]["unit"] != "throughput_per_call":
            raise MeasurementContractError("fixture requires graph replay throughput semantics")
        if expected_contract_id != case["measurement_contract_id"] or \
           measurement_contract_id(case["measurement_contract"]) != expected_contract_id:
            raise MeasurementContractError("fixture never mixes measurement contracts")
        if any(n.extra.get("output_shape") != case["shape"] or n.extra.get("dtype") != case["dtype"] for n in graph.nodes.values()):
            raise ValueError("fixture tensor shape/dtype does not match measured case")
        self.graph = graph
        self.device = case["device"]
        self.measurement_contract_id = expected_contract_id
        labels = {arm: MeasurementLabel(f"diagnostic_arm:{case['case_id']}:{arm}", expected_contract_id,
            copy.deepcopy(case["statistics"][arm]), case["admission"]) for arm in ("silu", "mul", "fused")}
        lookup = MeasurementLabelLookup({label.key_id: label for label in labels.values()})
        self.matches = {arm: lookup.lookup(label.key_id, expected_contract_id, allow_unbound=allow_conditional_dev)
                        for arm, label in labels.items()}

    def estimate_group(self, graph, group):
        if graph is not self.graph or group.device != self.device:
            raise ValueError("conditional fixture is bound to one graph and physical case index")
        ops = tuple(graph.nodes[n].op_type for n in group.node_ids)
        if ops == ("input",):
            return GroupCost(0., 0., 0., 0., 1., "logical_input")
        arm = "fused" if ops == ("silu", "mul") and group.pattern_id == "silu_mul" else ops[0] if len(ops) == 1 else None
        if arm not in self.matches:
            raise KeyError("unmeasured fixture group; no silent analytic fallback")
        match = self.matches[arm]; duration = match.label.statistics["median_us"]*1e-6
        reasons = (*match.reasons, "fixture_identity_not_production_kernel_key", "downstream_execution_contract_unconfirmed")
        return GroupCost(duration, duration, 0., 0., .15, "conditional_graph_fixture_arm", reasons)


def search_fixture(graph, profile, device, registry, method):
    scorer = _CachedScorer(graph, profile, capacity=tuple(d.max_mem for d in profile.hardware.devices))
    mode = _Mode("fusion_only", registry, False, (device,))
    plans, wall, budget = _run_search(method, graph, scorer, mode, seed=23, beam=8,
                                      rollouts=3, mcts_iterations=16, top_k=2)
    candidates = []
    for plan in plans:
        estimate = scorer.estimate(plan)
        candidates.append(dict(choice="fused" if any(len(g.node_ids)>1 for g in plan.groups) else "unfused",
            predicted_cost_s=estimate.makespan, plan_signature=digest(_plan_key(plan)), plan=plan_to_dict(plan),
            confidence=estimate.confidence, ood_reasons=list(estimate.ood_reasons)))
    if len(candidates) != 2:
        raise ValueError("fixture must retain both fixed-device fusion alternatives")
    return dict(method=method, budget=budget, observed_search_wall_s=wall, candidates=candidates), plans[0]


def run(source, output, *, allow_conditional_dev=False):
    source, output = Path(source), Path(output)
    verified = verify_bundle(source)
    if not verified["complete"]:
        raise ValueError("source GPU batch is incomplete")
    env = json.loads((source / "environment.json").read_text(encoding="utf-8"))
    hardware = load_hardware(HARDWARE)
    registry = development_registry(env["capability"], layout_policy="require_explicit")
    cases, prepared = [], []
    files = [source / "plan.json", source / "environment.json"]
    for device, seq in CASES:
        path = source / f"gpu{device}_seq{seq}/result.json"
        case = summarize_case(json.loads(path.read_text(encoding="utf-8")))
        graph = build_fixture_graph(case["shape"], case["dtype"])
        profile = ConditionalGraphProfile(hardware, graph, case,
            expected_contract_id=case["measurement_contract_id"], allow_conditional_dev=allow_conditional_dev)
        cases.append(case); prepared.append((graph, profile))
        files.extend((path, path.with_name("partial.json")))
    source_hashes = {p.relative_to(source).as_posix(): file_sha(p) for p in files}
    code_files = ("calibration/fusion_cost_validation.py", "cost_model/search_evaluation.py", "cost_model/legal_search.py",
        "cost_model/hardware_profile.py", "cost_model/structural.py", "cost_model/structure.py",
        "cost_model/measurement_labels.py", "calibration/kerv_silu_mul.py", "simulator/shape_model.py")
    code_hashes = {p: file_sha(ROOT/p) for p in code_files}
    output.mkdir(parents=True, exist_ok=False)
    for relative in source_hashes:
        target = output / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    save(output / "plan.json", dict(source_sha256=source_hashes, code_sha256=code_hashes,
        hardware_sha256=file_sha(HARDWARE), case_order=[c["case_id"] for c in cases],
        profiles=["legacy_analytic_prior", "conditional_graph_additive"], methods=["exact", "mcts"],
        seed=23, mcts_iterations=16, rollouts=3, fixed_device_per_case=True,
        allow_conditional_dev=allow_conditional_dev, admission="conditional_dev", deployable=False,
        downstream_contract_confirmed=False, analytic_launch_contract_aligned=False,
        preference_rule="all 8 paired signs agree and absolute median relative difference >= 1%; otherwise inconclusive",
        reference="same-batch fused vs joint chain; chain label never supplied to search",
        identity="diagnostic fixture arm IDs only; no canonical Kernel Key relabeling or accepted export",
        synthetic_slow_fusion="unit test only, not empirical data"))
    rows, searches = [], []
    for case, (graph, measured) in zip(cases, prepared):
        bests = {}
        model_costs = {}
        for name, profile in (("legacy_analytic_prior", AnalyticHardwareProfile(hardware)), ("conditional_graph_additive", measured)):
            for method in ("exact", "mcts"):
                result, winner = search_fixture(graph, profile, case["device"], registry, method)
                result.update(case_id=case["case_id"], profile=name)
                searches.append(result)
                if method == "exact":
                    model_costs[name] = {c["choice"]: c["predicted_cost_s"] for c in result["candidates"]}
                else:
                    bests[name] = result["candidates"][0]["choice"]
                    if result["candidates"][0]["predicted_cost_s"] != min(model_costs[name].values()):
                        raise ValueError("MCTS did not match local Exact reference")
                    estimate = _CachedScorer(graph, profile).estimate(winner)
                    save_placed_op_graph(output / "artifacts" / case["case_id"] / (name+".json"),
                        PlacedOpGraph(graph, hardware, winner, predicted_makespan=estimate.makespan,
                            predicted_peak_memory=estimate.peak_memory,
                            extra=dict(deployable=False, scope="conditional graph cost validation only",
                                measurement_contract_id=case["measurement_contract_id"], admission="conditional_dev")))
        stats = case["statistics"]; paired = case["paired"]
        med = {arm: stats[arm]["median_us"] for arm in ARMS}
        row = dict(case_id=case["case_id"], device=case["device"], shape=json.dumps(case["shape"]),
            measurement_contract_id=case["measurement_contract_id"], admission="conditional_dev",
            fused_us=med["fused"], chain_us=med["chain"], silu_us=med["silu"], mul_us=med["mul"],
            paired_fused_faster_rounds=sum(v["fused"]<v["chain"] for v in paired.values()),
            paired_fusion_saving_ratio=st.median((v["chain"]-v["fused"])/v["chain"] for v in paired.values()),
            paired_additivity_residual_us=st.median(v["silu"]+v["mul"]-v["chain"] for v in paired.values()),
            legacy_analytic_fused_us=model_costs["legacy_analytic_prior"]["fused"]*1e6,
            legacy_analytic_unfused_us=model_costs["legacy_analytic_prior"]["unfused"]*1e6,
            migration_prior_fused_relative_error=(model_costs["legacy_analytic_prior"]["fused"]*1e6-med["fused"])/med["fused"],
            initial_choice=bests["legacy_analytic_prior"], calibrated_choice=bests["conditional_graph_additive"],
            output_choice_changed=bests["legacy_analytic_prior"]!=bests["conditional_graph_additive"],
            paired_preference=paired_preference(paired),
            same_contract_adverse_fusion=paired_preference(paired)=="unfused_preferred")
        rows.append(row)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    with (output / "searches.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for result in searches:
            f.write(json.dumps(result,allow_nan=False)+"\n")
    if any(file_sha(source/p)!=h for p,h in source_hashes.items()) or any(file_sha(ROOT/p)!=h for p,h in code_hashes.items()):
        raise ValueError("source batch or code changed")
    report=dict(complete=True,cases=len(rows), shapes=2, physical_case_indices=2, mcts_runs=8, local_exact_runs=8,
        empirical_adverse_fusion_cases=sum(r["same_contract_adverse_fusion"] for r in rows),
        empirical_output_choice_changes=sum(r["output_choice_changed"] for r in rows),
        admission="conditional_dev", canonical_kernel_labels_exported=0, new_gpu_measurements=0,
        downstream_execution_contract_confirmed=False, ideal_fusion_hypothesis_confirmed=False,
        analytic_implementation_error_isolated=False, statistical_significance_claim=False,
        source_batch_verified=True, source_and_code_unchanged=True,
        evaluation_endpoint="local optimized graph under a conditional replay-work proxy; no whole-plan execution",
        conclusion="workflow regression only; sampled graph cases favor fusion, no empirical decision reversal")
    save(output / "report.json", report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-conditional-dev",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.source_dir,args.output,allow_conditional_dev=args.allow_conditional_dev),indent=2))


if __name__=="__main__":
    main()
