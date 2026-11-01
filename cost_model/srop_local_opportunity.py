"""Exposed-development eight-mask audit; no SROP scoring or model writeback."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from calibration.e2_manifest import content_id
from calibration.kerv_silu_mul import ROOT
from calibration.probe_kerv_additivity import verify_bundle
from calibration.probe_kerv_graph import save
from simulator.hardware import load_hardware
from simulator.model_graphs import build_decoder_layer
from simulator.schema import FusionGroup, Plan
from .hardware_profile import AnalyticHardwareProfile
from .search_evaluation import qwen3_spec
from .structure import analyze_plan_structure

KEYS = ("silu", "mul", "fused")
HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
SCENARIOS = (("A_transfer", None, 1.), ("C_hypothesis_minus50", 3.2, .5),
             ("C_hypothesis_center", 3.2, 1.), ("C_hypothesis_plus50", 3.2, 1.5))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_prior(device, seq, hardware_path=HARDWARE, launch_us=None):
    hardware = load_hardware(hardware_path)
    if launch_us is not None:
        hardware = replace(hardware, launch_overhead=launch_us * 1e-6)
    graph = build_decoder_layer(replace(qwen3_spec(1), seq_length=seq))
    silu, mul = "L000.mlp.silu", "L000.mlp.gated_mul"
    result, details = {}, {}
    for fused in (False, True):
        plan = Plan([FusionGroup([silu, mul] if fused and n == silu else [n], device)
            for n in graph.nodes if not (fused and n == mul)])
        structure = analyze_plan_structure(graph, plan, hardware.n_devs, strict_fusion=False)
        for group in structure.groups:
            if silu not in group.node_ids and mul not in group.node_ids:
                continue
            key = "fused" if len(group.node_ids) == 2 else "silu" if silu in group.node_ids else "mul"
            cost = AnalyticHardwareProfile(hardware).estimate_group(graph, group)
            result[key] = cost.duration * 1e6
            details[key] = {"duration_us": cost.duration * 1e6, "launch_us": cost.launch_time * 1e6,
                "compute_us": cost.compute_time * 1e6, "memory_us": cost.memory_time * 1e6,
                "input_bytes": group.input_bytes, "output_bytes": group.output_bytes}
    return result, details


def visible_keys(mask):
    return tuple(key for i, key in enumerate(KEYS) if mask & (1 << i))


def select(prior, visible):
    """Selector receives only public predictions and revealed Kernel labels."""
    if set(prior) != set(KEYS) or not set(visible) <= set(KEYS):
        raise ValueError("selector accepts only the three Kernel keys; chain is evaluator-only")
    values = {k: visible[k] if k in visible else prior[k] for k in KEYS}
    if any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("invalid Kernel prediction")
    costs = {"fused": values["fused"], "unfused": values["silu"] + values["mul"]}
    return ("fused" if costs["fused"] <= costs["unfused"] else "unfused"), costs


def evaluate(choice, reference):
    best = min(reference.values())
    excess = reference[choice] - best
    return {"reference_cost_us": reference[choice], "reference_excess_us": excess,
            "reference_regret": excess / best}


def mask_audit(prior, labels, reference):
    """All 8 states and all 12 directed reveal edges; no hidden labels to select."""
    states, actions = [], []
    full_choice, full_costs = select(prior, labels)
    for mask in range(8):
        visible = {k: labels[k] for k in visible_keys(mask)}
        choice, costs = select(prior, visible)
        row = {"visible_mask": mask, "visible_keys": list(visible), "available_actions": 3-len(visible),
            "choice": choice, "predicted_fused_us": costs["fused"], "predicted_unfused_us": costs["unfused"],
            "predicted_unfused_minus_fused_us": costs["unfused"]-costs["fused"],
            "tied": costs["fused"] == costs["unfused"], **evaluate(choice, reference),
            "mask_minus_full_fused_us": costs["fused"]-full_costs["fused"],
            "mask_minus_full_unfused_us": costs["unfused"]-full_costs["unfused"]}
        for k in KEYS:
            if k in visible:
                continue
            after, _ = select(prior, {**visible, k: labels[k]})
            post = evaluate(after, reference)
            gain = row["reference_excess_us"] - post["reference_excess_us"]
            actions.append({"visible_mask": mask, "key": k, "choice_before": choice,
                "choice_after": after, "selection_changed": choice != after,
                "reference_gain_us": gain, "beneficial": gain > 0, "harmful": gain < 0,
                "reference_regret_after": post["reference_regret"]})
        own = [a for a in actions if a["visible_mask"] == mask]
        row.update(beneficial_actions=sum(a["beneficial"] for a in own),
                   harmful_actions=sum(a["harmful"] for a in own))
        states.append(row)
    return states, actions


def context_diagnostic(states, labels, reference):
    """Post hoc oracle offset, never used by main selection or reveal edges."""
    offset = labels["silu"] + labels["mul"] - reference["unfused"]
    rows = []
    for r in states:
        unfused = r["predicted_unfused_us"] - offset
        choice = "fused" if r["predicted_fused_us"] <= unfused else "unfused"
        rows.append({"visible_mask": r["visible_mask"], "additive_choice": r["choice"],
            "oracle_context_choice": choice, "oracle_context_unfused_us": unfused,
            "oracle_context_regret": evaluate(choice, reference)["reference_regret"],
            "choice_changed": choice != r["choice"], "nonpositive_adjusted_cost": unfused <= 0})
    return {"scope": "post_hoc_oracle_sensitivity_not_an_acquisition_input",
        "hidden_reference_used": True, "single_local_offset_us": offset,
        "offset_definition": "sum of singleton medians minus chain median; not median paired residual or a per-adjacency parameter",
        "rows": rows}


def run_case(report, device, seq, hardware_path=HARDWARE):
    raw = report["rows"]
    by_round = [{r["method"]: r["graph_event_total_s"] * 1e6 / r["graph_calls"]
                 for r in raw if r["round"] == i} for i in range(8)]
    medians = {k: st.median(v[k] for v in by_round) for k in (*KEYS, "chain")}
    labels = {k: medians[k] for k in KEYS}
    reference = {"fused": medians["fused"], "unfused": medians["chain"]}
    scenarios = {}
    for name, launch, multiplier in SCENARIOS:
        prior, details = analytic_prior(device, seq, hardware_path, launch)
        prior["fused"] *= multiplier
        states, actions = mask_audit(prior, labels, reference)
        # Descriptive round sensitivity: each round supplies visible labels;
        # every replicate is evaluated against the SAME frozen median reference.
        round_audits = []
        for i, v in enumerate(by_round):
            rr, aa = mask_audit(prior, {k: v[k] for k in KEYS}, reference)
            round_audits.append({"round": i, "states": rr, "actions": aa})
        scenarios[name] = {"evidence_role": "migration_prior_only" if launch is None else "hypothetical_sensitivity_only",
            "counts_as_same_contract_SROP_evidence": False, "assumed_launch_us": launch,
            "hidden_fused_multiplier": multiplier, "prior_us": prior, "analytic_components": details,
            "states": states, "actions": actions, "round_audits": round_audits,
            "context_diagnostic": context_diagnostic(states, labels, reference)}
    return {"case_id": report["case_id"], "measurement_contract_id": report["measurement_contract_id"],
        "admission": "conditional_dev", "frozen_reference_us": reference,
        "visible_label_medians_us": labels, "B_contract_consistent": {"status": "pending_independent_calibration",
            "states": [{"visible_mask": m, "visible_keys": list(visible_keys(m)), "choice": None,
                        "reference_regret": None} for m in range(8)]}, "scenarios": scenarios}


def flatten(cases, kind):
    return [{"case_id": c["case_id"], "scenario": name, **r}
            for c in cases for name, s in c["scenarios"].items() for r in s[kind]]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})


def run(bundle, output, hardware_path=HARDWARE):
    bundle, output, hardware_path = Path(bundle), Path(output), Path(hardware_path)
    before = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    verification = verify_bundle(bundle)
    if not verification["complete"]:
        raise ValueError("incomplete four-arm bundle")
    output.mkdir(parents=True, exist_ok=False)
    sources = ("cost_model/srop_local_opportunity.py", "cost_model/hardware_profile.py", "cost_model/structure.py",
        "cost_model/search_evaluation.py", "simulator/hardware.py", "simulator/shape_model.py",
        "simulator/model_graphs/dense_layer.py", "simulator/model_graphs/spec.py")
    plan = {"scope": "already_exposed_development_audit_not_preregistered_test_or_SROP_benchmark",
        "raw_sha256": before, "hardware_sha256": sha(hardware_path), "mask_bits": list(KEYS),
        "bit_meaning": "1=visible", "update": "exact queried-key replacement; no refit or MCTS",
        "reference": "fixed per-case medians of fused and chain; chain never a query key",
        "primary_aggregation": "singleton additive; ties choose fused",
        "B_status": "pending_independent_calibration", "scenarios": [list(x) for x in SCENARIOS],
        "round_sensitivity": "8 within-case label replicates evaluated by unchanged median reference; not independent tasks",
        "context_branch": "post hoc hidden-reference local-offset diagnostic only; no fit or main-branch reuse",
        "source_sha256": {s: sha(ROOT / s) for s in sources}, "automatic_model_writeback": False}
    plan["audit_id"] = content_id("local-opportunity-audit", plan)
    save(output / "plan.json", plan)
    cases = []
    for d, seq in ((0,256), (0,1), (1,1), (1,256)):
        report = json.loads((bundle / f"gpu{d}_seq{seq}/result.json").read_text(encoding="utf-8"))
        cases.append(run_case(report, d, seq, hardware_path))
    after = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("input changed during audit")
    result = {"audit_id": plan["audit_id"], "cases": cases, "raw_unchanged": True,
        "same_contract_SROP_evidence": False, "strategy_comparison_executed": False,
        "statistical_significance_claim": False, "automatic_model_writeback": False}
    save(output / "verification.json", verification)
    save(output / "report.json", result)
    write_csv(output / "states.csv", flatten(cases, "states"))
    write_csv(output / "actions.csv", flatten(cases, "actions"))
    round_states = [{"case_id": c["case_id"], "scenario": name, "round": rr["round"], **r}
        for c in cases for name,s in c["scenarios"].items() for rr in s["round_audits"] for r in rr["states"]]
    write_csv(output / "round_states.csv", round_states)
    round_actions = [{"case_id": c["case_id"], "scenario": name, "round": rr["round"], **r}
        for c in cases for name,s in c["scenarios"].items() for rr in s["round_audits"] for r in rr["actions"]]
    write_csv(output / "round_actions.csv", round_actions)
    write_csv(output / "B_pending.csv", [{"case_id": c["case_id"], "prior_regime": "B_contract_consistent",
        "status": c["B_contract_consistent"]["status"], **r}
        for c in cases for r in c["B_contract_consistent"]["states"]])
    write_csv(output / "oracle_context_diagnostic.csv", [{"case_id": c["case_id"], "scenario": name,
        "scope": s["context_diagnostic"]["scope"], **r}
        for c in cases for name,s in c["scenarios"].items() for r in s["context_diagnostic"]["rows"]])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, default=HARDWARE)
    args = parser.parse_args(argv)
    result = run(args.bundle, args.out_dir, args.hardware)
    print("Completed", len(result["cases"]), "development cases; B pending; no SROP superiority claim")


if __name__ == "__main__":
    main()
