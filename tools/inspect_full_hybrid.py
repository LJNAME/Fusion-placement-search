"""One post-hoc hybrid feasibility witness, never added to search results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import build_evaluation_graph
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from tools.evaluate_full_joint import FullLedger, assert_shared_colocated, reference_plans
from tools.evaluate_stage_search import HARDWARE, save


def run(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    graph = build_evaluation_graph(report["scope"]["scale"])
    library = load_library()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    observations, _ = load_calibration()
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile = CalibratedFusionProfile(base, library, corrections)
    registry = library.registry("full")
    space = LegalFusionSearchSpace(graph, 2, registry, placement_policy="hybrid", tie_shared_params=True)
    state = space.initial_state()
    while not space.is_terminal(state):
        node = space.anchor(state)
        device = 1 if node in space.branch_unit else 0
        action = next(a for a in space.actions(state)
                      if a.device == device and len(a.candidate.node_ids) == 1)
        state = space.apply(state, action)
    plan = space.to_plan(state, name="post_hoc_all_branches_gpu1_trunk_gpu0")
    assert_shared_colocated(graph, plan)
    ledger = FullLedger(graph, profile, registry, report["tasks"]["cap75"]["capacity_bytes"])
    ledger.phase = "post_hoc_diagnostic_not_search"
    witness = ledger.evaluate(plan)
    # Check the known legal split reference against this restricted domain.
    reference = reference_plans(graph, registry)["half_layers_owner0_addnorm"]
    target = {n: group for group in reference.groups for n in group.node_ids}
    state, rejection = space.initial_state(), None
    while not space.is_terminal(state):
        node = space.anchor(state)
        group = target[node]
        match = [a for a in space.actions(state) if a.device == group.device
                 and set(a.candidate.node_ids) == set(group.node_ids)
                 and a.candidate.pattern_id == group.extra["pattern_id"]]
        if not match:
            rejection = {"anchor": node, "desired_device": group.device,
                         "desired_nodes": list(group.node_ids),
                         "available_devices": sorted({a.device for a in space.actions(state)}),
                         "reason": "reference_not_representable_in_current_hybrid_actions"}
            break
        state = space.apply(state, match[0])
    result = {"scope": "post-hoc CPU witness; excluded from all 64-call search runs",
              "additional_model_calls": 1, "additional_gpu_measurements": 0,
              "witness": witness, "witness_plan": plan_to_dict(plan),
              "half_layer_reference_rejection": rejection,
              "trunk_nodes": len(space.trunk_nodes), "branch_nodes": len(space.branch_unit),
              "not_a_global_infeasibility_proof": True}
    save(folder.parent / "hybrid_feasibility_diagnostic.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "witness_plan"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    run(parser.parse_args().folder)
