"""MCTS observer preserves behavior and separates search/reference scopes."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import _CachedScorer,_modes,_run_search,build_evaluation_graph
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import load_placed_op_graph
from tools.evaluate_mcts_mainline import ObservedScorer,gap,one_run,run

ROOT=Path(__file__).resolve().parents[2]


def profile():
    return AnalyticHardwareProfile(load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json"))


class TestMCTSMainlineAudit(unittest.TestCase):
    def test_observer_does_not_change_plans_or_scoring_counts(self):
        p=profile();g=build_evaluation_graph("small")
        for mode in _modes(p.n_devs):
            plain=_CachedScorer(g,p);observed=ObservedScorer(g,p)
            kwargs=dict(seed=23,beam=8,rollouts=3,mcts_iterations=16,top_k=4)
            a,_,ma=_run_search("mcts",g,plain,mode,**kwargs)
            b,_,mb=_run_search("mcts",g,observed,mode,**kwargs)
            self.assertEqual(list(map(plan_to_dict,a)),list(map(plan_to_dict,b)))
            for key in ("scorer_requests","unique_requested_plans","cost_model_evaluations"):
                self.assertEqual(ma[key],mb[key])
            self.assertEqual(len(observed.observed),observed.calls)
            self.assertEqual(len(observed.requests),mb["scorer_requests"])

    def test_joint_scores_both_axes_and_sequential_placement_freezes_one(self):
        p=profile()
        mode=next(m for m in _modes(p.n_devs) if m.name=="joint")
        result,_,_,_=one_run(p,"layer",mode,"mcts",16,23)
        self.assertEqual(result["observations"]["unique_fusion_partitions"],2)
        self.assertGreater(result["observations"]["unique_node_device_assignments"],1)
        self.assertGreater(result["observations"]["cross_device_candidates"],0)
        mode=next(m for m in _modes(p.n_devs) if m.name=="sequential")
        result,_,_,_=one_run(p,"layer",mode,"mcts",16,23)
        self.assertEqual(result["stage_observations"][1]["unique_fusion_partitions"],1)

    def test_small_exact_and_layer_references_keep_their_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/"audit"
            with patch("cost_model.search_evaluation.legal_beam_search",side_effect=AssertionError("Beam called")):
                report=run(out,seeds=(0,),budgets=(2,),reference_iterations=4,small_iterations=2)
            self.assertEqual(report["exact_runs"],4)
            self.assertEqual(report["mcts_runs"],12)
            self.assertEqual(report["small_joint_space"],48)
            self.assertFalse(report["beam_executed"])
            runs=[json.loads(line) for line in (out/"runs.jsonl").read_text().splitlines()]
            by_id={r["run_id"]:r for r in runs}
            frozen=json.loads((out/"layer_frozen_reference.json").read_text())
            self.assertTrue(all(by_id[i]["phase"]=="layer_high_reference" for i in frozen["all_reference_run_ids"]))
            grid_ids=[r["run_id"] for r in runs if r["phase"]=="layer_grid"]
            self.assertLess(max(frozen["all_reference_run_ids"]),min(grid_ids))
            rows=list(csv.DictReader((out/"results.csv").open(encoding="utf-8")))
            placement=next(r for r in rows if r["scale"]=="small" and r["mode"]=="placement_only")
            self.assertEqual(float(placement["restricted_optimality_gap"]),0.)
            self.assertGreater(float(placement["joint_exact_gap"]),0.)
            sequential=next(r for r in rows if r["scale"]=="small" and r["mode"]=="sequential")
            self.assertEqual(sequential["restricted_optimality_gap"],"")
            self.assertEqual(float(sequential["two_stage_exact_reference_gap"]),0.)
            with gzip.open(out/"score_requests.csv.gz","rt",encoding="utf-8") as f:
                requests=list(csv.DictReader(f))
            self.assertEqual(len(requests),sum(r["budget"]["scorer_requests"] for r in runs))
            for entry in json.loads((out/"best_plans.json").read_text()).values():
                placed=load_placed_op_graph(out/entry["artifact"])
                self.assertFalse(placed.extra["deployable"])
            from tools.verify_mcts_mainline import verify
            self.assertTrue(verify(out)["pass_checks"])
            rows[0]["joint_exact_gap"]="123"
            with (out/"results.csv").open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
                writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,"reference gap mismatch"):
                verify(out)

    def test_gap_to_discovered_reference_is_signed(self):
        self.assertLess(gap(4.,5.),0.)
        self.assertEqual(gap(4.,4.),0.)


if __name__=="__main__":
    unittest.main()
