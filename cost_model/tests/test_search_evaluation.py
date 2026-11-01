from __future__ import annotations

import unittest
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

from simulator.model_graphs import validate_model_graph
from simulator.dataset import plan_from_dict
from simulator.placement_artifacts import load_placed_op_graph

from cost_model.search_evaluation import (
    _CachedScorer, _contract_partition, _modes, _partition_key, _run_search,
    build_evaluation_graph, evaluate_searches, qwen25_3b_spec, save_evaluation,
)
from cost_model.measured_profile import MeasuredHardwareProfile
from cost_model.hardware_profile import AnalyticHardwareProfile
from simulator.hardware import load_hardware
from simulator import make_reference_plan
from cost_model.verify_search_evaluation import verify_search_evaluation

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return MeasuredHardwareProfile.from_files(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json",
        ROOT / "data/bw1000_2card/raw_measurements.json", allow_unbound=True)


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k,v in value.items() if k != "elapsed_s"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


class TestSearchEvaluation(unittest.TestCase):
    def test_small_search_evaluation(self) -> None:
        result = verify_search_evaluation(allow_unbound=True)
        self.assertEqual(result["nodes"], 5)
        self.assertEqual(result["searches"], 12)
        self.assertEqual(result["joint_space"], 48)

    def test_qwen25_3b_scale_uses_real_model_dimensions(self) -> None:
        spec = qwen25_3b_spec()
        graph = build_evaluation_graph("model_3b")
        summary = validate_model_graph(graph, spec)
        self.assertEqual(summary.parameter_elements, 3_085_846_528)
        self.assertEqual(len(graph.nodes), 687)
        self.assertEqual(graph.extra["num_layers"], 36)

    def test_sequential_odd_budget_and_partition_are_frozen(self):
        p = profile()
        graph = build_evaluation_graph("small")
        scorer = _CachedScorer(graph, p)
        mode = next(m for m in _modes(p.n_devs) if m.name == "sequential")
        plans, _, budget = _run_search("mcts", graph, scorer, mode, seed=23,
            beam=4, rollouts=2, mcts_iterations=17, top_k=3)
        self.assertEqual([s["iterations"] for s in budget["stages"]], [8,9])
        self.assertEqual(sum(s["iterations"] for s in budget["stages"]), 17)
        expected = tuple((r["pattern_id"], tuple(r["node_ids"])) for r in budget["frozen_partition"])
        self.assertTrue(all(_partition_key(plan) == expected for plan in plans))
        self.assertTrue(any(len(g.node_ids)>1 for g in plans[0].groups))
        self.assertEqual([s["allowed_devices"] for s in budget["stages"]], [[0],[0,1]])
        with self.assertRaisesRegex(ValueError, "at least two"):
            _run_search("mcts", graph, scorer, mode, seed=23, beam=4,
                        rollouts=2, mcts_iterations=1, top_k=1)

    def test_macro_device_changes_preserve_every_original_group_attribute(self):
        p = profile()
        graph = build_evaluation_graph("small")
        scorer = _CachedScorer(graph, p)
        mode = next(m for m in _modes(p.n_devs) if m.name == "fusion_only")
        plans, _, _ = _run_search("exact", graph, scorer, mode, seed=23,
            beam=4, rollouts=2, mcts_iterations=16, top_k=1)
        template = plans[0]
        snapshot = copy.deepcopy(template)
        macro, expand = _contract_partition(graph, template, mode.registry, p.n_devs)
        placement_mode = next(m for m in _modes(p.n_devs) if m.name == "placement_only")
        from simulator import make_reference_plan
        placement = make_reference_plan(macro, device=1, registry=placement_mode.registry)
        restored = expand(placement)
        self.assertEqual(_partition_key(restored), _partition_key(template))
        for old,new in zip(template.groups, restored.groups):
            self.assertEqual(old.extra, new.extra)
            self.assertEqual(old.node_ids, new.node_ids)
            self.assertEqual(new.device, 1)
        self.assertEqual(template, snapshot)

    def test_replay_and_export_cover_each_original_node_once(self):
        kwargs = dict(seed=23, beam=4, rollouts=2, mcts_iterations=16, top_k=2)
        a = evaluate_searches(profile(), ("small",), **kwargs)
        b = evaluate_searches(profile(), ("small",), **kwargs)
        self.assertEqual(without_timing(a), without_timing(b))
        scale = a["scales"][0]
        self.assertEqual(scale["configuration"]["seed"], 23)
        for search in scale["searches"]:
            expected = None
            if search["mode"] == "sequential":
                self.assertIsNone(search["optimality_gap"])
                self.assertEqual(search["reference_kind"], "two_stage_exact_not_global_optimum")
                expected = tuple((r["pattern_id"],tuple(r["node_ids"]))
                                 for r in search["budget"]["frozen_partition"])
            for c in search["candidates"]:
                plan = plan_from_dict(c["plan"])
                nodes = [n for g in plan.groups for n in g.node_ids]
                self.assertEqual(len(nodes), len(set(nodes)))
                self.assertEqual(set(nodes), set(build_evaluation_graph("small").nodes))
                self.assertFalse(c["deployable"])
                if expected is not None:
                    self.assertEqual(_partition_key(plan), expected)
        with tempfile.TemporaryDirectory() as directory:
            save_evaluation(Path(directory)/"search.json", a)
            loaded_report = json.loads((Path(directory)/"search.json").read_text())
            for search in loaded_report["scales"][0]["searches"]:
                path = Path(directory)/search["best_placed_op_graph"]
                placed = load_placed_op_graph(path)
                self.assertEqual(len(placed.graph.nodes), 5)
                self.assertFalse(placed.extra["deployable"])
            from tools.verify_search_replay import verify
            second = Path(directory)/"again/search.json"
            save_evaluation(second, b)
            self.assertTrue(verify(Path(directory)/"search.json", second)["pass"])
            changed = json.loads(second.read_text())
            changed["scales"][0]["configuration"]["seed"] = 999
            second.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside elapsed"):
                verify(Path(directory)/"search.json", second)

    def test_sequential_reference_gap_can_be_negative(self):
        # A heuristic can pick a different first-stage fusion partition and beat
        # the two-stage exact procedure; do not label it a global optimum.
        from cost_model.search_evaluation import evaluate_scale
        original = _run_search
        def intercepted(method, graph, scorer, mode, **kwargs):
            plans, elapsed, budget = original(method, graph, scorer, mode, **kwargs)
            if method == "exact" and mode.name == "sequential":
                # Return a worse legal candidate from placement-only space.
                placement = next(m for m in _modes(scorer.profile.n_devs) if m.name=="placement_only")
                plans, _, _ = original("exact", graph, scorer, placement, **kwargs)
            return plans, elapsed, budget
        with patch("cost_model.search_evaluation._run_search", side_effect=intercepted):
            report = evaluate_scale("small", profile(), seed=23, beam=4, rollouts=2, mcts_iterations=16, top_k=1)
        row = next(s for s in report["searches"] if s["mode"]=="sequential" and s["method"]=="mcts")
        self.assertIsNone(row["optimality_gap"])
        self.assertLess(row["reference_gap"], 0.)

    def test_analytic_profile_keeps_logical_input_free(self):
        graph = build_evaluation_graph("layer")
        p = AnalyticHardwareProfile(load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json"))
        scorer = _CachedScorer(graph, p)
        plan = make_reference_plan(graph, device=0)
        prepared = scorer.model.prepare(graph, plan, p.n_devs)
        inputs = [g for g in prepared.structure.groups if graph.nodes[g.node_ids[0]].op_type=="input"]
        self.assertTrue(inputs)
        for group in inputs:
            cost = p.estimate_group(graph, group)
            self.assertEqual((cost.duration,cost.compute_time,cost.memory_time,cost.launch_time), (0.,0.,0.,0.))
            self.assertEqual(cost.source,"logical_input")

    def test_analytic_cli_does_not_load_raw_labels_or_allow_unbound(self):
        from cost_model.search_evaluation import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"search.json"
            args = SimpleNamespace(profile="analytic",scales="small",hardware=ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json",
                raw=Path(directory)/"missing-raw.json",seed=23,beam=4,rollouts=2,mcts_iterations=16,top_k=1,
                quick=False,model_beam=False,measurement_contract_id=None,allow_unbound=False,output=output)
            with patch("cost_model.search_evaluation._parse_args",return_value=args), patch.object(
                    MeasuredHardwareProfile,"from_files",side_effect=AssertionError("raw labels loaded")):
                main()
            report = json.loads(output.read_text())
            self.assertEqual(set(report["inputs"]),{"hardware"})
            self.assertEqual(report["profile_policy"]["kind"],"analytic")
            self.assertFalse(report["profile_policy"]["allow_unbound"])
            args.allow_unbound = True
            with patch("cost_model.search_evaluation._parse_args",return_value=args):
                with self.assertRaisesRegex(ValueError,"omit measurement admission"):
                    main()

    def test_all_beam_modes_respect_their_own_reference_space(self):
        graph=build_evaluation_graph("layer")
        p=AnalyticHardwareProfile(load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json"))
        for mode in _modes(p.n_devs):
            scorer=_CachedScorer(graph,p)
            plans,_,budget=_run_search("beam",graph,scorer,mode,seed=23,
                beam=1,rollouts=1,mcts_iterations=16,top_k=1)
            if mode.name=="sequential":
                # A macro singleton is an already-fused original group. Reusing
                # original-graph singleton references would violate this mode.
                references=[]
                for device in mode.allowed_devices:
                    ref=copy.deepcopy(plans[0])
                    for group in ref.groups:
                        group.device=device
                    references.append(ref)
                expected=tuple((r["pattern_id"],tuple(r["node_ids"])) for r in budget["frozen_partition"])
                self.assertEqual(_partition_key(plans[0]),expected)
            else:
                references=[make_reference_plan(graph,device=d,registry=mode.registry) for d in mode.allowed_devices]
            self.assertLessEqual(scorer.estimate(plans[0]).makespan,
                                 min(scorer.estimate(ref).makespan for ref in references))


if __name__ == "__main__":
    unittest.main()
