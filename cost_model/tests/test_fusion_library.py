"""B: the manifest registers library capabilities; measurements only calibrate the fused cost."""
import json
from pathlib import Path
import tempfile
import unittest

from cost_model.fusion_library import (
    CalibratedFusionProfile, FusionLibraryError, _one, fit_corrections, load_calibration, load_library,
)
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import (
    LegalFusionSearchSpace, _unfused_guided_rollout_legal_plan, legal_mcts_search,
)
from cost_model.search_evaluation import build_evaluation_graph, evaluate_searches
from cost_model.structure import analyze_plan_structure
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.hardware import load_hardware

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/fusion_library_lightop_v1/manifest.json"
SILU = ("L000.mlp.silu", "L000.mlp.gated_mul")


def bw1000():
    return AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))


def durations(profile, graph, ids, pid, registry):
    def total(fused):
        groups = analyze_plan_structure(graph, _one(graph, ids, pid, fused), profile.n_devs,
                                        registry=registry).groups
        return sum(profile.estimate_group(graph, g).duration for g in groups if set(g.node_ids) <= set(ids))
    return total(True), total(False)


def edited_manifest(edit):
    data = json.loads(MANIFEST.read_text())
    edit(data)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    return handle.name


def fused_ids(graph, registry):
    return {c.pattern_id for c in enumerate_fusion_candidates(graph, registry) if len(c.node_ids) > 1}


class ManifestRegistersCapabilities(unittest.TestCase):
    def test_registry_is_registered_fusions_plus_singletons(self):
        lib = load_library()
        self.assertEqual(fused_ids(build_evaluation_graph("layer"), lib.registry()),
                         {"lightop_silu_mul", "lightop_add_rmsnorm"})  # no template fusions
        self.assertEqual(lib.training_backward_policies(), {})  # provider has not declared training

    def test_checks_interface_build_and_status_only(self):
        def unknown(d): d["fusions"][0]["implementation"] = "no_such_op"
        def other_build(d): d["wheel_sha256"] = "0" * 64
        def bad_status(d): d["fusions"][0]["status"] = "ready"
        for edit in (unknown, other_build, bad_status):
            with self.assertRaises(FusionLibraryError, msg=edit.__name__):
                load_library(edited_manifest(edit))

    def test_planned_fusion_is_registered_but_never_searched(self):
        def plan(d):
            d["fusions"][1].update(status="planned", implementation="not_shipped_yet")
        lib = load_library(edited_manifest(plan))
        self.assertEqual(fused_ids(build_evaluation_graph("layer"), lib.registry()), {"lightop_silu_mul"})

    def test_training_support_is_the_providers_declaration(self):
        def declare(d): d["fusions"][1]["training_supported"] = True
        lib = load_library(edited_manifest(declare))
        self.assertEqual(lib.training_backward_policies(), {"lightop_add_rmsnorm": "save"})

    def test_declared_shape_range_limits_matches(self):
        def narrow(d): d["fusions"][0]["supported_shapes"] = [{"dim": -1, "max": 4096}]
        lib = load_library(edited_manifest(narrow))
        self.assertIn("lightop_silu_mul", fused_ids(build_evaluation_graph("layer"), lib.registry()))
        self.assertNotIn("lightop_silu_mul", fused_ids(build_evaluation_graph("model_3b"), lib.registry()))

    def test_unmeasured_shapes_and_fusions_stay_eligible(self):
        graph = build_evaluation_graph("model_3b")  # 11008-wide silu_mul, never measured
        fused = [c for c in enumerate_fusion_candidates(graph, load_library().registry()) if len(c.node_ids) > 1]
        layers = sum(n.op_type == "silu" for n in graph.nodes.values())
        self.assertEqual(sum(c.pattern_id == "lightop_silu_mul" for c in fused), layers)
        self.assertEqual(sum(c.pattern_id == "lightop_add_rmsnorm" for c in fused), layers)


class Scopes(unittest.TestCase):
    def test_development_is_calibrated_subset_full_is_all_implemented(self):
        lib = load_library()
        self.assertEqual({e.fusion_id for e in lib.eligible("full")},
                         {"lightop_silu_mul", "lightop_add_rmsnorm", "lightop_qk_rmsnorm_rope"})
        self.assertEqual({e.fusion_id for e in lib.eligible("development")}, {"lightop_silu_mul"})

    def test_calibration_flags_must_match_observations(self):
        lib = load_library()
        obs, _ = load_calibration()
        lib.check_calibration(obs)
        with self.assertRaises(FusionLibraryError):
            lib.check_calibration([])  # silu_mul declares calibration=true
        with self.assertRaises(FusionLibraryError):
            lib.check_calibration(obs + [{**obs[0], "pattern_id": "lightop_add_rmsnorm"}])

    def test_development_scope_requires_calibration(self):
        with self.assertRaises(FusionLibraryError):
            evaluate_searches(bw1000(), ["layer"], mcts_iterations=2, beam=2, rollouts=1,
                              fusion_library=load_library(), fusion_scope="development")


class CalibrationOnlyScalesCost(unittest.TestCase):
    def setUp(self):
        self.lib, self.base = load_library(), bw1000()
        self.obs, _ = load_calibration()
        self.corr, _ = fit_corrections(self.lib, self.base, self.obs, build_evaluation_graph)

    def test_calibrated_cost_reproduces_the_measured_ratio(self):
        graph = build_evaluation_graph("layer")
        cal = CalibratedFusionProfile(self.base, self.lib, self.corr)
        fused, _ = durations(cal, graph, SILU, "lightop_silu_mul", self.lib.registry())
        _, native = durations(self.base, graph, SILU, "lightop_silu_mul", self.lib.registry())
        self.assertAlmostEqual(fused / native, self.obs[0]["fused_us"] / self.obs[0]["native_us"], places=9)

    def test_correction_transfers_to_an_unmeasured_shape(self):
        graph = build_evaluation_graph("model_3b")
        ids = next(c.node_ids for c in enumerate_fusion_candidates(graph, self.lib.registry())
                   if c.pattern_id == "lightop_silu_mul")
        cal = CalibratedFusionProfile(self.base, self.lib, self.corr)
        analytic, _ = durations(self.base, graph, ids, "lightop_silu_mul", self.lib.registry())
        calibrated, _ = durations(cal, graph, ids, "lightop_silu_mul", self.lib.registry())
        self.assertAlmostEqual(calibrated / analytic, self.corr["lightop_silu_mul"], places=9)

    def test_uncalibrated_fusion_is_priced_and_flagged(self):
        graph = build_evaluation_graph("layer")
        cal = CalibratedFusionProfile(self.base, self.lib, self.corr)
        ids = ("L000.residual_add_1", "L000.pre_mlp_norm")
        groups = analyze_plan_structure(graph, _one(graph, ids, "lightop_add_rmsnorm", True), 2,
                                        registry=self.lib.registry()).groups
        fused = next(g for g in groups if len(g.node_ids) > 1)
        cost = cal.estimate_group(graph, fused)
        self.assertEqual(cost.duration, self.base.estimate_group(graph, fused).duration)
        self.assertIn("pattern:lightop_add_rmsnorm:no_calibration", cost.ood_reasons)


class UnfusedIncumbent(unittest.TestCase):
    def test_mcts_never_returns_worse_than_unfused(self):
        graph, lib = build_evaluation_graph("model_3b"), load_library()
        obs, _ = load_calibration()
        profile = CalibratedFusionProfile(bw1000(), lib, fit_corrections(lib, bw1000(), obs,
                                                                        build_evaluation_graph)[0])
        from cost_model.structural import StructuralCostModel
        model = StructuralCostModel(registry=lib.registry())
        score = lambda plan: model.predict(graph, plan, profile).makespan  # noqa: E731
        common = dict(iterations=4, rollouts=1, fusion_probability=.7, seed=23, n_best=1,
                      registry=lib.registry(), require_executable=False, allowed_devices=(0,))
        seeded = legal_mcts_search(graph, score, 2, unfused_incumbent=True, **common)[0]
        space = LegalFusionSearchSpace(graph, 2, lib.registry(), require_executable=False,
                                       allowed_devices=(0,))
        unfused = _unfused_guided_rollout_legal_plan(space, space.initial_state())
        self.assertLessEqual(score(seeded), score(unfused))
        plain = legal_mcts_search(graph, score, 2, **common)[0]
        self.assertGreater(score(plain), score(unfused))  # the gap the incumbent closes


class SearchUsesTheLibrary(unittest.TestCase):
    def test_full_scope_forward_and_training(self):
        lib = load_library()
        common = dict(seed=23, mcts_iterations=8, beam=2, rollouts=1, top_k=2, fusion_library=lib)
        forward = evaluate_searches(bw1000(), ["layer"], **common)["scales"][0]
        training = evaluate_searches(bw1000(), ["layer"], memory_model="training", constraint="hard",
                                     **common)["scales"][0]
        self.assertTrue(forward["configuration"]["unfused_incumbent"])
        used = {p for s in forward["searches"] for c in s["candidates"] for p in c["patterns"]}
        self.assertLessEqual(used, {"lightop_silu_mul", "lightop_add_rmsnorm"})
        self.assertTrue(used)
        for search in training["searches"]:  # nothing is declared trainable
            self.assertIsNotNone(search["plan"])
            self.assertTrue(all(not c["patterns"] for c in search["candidates"]))


if __name__ == "__main__":
    unittest.main()
