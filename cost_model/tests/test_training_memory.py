"""P1-A training-memory static model and Evaluation contract."""

from __future__ import annotations

import unittest

from cost_model import StructuralCostModel
from cost_model.training_memory import (
    Evaluation,
    MemoryBaseline,
    TrainingLifetimeConfig,
    TrainingMemoryConfig,
    estimate_training_lifetime,
    estimate_training_state,
    evaluate_training_candidate,
)
from simulator import FusionGroup, OpGraph, OpNode, Plan, TensorEdge, hardware_from_dict

GB = 1e9


def _hardware(max_mem: float = 1e12):
    devices = [
        {"name": f"d{i}", "flops": 1e12, "mem_bw": 1e11, "max_mem": max_mem, "calibrated": True}
        for i in range(2)
    ]
    return hardware_from_dict(
        {"name": "t", "comm_bw": 1e10, "comm_latency": 1e-6, "link_bw": 1e10, "devices": devices}
    )


def _tied_graph() -> OpGraph:
    """embedding -> mid -> lm_head, lm_head tied to embedding (bf16 bytes)."""

    graph = OpGraph(name="tied", extra={"dtype": "bf16"})
    graph.add_node(OpNode("embedding", "embedding", 0.0, 64.0, 2000.0, extra={}))
    graph.add_node(OpNode("mid", "matmul", 1e6, 64.0, 400.0, extra={}))
    graph.add_node(
        OpNode("lm_head", "matmul", 1e6, 128.0, 0.0, extra={"shared_param_with": "embedding"})
    )
    graph.add_edge(TensorEdge("embedding", "mid", 64.0))
    graph.add_edge(TensorEdge("mid", "lm_head", 64.0))
    return graph


def _plan(devices, extra=None) -> Plan:
    return Plan(
        name="p",
        groups=[
            FusionGroup([nid], dev, dict(extra or {"pattern_id": "unfused"}))
            for nid, dev in zip(("embedding", "mid", "lm_head"), devices)
        ],
    )


class ConfigTest(unittest.TestCase):
    def test_flagscale_default_is_18_bytes(self):
        self.assertEqual(TrainingMemoryConfig().bytes_per_param, 18)
        self.assertEqual(TrainingMemoryConfig.from_flagscale_args().bytes_per_param, 18)

    def test_flagscale_variants(self):
        self.assertEqual(
            TrainingMemoryConfig.from_flagscale_args(grad_reduce_in_bf16=True).bytes_per_param, 16
        )
        # dtype flags are ignored without the precision-aware optimizer
        self.assertEqual(
            TrainingMemoryConfig.from_flagscale_args(exp_avg_dtype="bf16").bytes_per_param, 18
        )
        cfg = TrainingMemoryConfig.from_flagscale_args(
            use_precision_aware_optimizer=True, exp_avg_dtype="bf16", exp_avg_sq_dtype="bf16"
        )
        self.assertEqual(cfg.bytes_per_param, 14)
        with self.assertRaises(NotImplementedError):
            TrainingMemoryConfig.from_flagscale_args(use_distributed_optimizer=True)

    def test_uncalibrated_baseline_cannot_claim_calibration(self):
        with self.assertRaises(ValueError):
            MemoryBaseline((1.0, 1.0), calibrated=True)


class TrainingStateTest(unittest.TestCase):
    def test_per_device_state_and_tied_dedup(self):
        state = estimate_training_state(_tied_graph(), _plan((0, 1, 0)), 2, TrainingMemoryConfig())
        self.assertEqual(state.params_per_device, (1000.0, 200.0))
        self.assertEqual(state.state_bytes, (18000.0, 3600.0))
        self.assertEqual(state.violations, ())

    def test_split_tied_embedding_is_violation(self):
        state = estimate_training_state(_tied_graph(), _plan((0, 0, 1)), 2, TrainingMemoryConfig())
        self.assertEqual(state.violations, ("shared_param_split:embedding:[0, 1]",))
        self.assertEqual(sum(state.params_per_device), 1200.0)  # still counted once

    def test_qwen25_3b_state(self):
        from cost_model.search_evaluation import qwen25_3b_spec
        from simulator.model_graphs.transformer import build_model_graph

        graph = build_model_graph(qwen25_3b_spec())
        plan = Plan(name="one-device", groups=[FusionGroup([nid], 0, {}) for nid in graph.nodes])
        state = estimate_training_state(graph, plan, 2, TrainingMemoryConfig())
        self.assertAlmostEqual(state.params_per_device[0] / 1e9, 3.086, places=3)
        self.assertAlmostEqual(state.state_bytes[0] / GB, 55.55, places=1)


class EvaluationTest(unittest.TestCase):
    def test_contract(self):
        with self.assertRaises(ValueError):
            Evaluation(False, {}, (0,), 1.0, "x")
        with self.assertRaises(ValueError):
            Evaluation(True, {}, (), None, None)

    def test_feasible_peak_adds_static_state_once(self):
        graph, plan, hw = _tied_graph(), _plan((0, 1, 0)), _hardware()
        base = MemoryBaseline((10.0, 20.0))
        result = evaluate_training_candidate(graph, plan, hw, (1e12, 1e12), baseline=base)
        self.assertTrue(result.feasible)
        dynamic = result.breakdown["lifetime_bytes_excluding_weights"]
        self.assertEqual(
            tuple(result.peak_per_device.values()),
            (10.0 + 18000.0 + dynamic[0], 20.0 + 3600.0 + dynamic[1]),
        )
        # weights (2000 / 400 bytes) never appear in the lifetime part
        self.assertLess(dynamic[0], 2000.0)
        self.assertIsNotNone(result.latency)

    def test_static_overflow_checked_per_device(self):
        result = evaluate_training_candidate(
            _tied_graph(), _plan((0, 1, 0)), _hardware(), (17999.0, 1e12)
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.overflow_devices, (0,))
        self.assertEqual(result.reason, "static_memory_overflow")
        self.assertIsNone(result.latency)

    def test_lifetime_overflow(self):
        graph, plan, hw = _tied_graph(), _plan((0, 1, 0)), _hardware()
        # device 1 static is 3600; lifetime adds activations on top
        result = evaluate_training_candidate(graph, plan, hw, (1e12, 3600.0))
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "peak_memory_overflow")
        self.assertEqual(result.overflow_devices, (1,))

    def test_latency_not_computed_for_infeasible(self):
        calls = []
        evaluate_training_candidate(
            _tied_graph(), _plan((0, 0, 1)), _hardware(), (1e12, 1e12),
            latency_fn=lambda g, p: calls.append(1) or 1.0,
        )
        self.assertEqual(calls, [])

    def test_unverified_fused_backward_rejected(self):
        graph = _tied_graph()
        plan = Plan(
            name="fused", groups=
            [
                FusionGroup(["embedding"], 0, {"pattern_id": "unfused"}),
                FusionGroup(["mid", "lm_head"], 0, {"pattern_id": "matmul_chain"}),
            ],
        )
        result = evaluate_training_candidate(
            graph, plan, _hardware(), (1e12, 1e12),
            lifetime_model=StructuralCostModel(strict_fusion=False),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "fusion_backward_unverified:matmul_chain")


def _chain(devices, output_op="add"):
    graph = OpGraph(name="chain", extra={"dtype": "bf16"})
    for nid in ("a", "b", "c"):
        graph.add_node(OpNode(nid, output_op if nid == "a" else "add", 1e3, 64.0, 0.0,
                              extra={"dtype": "bf16"}))
    graph.add_edge(TensorEdge("a", "b", 64.0))
    graph.add_edge(TensorEdge("b", "c", 64.0))
    plan = Plan(name="chain", groups=[
        FusionGroup([nid], dev, {"pattern_id": "unfused"}) for nid, dev in zip("abc", devices)
    ])
    return graph, plan


def _lifetime(graph, plan, **kw):
    model = StructuralCostModel()
    prepared = model.prepare(graph, plan, 2)
    est = model.predict_prepared(prepared, _hardware())
    times = [(g["start"], g["end"]) for g in est.details["groups"]]
    ends = {t["dependency_index"]: t["end"] for t in est.details["transfers"]}
    return estimate_training_lifetime(graph, plan, prepared.structure, times, ends,
                                      est.makespan, **kw)


class TrainingLifetimeTest(unittest.TestCase):
    def test_single_device_chain_hand_computed(self):
        # STRUCTURAL_V1 (training_lifetime_v1): at the start of b's backward:
        # saved a,b,c (3x64) + FP32 loss copy of c (32 elems x 4) + grads of
        # c, b and the new grad of a (3x64); logits held to c's backward.
        life = _lifetime(*_chain((0, 0, 0)))
        self.assertEqual(life.peak_per_device, (512.0, 0.0))
        self.assertEqual(life.m_lm_bytes, (128.0, 0.0))
        self.assertEqual(life.backward_cross_device_bytes, 0.0)

    def test_saved_activation_kept_on_consumer_device(self):
        life = _lifetime(*_chain((0, 1, 1)))
        # a is consumed only on device 1, and add does not save its output,
        # so device 0 holds a only as a send buffer until the transfer ends.
        self.assertEqual(life.backward_cross_device_bytes, 64.0)
        self.assertGreater(life.peak_per_device[1], life.peak_per_device[0])
        self.assertIn("saved_activation", life.component_at_peak[1])

    def test_output_saving_op_keeps_copy_on_producer(self):
        plain = _lifetime(*_chain((0, 1, 1)))
        softmax = _lifetime(*_chain((0, 1, 1), output_op="softmax"))
        self.assertGreater(softmax.peak_per_device[0], plain.peak_per_device[0])

    def test_loss_factor_scales_m_lm(self):
        from cost_model.training_semantics import STRUCTURAL_V1

        life = _lifetime(*_chain((0, 0, 0)), config=TrainingLifetimeConfig(
            semantics=STRUCTURAL_V1.with_loss(fp32_copy_factor=2.0)))
        self.assertEqual(life.m_lm_bytes, (256.0, 0.0))

    def test_fused_recompute_holds_less_than_save(self):
        graph = OpGraph(name="fused", extra={"dtype": "bf16"})
        for nid in ("a", "b", "c", "d", "e"):
            graph.add_node(OpNode(nid, "add", 1e3, 64.0, 0.0, extra={"dtype": "bf16"}))
        graph.add_edge(TensorEdge("a", "b", 64.0))
        graph.add_edge(TensorEdge("b", "c", 64.0))
        graph.add_edge(TensorEdge("c", "d", 64.0))
        graph.add_edge(TensorEdge("d", "e", 64.0))
        # downstream groups make the peak land outside ab's own backward
        plan = Plan(name="f", groups=[
            FusionGroup(["a", "b"], 0, {"pattern_id": "ab"}),
            *[FusionGroup([n], 0, {"pattern_id": "unfused"}) for n in "cde"],
        ])
        model = StructuralCostModel(strict_fusion=False)
        hw = _hardware()
        save = evaluate_training_candidate(graph, plan, hw, (1e12, 1e12), lifetime_model=model,
                                           verified_backward_patterns={"ab": "save"})
        rec = evaluate_training_candidate(graph, plan, hw, (1e12, 1e12), lifetime_model=model,
                                          verified_backward_patterns={"ab": "recompute"})
        self.assertTrue(save.feasible and rec.feasible)
        self.assertLess(rec.peak_per_device[0], save.peak_per_device[0])
        with self.assertRaises(ValueError):
            evaluate_training_candidate(graph, plan, hw, (1e12, 1e12), lifetime_model=model,
                                        verified_backward_patterns={"ab": "maybe"})


def _param_chain(n=6, param=1000.0):  # 1000 bf16 bytes = 500 params
    graph = OpGraph(name="pchain", extra={"dtype": "bf16"})
    ids = [f"n{i}" for i in range(n)]
    for nid in ids:
        graph.add_node(OpNode(nid, "add", 1e3, 64.0, param, extra={"dtype": "bf16"}))
    for a, b in zip(ids, ids[1:]):
        graph.add_edge(TensorEdge(a, b, 64.0))
    return graph


class MctsHardConstraintTest(unittest.TestCase):
    def setUp(self):
        from cost_model.legal_search import legal_mcts_search
        from cost_model.training_memory import make_static_memory_prune, make_training_evaluator

        self.search = legal_mcts_search
        self.graph = _param_chain()
        self.hw = _hardware()
        # 6 nodes x 500 params x 18 B = 54000; one device holds at most 3 nodes
        self.capacity = (30000.0, 30000.0)
        # Mark the toy chain fusion as verified so memory is the binding constraint.
        verified = {"elementwise_chain": "save"}
        self.evaluator = make_training_evaluator(
            self.graph, self.hw, self.capacity,
            lifetime_model=StructuralCostModel(strict_fusion=False),
            verified_backward_patterns=verified,
        )
        self.prune = make_static_memory_prune(
            self.graph, self.capacity, verified_backward_patterns=verified
        )

    def _run(self, **kw):
        return self.search(self.graph, None, 2, iterations=60, rollouts=2, seed=1, n_best=8,
                           require_executable=False, evaluator=self.evaluator, **kw)

    def test_returns_only_feasible_plans(self):
        for prune in (None, self.prune):
            plans = self._run(prune=prune)
            self.assertTrue(plans)
            for plan in plans:
                self.assertTrue(self.evaluator(plan).feasible)
                devices = [g.device for g in plan.groups for _ in g.node_ids]
                self.assertEqual(devices.count(0), 3)

    def test_soft_penalty_path_can_pick_infeasible(self):
        # Reference: a latency-only scorer prefers colocating everything.
        plans = self.search(self.graph, lambda p: float(len({g.device for g in p.groups})), 2,
                            iterations=60, rollouts=2, seed=1, require_executable=False)
        self.assertFalse(self.evaluator(plans[0]).feasible)

    def test_no_feasible_plan_returns_empty(self):
        from cost_model.training_memory import make_static_memory_prune, make_training_evaluator

        cap = (1000.0, 1000.0)
        ev = make_training_evaluator(self.graph, self.hw, cap,
                                     lifetime_model=StructuralCostModel(strict_fusion=False))
        for prune in (None, make_static_memory_prune(self.graph, cap)):
            self.assertEqual(
                self.search(self.graph, None, 2, iterations=20, rollouts=2, seed=1,
                            require_executable=False, evaluator=ev, prune=prune), [])

    def test_prune_is_sound(self):
        from cost_model.legal_search import LegalFusionSearchSpace, rollout_legal_plan
        import random

        space = LegalFusionSearchSpace(self.graph, 2, require_executable=False)
        rng = random.Random(0)
        for _ in range(200):
            state = space.initial_state()
            while not space.is_terminal(state):
                pruned = self.prune(space, state)
                plan = rollout_legal_plan(space, state, rng, validate=False)
                if pruned:
                    self.assertFalse(self.evaluator(plan).feasible)
                state = space.apply(state, rng.choice(space.actions(state)))


class MemoryBudgetTest(unittest.TestCase):
    """P2: hard budget = utilization * max_mem, wired into legal MCTS."""

    def test_default_is_95_percent_of_max_mem(self):
        from cost_model.training_memory import DEFAULT_MEMORY_UTILIZATION, memory_budget_bytes

        self.assertEqual(DEFAULT_MEMORY_UTILIZATION, 0.95)
        hw = _hardware(1000.0)
        self.assertEqual(memory_budget_bytes(hw), (950.0, 950.0))
        self.assertEqual(memory_budget_bytes(hw, 0.5), (500.0, 500.0))
        for bad in (0.0, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                memory_budget_bytes(hw, bad)

    def test_utilization_binds_in_search(self):
        from cost_model.memory_constrained_search import run_memory_constrained_search

        graph = _param_chain()  # 500 params x 18 B = 9000 B per node, 54000 total
        hw = _hardware(30000.0 / 0.95)  # 95% budget = 30000: three nodes per device fit
        kw = dict(lifetime_model=StructuralCostModel(strict_fusion=False), iterations=60,
                  rollouts=2, seed=1, n_best=8, require_executable=False)
        result = run_memory_constrained_search(graph, hw, **kw)
        self.assertTrue(result["plans"])
        for cand in result["report"]["candidates"]:
            self.assertTrue(cand["feasible"])
            self.assertEqual(cand["nodes_per_device"], [3, 3])
            self.assertTrue(all(p <= b for p, b in zip(cand["peak_bytes"], cand["budget_bytes"])))
        self.assertFalse(result["report"]["reference_all_dev0"]["feasible"])
        # At 80% (25263 B) a device holds at most two nodes: 4 < 6, nothing fits.
        tight = run_memory_constrained_search(graph, hw, utilization=0.8, **kw)
        self.assertEqual(tight["plans"], [])
        self.assertEqual(tight["report"]["reason"], "evaluated_no_feasible")


if __name__ == "__main__":
    unittest.main()


class ExecutionSemanticsTest(unittest.TestCase):
    """Backend profiles drive the generic estimator; HF profile vs BW1000 audit."""

    AUDIT_LAYER_BYTES = 41_420_800  # data/training_memory_p1d_v1_audit, seq 256, mbs 1

    @staticmethod
    def _retained(num_layers, semantics):
        from dataclasses import replace

        from cost_model.search_evaluation import qwen25_3b_spec
        from simulator.model_graphs.transformer import build_model_graph

        spec = replace(qwen25_3b_spec(), seq_length=256, micro_batch_size=1,
                       num_layers=num_layers)
        graph = build_model_graph(spec)
        plan = Plan(name="p", groups=[FusionGroup([n], 0, {"pattern_id": "unfused"})
                                      for n in graph.nodes])
        ev = evaluate_training_candidate(
            graph, plan, _hardware(1e15), (1e15, 1e15),
            lifetime_model=StructuralCostModel(strict_fusion=False),
            lifetime_config=TrainingLifetimeConfig(semantics=semantics),
        )
        return ev.breakdown["retained_at_forward_end"][0], ev.breakdown

    def test_hf_profile_matches_audited_layer(self):
        from cost_model.backend_semantics import get_semantics

        hf = get_semantics("hf_qwen2_eager_saved_v1")
        three, breakdown = self._retained(3, hf)
        two, _ = self._retained(2, hf)
        self.assertEqual(three - two, self.AUDIT_LAYER_BYTES)
        self.assertIn(hf.name, breakdown["activation_semantics"])
        self.assertEqual(breakdown["execution_semantics"]["name"], hf.name)

    def test_structural_default_unchanged(self):
        from cost_model.training_semantics import STRUCTURAL_V1

        three, breakdown = self._retained(3, STRUCTURAL_V1)
        two, _ = self._retained(2, STRUCTURAL_V1)
        self.assertEqual(three - two, 37_617_664)  # training_lifetime_v1, 35.875 MiB
        self.assertFalse(breakdown["execution_semantics"]["verified"])

    def test_estimator_names_no_backend(self):
        import inspect

        import cost_model.training_memory as tm

        src = inspect.getsource(tm).lower()
        for word in ("hf_", "qwen2_eager", "transformers==", "forcausallmloss", "repeat_kv"):
            self.assertNotIn(word, src)

    def test_add_keeps_no_input(self):
        from cost_model.training_semantics import ExecutionSemantics, OpSaveRule

        none = ExecutionSemantics("t", ops={"add": OpSaveRule(saves_inputs=False)})
        keep = ExecutionSemantics("t", ops={"add": OpSaveRule(saves_inputs=True)})
        a = _lifetime(*_chain((0, 0, 0)), config=TrainingLifetimeConfig(semantics=none))
        b = _lifetime(*_chain((0, 0, 0)), config=TrainingLifetimeConfig(semantics=keep))
        self.assertLess(a.retained_at_forward_end[0], b.retained_at_forward_end[0])

    def test_internal_fp32_copy(self):
        from cost_model.training_semantics import ExecutionSemantics, OpSaveRule

        sem = ExecutionSemantics("t", ops={"add": OpSaveRule(internal=(("output", "fp32"),))})
        base = _lifetime(*_chain((0, 0, 0)), config=TrainingLifetimeConfig(
            semantics=ExecutionSemantics("t")))
        life = _lifetime(*_chain((0, 0, 0)), config=TrainingLifetimeConfig(semantics=sem))
        # a and b (32 BF16 elements) get one FP32 copy each = 2 x 128 B; the
        # graph output c goes through the loss branch, which has no op rules.
        self.assertEqual(life.retained_at_forward_end[0] - base.retained_at_forward_end[0], 256.0)

    def test_bad_internal_basis_rejected(self):
        from cost_model.training_semantics import OpSaveRule

        with self.assertRaises(ValueError):
            OpSaveRule(internal=(("column", "fp32"),))

    def test_registry(self):
        from cost_model.backend_semantics import available_semantics, get_semantics

        self.assertIn("structural_v1", available_semantics())
        with self.assertRaises(KeyError):
            get_semantics("flagscale_megatron_unaudited")


class MemoryBaselineComponentsTest(unittest.TestCase):
    def test_charged_excludes_slack(self):
        b = MemoryBaseline.from_components((100.0, 50.0), (64.0, 64.0),
                                           allocator_reserved_slack_bytes=(1e9, 0.0),
                                           source="test")
        self.assertEqual(b.per_device, (164.0, 114.0))
        self.assertEqual(MemoryBaseline.from_dict(b.to_dict()), b)

    def test_inconsistent_total_rejected(self):
        with self.assertRaises(ValueError):
            MemoryBaseline((1.0, 1.0), calibrated=True, source="x",
                           runtime_baseline_bytes=(5.0, 5.0))

    def test_margin_is_charged(self):
        b = MemoryBaseline.from_components((10.0,), (0.0,), capacity_safety_margin_bytes=(5.0,),
                                           source="test")
        self.assertEqual(b.per_device, (15.0,))


class TiedWeightGradTest(unittest.TestCase):
    def test_tied_wgrad_held_until_last_user_and_3x_inside(self):
        graph = _tied_graph()
        plan = _plan((0, 0, 0))
        model = StructuralCostModel()
        prepared = model.prepare(graph, plan, 2)
        est = model.predict_prepared(prepared, _hardware())
        times = [(g["start"], g["end"]) for g in est.details["groups"]]
        ends = {t["dependency_index"]: t["end"] for t in est.details["transfers"]}
        from cost_model.training_semantics import ExecutionSemantics, WeightGradSemantics

        sem = ExecutionSemantics("t", weight_grad=WeightGradSemantics(
            accumulate_into_main_grad=False, shared_held_across_users=True,
            shared_extra_copies_per_later_user=2.0))
        life = estimate_training_lifetime(
            graph, plan, prepared.structure, times, ends, est.makespan,
            config=TrainingLifetimeConfig(semantics=sem))
        # embedding (2000 B, tied with lm_head) backward runs last: held
        # first wgrad + new wgrad + out-of-place sum = 3 x 2000, plus mid's
        # 400 B wgrad, whose slot ends exactly when embedding's starts
        # (allocations precede releases at equal times).
        self.assertEqual(life.component_at_peak[0].get("weight_grad_transient"), 6400.0)


class LossBackwardTransientTest(unittest.TestCase):
    def test_loss_backward_fp32_grads_transient(self):
        from cost_model.training_semantics import (
            LOSS_RELEASE_AT_LOSS_BACKWARD, ExecutionSemantics, LossSemantics)

        sem = ExecutionSemantics("t", loss=LossSemantics(
            saves_logits=False, backward_fp32_grads=2.0, release=LOSS_RELEASE_AT_LOSS_BACKWARD))
        life = _lifetime(*_chain((0, 0, 0)), config=TrainingLifetimeConfig(semantics=sem))
        # Loss slot: saved a,b (2x64) + FP32 copy (128) + 2 FP32 grads (256);
        # the BF16 d(logits) is only cast after the FP32 grads are freed.
        self.assertEqual(life.peak_per_device[0], 512.0)
        self.assertEqual(life.component_at_peak[0]["loss_backward_transient"], 256.0)
