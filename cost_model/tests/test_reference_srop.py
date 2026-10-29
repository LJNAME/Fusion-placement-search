import copy
import json
from pathlib import Path
import unittest

from calibration.e2_records import load_jsonl_records
from cost_model.e2_srop_proxy import WORKLOADS, build_workload_graph, object_map_for_workload
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.reference_srop import ReferenceCandidateRanker, replay, structural_candidates
from cost_model.search_sensitivity import plan_id, permute_plan_devices
from cost_model.tests.test_reference_semantics import FixedProfile, fixture
from simulator import FusionGroup
from simulator.hardware import load_hardware


def decisions(report):
    return [[{key: point[key] for key in ("budget", "selected_object_ids", "next_selection",
             "next_selection_score", "predicted_top1_plan_id")} for point in run["trace"]]
            for run in report["runs"]]


class ReferenceRankerTests(unittest.TestCase):
    def test_scores_match_existing_model_and_physical_directions_survive(self):
        graph, plan = fixture()
        opposite = permute_plan_devices(plan, (1, 0))
        host = HostSubmissionProfile(node_submit_s=.25)
        ranker = ReferenceCandidateRanker(graph, [plan, opposite], 2, host)
        expected = {plan_id(p): ReferenceEagerCostModel(host).predict(graph, p, FixedProfile()).makespan
                    for p in (plan, opposite)}
        self.assertEqual({row.plan_id: row.makespan for row in ranker.rank(FixedProfile())}, expected)
        self.assertEqual(len(ranker.plans), 2)
        self.assertEqual(len(ranker.rank_many([FixedProfile(), FixedProfile()])), 2)
        self.assertEqual(ranker.calls, 3)

    def test_candidates_and_host_are_snapshots(self):
        graph, plan = fixture()
        host = HostSubmissionProfile(operator_submit_s={"matmul": .5})
        ranker = ReferenceCandidateRanker(graph, [plan], 2, host)
        before = ranker.rank(FixedProfile())
        host.operator_submit_s["matmul"] = 1000
        graph.nodes["a"].extra["output_shape"] = [100]
        plan.groups[0].device = 1
        after = ranker.rank(FixedProfile())
        self.assertEqual([(r.plan_id, r.makespan) for r in before],
                         [(r.plan_id, r.makespan) for r in after])

    def test_multinode_fusion_cannot_become_reference_evidence(self):
        graph, plan = fixture()
        plan.groups[1:3] = [FusionGroup(["a", "b"], 0, {"pattern_id": "planned"})]
        with self.assertRaisesRegex(ValueError, "singleton"):
            ReferenceCandidateRanker(graph, [plan], 2, HostSubmissionProfile())


class ReferenceReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        cls.contract = json.loads((root / "data/bw1000_e2/split_contract.json").read_text(encoding="utf-8"))
        cls.hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        cls.workload = WORKLOADS["model_3b_s256"]
        cls.graph = build_workload_graph(cls.workload, "layer")
        # A small explicit test split; this is not a new research workload.
        split = next(s for s in cls.contract["splits"] if s["split_id"] == cls.workload.split_id)
        mapping = object_map_for_workload(cls.records, cls.workload)
        used = {mapping[str(node.extra.get("local_name", node.id))]
                for node in cls.graph.nodes.values() if node.op_type != "input"}
        cls.hidden = sorted(set(split["hidden_object_ids"]) & used)[:2]
        split["train_object_ids"] += [key for key in split["hidden_object_ids"] if key not in cls.hidden]
        split["hidden_object_ids"] = cls.hidden
        cls.plans = structural_candidates(cls.graph, cls.hardware.n_devs)
        cls.host = HostSubmissionProfile(node_submit_s=2e-6, source="test_fixture")
        cls.original = cls.run_replay()

    @classmethod
    def run_replay(cls, records=None, measured=None):
        return replay(cls.graph, cls.plans, cls.records if records is None else records,
                      cls.contract, cls.hardware, cls.workload, cls.host,
                      measured_plan_times=measured)

    def test_shared_initial_decision_and_full_model_recovery(self):
        report = self.original
        self.assertEqual(len(report["runs"]), 6)
        self.assertEqual(len(report["measurement_pool_ids"]), 2)
        self.assertEqual(len({run["trace"][0]["predicted_top1_plan_id"] for run in report["runs"]}), 1)
        self.assertEqual(report["physical_device_policy"], "identity_only_no_exchange_deduplication")
        self.assertIn("not_gpu_plan_truth", report["metric_reference"])
        self.assertEqual(report["runs"][0]["trace"][0]["budget"], 0)
        self.assertEqual(len(report["runs"][0]["trace"]), 1)
        for run in report["runs"][1:]:
            self.assertEqual(run["trace"][-1]["budget"], len(report["measurement_pool_ids"]))
            self.assertAlmostEqual(run["trace"][-1]["top1_regret_ms"], 0.)

    def test_hidden_labels_do_not_affect_initial_decisions_or_queries(self):
        tampered = copy.deepcopy(self.records)
        for row in tampered:
            if row["object_id"] in self.hidden:
                row["measurement"]["duration_s"] *= 100
                row["target"]["log_residual"] += 5
        changed = self.run_replay(records=tampered)
        original = [rows[0] for rows in decisions(self.original)]
        self.assertEqual(original, [rows[0] for rows in decisions(changed)])

    def test_gpu_labels_are_evaluation_only_and_full_kernel_reveal_need_not_win(self):
        chosen = self.original["runs"][-1]["trace"][-1]["predicted_top1_plan_id"]
        measured = {plan_id(p): .001 for p in self.plans}
        measured[chosen] = .010
        changed = self.run_replay(measured=measured)
        self.assertEqual(decisions(self.original), decisions(changed))
        self.assertEqual(changed["metric_reference"], "supplied_gpu_plan_measurements_evaluator_only")
        self.assertAlmostEqual(changed["runs"][-1]["trace"][-1]["top1_regret_ms"], 9.)

    def test_incomplete_gpu_measurements_and_missing_kernel_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.run_replay(measured={plan_id(self.plans[0]): .001})
        graph = copy.deepcopy(self.graph)
        node = next(node for node in graph.nodes.values() if node.op_type != "input")
        node.extra["local_name"] = "unmapped_operator"
        with self.assertRaisesRegex(ValueError, "missing full measurement keys"):
            replay(graph, self.plans, self.records, self.contract, self.hardware, self.workload, self.host)


if __name__ == "__main__":
    unittest.main()
