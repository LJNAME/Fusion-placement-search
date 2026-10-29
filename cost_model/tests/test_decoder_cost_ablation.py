import copy
from itertools import product
from types import SimpleNamespace
import unittest

from cost_model.decoder_cost_ablation import metrics, serial_duration
from cost_model.hardware_profile import GroupCost, TransferCost
from cost_model.reference_semantics import HostSubmissionProfile
from execution.verify_reference import _placed


def fixture_profile():
    def group(graph, group):
        duration = {"input": 999., "matmul": 2., "silu": 3., "add": 5.}[group.node_ids[0]]
        return GroupCost(duration, duration, 0., 0., 1., "CPU-schema-only")
    return SimpleNamespace(n_devs=2, estimate_group=group,
        estimate_transfer=lambda src, dst, size: TransferCost(7., 1., "CPU-schema-only"))


class DecoderCostAblationTests(unittest.TestCase):
    def test_serial_uses_zero_cached_input_and_full_copy_cost(self):
        placed = _placed()
        profile = fixture_profile()
        self.assertEqual(serial_duration(placed.graph, placed.plan, profile), 38.)
        host = HostSubmissionProfile(node_submit_s=.5, copy_submit_s=.25, start_s=1., finish_s=2.)
        self.assertEqual(serial_duration(placed.graph, placed.plan, profile, host), 44.)

    def test_positive_copy_cost_and_same_service_make_uniform_optimal_in_serial_model(self):
        placed = _placed()
        profile = fixture_profile()
        uniform = copy.deepcopy(placed.plan)
        for group in uniform.groups:
            group.device = 0
        best = serial_duration(placed.graph, uniform, profile)
        for devices in product((0, 1), repeat=4):
            plan = copy.deepcopy(placed.plan)
            for group, device in zip(plan.groups, devices):
                group.device = device
            self.assertGreaterEqual(serial_duration(placed.graph, plan, profile), best)

    def test_equal_regret_does_not_imply_equal_prediction_error(self):
        measured = {"a": .002, "b": .003}
        exact = metrics(measured, measured)
        biased = metrics({"a": .004, "b": .006}, measured)
        self.assertEqual(exact["relative_regret"], biased["relative_regret"])
        self.assertEqual(exact["mape"], 0.)
        self.assertEqual(biased["mape"], 1.)

    def test_roundoff_ties_cannot_choose_the_faster_physical_card_posthoc(self):
        result = metrics({"a": .001, "b": .001 + 1e-13}, {"a": .003, "b": .002})
        self.assertEqual(result["selected_id"], "a")
        self.assertAlmostEqual(result["relative_regret"], .5)
        self.assertEqual(result["predicted_tie_regret_range"], [0., .5])


if __name__ == "__main__":
    unittest.main()
