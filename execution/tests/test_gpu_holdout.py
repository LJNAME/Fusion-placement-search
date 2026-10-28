import copy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from execution.gpu_holdout import CanonicalReferenceExecutor, compare_outputs, measure_candidate, summarize_execution, main
from execution.verify_reference import _placed


class GPUHoldoutExecutionTests(unittest.TestCase):
    def test_canonical_tensors_and_outputs_are_identical_across_placements(self):
        placed = _placed()
        reference = copy.deepcopy(placed)
        for group in reference.plan.groups:
            group.device = 0
        templates = {}
        with torch.no_grad():
            expected, _, _, _ = CanonicalReferenceExecutor(reference, ("cpu", "cpu"), 1234, templates).run_once(False)
        parameters = {"seed": 1234, "rtol": .01, "atol": .001, "warmup": 1, "iterations": 2}
        result = measure_candidate(placed, ("cpu", "cpu"), parameters, templates, expected)
        self.assertTrue(result["correctness"]["pass"])
        self.assertEqual(result["correctness"]["max_abs_error"], 0)
        self.assertEqual(len(result["samples_s"]), 2)

    def test_numeric_failure_and_performance_failure_are_not_hidden(self):
        self.assertFalse(compare_outputs(torch.tensor([float("nan")]), torch.ones(1), rtol=.01, atol=.001)["pass"])
        freeze = {"freeze_id": "fixture", "protocol": {"execution": {"repeats": 2, "iterations": 2, "top1_relative_regret_target": .05}},
            "workloads": [{"model_id": "fixture", "predicted_ranking": [{"plan_id": "reference", "makespan_s": 2},
                {"plan_id": "predicted", "makespan_s": 1}], "execution_plan_ids": ["reference", "predicted"],
                "predicted_top1_id": "predicted", "reference_plan_id": "reference"}]}
        records = [dict(status="ok", model_id="fixture", plan_id=key, repeat=repeat, samples_s=[time, time],
                        correctness={"pass": True, "max_abs_error": 0}, peak_memory_bytes=[1, 1])
                   for key, time in (("reference", 1), ("predicted", 2)) for repeat in range(2)]
        report = summarize_execution(freeze, records)
        self.assertTrue(report["correctness_pass"])
        self.assertFalse(report["all_top1_within_5pct"])
        self.assertEqual(report["workloads"][0]["top1_relative_regret"], 1)

    def test_gpu_command_cannot_silently_fall_back_to_cpu(self):
        with TemporaryDirectory() as temp, patch("execution.gpu_holdout.load_freeze", return_value=({}, {})), \
                patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no HIP/CUDA"):
                main(["--out-dir", temp])
