"""CPU control-flow fixtures; never claim a GPU smoke pass."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

from calibration.ffn_b4 import contract, plan, run_smoke
from calibration.ffn_b4_policy import POLICY, active_recipe, collect_grid, diagnostic_time, smoke_pass, formal_quality_review
from cost_model.ffn_measurement_b3 import RECIPE
from cost_model.measurement_labels import measurement_contract_id
from calibration.ffn_b4_runtime import GraphCase, Runtime


def report(row_id, value=1.0):
    return {"row_id": row_id, **{name: True for name in POLICY["smoke"]["required_booleans"]},
            "diagnostic_timings": [diagnostic_time(value)]}


class TestFfnB4Policy(unittest.TestCase):
    def test_smoke_is_boolean_only_even_for_slow_or_unavailable_timing(self):
        for value in (1e12, -1., float("nan"), float("inf")):
            result = report("fixture", value)
            self.assertTrue(smoke_pass(result))
            json.dumps(result, allow_nan=False)
        for flag in POLICY["smoke"]["required_booleans"]:
            result = report("fixture")
            result[flag] = False
            self.assertFalse(smoke_pass(result))

    def test_timing_api_failure_is_diagnostic_not_smoke_failure(self):
        result = report("fixture")
        result["diagnostic_timings"] = [diagnostic_time(error=RuntimeError("Event timing unsupported"))]
        self.assertTrue(smoke_pass(result))

    def test_actual_smoke_replay_still_runs_when_diagnostic_event_api_fails(self):
        calls = []
        case = GraphCase.__new__(GraphCase)
        case.device, case.devices, case.stream = None, (), "fixture_stream"
        case.reset = lambda _: calls.append("reset")
        case.graph = SimpleNamespace(replay=lambda: calls.append("graph_replay"))
        case.outputs = []
        case.check = lambda *_: {"fixture_numeric_ok": True}
        with patch("calibration.ffn_b4_runtime.sync"), patch("torch.cuda.device", return_value=nullcontext()), \
             patch("torch.cuda.stream", return_value=nullcontext()), patch("torch.cuda.Event", side_effect=RuntimeError("timer unavailable")):
            timing, checks = case.replay(0, diagnostic=True)
        self.assertEqual(calls, ["reset", "graph_replay"])
        self.assertIsNone(timing["elapsed_ms"])
        self.assertIn("timer unavailable", timing["error"])
        self.assertTrue(checks["fixture_numeric_ok"])

    def test_cpu_runtime_rejected_before_any_gpu_collection(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU/CUDA fallback is forbidden"):
                Runtime("fixture", {})

    def test_loaded_implementation_identity_preserves_fused_source_version(self):
        runtime = Runtime.__new__(Runtime)
        runtime.build_identity = {"collector_source_sha256": "unit-fixture"}
        runtime.observations = {0: ({"uuid": "unit-fixture"}, {"torch": "unit-fixture"})}
        runtime.loaded, runtime.peer = {}, {}
        native = runtime.identity((0,), {"operator": "torch.matmul"})["implementation"]
        self.assertTrue(native["id"] and native["version"])
        fused = runtime.identity((0,), {"kerv_binding": {"id": "unit-fixture-fused", "version": "source:unit-fixture", "backend": "triton"}})["implementation"]
        self.assertEqual(fused["version"], "source:unit-fixture")
        self.assertNotEqual(native["id"], fused["id"])

    def test_minimum_smoke_contains_the_cross_gpu_link(self):
        rows = POLICY["smoke"]["minimum_rows"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], "down_proj@gpu0")
        self.assertIn("copy/524288/0->1", rows)
        self.assertIn("silu_mul/native@gpu0", rows)
        self.assertIn("silu_mul/fused@gpu0", rows)

    def test_smoke_fail_fast_retains_trace_pool_memory_and_stops(self):
        called = []
        def run(row_id):
            called.append(row_id)
            if row_id == "bad":
                raise RuntimeError("capture fixture failure")
            return report(row_id)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "capture fixture failure"):
                run_smoke(["ok", "bad", "never"], run, directory)
            failure = json.loads((Path(directory)/"failure.json").read_text())
            self.assertEqual(called, ["ok", "bad"])
            self.assertIn("capture fixture failure", failure["traceback"])
            self.assertIn("pool_config", failure)
            self.assertIn("memory", failure)
            self.assertFalse(failure["formal_started"])

    def test_formal_failure_keeps_the_entire_19_cell_grid_without_retry(self):
        called = []
        ids = [f"cell{i}" for i in range(19)]
        def run(row_id):
            called.append(row_id)
            if row_id in {"cell0", "cell8"}:
                raise RuntimeError("cell fixture failure")
            return {"row_id": row_id, "status": "quality_passed"}
        grid = collect_grid(ids, run)
        self.assertEqual(called, ids)
        self.assertEqual(set(grid), set(ids))
        self.assertEqual(grid["cell8"]["status"], "quality_failed")
        self.assertEqual(grid["cell18"]["status"], "quality_passed")

    def test_active_fixed_seven_policy_preserves_old_b3_recipe(self):
        before = copy.deepcopy(RECIPE)
        base = {**copy.deepcopy(RECIPE), "recipe_id": "historical-recipe"}
        active = active_recipe(base)
        self.assertEqual(active["max_attempts"], 7)
        self.assertEqual(POLICY["formal"]["extra_attempts"], 0)
        self.assertEqual(RECIPE, before)
        self.assertEqual(active["base_b3_recipe_id"], "historical-recipe")
        self.assertNotEqual(active["recipe_id"], base["recipe_id"])

    def test_smoke_scope_does_not_change_formal_recipe_identity(self):
        base = {**copy.deepcopy(RECIPE), "recipe_id": "historical-recipe"}
        before = active_recipe(base)["recipe_id"]
        with patch.dict(POLICY["smoke"], {"replays": 99}):
            self.assertEqual(active_recipe(base)["recipe_id"], before)

    def test_invalid_formal_round_does_not_request_an_eighth_attempt(self):
        rows = [{"attempt": i, "numeric_ok": True, "capture_ok": True, "path_ok": True,
                 "arm_us": {"body": 10.}, "anchors": {"gpu0": {"before_us": 100., "after_us": 111. if i == 2 else 100.}}} for i in range(7)]
        quality = formal_quality_review(rows, ("body",), ("gpu0",))
        self.assertFalse(quality["timing_eligible"])
        self.assertEqual(quality["status"], "quality_failed_invalid_round")
        self.assertEqual(quality["extra_attempts_allowed"], 0)
        with self.assertRaises(ValueError):
            formal_quality_review(rows + [rows[-1]], ("body",), ("gpu0",))

    def test_cpu_plan_preserves_superseded_records_and_disables_auto_admission(self):
        frozen = plan()
        self.assertEqual(len(frozen["superseded_profiles"]), 2)
        self.assertTrue(all(r["superseded_by"] == "c4fc415" and r["original_admission"] == "accepted" for r in frozen["superseded_profiles"]))
        self.assertFalse(frozen["gpu_validated"])
        self.assertFalse(frozen["automatic_admission"])
        self.assertEqual(len(frozen["matrix"]["core_rows"]), 18)
        self.assertTrue(all(r["recipe_id"] == frozen["active_recipe"]["recipe_id"] for r in frozen["matrix"]["core_rows"]))

    def test_actual_environment_contract_and_same_device_pair_remain_equal(self):
        frozen = plan()
        rows = [r for r in frozen["matrix"]["core_rows"] if r["row_id"] in ("silu_mul/native@gpu0", "silu_mul/fused@gpu0")]
        identity = {"devices": {"0": {"environment": {"torch": "fixture", "hip": "fixture", "triton": "fixture", "device_model": "fixture"}}}}
        case = SimpleNamespace(device=SimpleNamespace(index=0), identity=identity)
        first, second = (contract(row, case, frozen) for row in rows)
        self.assertEqual(measurement_contract_id(first), measurement_contract_id(second))
        self.assertEqual(first["extra_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
