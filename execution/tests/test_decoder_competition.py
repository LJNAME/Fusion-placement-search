import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from calibration.decoder_competition import build_freeze, build_schedule, check
from calibration.e2_manifest import content_id
from execution.audit_gpu_holdout import write_json
from execution.decoder_competition import summarize, collect_environment
from simulator.dataset import plan_from_dict
from cost_model.search_sensitivity import plan_id, permute_plan_devices


def schema_fixture():
    protocol = {"iterations": 20, "repeats": 3, "order_seed": 20261029,
        "anchor_drift_limit": .1, "noise_relative_floor": .001, "noise_mad_multiplier": 3.,
        "epsilon_values": [.001, .005, .01], "primary_epsilon": .01}
    task = {"task_id": "CPU-schema-fixture", "plans": {"u0": {}, "u1": {}, "cross": {}},
        "uniform_plan_ids": ["u0", "u1"], "initial_selected_id": "u0", "initial_predicted_ties": ["u0", "u1"]}
    frozen = {"freeze_id": "schema-only-not-GPU", "protocol": protocol, "workloads": [task]}
    frozen["schedule"] = build_schedule([task], protocol)
    environment = {"host": "CPU-schema-fixture"}
    rows = []
    for slot in frozen["schedule"]:
        duration = {"u0": .006, "u1": .004, "cross": .003}[slot["plan_id"]]
        row = {**slot, "freeze_id": frozen["freeze_id"], "environment_id": content_id("a3-environment", environment),
            "collection_run_id": "fixture-run", "status": "ok", "samples_s": [duration] * 20,
            "median_s": duration, "correctness": {"pass": True, "finite": True}}
        seal(row)
        rows.append(row)
    return frozen, environment, rows


def seal(row):
    row.pop("record_id", None)
    row["record_id"] = content_id("a3-execution-block", row)


class DecoderCompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = build_freeze()

    def test_structure_selection_counts_directions_and_schedule(self):
        f = self.frozen
        self.assertEqual([len(w["plans"]) for w in f["workloads"]], [28, 32, 28, 32])
        self.assertEqual(len(f["schedule"]), 384)
        self.assertEqual(len({s["slot_id"] for s in f["schedule"]}), 384)
        self.assertEqual(f["schedule"], build_schedule(f["workloads"], f["protocol"]))
        for task in f["workloads"]:
            for raw in task["plans"].values():
                mirror = plan_id(permute_plan_devices(plan_from_dict(raw), (1, 0)))
                self.assertIn(mirror, task["plans"])
            self.assertEqual(task["initial_selected_id"], min(task["initial_predicted_ties"]))

    def test_freeze_roundtrip_and_source_tampering(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            write_json(path / "freeze.json", self.frozen)
            self.assertEqual(check(path)["freeze_id"], self.frozen["freeze_id"])
            changed = copy.deepcopy(self.frozen)
            changed["source_sha256"]["execution/decoder_competition.py"] = "0" * 64
            changed["freeze_id"] = content_id("a3-decoder-competition", {k: v for k, v in changed.items() if k != "freeze_id"})
            write_json(path / "freeze.json", changed)
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                check(path)

    def test_preselected_slow_physical_tie_is_not_replaced_with_best_card(self):
        f, env, rows = schema_fixture()
        task = summarize(f, rows, env)["workloads"][0]
        self.assertEqual(task["initial_preselected_id"], "u0")
        self.assertAlmostEqual(task["initial_relative_regret"], 1.)
        self.assertEqual(task["quality_windows"], 3)
        self.assertEqual(task["repeat_consistent_cross_gpu_advantages"], 1)
        self.assertFalse(task["initial_epsilon_success"]["0.01"])

    def test_drift_flags_preserve_all_samples_and_suppress_stability_claim(self):
        f, env, rows = schema_fixture()
        row = next(r for r in rows if r["repeat"] == 0 and r["phase"] == "after" and r["plan_id"] == "u0")
        row["samples_s"], row["median_s"] = [.009] * 20, .009
        seal(row)
        report = summarize(f, rows, env)
        self.assertEqual(report["record_count"], len(rows))
        self.assertEqual(report["workloads"][0]["quality_windows"], 2)
        self.assertEqual(report["workloads"][0]["repeat_consistent_cross_gpu_advantages"], 0)

    def test_partial_failed_and_duplicate_records_do_not_produce_regret(self):
        f, env, rows = schema_fixture()
        self.assertFalse(summarize(f, rows[:-1], env)["complete"])
        self.assertEqual(summarize(f, rows[:-1], env)["workloads"], [])
        failed = copy.deepcopy(rows)
        failed[0]["status"] = "error"
        seal(failed[0])
        self.assertFalse(summarize(f, failed, env)["complete"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(f, rows + [rows[0]], env)

    def test_resume_window_is_flagged_without_deleting_it(self):
        f, env, rows = schema_fixture()
        rows[0]["collection_run_id"] = "another-process"
        seal(rows[0])
        report = summarize(f, rows, env)
        self.assertTrue(report["complete"])
        self.assertEqual(report["workloads"][0]["quality_windows"], 2)
        self.assertTrue(report["workloads"][0]["windows"][0]["spans_collection_processes"])

    def test_gpu_entry_has_no_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                collect_environment({"runtime": {}})

    def test_pooled_advantage_does_not_hide_window_rank_reversal(self):
        f, env, rows = schema_fixture()
        row = next(r for r in rows if r["repeat"] == 2 and r["plan_id"] == "cross")
        row["samples_s"], row["median_s"] = [.005] * 20, .005
        seal(row)
        report = summarize(f, rows, env)
        self.assertEqual(report["workloads"][0]["repeat_consistent_cross_gpu_advantages"], 0)
        candidate = next(c for c in report["workloads"][0]["candidates"] if c["plan_id"] == "cross")
        self.assertEqual(candidate["median_s"], .003)

    def test_timing_tampering_is_not_accepted(self):
        f, env, rows = schema_fixture()
        rows[0]["samples_s"][0] = .9
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            summarize(f, rows, env)


if __name__ == "__main__":
    unittest.main()
