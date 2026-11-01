"""Additivity sign, input semantics, balanced schedule, and rejection checks."""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from calibration.probe_kerv_additivity import (
    ARMS, CALLS, CASES, ORDERS, main, prepare_case, protocol, save, summarize, verify_bundle,
)
from calibration.kerv_silu_mul import DEFAULT_BUNDLE
from calibration.probe_kerv_graph import POOL
from cost_model.measurement_labels import measurement_contract_id


def rows():
    # Deliberately makes the additive and chain decisions disagree.
    costs = {"silu": 6., "mul": 7., "chain": 10., "fused": 11.}
    return [{"round": i, "method": arm, "graph_calls": CALLS,
        "graph_event_total_s": costs[arm] * CALLS / 1e6,
        "graph_batch_amortized_per_call_us": costs[arm],
        "graph_submission_wall_s": .000001, "graph_completion_wall_s": .001,
        "numeric_passed": True,
        "numeric_coverage": "all_50_gate_outputs" if arm == "fused" else "last_final_output_only"}
        for i, order in enumerate(ORDERS) for arm in order]


def fixture_bundle(directory):
    plan = protocol()
    save(directory / "plan.json", plan)
    env = {"capability": json.loads((DEFAULT_BUNDLE / "capability.json").read_text()),
           "torch": "CPU-schema-test-not-GPU-data", "hip": "test", "triton": "test",
           "devices": {"0": "test", "1": "test"}}
    save(directory / "environment.json", env)
    for device, seq in CASES:
        path = directory / f"gpu{device}_seq{seq}"
        path.mkdir()
        contract = {"executor": "graph_replay", "submission_included": True,
            "allocation_included": False, "granularity": "graph_replay", "unit": "throughput_per_call",
            "serial_plan_only": True, "environment": {"torch": env["torch"], "hip": "test",
                "triton": "test", "device_model": "test"}}
        report = {"case_id": path.name, "shape": [1, seq, 3072], "dtype": "bf16",
            "gate_pool_bytes": CALLS * seq * 3072 * 2, "protocol_id": plan["protocol_id"],
            "measurement_contract": contract, "measurement_contract_id": measurement_contract_id(contract),
            "admission": "conditional_dev", "numeric_tolerance": {"rtol": .01, "atol": .001},
            "captures": {arm: {"single_stream": True, "retained_outputs": "last_final_output_only",
                               "attempted_pool_config": dict(POOL)} for arm in ARMS}, "rows": rows()}
        save(path / "partial.json", report)
        report["summary"] = summarize(report["rows"])
        save(path / "result.json", report)


class TestAdditivityDiagnostic(unittest.TestCase):
    def test_order_balances_position_and_within_round_carryover(self):
        self.assertEqual(len(ORDERS), 8)
        for position in range(4):
            self.assertEqual(Counter(order[position] for order in ORDERS), Counter({a: 2 for a in ARMS}))
        adjacent = Counter(pair for order in ORDERS for pair in zip(order, order[1:]))
        self.assertEqual(len(adjacent), 12)
        self.assertEqual(set(adjacent.values()), {2})

    def test_positive_residual_overstates_fusion_saving_and_can_change_decision(self):
        result = summarize(rows())
        self.assertEqual(result["additivity_residual_us"]["median"], 3.)
        self.assertEqual(result["predicted_fusion_saving_us"]["median"], 2.)
        self.assertEqual(result["chain_fusion_saving_us"]["median"], -1.)
        self.assertEqual(result["fusion_saving_prediction_error_us"]["median"], 3.)
        self.assertEqual(result["decision_disagreement_rounds"], 8)

    def test_negative_residual_is_retained(self):
        values = rows()
        for row in values:
            if row["method"] == "chain":
                row["graph_batch_amortized_per_call_us"] = 14.
        result = summarize(values)
        self.assertEqual(result["additivity_residual_us"]["median"], -1.)
        self.assertEqual(result["decision_disagreement_rounds"], 0)

    def test_mul_receives_the_chain_intermediate_without_mutating_original(self):
        import torch
        import torch.nn.functional as F
        module = SimpleNamespace(kerv_silu_mul=lambda gate, up: gate.copy_(F.silu(gate) * up))
        gates, original, up, _, functions, sources, refs = prepare_case(torch, F, module, "cpu", 1)
        saved = original.clone()
        torch.testing.assert_close(sources["mul"], F.silu(original), rtol=0, atol=0)
        for arm in ARMS:
            gates.copy_(sources[arm].unsqueeze(0).expand_as(gates))
            result = functions[arm](gates[0], up)
            torch.testing.assert_close(result, refs[arm], rtol=.01, atol=.001)
        self.assertTrue(torch.equal(original, saved))

    def test_plan_only_never_queries_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "plan"
            with patch("torch.cuda.is_available", side_effect=AssertionError("GPU queried")):
                main(["--out-dir", str(out), "--plan-only"])
            self.assertEqual(json.loads((out / "plan.json").read_text())["timed_body_calls"], 6400)
            self.assertFalse((out / "failure.json").exists())

    def test_no_gpu_records_environment_failure_without_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no CPU/eager fallback"):
                    main(["--out-dir", str(out)])
            failure = json.loads((out / "failure.json").read_text())
            self.assertFalse(failure["context"]["capture_attempted"])
            self.assertIn("Traceback", failure["traceback"])
            self.assertFalse(verify_bundle(out)["complete"])

    def test_verifier_recomputes_summary_and_refuses_admission_or_contract_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            fixture_bundle(out)
            result = verify_bundle(out)
            self.assertTrue(result["complete"])
            self.assertEqual(result["timed_body_calls"], 6400)
            path = out / "gpu0_seq256/result.json"
            old = json.loads(path.read_text())
            for edit in ("summary", "contract", "admission", "missing_arm"):
                changed = copy.deepcopy(old)
                if edit == "summary": changed["summary"]["additivity_residual_us"]["median"] = 0
                if edit == "contract": changed["measurement_contract"]["serial_plan_only"] = False
                if edit == "admission": changed["admission"] = "accepted"
                if edit == "missing_arm": changed["rows"].pop()
                save(path, changed)
                with self.assertRaises((ValueError, AssertionError)):
                    verify_bundle(out)
            save(path, old)
            self.assertTrue(verify_bundle(out)["complete"])


if __name__ == "__main__":
    unittest.main()
