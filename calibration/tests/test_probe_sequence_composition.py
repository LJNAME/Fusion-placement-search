"""CPU checks for composition, graph-boundary confounding, and bundle admission."""
from collections import Counter
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from calibration.probe_sequence_composition import (
    ARMS, BATCHES, BUFFERS, ORDERS, POOL, capture, contract_for, main, measurement_contract_id,
    prepare_case, protocol, sample, save, schedule, summarize, verify_bundle,
)


ENV = {"torch": "CPU-fixture-not-GPU-data", "hip": "test", "triton": "test", "device_model": "test"}


def observations(cost):
    ids = {n: measurement_contract_id(contract_for(ENV, n)) for n in BATCHES}
    return [{"round": r, "graph_calls": n, "method": a, "measurement_contract_id": ids[n],
        "graph_event_total_s": cost(n, a) * n / 1e6,
        "graph_batch_amortized_per_call_us": cost(n, a),
        "graph_submission_wall_s": .000001, "graph_completion_wall_s": .001,
        "memory_before_numeric_checks": {"memory_allocated": 123},
        "numeric_passed": True, "numeric_coverage": protocol()["numeric_coverage"]}
        for r, n, a in schedule()]


def boundary_cost(n, arm):
    length = dict(silu=1, mul=1, add=1, silu_mul=2, mul_add=2, silu_mul_add=3)[arm]
    return length * 3. + 50. / n


def fixture(directory):
    plan = protocol()
    save(directory / "plan.json", plan)
    save(directory / "environment.json", ENV)
    contracts = {str(n): contract_for(ENV, n) for n in BATCHES}
    report = {"case_id": "gpu0_seq256", "shape": [1, 256, 3072], "dtype": "bf16",
        "input_pool_bytes": BUFFERS * 256 * 3072 * 2, "protocol_id": plan["protocol_id"],
        "measurement_contracts": contracts, "measurement_contract_ids": {
            n: measurement_contract_id(c) for n, c in contracts.items()}, "admission": "conditional_dev",
        "captures": {f"n{n}_{a}": {"single_stream": True, "graph_calls": n,
            "retained_outputs": "last_final_output_only", "attempted_pool_config": dict(POOL)}
            for n in BATCHES for a in ARMS}, "rows": observations(boundary_cost)}
    save(directory / "partial.json", report)
    report["summary"] = summarize(report["rows"])
    save(directory / "result.json", report)


class TestSequenceComposition(unittest.TestCase):
    def test_balanced_schedule_and_counts(self):
        for position in range(6):
            self.assertEqual(Counter(o[position] for o in ORDERS), Counter({a: 2 for a in ARMS}))
        pairs = Counter(p for o in ORDERS for p in zip(o, o[1:]))
        self.assertEqual(len(pairs), 30)
        self.assertEqual(set(pairs.values()), {2})
        self.assertEqual(len(schedule()), 144)
        self.assertEqual(sum(n for _, n, _ in schedule()), 10800)
        self.assertEqual([n for r, n, a in schedule() if a == ORDERS[r][0]], [50, 100, 100, 50] * 6)

    def test_graph_boundary_alone_mimics_two_adjacencies(self):
        result = summarize(observations(boundary_cost))
        for n, d in (("50", 1.), ("100", .5)):
            b = result["by_calls_per_graph"][n]
            self.assertEqual(b["delta_sm_us"]["median"], d)
            self.assertEqual(b["delta_ma_us"]["median"], d)
            self.assertEqual(b["delta_sma_us"]["median"], 2 * d)
            self.assertEqual(b["interaction_us"]["median"], 0.)
        self.assertEqual(result["n100_minus_n50"]["delta_sma_us"]["median"], -1.)
        self.assertFalse(result["mechanism_identified"])

    def test_pair_specific_residuals_and_nonadditive_interaction(self):
        values = dict(silu=4., mul=5., add=6., silu_mul=8., mul_add=9., silu_mul_add=10.)
        result = summarize(observations(lambda n, a: values[a]))
        b = result["by_calls_per_graph"]["50"]
        self.assertEqual([b[f]["median"] for f in ("delta_sm_us", "delta_ma_us", "delta_sma_us", "interaction_us")],
                         [1., 2., 5., 2.])
        values["silu_mul_add"] = 18.
        self.assertEqual(summarize(observations(lambda n, a: values[a]))[
            "by_calls_per_graph"]["100"]["delta_sma_us"]["median"], -3.)

    def test_chain_inputs_and_100_calls_reread_without_mutation(self):
        import torch
        import torch.nn.functional as F
        gates, up, addend, functions, sources, refs = prepare_case(torch, F, "cpu", seq=1)
        torch.testing.assert_close(sources["mul"], F.silu(sources["silu"]), rtol=0, atol=0)
        torch.testing.assert_close(sources["add"], sources["mul"] * up, rtol=0, atol=0)
        stream = SimpleNamespace(wait_stream=lambda other: None)
        cuda = SimpleNamespace(current_stream=lambda d: stream, Stream=lambda **k: stream,
            stream=lambda s: nullcontext(), synchronize=lambda: None, reset_peak_memory_stats=lambda d: None,
            CUDAGraph=lambda: object(), graph=lambda *a, **k: nullcontext())
        shim = SimpleNamespace(cuda=cuda)
        for arm in ARMS:
            for n in BATCHES:
                seen = []
                def body(x, u):
                    seen.append(x.data_ptr())
                    return functions[arm](x, u)
                _, result, record = capture(shim, body, gates, sources[arm], up, n, {})
                self.assertEqual(len(seen), n * 2)  # Full warmup plus captured bodies.
                self.assertEqual(len(set(seen)), BUFFERS)
                if n == 100:
                    self.assertEqual(seen[:50], seen[50:100])
                torch.testing.assert_close(result, refs[arm], rtol=0, atol=0)
                torch.testing.assert_close(gates, sources[arm].unsqueeze(0).expand_as(gates), rtol=0, atol=0)
                self.assertEqual(record["graph_calls"], n)
        self.assertTrue(torch.isfinite(addend).all())

    def test_event_normalization_uses_the_actual_batch_length(self):
        event = SimpleNamespace(record=lambda: None, elapsed_time=lambda end: 1.)
        cuda = SimpleNamespace(reset_peak_memory_stats=lambda d: None,
            Event=lambda **k: event, synchronize=lambda: None)
        graph = SimpleNamespace(replay=lambda: None)
        for n, expected in ((50, 20.), (100, 10.)):
            row = sample(SimpleNamespace(cuda=cuda), graph, "test", n)
            self.assertEqual(row["graph_batch_amortized_per_call_us"], expected)

    def test_contract_separates_N_without_changing_kernel_identity(self):
        self.assertNotEqual(measurement_contract_id(contract_for(ENV, 50)),
                            measurement_contract_id(contract_for(ENV, 100)))
        self.assertNotIn("kernel_key", protocol())

    def test_plan_only_and_no_gpu_failure_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "plan"
            with patch("torch.cuda.is_available", side_effect=AssertionError("GPU queried")):
                main(["--out-dir", str(out), "--plan-only"])
            self.assertFalse((out / "failure.json").exists())
            with self.assertRaises(FileNotFoundError):
                verify_bundle(out)
            failed = Path(directory) / "failed"
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no CPU/eager fallback"):
                    main(["--out-dir", str(failed)])
            failure = json.loads((failed / "failure.json").read_text())
            self.assertFalse(failure["context"]["capture_attempted"])
            self.assertIn("Traceback", failure["traceback"])
            self.assertFalse(verify_bundle(failed)["complete"])

    def test_verifier_recomputes_and_rejects_mixing_or_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture(directory)
            self.assertTrue(verify_bundle(directory)["complete"])
            path = directory / "result.json"
            original = json.loads(path.read_text())
            for edit in ("summary", "contract", "admission", "missing", "normalization", "row_contract"):
                changed = copy.deepcopy(original)
                if edit == "summary": changed["summary"]["by_calls_per_graph"]["50"]["delta_sm_us"]["median"] = 99
                if edit == "contract": changed["measurement_contracts"]["50"]["calls_per_graph"] = 100
                if edit == "admission": changed["admission"] = "accepted"
                if edit == "missing": changed["rows"].pop()
                if edit == "normalization": changed["rows"][6]["graph_batch_amortized_per_call_us"] *= 2
                if edit == "row_contract": changed["rows"][6]["measurement_contract_id"] = changed["measurement_contract_ids"]["50"]
                save(path, changed)
                save(directory / "partial.json", {k: v for k, v in changed.items() if k != "summary"})
                with self.assertRaises(ValueError, msg=edit):
                    verify_bundle(directory)
            save(path, original)
            save(directory / "partial.json", {k: v for k, v in original.items() if k != "summary"})
            self.assertTrue(verify_bundle(directory)["complete"])


if __name__ == "__main__":
    unittest.main()
