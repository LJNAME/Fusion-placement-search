"""Graph failures and metric naming remain reviewable without a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.probe_kerv_graph import POOL, main, write_failure


class TestGraphDiagnostic(unittest.TestCase):
    def test_capture_exception_keeps_traceback_pool_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            context = {"stage": "capture", "capture_attempted": True, "device": "cuda:0",
                "memory_before_capture": {"memory_allocated": 1, "memory_reserved": 2}}
            with patch("calibration.probe_kerv_graph.memory_snapshot", return_value={"memory_allocated": 3, "memory_reserved": 4}):
                try:
                    raise RuntimeError("capture deliberately failed for test")
                except RuntimeError as error:
                    write_failure(output, error, context, object())
            result = json.loads(output.read_text())
            self.assertIn("Traceback (most recent call last)", result["traceback"])
            self.assertIn("capture deliberately failed", result["traceback"])
            self.assertEqual(result["attempted_pool_config"], POOL)
            self.assertEqual(result["memory_at_failure"]["memory_reserved"], 4)
            self.assertEqual(result["context"]["memory_before_capture"]["memory_reserved"], 2)

    def test_no_gpu_fails_without_claiming_capture_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no CPU/eager fallback"):
                    main(["--out-dir", str(output)])
            failure = json.loads((output / "failure.json").read_text())
            plan = json.loads((output / "plan.json").read_text())
            self.assertFalse(failure["context"]["capture_attempted"])
            self.assertEqual(plan["metric"], "graph_batch_amortized_per_call_us")
            self.assertFalse(list(output.rglob("result.json")))

    def test_source_has_no_bare_latency_json_field(self):
        import ast
        from calibration import probe_kerv_graph
        tree = ast.parse(Path(probe_kerv_graph.__file__).read_text())
        keys = [key.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
        self.assertFalse(any("latency" in key for key in keys))
        self.assertIn("graph_batch_amortized_per_call_us", keys)


if __name__ == "__main__":
    unittest.main()
