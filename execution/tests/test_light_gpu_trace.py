from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import DEFAULT_BUNDLE
from execution.diagnose_gpu_holdout import main
from execution.gpu_holdout import CanonicalReferenceExecutor
from execution.light_gpu_trace import analyze_trace_frames, assess_trace_timing, capture_light_trace, overlap_ms
from execution.verify_reference import _placed


class LightGPUTraceTests(unittest.TestCase):
    def test_physical_intervals_reproduce_received_gpu_trace_without_flow_or_annotation_ranges(self):
        base = DEFAULT_BUNDLE / "results_received_20261028/diagnostics_3b_s2048"
        for role, copies in (("single_gpu_reference", 0), ("predicted_top1", 2), ("intermediate", 2)):
            events = json.loads((base / f"{role}.trace.json").read_text())["traceEvents"]
            # A large annotation span must never add physical GPU activity.
            events.append({"ph": "X", "cat": "gpu_user_annotation", "name": "fake_span", "ts": 0, "dur": 1e12, "args": {"device": 1}})
            parsed = analyze_trace_frames(events, prefix="fusion_diagnostic_forward")
            self.assertTrue(parsed["gpu_event_attribution_complete"])
            self.assertEqual(len(parsed["frames"]), 1)
            frame = parsed["frames"][0]
            self.assertEqual(frame["physical_kernels"], 37)
            self.assertEqual(frame["gpu_memcpy_events"], copies)
            self.assertEqual(frame["gpu0_gpu1_kernel_overlap_ms"], 0)
            self.assertTrue(all(pair["available"] for pair in frame["branch_overlap"].values()))

    def test_missing_gpu_node_association_is_unavailable_not_zero_overlap(self):
        path = DEFAULT_BUNDLE / "results_received_20261028/diagnostics_3b_s2048/predicted_top1.trace.json"
        events = json.loads(path.read_text())["traceEvents"]
        kernel = next(event for event in events if event.get("cat") == "kernel" and event.get("args", {}).get("device") == 1)
        kernel["args"].pop("External id")
        parsed = analyze_trace_frames(events, prefix="fusion_diagnostic_forward")
        self.assertFalse(parsed["gpu_event_attribution_complete"])
        pair = parsed["frames"][0]["branch_overlap"]["gate_up"]
        self.assertFalse(pair["available"])
        self.assertIsNone(pair["overlap_ms"])

    def test_overlap_uses_unions_not_sum_of_nested_intervals(self):
        self.assertEqual(overlap_ms([(0, 2), (1, 3)], [(2, 4)]), 1)
        self.assertEqual(overlap_ms([(0, 1)], [(1, 2)]), 0)

    def test_timing_check_rejects_large_perturbation_and_baseline_shift(self):
        self.assertTrue(assess_trace_timing({"traced_samples_s": [.0041] * 3}, 4, 4.05)["timing_target_pass"])
        self.assertFalse(assess_trace_timing({"traced_samples_s": [.006] * 3}, 4, 4)["timing_target_pass"])
        self.assertFalse(assess_trace_timing({"traced_samples_s": [.0043] * 3}, 4, 4.5)["timing_target_pass"])
        self.assertFalse(assess_trace_timing({}, 4, 4)["available"])

    def test_light_capture_keeps_outputs_and_profiles_three_forwards_without_metadata(self):
        placed, templates = _placed(), {}
        parameters = {"seed": 1234, "rtol": .01, "atol": .001, "warmup": 1}
        with torch.no_grad():
            expected, _, _, _ = CanonicalReferenceExecutor(placed, ("cpu", "cpu"), 1234, templates).run_once(False)
        real_profile = torch.profiler.profile
        with TemporaryDirectory() as temp, \
                patch("torch.profiler.supported_activities", return_value={torch.profiler.ProfilerActivity.CPU}), \
                patch("torch.profiler.profile", wraps=real_profile) as profiler:
            path = Path(temp) / "light.trace.json"
            result = capture_light_trace(placed, ("cpu", "cpu"), parameters, templates, expected, path)
            self.assertTrue(result["correctness"]["pass"])
            self.assertEqual(result["correctness"]["max_abs_error"], 0)
            self.assertEqual(len(result["traced_samples_s"]), 3)
            self.assertEqual(len(result["frames"]), 3)
            self.assertFalse(result["gpu_timeline_available"])
            self.assertTrue(all(frame["gpu0_gpu1_kernel_overlap_ms"] is None for frame in result["frames"]))
            self.assertEqual(profiler.call_count, 2)
            for call in profiler.call_args_list:
                self.assertFalse(call.kwargs["record_shapes"])
                self.assertFalse(call.kwargs["profile_memory"])
                self.assertFalse(call.kwargs["with_stack"])
            self.assertFalse(result["collection"]["dispatch_interception"])
            self.assertFalse(result["collection"]["per_node_metadata"])

    def test_light_command_persists_pre_and_post_blocks_and_reports_perturbation(self):
        for seconds, status in ((.004, "complete"), (.008, "complete_light_trace_perturbed")):
            with self.subTest(status=status), TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                    patch("execution.diagnose_gpu_holdout._validate_runtime"), \
                    patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="BW", total_memory=1024)), \
                    patch("torch.cuda.can_device_access_peer", return_value=True), \
                    patch("execution.diagnose_gpu_holdout.CanonicalReferenceExecutor") as executor, \
                    patch("execution.diagnose_gpu_holdout.measure_candidate", return_value={"samples_s": [.004] * 20,
                        "median_s": .004, "correctness": {"pass": True}, "peak_memory_bytes": [1, 1]}), \
                    patch("execution.diagnose_gpu_holdout.capture_trace", side_effect=AssertionError("heavy tracing used")), \
                    patch("execution.diagnose_gpu_holdout.capture_light_trace") as capture:
                def fake_capture(*args):
                    role = args[-1].name.removesuffix(".trace.json")
                    count, size = {"single_gpu_reference": (0, 0), "predicted_top1": (2, 51 * 1024**2), "intermediate": (2, 2 * 1024**2)}[role]
                    return {"correctness": {"pass": True}, "gpu_timeline_available": True,
                        "gpu_event_attribution_complete": True, "traced_samples_s": [seconds] * 3,
                        "frames": [{"physical_kernels": 37, "gpu_memcpy_events": count, "gpu_memcpy_bytes": size,
                            "branch_overlap": {key: {"available": True} for key in ("gate_up", "k_rope_q_proj")}} for _ in range(3)]}
                capture.side_effect = fake_capture
                executor.return_value.run_once.return_value = (torch.ones(1), [], [], .001)
                output = Path(temp) / "light"
                main(["--trace-mode", "light", "--out-dir", str(output)])
                rows = load_jsonl_records(output / "baseline_samples.jsonl")
                report = json.loads((output / "diagnostic_report.json").read_text())
                self.assertEqual(len(rows), 12)
                self.assertEqual(sum(len(row["samples_s"]) for row in rows), 240)
                self.assertEqual(sum(row["phase"] == "before_trace" for row in rows), 9)
                self.assertEqual(sum(row["phase"] == "after_trace" for row in rows), 3)
                self.assertEqual(len({row["record_id"] for row in rows}), 12)
                for row in rows:
                    self.assertEqual(row["record_id"], content_id("gpu-three-plan-diagnostic-baseline",
                        {key: value for key, value in row.items() if key != "record_id"}))
                self.assertEqual(report["status"], status)
                self.assertEqual(len(report["post_trace_baselines"]), 3)
                self.assertTrue(report["all_light_activity_checks_pass"])
                self.assertEqual(report["all_light_timing_targets_pass"], status == "complete")


if __name__ == "__main__":
    unittest.main()
