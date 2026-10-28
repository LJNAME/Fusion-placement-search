"""Check lifecycle controls and persistence without claiming CPU timings model a GPU."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import load_freeze
from calibration.tests.test_reference_host_cpu import coherent_fake_probe
from execution.diagnose_reference_state import audit, build_protocol, cache_state, main, measure_block, summarize
from execution.gpu_holdout import CanonicalReferenceExecutor
from execution.host_submission import SubmissionProbeExecutor
from execution.verify_reference import _placed


def fake_executor(placed, devices, seed, templates, **kwargs):
    return SimpleNamespace(placed=placed, environment_id=kwargs["environment_id"],
        _weights={"fixture": torch.ones(1)}, _inputs={}, _norm_weights={}, _rope_tables={})


def fake_block(executor, parameters, reference, *, probe):
    if probe:
        result = coherent_fake_probe(executor.placed, None, parameters, None, None,
            mode="cpu_probe", environment_id=executor.environment_id)
    else:
        result = {"samples_s": [.004] * parameters["iterations"], "probe_samples": [], "catalog": {}, "correctness": {"pass": True}}
    return {**result, "cache_before_samples": cache_state(executor), "cache_after_samples": cache_state(executor)}


class ReferenceStateTests(unittest.TestCase):
    def test_protocol_crosses_lifecycle_and_sham_with_only_three_old_placements(self):
        protocol = build_protocol(*load_freeze())
        self.assertEqual({x["plan_name"] for x in protocol["cases"]}, {"uniform0", "uniform1", "mlp_up"})
        self.assertEqual({x["model_id"] for x in protocol["cases"]}, {"qwen3-0.6b-s1024-b1"})
        self.assertEqual(len(protocol["schedule"]), 108)
        for repeat in range(3):
            rows = [x for x in protocol["schedule"] if x["repeat"] == repeat]
            self.assertEqual(len(rows), 36)
            self.assertEqual(len({(x["case_id"], x["lifecycle"], x["middle_kind"]) for x in rows}), 12)

    def test_same_cpu_executor_preserves_cache_and_outputs_across_baseline_probe_baseline(self):
        placed, templates = _placed(), {}
        with torch.no_grad():
            output, _, _, _ = CanonicalReferenceExecutor(placed, ("cpu", "cpu"), 1234, templates).run_once(False)
            reference = output.clone()
        executor = SubmissionProbeExecutor(placed, ("cpu", "cpu"), 1234, templates)
        parameters = {"seed": 1234, "warmup": 1, "iterations": 2, "rtol": .01, "atol": .001}
        with patch("torch.cuda.Event", side_effect=AssertionError("unexpected device events")):
            rows = [measure_block(executor, parameters, reference, probe=probe) for probe in (False, True, False)]
        self.assertTrue(all(x["correctness"]["pass"] for x in rows))
        self.assertEqual(len({x["cache_before_samples"]["identity"] for x in rows}), 1)
        self.assertEqual([len(x["probe_samples"]) for x in rows], [0, 2, 0])

    def test_all_108_blocks_persist_and_reused_instances_are_not_reconstructed(self):
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch("execution.diagnose_reference_state._validate_runtime"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="BW")), \
                patch("torch.cuda.can_device_access_peer", return_value=True), \
                patch("execution.diagnose_reference_state.memory_state", return_value=[]), \
                patch("execution.diagnose_reference_state.CanonicalReferenceExecutor") as canonical, \
                patch("execution.diagnose_reference_state.SubmissionProbeExecutor", side_effect=fake_executor) as factory, \
                patch("execution.diagnose_reference_state.measure_block", side_effect=fake_block), \
                patch("torch.cuda.Event", side_effect=AssertionError("unexpected device events")):
            canonical.return_value.run_once.return_value = (torch.ones(1), [], [], .001)
            directory = Path(temp) / "state"
            main(["--out-dir", str(directory)])
            result = audit(directory)
            self.assertEqual((result["records"], result["measured_forwards"]), (108, 2160))
            self.assertFalse(result["calibration_promoted"])
            self.assertEqual(factory.call_count, 72)  # 18 reuse objects + 18*3 rebuild objects.
            self.assertFalse((directory / "host_submission_profile.json").exists())
            rows = load_jsonl_records(directory / "blocks.jsonl")
            self.assertEqual(len({x["record_id"] for x in rows}), 108)
            report = json.loads((directory / "diagnostic_report.json").read_text())
            self.assertEqual(len(report["triplets"]), 36)
            self.assertEqual(len(report["cells"]), 12)
            main(["--audit-dir", str(directory)])
            rows[2]["samples_s"] = [.006] * 20
            protocol = json.loads((directory / "protocol.json").read_text())
            changed = summarize(protocol, rows)
            self.assertFalse(changed["triplets"][0]["baseline_within_reference_10pct"])
            self.assertEqual(len(changed["triplets"]), 36)  # No unfavorable triplet is dropped.
            # A raw timing change cannot silently enter the replayed report.
            lines = (directory / "blocks.jsonl").read_text().splitlines()
            row = json.loads(lines[0])
            row["samples_s"][0] *= 2
            lines[0] = json.dumps(row)
            (directory / "blocks.jsonl").write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "record hash mismatch"):
                audit(directory)

    def test_gpu_command_has_no_cpu_fallback(self):
        with TemporaryDirectory() as temp, patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no HIP/CUDA"):
                main(["--out-dir", temp])


if __name__ == "__main__":
    unittest.main()
