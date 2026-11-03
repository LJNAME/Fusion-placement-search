import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from calibration import lightop_vendor_suite as suite
import importlib

provider = importlib.import_module("integration.lightop.catalog")


class VendorSuiteTests(unittest.TestCase):
    def test_catalog_does_not_promote_api_to_graph_adapter(self):
        data = provider.catalog()
        records = data["capabilities"]
        self.assertEqual(len({r["id"] for r in records}), data["fusion_interface_count"])
        self.assertTrue(all(r["implementation_status"] == "implemented" for r in records))
        self.assertTrue(all(not r["placed_graph_adapter"] for r in records))
        root = next(r for r in records if r["id"] == "fuse_silu_and_mul")
        optimized = next(r for r in records if r["id"] == "op.silu_and_mul_opt")
        self.assertNotEqual(root["binding"], optimized["binding"])
        self.assertEqual(root["gpu_measurement"], "not_yet_measured_in_project")
        self.assertEqual(optimized["gpu_measurement"], "trial02_exploratory")

    def test_version_mismatch_blocks_dispatch_before_native_import(self):
        class Dist:
            version = "different"
        with patch.object(provider.importlib.metadata, "distribution", return_value=Dist()):
            with self.assertRaisesRegex(ValueError, "version differs"):
                provider.check_installed()

    def test_source_mismatch_blocks_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "fake.so").write_bytes(b"tampered")
            class Dist:
                version = "fixed"

                def locate_file(self, name):
                    return p / name
            data = {"version": "fixed", "op_so_path": "fake.so", "op_so_sha256": "wrong",
                    "init_sha256": "also_wrong", "capabilities": []}
            with patch.object(provider, "catalog", return_value=data), patch.object(provider.importlib.metadata, "distribution", return_value=Dist()):
                with self.assertRaisesRegex(ValueError, "source differs"):
                    provider.check_installed()

    def test_unknown_api_has_no_fallback(self):
        with self.assertRaises(KeyError):
            provider.resolve("invented_fusion")

    def test_failed_vendor_job_does_not_discard_later_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            original = p / "original"
            (original / "test").mkdir(parents=True)
            (original / "test/helper.py").write_text("VALUE = 23\n")
            (original / "test/fails.py").write_text("import sys\nprint('before crash', flush=True)\nprint('trace detail',file=sys.stderr)\nsys.exit(3)\n")
            (original / "test/succeeds.py").write_text("from helper import VALUE\nprint(VALUE)\nprint('False')\nprint('vendor latency 1.25 us')\n")
            jobs = [{"id": name, "script": "test/" + name + ".py", "args": [], "capability_references": ["test_api"], "launch_mode": "script"} for name in ("fails", "succeeds")]
            frozen = {"scope": "vendor_original_tests_not_profile_admission", "jobs": jobs}
            output = p / "output"
            output.mkdir()
            with patch.object(suite, "plan", return_value=frozen), patch.object(suite, "check_installed", return_value={}), patch.object(suite, "extract_vendor_tests", side_effect=lambda _, target: shutil.copytree(original, target)), contextlib.redirect_stdout(io.StringIO()):
                status = suite.run(output, "unused.whl", sys.executable)
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(status, 1)
            self.assertEqual([r["returncode"] for r in results], [3, 0])
            self.assertEqual([r["attempts"] for r in results], [1, 1])
            self.assertIn("trace detail", (output / "jobs/fails/stderr.log").read_text())
            self.assertIn("False", results[1]["validation_lines"])
            self.assertEqual(results[1]["validation_status"], "vendor_output_requires_review")
            self.assertEqual(results[1]["timing_lines"], ["vendor latency 1.25 us"])

    def test_wrong_wheel_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "wrong.whl").write_bytes(b"wrong wheel")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                suite.extract_vendor_tests(p / "wrong.whl", p / "vendor")
            self.assertFalse((p / "vendor").exists())

    def test_optimized_activation_cli_variants_are_independent_jobs(self):
        jobs = suite.plan()["jobs"]
        variants = [j for j in jobs if j["script"] == "test/test_act_and_mul.py"]
        self.assertEqual({tuple(j["args"]) for j in variants}, {("--act", "silu"), ("--act", "gelu"), ("--act", "gelu_tanh")})
        self.assertTrue(all(len(j["capability_references"]) == 1 for j in variants))


if __name__ == "__main__":
    unittest.main()
