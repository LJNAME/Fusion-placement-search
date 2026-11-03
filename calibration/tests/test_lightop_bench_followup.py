import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import prepare_lightop_bench_followup as prepare


class FollowupTests(unittest.TestCase):
    def test_selects_only_dependency_failures(self):
        data = prepare.manifest()
        self.assertEqual(len(data["jobs"]), 16)
        self.assertTrue(all("dropout" not in j for j in data["jobs"]))
        self.assertNotIn("test_rms_mrope", data["jobs"])
        self.assertNotIn("test_fuse_rmsnorm_rope_qkv_no_quant", data["jobs"])

    def test_numeric_stack_replacement_is_rejected_before_install(self):
        for name in ("torch", "triton", "lightop", "numpy", "pytorch_triton_rocm", "torchvision", "nvidia-cublas-cu12"):
            report = {"install": [{"metadata": {"name": name}}]}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "protected"):
                prepare.approved_lock(report, ["torch", "triton", "lightop", "numpy"])

    def test_host_dependency_download_is_hash_locked(self):
        report = {"install": [{"metadata": {"name": "tqdm"}, "download_info": {"url": "https://files.pythonhosted.org/tqdm.whl", "archive_info": {"hashes": {"sha256": "a" * 64}}}}]}
        self.assertEqual(prepare.approved_lock(report, []), ["tqdm @ https://files.pythonhosted.org/tqdm.whl --hash=sha256:" + "a" * 64])

    def test_unknown_hash_does_not_produce_an_install_lock(self):
        report = {"install": [{"metadata": {"name": "tqdm"}, "download_info": {"url": "http://unverified/tqdm.whl", "archive_info": {"hashes": {"sha256": "unknown"}}}}]}
        with self.assertRaisesRegex(ValueError, "URL/hash"):
            prepare.approved_lock(report, [])

    def test_changed_cached_vendor_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "vendor.whl").write_bytes(b"changed")
            with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("no download expected")):
                with self.assertRaisesRegex(ValueError, "Cached wheel SHA"):
                    prepare.fetch({"filename": "vendor.whl", "sha256": "wrong"}, p)

    def test_selected_package_preserves_old_rms_argument_order(self):
        data = prepare.manifest()
        self.assertEqual(data["required_rms_norm_opt_parameters"], ["out", "input", "weight", "epsilon"])
        vllm = next(w for w in data["wheels"] if w["distribution"] == "vllm")
        self.assertTrue(vllm["version"].startswith("0.11.0+das."))
        self.assertFalse(data["gpu_validated"])
        self.assertTrue(data["rejected_candidate"]["version"].startswith("0.15.1"))


if __name__ == "__main__":
    unittest.main()
