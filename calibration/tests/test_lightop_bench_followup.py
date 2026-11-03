import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import prepare_lightop_bench_followup as prepare
from tools.build_lightop_bench_bundle import linux310_closure


class FollowupTests(unittest.TestCase):
    def test_cross_build_checks_linux_python310_markers(self):
        requirements = ["exceptiongroup>=1; python_version<'3.11'", "colorama; sys_platform=='win32'"]
        missing = linux310_closure([], requirements, {})
        self.assertEqual(missing, [requirements[0]])
        self.assertEqual(linux310_closure([], requirements, {"exceptiongroup": "1.3.0"}), [])

    def test_pip_configuration_is_isolated_from_parent(self):
        with patch.dict(os.environ, {"PIP_NO_INDEX": "1", "PIP_CONFIG_FILE": "wrong.ini", "PIP_INDEX_URL": "http://wrong", "HTTPS_PROXY": "http://proxy"}, clear=True):
            env = prepare.pip_environment()
        self.assertNotIn("PIP_NO_INDEX", env)
        self.assertNotIn("PIP_INDEX_URL", env)
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(env["HTTPS_PROXY"], "http://proxy")

    def test_real_pip_does_not_load_an_inherited_no_index_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "pip.ini"
            config.write_text("[global]\nno-index = true\n")
            with patch.dict(os.environ, {"PIP_CONFIG_FILE": str(config), "PIP_NO_INDEX": "1"}):
                result = subprocess.run([sys.executable, "-m", "pip", "--isolated", "config", "list"],
                                        env=prepare.pip_environment(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("no-index", result.stdout)

    def test_offline_lock_uses_only_local_verified_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "host").mkdir()
            wheel = root / "host/test.whl"
            wheel.write_bytes(b"verified payload")
            report = {"install": [{"metadata": {"name": "pytest"}, "download_info": {"url": wheel.as_uri(), "archive_info": {"hashes": {"sha256": prepare.sha(wheel)}}}}]}
            self.assertEqual(len(prepare.approved_lock(report, [], root)), 1)
            report["install"][0]["download_info"]["url"] = "https://files.pythonhosted.org/test.whl"
            with self.assertRaisesRegex(ValueError, "URL/hash"):
                prepare.approved_lock(report, [], root)

    def test_bundle_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "test.whl"
            wheel.write_bytes(b"original")
            data = {"source_manifest_sha256": prepare.sha(prepare.MANIFEST), "files": [{"path": wheel.name, "sha256": prepare.sha(wheel)}]}
            (root / "bundle.json").write_text(json.dumps(data))
            prepare.verify_bundle(root, {"wheels": []})
            wheel.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare.verify_bundle(root, {"wheels": []})

    def test_offline_wheel_outside_host_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "outside.whl"
            wheel.write_bytes(b"payload")
            report = {"install": [{"metadata": {"name": "pytest"}, "download_info": {"url": wheel.as_uri(), "archive_info": {"hashes": {"sha256": prepare.sha(wheel)}}}}]}
            with self.assertRaisesRegex(ValueError, "URL/hash"):
                prepare.approved_lock(report, [], root)

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
