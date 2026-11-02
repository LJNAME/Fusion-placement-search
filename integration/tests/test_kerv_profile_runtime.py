"""Metadata observation tests use fake runtime objects, never GPU evidence."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from calibration.kerv_silu_mul import CORE
from integration.flagscale import kerv_profile_runtime as runtime


class TestKervRuntimeObservation(unittest.TestCase):
    def test_source_hash_comes_from_loaded_module_bytes_and_revision_from_git(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CORE
            path.parent.mkdir(parents=True)
            path.write_bytes(b"# synthetic test source\n")
            module = SimpleNamespace(__file__=str(path), _silu_mul_kernel=SimpleNamespace(
                fn=SimpleNamespace(__code__=SimpleNamespace(co_filename=str(path)))))
            with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout="test-actual-revision\n")):
                proof = runtime.loaded_source(module, directory)
            self.assertEqual(proof["module_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(proof["git_revision"], "test-actual-revision")
            self.assertEqual(Path(proof["module_path"]), path.resolve())
            module._silu_mul_kernel.fn.__code__.co_filename = str(path.with_name("other.py"))
            with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout="test-revision\n")):
                with self.assertRaisesRegex(ValueError, "different source file"):
                    runtime.loaded_source(module, directory)

    def test_architecture_and_distribution_builds_are_observed_not_guessed(self):
        props = SimpleNamespace(name="BW", total_memory=123456, uuid="test-uuid", gcnArchName="test-architecture")
        with patch.object(runtime.torch.cuda, "get_device_properties", return_value=props), patch.object(
                runtime, "driver_version", return_value="test-driver"), patch.object(
                runtime.importlib.metadata, "version", side_effect=lambda name: name+"-test-build"):
            device, env = runtime.observe_environment(SimpleNamespace(triton=SimpleNamespace(__version__="test-module")), "cuda:1")
        self.assertEqual(device["architecture"], "test-architecture")
        self.assertEqual(env["torch"], "torch-test-build")
        self.assertEqual(env["triton"], "triton-test-build")
        self.assertEqual(env["device_uuid"], "test-uuid")
        self.assertEqual(env["driver_version"], "test-driver")

    def test_unknown_hip_architecture_and_driver_stay_unknown(self):
        props = SimpleNamespace(name="BW", total_memory=123456, uuid="test-uuid")
        with patch.object(runtime.torch.cuda, "get_device_properties", return_value=props), patch.object(
                runtime.torch.version, "hip", "test-hip"), patch.object(runtime, "driver_version", return_value=None), patch.object(
                runtime.importlib.metadata, "version", return_value="test-build"):
            device, env = runtime.observe_environment(SimpleNamespace(triton=SimpleNamespace(__version__="test-module")), "cuda:0")
        self.assertIsNone(device["architecture"])
        self.assertIsNone(env["driver_version"])


if __name__ == "__main__":
    unittest.main()
