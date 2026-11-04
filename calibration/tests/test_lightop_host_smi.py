from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from tools import run_lightop_host_smi as host
from tools import prepare_lightop_bench_followup as prepare
from integration.lightop import hygon_platform


class HostSmiTests(unittest.TestCase):
    def test_uses_the_host_sdk_verified_in_the_diagnostic_receipt(self):
        data = host.manifest()
        self.assertEqual(data["sdk_package_path"], "/opt/hyhal/share/amd_smi/amdsmi")
        self.assertEqual(data["sdk_version"], "7.6.0+995313d1")
        self.assertEqual(len(data["python_sources_sha256"]), 5)
        self.assertFalse(data["vllm_baseline_probe_gpu_validated"])

    def test_path_binding_reaches_child_imports_and_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "sdk/amdsmi"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("origin = 'host_sdk_fixture'\n")
            site = root / "private_site"
            site.mkdir()
            with host.sdk_search_path(site, package, root) as binding:
                self.assertTrue(binding.is_file())
                child = subprocess.run([sys.executable, "-c", "import site,sys; site.addsitedir(sys.argv[1]); import amdsmi; print(amdsmi.origin)", str(site)], capture_output=True, text=True)
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertEqual(child.stdout.strip(), "host_sdk_fixture")
            self.assertFalse(binding.exists())

    def test_failed_probe_cleans_binding_and_never_starts_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "private_site"
            site.mkdir()
            data = host.manifest()
            state = {"packages": data["protected_versions"], "python": "3.10", "executable": "private_python"}
            with patch.object(host.sys, "platform", "linux"), patch.object(host.sys, "version_info", (3, 10)), patch.object(host.sys, "prefix", str(root)), patch.object(host.sysconfig, "get_path", return_value=str(site)), patch.object(prepare, "snapshot", return_value=state), patch.object(prepare, "version", return_value=None), patch.object(host, "check_sources"), patch.object(host, "run_child", return_value={"returncode": -11, "timed_out": False}), patch.object(host.subprocess, "run") as bench:
                with self.assertRaisesRegex(RuntimeError, "no benchmarks started"):
                    host.run(root, None)
                bench.assert_not_called()
            self.assertFalse((site / "fusion_lightop_host_smi.pth").exists())
            self.assertTrue(json.loads((root / "path_cleanup.json").read_text())["removed"])

    def test_existing_path_binding_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "fusion_lightop_host_smi.pth"
            binding.write_text("existing\n")
            with self.assertRaises(FileExistsError):
                with host.sdk_search_path(root, root / "amdsmi", root):
                    self.fail("must not enter")
            self.assertEqual(binding.read_text(), "existing\n")

    def test_successful_preflight_runs_exact_original_subset_then_cleans_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "private_site"
            site.mkdir()
            data = host.manifest()
            state = {"packages": data["protected_versions"]}
            def probe(output, label, arguments, timeout):
                prepare.save(output / "probe.json", {"ready": True})
                return {"returncode": 0, "timed_out": False}
            with patch.object(host.sys, "platform", "linux"), patch.object(host.sys, "version_info", (3, 10)), patch.object(host.sys, "prefix", str(root)), patch.object(host.sysconfig, "get_path", return_value=str(site)), patch.object(prepare, "snapshot", return_value=state), patch.object(prepare, "version", return_value=None), patch.object(host, "check_sources"), patch.object(host, "run_child", side_effect=probe), patch.object(host.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as bench:
                code = host.run(root, None)
                self.assertEqual(code, 1)
                command = bench.call_args.args[0]
                self.assertEqual(command[command.index("--jobs") + 1:], prepare.manifest()["jobs"])
                self.assertIn(str(host.ROOT / "calibration/lightop_vendor_suite.py"), command)
            self.assertFalse((site / "fusion_lightop_host_smi.pth").exists())

    def test_changed_host_sdk_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            source = package / "__init__.py"
            source.write_text("original")
            data = {"sdk_package_path": str(package), "python_sources_sha256": {"__init__.py": prepare.sha(source)}, "library_path": str(source), "library_sha256": prepare.sha(source)}
            host.check_sources(data)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "wrapper differs"):
                host.check_sources(data)

    def test_platform_entrypoint_is_discovered_and_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(host.os.environ, {}, clear=True):
                with host.platform_registration(root, root):
                    entries = [e for d in importlib.metadata.distributions(path=[str(root)])
                               for e in d.entry_points if e.group == "vllm.platform_plugins"]
                    self.assertEqual(len(entries), 1)
                    self.assertEqual(entries[0].name, hygon_platform.ENTRY_POINT)
                    self.assertIs(entries[0].load(), hygon_platform.register)
            self.assertFalse(list(importlib.metadata.distributions(path=[str(root)])))
            self.assertTrue(json.loads((root / "platform_cleanup.json").read_text())["removed"])

    def test_platform_binding_cleans_up_on_probe_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(host.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "probe failed"):
                    with host.platform_registration(root, root):
                        raise RuntimeError("probe failed")
            self.assertFalse(list(root.glob("*.dist-info")))

    def test_existing_plugin_allowlist_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(host.os.environ, {"VLLM_PLUGINS": "another_plugin"}):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "VLLM_PLUGINS excludes"):
                with host.platform_registration(root, root):
                    self.fail("must not create a binding")
            self.assertFalse(list(root.glob("*.dist-info")))

    def test_plugin_requires_hip_and_a_real_smi_device(self):
        smi = Mock()
        smi.amdsmi_get_processor_handles.return_value = [object()]
        for hip, handles, expected in ((None, [object()], None), ("6.3", [], None), ("6.3", [object()], hygon_platform.PLATFORM)):
            smi.reset_mock()
            smi.amdsmi_get_processor_handles.return_value = handles
            with self.subTest(hip=hip, count=len(handles)), patch.dict(sys.modules, {"torch": SimpleNamespace(version=SimpleNamespace(hip=hip)), "amdsmi": smi}):
                self.assertEqual(hygon_platform.register(), expected)
            if hip:
                smi.amdsmi_shut_down.assert_called_once()
            else:
                smi.amdsmi_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
