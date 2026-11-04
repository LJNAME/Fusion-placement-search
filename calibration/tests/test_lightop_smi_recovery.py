import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_lightop_bench_followup as prepare
from tools import recover_lightop_smi as recovery


class SmiRecoveryTests(unittest.TestCase):
    def receipt(self, root):
        data = prepare.manifest()
        before = {"executable": sys.executable, "packages": {
            "torch": "2.9.0+das.opt1.dtk2604.test", "triton": "existing", "lightop": "existing", "numpy": "1.26.4"}}
        after = copy.deepcopy(before)
        after["packages"]["amdsmi"] = "1.0.0+a788f784"
        root.mkdir(exist_ok=True)
        prepare.save(root / "before.json", before)
        prepare.save(root / "after.json", after)
        (root / "plan.json").write_bytes(prepare.MANIFEST.read_bytes())
        (root / "probe.log").write_text("AttributeError: /opt/hyhal/lib/libamd_smi.so: undefined symbol: amdsmi_get_node_handle")
        return data, before, after

    def test_exact_newly_installed_failure_is_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, before, after = self.receipt(root)
            selected = recovery.validate_receipt(root, after, sys.executable, data)
            self.assertEqual(selected["distribution"], "amdsmi")

    def test_preexisting_smi_is_never_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, before, after = self.receipt(root)
            before["packages"]["amdsmi"] = "1.0.0+a788f784"
            prepare.save(root / "before.json", before)
            with self.assertRaisesRegex(ValueError, "existed before"):
                recovery.validate_receipt(root, after, sys.executable, data)

    def test_different_environment_or_package_is_never_removed(self):
        for change, message in (("python", "same private Python"), ("amdsmi", "exact newly installed"), ("torch", "Numeric versions"), ("error", "specific to receipt03"), ("plan", "plan differs")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data, _, after = self.receipt(root)
                executable = sys.executable
                if change == "python": executable = "/another/python"
                if change in ("amdsmi", "torch"): after["packages"][change] = "different"
                if change == "error": (root / "probe.log").write_text("different failure")
                if change == "plan": (root / "plan.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, message):
                    recovery.validate_receipt(root, after, executable, data)

    def test_failed_numeric_import_prevents_dependency_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, before, _ = self.receipt(root)
            with patch.object(prepare, "snapshot", return_value=before), patch.object(prepare.platform, "system", return_value="Linux"), patch.object(prepare.platform, "machine", return_value="x86_64"), patch.object(prepare.sys, "version_info", (3, 10)), patch.object(prepare, "numeric_import_probe", side_effect=RuntimeError("Existing Torch import failed")), patch.object(prepare, "pip") as pip:
                with self.assertRaisesRegex(RuntimeError, "Torch import failed"):
                    prepare.install(root)
                pip.assert_not_called()

    def test_frozen_bundle_cannot_reinstall_bad_optional_smi(self):
        data = prepare.manifest()
        self.assertIn("amdsmi", [w["distribution"] for w in data["wheels"]])
        self.assertEqual([w["distribution"] for w in prepare.baseline_wheels(data)], ["vllm", "lmslim"])

    def test_recovery_backs_up_package_and_only_uninstalls_new_smi(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "receipt"
            data, before, current = self.receipt(receipt)
            prefix = root / "private_python"
            package_root = prefix / "site-packages"
            relative = "amdsmi/amdsmi_wrapper.py"
            source = package_root / relative
            source.parent.mkdir(parents=True)
            source.write_text("# original installed wrapper\n")
            selected = next(w for w in data["wheels"] if w["distribution"] == "amdsmi")
            selected["source_sha256"] = {relative: prepare.sha(source)}
            output = root / "recovery"
            output.mkdir()
            from unittest.mock import Mock
            dist = Mock(files=[relative])
            dist.locate_file.side_effect = lambda name: package_root / name
            with patch.object(recovery.sys, "platform", "linux"), patch.object(recovery.sys, "version_info", (3, 10)), patch.object(recovery.sys, "prefix", str(prefix)), patch.object(prepare, "manifest", return_value=data), patch.object(prepare, "snapshot", side_effect=[current, before]), patch.object(prepare, "version", return_value=None), patch.object(recovery.metadata, "distribution", return_value=dist), patch.object(prepare, "pip") as pip, patch.object(recovery, "collect_diagnostics") as collect:
                recovery.recover(receipt, output)
            pip.assert_called_once_with(output, "remove_receipt03_smi", ["uninstall", "-y", "amdsmi"])
            self.assertEqual((output / "removed_distribution" / relative).read_bytes(), source.read_bytes())
            collect.assert_called_once_with(output)

    def test_installed_source_mismatch_blocks_uninstall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "receipt"
            _, _, current = self.receipt(receipt)
            output = root / "out"
            output.mkdir()
            source = root / "modified_source"
            source.write_text("different from pinned SMI wrapper")
            from unittest.mock import Mock
            dist = Mock()
            dist.locate_file.return_value = source
            with patch.object(recovery.sys, "platform", "linux"), patch.object(recovery.sys, "version_info", (3, 10)), patch.object(prepare, "snapshot", return_value=current), patch.object(recovery.metadata, "distribution", return_value=dist), patch.object(prepare, "pip") as pip:
                with self.assertRaisesRegex(ValueError, "source differs"):
                    recovery.recover(receipt, output)
                pip.assert_not_called()

    def test_child_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(recovery.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 20)):
            result = recovery.run_child(Path(temporary), "probe", ["-c", "pass"], timeout=20)
            self.assertTrue(result["timed_out"])
            self.assertIsNone(result["returncode"])
            saved = json.loads((Path(temporary) / "probe.process.json").read_text())
            self.assertTrue(saved["timed_out"])

    def test_child_signal_is_persisted(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary, patch.object(recovery.subprocess, "run", return_value=SimpleNamespace(returncode=-11)):
            result = recovery.run_child(Path(temporary), "probe", ["-c", "pass"])
            self.assertEqual(result["signal"], "SIGSEGV")
            self.assertEqual(json.loads((Path(temporary) / "probe.process.json").read_text())["returncode"], -11)

    def test_failed_import_process_does_not_prevent_sdk_inventory(self):
        def child(output, label, arguments, timeout=60):
            (output / (label + ".log")).write_text("IMPORT_BODY_COMPLETE\n")
            return {"returncode": -11 if label == "torch_lightop" else 0, "timed_out": False}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(recovery, "run_child", side_effect=child) as run, patch.object(recovery, "sdk_paths", return_value=[]):
                recovery.collect_diagnostics(output)
            self.assertEqual(run.call_args_list[-1].args[1], "host_library")
            saved = json.loads((output / "summary.json").read_text())
            self.assertFalse(saved["torch_and_lightop_import_restored"])
            self.assertTrue(saved["import_probes"]["torch_lightop"]["import_body_completed"])
            self.assertEqual(saved["import_probes"]["torch_lightop"]["returncode"], -11)
            self.assertEqual(saved["gpu_measurement_jobs"], 0)

    def test_resume_checks_removal_evidence_without_repeating_uninstall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, before, after = self.receipt(root)
            prepare.save(root / "before.json", after)
            prepare.save(root / "after.json", before)
            source = root / "removed_distribution/amdsmi/amdsmi_wrapper.py"
            source.parent.mkdir(parents=True)
            source.write_text("# saved wrapper")
            selected = next(w for w in data["wheels"] if w["distribution"] == "amdsmi")
            selected["source_sha256"] = {"amdsmi/amdsmi_wrapper.py": prepare.sha(source)}
            (root / "failure.json").write_text("{}")
            (root / "torch_import.log").write_text("2.9.0\n")
            output = root / "diagnostic"
            output.mkdir()
            with patch.object(recovery.sys, "platform", "linux"), patch.object(recovery.sys, "version_info", (3, 10)), patch.object(prepare, "manifest", return_value=data), patch.object(prepare, "snapshot", return_value=before), patch.object(prepare, "pip") as pip, patch.object(recovery, "collect_diagnostics") as collect:
                recovery.resume(root, output)
                pip.assert_not_called()
                collect.assert_called_once_with(output)
                changed = copy.deepcopy(before)
                changed["packages"]["torch"] = "different"
                with self.assertRaisesRegex(ValueError, "unchanged post-removal"):
                    recovery.validate_removed_receipt(root, changed)


if __name__ == "__main__":
    unittest.main()
