"""CPU protocol tests. Synthetic raw fixtures are never written to datasets."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from calibration import group_key_remeasurement as remeasure
from cost_model.fusion_kernel_profile import key_id


def environment():
    return remeasure.sealed({
        "host": "unit-test-only", "runtime": {
            "framework": "pytorch", "framework_version": "fixture",
            "accelerator": "hip", "accelerator_version": "fixture",
            "triton_version": "fixture", "torch_build_config_sha256": "fixture"},
        "devices": [{"device_index": d, "name": "fixture", "uuid": f"fixture-{d}",
                     "physical_index": None} for d in (0, 1)],
    }, "environment_id")


def fixture_bundle(path, manifest):
    remeasure.write_json(path / "frozen_manifest.json", manifest)
    with (path / "setup_records.jsonl").open("x", encoding="utf-8") as stream:
        for e in manifest["entries"]:
            fused = e["pattern_id"] == "silu_mul"
            remeasure.append_record(stream, {
                "slot_id": e["slot_id"], "kernel_id": e["kernel_id"],
                "manifest_id": manifest["manifest_id"], "status": "ok", "setup_wall_s": 1.,
                "correctness": {"passed": True, "finite": True, "output_alias_gate": fused,
                                "inputs_preserved_except_declared_gate": True,
                                "observed_fused_JIT_dispatches": 1 if fused else None},
            })
    by_slot = {e["slot_id"]: e for e in manifest["entries"]}
    with (path / "raw_measurements.jsonl").open("x", encoding="utf-8") as stream:
        for b in manifest["schedule"]:
            remeasure.append_record(stream, {
                **b, "kernel_id": by_slot[b["slot_id"]]["kernel_id"],
                "manifest_id": manifest["manifest_id"],
                "environment_id": manifest["environment"]["environment_id"],
                "status": "ok", "block_wall_s": .1,
                "kernel_samples_s": [.001 * (b["sample"] + 1)] * manifest["recipe"]["iterations_per_sample"],
            })


def rewrite_records(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            remeasure.append_record(stream, {k: v for k, v in row.items() if k != "record_id"})


class GroupRemeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft = remeasure.load_draft()

    def manifest(self, **kwargs):
        return remeasure.bind_manifest(self.draft, environment(), **kwargs)

    def test_draft_remains_unbound_and_label_free(self):
        self.assertEqual(len(self.draft["entries"]), 12)
        self.assertTrue(all(e["kernel_id"] is None and e["latency_s"] is None
                            for e in self.draft["entries"]))

    def test_draft_rejects_edits_even_after_manifest_rehash(self):
        d = copy.deepcopy(self.draft)
        d["entries"][0]["kernel_key_template"]["operators"][0]["output"]["dtype"] = "fp32"
        d = remeasure.sealed({k: v for k, v in d.items() if k != "draft_id"}, "draft_id")
        with TemporaryDirectory() as directory:
            p = Path(directory) / "draft.json"
            remeasure.write_json(p, d)
            with self.assertRaisesRegex(ValueError, "specification"):
                remeasure.load_draft(p)

    def test_mm_physical_operands_and_logical_view_are_explicit(self):
        c = remeasure.operation_contract(self.draft["entries"][0]["kernel_key_template"])
        self.assertEqual([v["shape"] for v in c["inputs"]], [[256, 1024], [1024, 3072]])
        self.assertEqual(c["output"], {"shape": [1, 256, 3072], "strides": [786432, 3072, 1]})

    def test_final_keys_include_real_environment_layout_and_collector(self):
        m = self.manifest()
        self.assertEqual(len({e["kernel_id"] for e in m["entries"]}), 12)
        for e in m["entries"]:
            key = e["kernel_key"]
            self.assertEqual(e["kernel_id"], key_id(key))
            self.assertEqual(key["hardware"]["source"], m["environment"]["environment_id"])
            self.assertIsNone(key["hardware"]["physical_index"])
            self.assertIn("measurement_contract", key["implementation"])
            self.assertTrue(all(t["strides"] for t in key["external_inputs"]))
        self.assertTrue(all(e["kernel_id"] is None for e in self.draft["entries"]))

    def test_different_device_and_runtime_produce_different_keys(self):
        a = self.manifest()
        env = environment()
        env["runtime"]["accelerator_version"] = "different-build"
        env = remeasure.sealed({k: v for k, v in env.items() if k != "environment_id"}, "environment_id")
        b = remeasure.bind_manifest(self.draft, env)
        self.assertTrue(all(x["kernel_id"] != y["kernel_id"] for x, y in zip(a["entries"], b["entries"])))

    def test_missing_runtime_cannot_become_formal_key(self):
        env = environment()
        env["runtime"]["triton_version"] = None
        env = remeasure.sealed({k: v for k, v in env.items() if k != "environment_id"}, "environment_id")
        with self.assertRaisesRegex(ValueError, "unbound"):
            remeasure.bind_manifest(self.draft, env)

    def test_predeclared_schedule_covers_all_keys_and_interleaves_devices(self):
        for mode, blocks in (("smoke", 36), ("formal", 84)):
            m = self.manifest(mode=mode)
            schedule = m["schedule"]
            self.assertEqual(len(schedule), blocks)
            self.assertEqual(schedule, self.manifest(mode=mode)["schedule"])
            by_slot = {e["slot_id"]: e for e in m["entries"]}
            for start in range(0, blocks, 2):
                pair = schedule[start:start + 2]
                self.assertEqual({by_slot[r["slot_id"]]["device_index"] for r in pair}, {0, 1})
                self.assertEqual(pair[0]["slot_id"].split("@")[0], pair[1]["slot_id"].split("@")[0])
        self.assertEqual(84 * m["recipe"]["iterations_per_sample"], 4200)

    def test_cpu_primitives_validate_semantics_without_gpu_measurements(self):
        import torch
        m = self.manifest()
        with torch.no_grad():
            for kind in ("mm", "silu", "mul"):
                e = copy.deepcopy(next(e for e in m["entries"]
                    if e["kernel_key"]["implementation"]["measurement_contract"]["kind"] == kind))
                c = e["kernel_key"]["implementation"]["measurement_contract"]
                shapes = [[2, 3], [3, 4]] if kind == "mm" else [[1, 2, 4]] * len(c["inputs"])
                c["inputs"] = [{"shape": s, "strides": remeasure.strides(s)} for s in shapes]
                c["output"] = {"shape": [1, 2, 4], "strides": [8, 4, 1]}
                p = remeasure.prepare_operation(e, torch, None, device="cpu")
                result = remeasure.validate_operation(p, torch, None, m["recipe"]["numeric"])
                self.assertTrue(result["passed"])
                self.assertFalse(result["output_alias_gate"])

    def test_inplace_restore_is_outside_every_event_pair(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        log = []

        class Event:
            def record(self): log.append("event")
            def elapsed_time(self, end): return .25

        torch = SimpleNamespace(cuda=SimpleNamespace(
            device=lambda index: nullcontext(), Event=lambda **kwargs: Event(),
            synchronize=lambda index: None))
        prepared = {"restore": lambda: log.append("restore"), "call": lambda: log.append("call")}
        with patch.object(remeasure, "telemetry", return_value={}):
            row = remeasure.sample_block(prepared, torch, 0, 3)
        self.assertEqual(log, ["restore", "event", "call", "event"] * 3)
        self.assertEqual(row["kernel_samples_s"], [.00025] * 3)

    def test_cpu_collection_is_rejected_and_failure_is_retained(self):
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            bundle = Path(directory) / "cpu-attempt"
            with self.assertRaisesRegex(RuntimeError, "no GPU"):
                remeasure.collect(self.draft, bundle, None, None)
            failure = json.loads((bundle / "failure.json").read_text())
            self.assertEqual(failure["validation_launch_attempts"], 0)
            self.assertFalse(failure["latency_labels_promoted"])
            self.assertFalse((bundle / "frozen_manifest.json").exists())
            with self.assertRaises(FileExistsError):
                remeasure.collect(self.draft, bundle, None, None)

    def test_collection_freezes_before_launch_and_retains_failed_key(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import torch
        for fail in (False, True):
            with TemporaryDirectory() as directory:
                bundle = Path(directory) / "simulated-protocol-only"
                module = SimpleNamespace(
                    configure_recording=lambda *args: None,
                    configure_backend=lambda *args: None,
                    configure_enabled_operators=lambda *args: None,
                    triton=object(), _use_triton=lambda name: True)

                def prepare(e, *args):
                    self.assertTrue((bundle / "frozen_manifest.json").exists())
                    return {"entry": e, "restore": lambda: None, "call": lambda: None}

                def validate(p, *args):
                    if fail and p["entry"]["slot_id"] == self.draft["entries"][0]["slot_id"]:
                        raise ValueError("simulated numeric failure")
                    fused = p["entry"]["pattern_id"] == "silu_mul"
                    return {"passed": True, "finite": True, "output_alias_gate": fused,
                            "inputs_preserved_except_declared_gate": True,
                            "observed_fused_JIT_dispatches": 1 if fused else None}

                def sample(p, torch, device, iterations):
                    return {"kernel_samples_s": [.001] * iterations, "block_wall_s": .000001}

                with patch("torch.cuda.is_available", return_value=True), \
                     patch("torch.cuda.device_count", return_value=2), \
                     patch("torch.version.hip", "simulated"), \
                     patch("torch.cuda.device", side_effect=lambda d: nullcontext()), \
                     patch("torch.cuda.synchronize"), \
                     patch("calibration.probe_kerv_silu_mul.load_kernel_module", return_value=(module, {"capability_id": self.draft["capability_id"]})), \
                     patch.object(remeasure, "capture_environment", return_value=environment()), \
                     patch.object(remeasure, "prepare_operation", side_effect=prepare), \
                     patch.object(remeasure, "validate_operation", side_effect=validate), \
                     patch.object(remeasure, "sample_block", side_effect=sample):
                    report = remeasure.collect(self.draft, bundle, None, None)
                self.assertEqual(report["accepted_keys"], 11 if fail else 12)
                self.assertEqual(report["complete"], not fail)
                self.assertEqual(report["recorded_blocks"], 36)
                rows = remeasure.read_records(bundle / "raw_measurements.jsonl")
                self.assertEqual(sum(r["status"] == "not_run" for r in rows), 3 if fail else 0)
                self.assertFalse((bundle / "verified_profile.json").exists())
                self.assertFalse((bundle / "failure.json").exists())

    def test_verifier_preserves_noise_and_accounts_per_key_wall(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            fixture_bundle(bundle, self.manifest())
            report = remeasure.verify_bundle(bundle)
            self.assertTrue(report["complete"])
            self.assertEqual(report["accepted_keys"], 12)
            for p in report["profiles"].values():
                self.assertEqual(p["duration_s"], .002)
                self.assertEqual(p["block_medians_s"], [.001, .002, .003])
                self.assertAlmostEqual(p["profiling_wall_s"], 1.3)
                self.assertEqual(p["first_to_last_relative_change"], 2.)
            self.assertFalse((bundle / "verified_profile.json").exists())

    def test_verifier_rejects_tampered_raw_samples(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            fixture_bundle(bundle, self.manifest())
            path = bundle / "raw_measurements.jsonl"
            rows = remeasure.read_records(path)
            rows[0]["kernel_samples_s"][0] = 100.
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record_id"):
                remeasure.verify_bundle(bundle)

    def test_verifier_rejects_wrong_key_even_after_record_rehash(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            fixture_bundle(bundle, self.manifest())
            path = bundle / "raw_measurements.jsonl"
            rows = remeasure.read_records(path)
            rows[0]["kernel_id"] = rows[1]["kernel_id"]
            rewrite_records(path, rows)
            with self.assertRaisesRegex(ValueError, "lineage"):
                remeasure.verify_bundle(bundle)

    def test_rebound_manifest_must_still_match_draft_and_environment(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            m = self.manifest()
            m["entries"][0]["kernel_key"]["hardware"]["uuid"] = "wrong-card"
            m["entries"][0]["kernel_id"] = key_id(m["entries"][0]["kernel_key"])
            m = remeasure.sealed({k: v for k, v in m.items() if k != "manifest_id"}, "manifest_id")
            fixture_bundle(bundle, m)
            with self.assertRaisesRegex(ValueError, "contract"):
                remeasure.verify_bundle(bundle)

    def test_missing_or_failed_blocks_cannot_be_full_profile(self):
        for remove in (True, False):
            with TemporaryDirectory() as directory:
                bundle = Path(directory)
                fixture_bundle(bundle, self.manifest())
                path = bundle / "raw_measurements.jsonl"
                rows = remeasure.read_records(path)
                if remove:
                    rows.pop()
                else:
                    rows[-1]["status"] = "error"
                rewrite_records(path, rows)
                report = remeasure.verify_bundle(bundle)
                self.assertFalse(report["complete"])
                self.assertEqual(report["accepted_keys"], 11)
                with patch("sys.stdout", new_callable=io.StringIO), self.assertRaisesRegex(ValueError, "incomplete"):
                    remeasure.main(["verify", "--bundle", str(bundle), "--out", str(bundle / "profile.json")])
                self.assertFalse((bundle / "profile.json").exists())

    def test_fused_label_requires_launch_and_alias_evidence(self):
        for field, value in (("observed_fused_JIT_dispatches", None), ("output_alias_gate", False)):
            with TemporaryDirectory() as directory:
                bundle = Path(directory)
                fixture_bundle(bundle, self.manifest())
                path = bundle / "setup_records.jsonl"
                rows = remeasure.read_records(path)
                row = next(r for r in rows if r["slot_id"].startswith("kerv_"))
                row["correctness"][field] = value
                rewrite_records(path, rows)
                with self.assertRaises(ValueError):
                    remeasure.verify_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
