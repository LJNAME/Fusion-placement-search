"""Identity, exact lookup and admission tests; generated fixtures are not GPU data."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from calibration.kerv_profile_workflow import demonstrate
from calibration.measure_kerv_profile_pair import RECIPE, inspect_trace, main as collect
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_profile import KervMeasuredProfile, KervProfileStore, make_record, normalize_identity
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from simulator.placement_artifacts import load_placed_op_graph


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/kerv_profile_identity_v1/profile.json"
EVIDENCE = BUNDLE.parent / "inputs/gpu0/fused"


def resign(record):
    record["record_id"] = key_id({k: v for k, v in record.items() if k != "record_id"})
    return record


class TestCanonicalKervProfile(unittest.TestCase):
    def setUp(self):
        self.placed = load_placed_op_graph(EVIDENCE / "placed.json")
        self.report = json.loads((EVIDENCE / "result.json").read_text())
        self.store = KervProfileStore.load(BUNDLE)
        self.record = next(iter(self.store.records.values()))

    def complete_fixture(self, device=0):
        # Explicit artificial metadata for CPU identity tests, never exported as
        # measured evidence. The source package remains the actual GPU0 receipt.
        report = copy.deepcopy(self.report)
        placed = copy.deepcopy(self.placed)
        placed.plan.groups[18].device = device
        report["logical_device"] = device
        report["runtime_device"] = f"cuda:{device}"
        report["observed_device"].update(architecture="test-architecture", uuid=f"test-device-{device}")
        report["loaded_source"] = {"module_sha256": "test-module-hash", "git_revision": "test-revision"}
        report["measurement_contract"].update(rounds=7, calls_per_round=50,
            warmup={"replays_per_arm": 10}, statistic="median", synchronization="test-sync")
        report["measurement_contract"]["environment"]["driver_version"] = "test-driver"
        report["measurement_contract_id"] = measurement_contract_id(report["measurement_contract"])
        return placed, report

    def test_real_returned_data_queries_only_with_explicit_diagnostic_flag(self):
        key, contract = self.record["kernel_key"], self.record["measurement_contract_id"]
        with self.assertRaises(MeasurementContractError):
            self.store.query(key, contract)
        with self.assertWarns(RuntimeWarning):
            result = self.store.query(key, contract, allow_conditional_dev=True)
        self.assertAlmostEqual(result["statistics"]["median_us"], 7.238399982452393)
        self.assertEqual(result["admission"], "conditional_dev")
        self.assertEqual(result["identity_gaps"], ["device.architecture", "implementation.loaded_source",
            "measurement_contract.recipe_incomplete", "runtime.driver_version"])

    def test_cost_model_consumes_real_duration_once_and_traces_identity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = demonstrate(BUNDLE, allow_conditional_dev=True)
        self.assertEqual(result["measured_candidates"], 1)
        row = result["results"][0]
        self.assertAlmostEqual(row["makespan_s"], 7.238399982452393e-6, places=15)
        used = [u for u in row["profile_usage"] if u["measured"]]
        self.assertEqual(len(used), 1)
        self.assertEqual(used[0]["record_id"], self.record["record_id"])
        self.assertEqual(used[0]["kernel_key_id"], self.record["kernel_key_id"])
        self.assertEqual(used[0]["measurement_contract_id"], self.record["measurement_contract_id"])

    def test_unknown_key_or_eager_contract_does_not_fall_back(self):
        key = copy.deepcopy(self.record["kernel_key"])
        key["implementation"]["version"] = "different-kernel-version"
        with self.assertRaises(KeyError):
            self.store.query(key, self.record["measurement_contract_id"], allow_conditional_dev=True)
        with self.assertRaises(KeyError):
            self.store.query(self.record["kernel_key"], "eager-contract", allow_conditional_dev=True)

    def test_hardware_architecture_uuid_implementation_dtype_shape_and_layout_change_key(self):
        placed, report = self.complete_fixture()
        _, _, base, gaps = normalize_identity(placed, report)
        self.assertEqual(gaps, [])
        variants = []
        for field, value in (("architecture", "another-architecture"), ("uuid", "another-device")):
            changed = copy.deepcopy(report)
            changed["observed_device"][field] = value
            variants.append((placed, changed))
        changed = copy.deepcopy(report)
        changed["loaded_source"]["module_sha256"] = "different-loaded-source"
        variants.append((placed, changed))
        changed = copy.deepcopy(report)
        changed["input"]["up_strides"] = [786432, 3072, 2]
        with self.assertRaisesRegex(ValueError, "contiguous layout"):
            normalize_identity(placed, changed)
        for name, value in (("dtype", "fp32"), ("output_shape", [1, 1, 3072])):
            other, changed = copy.deepcopy(placed), copy.deepcopy(report)
            for node in other.graph.nodes.values():
                if node.id in (*changed["node_ids"], changed["gate_node"], changed["up_node"]):
                    node.extra[name] = value
            changed["input"]["dtype" if name == "dtype" else "shape"] = value
            if name == "output_shape":
                changed["input"]["gate_strides"] = [3072, 3072, 1]
                changed["input"]["up_strides"] = [3072, 3072, 1]
            variants.append((other, changed))
        for other, changed in variants:
            self.assertNotEqual(key_id(base), key_id(normalize_identity(other, changed)[2]))

    def test_gpu0_and_gpu1_are_separate_records_without_profile_sharing(self):
        p0, r0 = self.complete_fixture(0)
        p1, r1 = self.complete_fixture(1)
        a = make_record(p0, r0, "fused_kernel", {}, case_id="gpu0")[0]
        b = make_record(p1, r1, "fused_kernel", {}, case_id="gpu1")[0]
        self.assertNotEqual(a["kernel_key_id"], b["kernel_key_id"])
        store = KervProfileStore([a])
        with self.assertRaises(KeyError):
            store.query(b["kernel_key"], b["measurement_contract_id"], allow_conditional_dev=True)
        self.assertFalse(RECIPE["hardware_profile_sharing"])

    def test_four_record_fixture_keeps_native_separate_and_scores_both_devices(self):
        records, cases = [], []
        for device in (0, 1):
            placed, fused_report = self.complete_fixture(device)
            fused, keys, group = make_record(placed, fused_report, "fused_kernel", {}, case_id=f"gpu{device}")
            native_report = copy.deepcopy(fused_report)
            native_report["binding"] = {"id": "test-native-chain", "version": "test-build"}
            native = make_record(placed, native_report, "native_chain", {}, case_id=f"gpu{device}")[0]
            self.assertNotEqual(native["kernel_key_id"], fused["kernel_key_id"])
            records.extend((fused, native))
            from cost_model.structure import analyze_plan_structure
            structure = analyze_plan_structure(keys.graph, placed.plan, keys.hardware.n_devs)
            cases.append((fused, keys, structure.groups[18]))
        store = KervProfileStore(records)
        self.assertEqual(len(store.records), 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for record, keys, group in cases:
                profile = KervMeasuredProfile(keys, store, {group.device: record["measurement_contract_id"]},
                                              allow_conditional_dev=True)
                cost = profile.estimate_group(keys.graph, group)
                self.assertAlmostEqual(cost.duration, record["statistics"]["median_us"] / 1e6, places=15)
                self.assertEqual(cost.launch_time, 0.)
                self.assertEqual(profile.usage[-1]["record_id"], record["record_id"])

    def test_measurement_contract_is_separate_from_kernel_identity(self):
        placed, report = self.complete_fixture()
        other = copy.deepcopy(report)
        other["measurement_contract"]["warmup"] = {"replays_per_arm": 20}
        other["measurement_contract_id"] = measurement_contract_id(other["measurement_contract"])
        first = make_record(placed, report, "fused_kernel", {}, case_id="test")[0]
        second = make_record(placed, other, "fused_kernel", {}, case_id="test")[0]
        self.assertEqual(first["kernel_key_id"], second["kernel_key_id"])
        self.assertNotEqual(first["measurement_contract_id"], second["measurement_contract_id"])

    def test_native_chain_is_a_distinct_comparison_record_not_a_fused_cost(self):
        placed, report = self.complete_fixture()
        report["binding"] = {"id": "test-native-chain", "version": "test-build", "inplace": False}
        native, keys, group = make_record(placed, report, "native_chain", {}, case_id="test")
        profile = KervMeasuredProfile(keys, KervProfileStore([native]),
            {0: native["measurement_contract_id"]}, allow_conditional_dev=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "comparison-only"):
                profile.estimate_group(keys.graph, group)

    def test_manual_status_flip_cannot_bypass_review_or_incomplete_identity(self):
        record = resign({**copy.deepcopy(self.record), "admission": "accepted"})
        store = KervProfileStore([record], approved_contract_ids=[record["measurement_contract_id"]])
        with self.assertRaisesRegex(MeasurementContractError, "reviewed contract and complete identity"):
            store.query(record["kernel_key"], record["measurement_contract_id"])

    def test_record_tampering_and_raw_evidence_tampering_are_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["statistics"]["median_us"] = 1
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            KervProfileStore([changed])
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            target = Path(directory) / "bundle"
            shutil.copytree(BUNDLE.parent, target)
            raw = target / "inputs/gpu0/fused/samples.json"
            raw.write_text("[]")
            with self.assertRaisesRegex(ValueError, "evidence missing or changed"):
                KervProfileStore.load(target / "profile.json")

    def test_bad_timing_or_recipe_does_not_create_record(self):
        placed, report = self.complete_fixture()
        report["rows"][0]["graph_event_total_s"] = -1
        with self.assertRaisesRegex(ValueError, "invalid timing"):
            make_record(placed, report, "fused_kernel", {}, case_id="test")
        placed, report = self.complete_fixture()
        report["measurement_contract"]["calls_per_round"] = 49
        report["measurement_contract_id"] = measurement_contract_id(report["measurement_contract"])
        with self.assertRaisesRegex(ValueError, "call count"):
            make_record(placed, report, "fused_kernel", {}, case_id="test")

    def test_pair_collector_refuses_cpu_without_claiming_any_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "two GPUs required"):
                    collect(["--placed", str(EVIDENCE / "placed.json"), "--out-dir", str(output)])
            self.assertFalse((output / "summary.json").exists())
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["profile_accepted"])

    def test_native_and_fused_trace_require_different_gpu_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            def trace(names, operators):
                kernels = [{"cat": "kernel", "name": n, "args": {"correlation": i}} for i, n in enumerate(names)]
                launches = [{"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "args": {"correlation": i}}
                            for i in range(len(names))]
                path.write_text(json.dumps({"traceEvents": [*kernels, *launches,
                    *[{"cat": "cpu_op", "name": n} for n in operators]]}))
            trace(["silu", "mul"], ["aten::silu", "aten::mul"])
            self.assertEqual(inspect_trace(path, "native")["device_kernel_count"], 2)
            with self.assertRaises(ValueError):
                inspect_trace(path, "fused")
            trace(["_silu_mul_kernel"], ["flagos_embodied::kerv_silu_mul"])
            self.assertEqual(inspect_trace(path, "fused")["device_kernel_count"], 1)
            with self.assertRaises(ValueError):
                inspect_trace(path, "native")


if __name__ == "__main__":
    unittest.main()
