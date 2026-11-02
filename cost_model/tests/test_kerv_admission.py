"""Scoped strict admission over the real two-device receipt; no GPU execution."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from calibration.kerv_profile_workflow import demonstrate
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_admission import USE, load_admitted_store
from cost_model.kerv_profile import KervMeasuredProfile, KervProfileStore, digest, normalize_identity
from cost_model.measurement_labels import MeasurementContractError
from cost_model.structure import analyze_plan_structure
from simulator.placement_artifacts import load_placed_op_graph


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/kerv_profile_admission_v1"
PROFILE = DATA / "measurements/profile.json"
DECISION = DATA / "admission.json"


class TestScopedKervAdmission(unittest.TestCase):
    def test_two_fused_records_pass_strict_query_without_warnings_or_ood(self):
        before = digest(PROFILE)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            result = demonstrate(PROFILE, admission_path=DECISION)
        self.assertEqual(emitted, [])
        self.assertEqual(result["measured_candidates"], 2)
        self.assertEqual(result["admission_use"], USE)
        expected = [7.2511798143386835e-6, 7.232000231742859e-6]
        for row, duration in zip(result["results"], expected):
            self.assertFalse(row["out_of_distribution"])
            self.assertAlmostEqual(row["makespan_s"], duration, places=15)
            measured = [u for u in row["profile_usage"] if u["measured"]]
            self.assertEqual(len(measured), 1)
            self.assertEqual(measured[0]["source"], "measured_group_kernel")
            self.assertEqual(measured[0]["admission"], "accepted")
            self.assertEqual(measured[0]["admission_use"], USE)
            self.assertNotEqual(measured[0]["source_record_id"], measured[0]["record_id"])
        self.assertEqual(digest(PROFILE), before)

    def test_original_measurements_still_require_diagnostic_opt_in(self):
        original = KervProfileStore.load(PROFILE)
        self.assertTrue(all(r["admission"] == "conditional_dev" for r in original.records.values()))
        with self.assertRaises(MeasurementContractError):
            demonstrate(PROFILE)

    def test_native_chains_remain_comparison_only(self):
        store = load_admitted_store(PROFILE, DECISION, usage_scope=USE)
        self.assertEqual(sum(r["admission"] == "accepted" for r in store.records.values()), 2)
        self.assertEqual(sum(r["role"] == "native_chain" for r in store.records.values()), 2)
        native = next(r for r in store.records.values() if r["role"] == "native_chain")
        self.assertEqual(native["admission"], "conditional_dev")
        with self.assertRaises(MeasurementContractError):
            store.query(native["kernel_key"], native["measurement_contract_id"])

    def test_incorrect_usage_scope_or_contract_is_rejected(self):
        with self.assertRaisesRegex(MeasurementContractError, "cannot authorize"):
            load_admitted_store(PROFILE, DECISION, usage_scope="whole_model_eager")
        store = load_admitted_store(PROFILE, DECISION, usage_scope=USE)
        fused = next(r for r in store.records.values() if r["admission"] == "accepted")
        with self.assertRaises(KeyError):
            store.query(fused["kernel_key"], "eager-contract")
        changed = copy.deepcopy(fused["kernel_key"])
        changed["hardware"]["source"]["uuid"] = "another-gpu"
        with self.assertRaises(KeyError):
            store.query(changed, fused["measurement_contract_id"])
        with self.assertRaisesRegex(ValueError, "must not enable"):
            demonstrate(PROFILE, admission_path=DECISION, allow_conditional_dev=True)

    def test_decision_hash_scope_and_exact_record_binding_are_checked(self):
        source = json.loads(DECISION.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.json"
            for mutation in ("hash", "scope", "record", "bundle"):
                changed = copy.deepcopy(source)
                if mutation == "hash":
                    changed["status"] = "rejected"
                elif mutation == "scope":
                    changed["scope"]["whole_plan_costs_admitted"] = True
                elif mutation == "record":
                    changed["entries"][0]["source_record_id"] = "unreviewed-record"
                else:
                    changed["source_profile_sha256"] = "another-bundle"
                if mutation != "hash":
                    changed["admission_id"] = key_id({k: v for k, v in changed.items() if k != "admission_id"})
                path.write_text(json.dumps(changed))
                with self.assertRaises(MeasurementContractError):
                    load_admitted_store(PROFILE, path, usage_scope=USE)

    def test_admitted_profile_cannot_score_singletons_or_transfers(self):
        store = load_admitted_store(PROFILE, DECISION, usage_scope=USE)
        record = next(r for r in store.records.values() if r["role"] == "fused_kernel")
        report_path = next(p for p in record["evidence"] if p.endswith("/result.json"))
        placed_path = next(p for p in record["evidence"] if p.endswith("/placed.json"))
        report = json.loads((PROFILE.parent / report_path).read_text())
        placed = load_placed_op_graph(PROFILE.parent / placed_path)
        keys, group, _, _ = normalize_identity(placed, report)
        profile = KervMeasuredProfile(keys, store, {group.device: record["measurement_contract_id"]})
        structure = analyze_plan_structure(keys.graph, placed.plan, keys.hardware.n_devs)
        singleton = next(g for g in structure.groups if report["gate_node"] in g.node_ids)
        with self.assertRaisesRegex(MeasurementContractError, "singleton"):
            profile.estimate_group(keys.graph, singleton)
        with self.assertRaisesRegex(MeasurementContractError, "communication"):
            profile.estimate_transfer(0, 1, 1024)


if __name__ == "__main__":
    unittest.main()
