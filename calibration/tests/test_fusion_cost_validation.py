"""Synthetic unit controls for the diagnostic adapter; never GPU evidence."""
import copy
import json
from pathlib import Path
import unittest
import warnings

from calibration.fusion_cost_validation import ConditionalGraphProfile, build_fixture_graph, paired_preference, search_fixture
from calibration.kerv_silu_mul import development_registry
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from cost_model.structure import analyze_plan_structure
from simulator.hardware import load_hardware

ROOT=Path(__file__).resolve().parents[2]


def case(fused=1.):
    contract=dict(executor="graph_replay",submission_included=True,allocation_included=False,
        granularity="graph_replay",unit="throughput_per_call",serial_plan_only=True,
        environment=dict(torch="test",hip="test",triton="test",device_model="test"))
    return dict(case_id="gpu0_seq1",shape=[1,1,3072],dtype="bf16",device=0,admission="conditional_dev",
        measurement_contract=contract,measurement_contract_id=measurement_contract_id(contract),
        statistics={arm:dict(median_us=value,mad_rel=0.,n_rounds=8,n_calls=400)
                    for arm,value in dict(silu=10.,mul=10.,fused=fused,chain=19.).items()})


class TestFusionCostValidation(unittest.TestCase):
    def setUp(self):
        self.hardware=load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json")
        self.graph=build_fixture_graph([1,1,3072])
        self.registry=development_registry(json.loads((ROOT/"data/kerv_silu_mul_proxy_v1/capability.json").read_text()),
                                           layout_policy="require_explicit")

    def profile(self, data):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profile=ConditionalGraphProfile(self.hardware,self.graph,data,
                expected_contract_id=data["measurement_contract_id"],allow_conditional_dev=True)
        self.assertEqual(len(caught),3)
        return profile

    def test_default_strict_admission_and_contract_mismatch_fail(self):
        data=case()
        with self.assertRaises(MeasurementContractError):
            ConditionalGraphProfile(self.hardware,self.graph,data,expected_contract_id=data["measurement_contract_id"])
        with self.assertRaises(MeasurementContractError):
            ConditionalGraphProfile(self.hardware,self.graph,data,expected_contract_id="wrong",allow_conditional_dev=True)
        data["measurement_contract"]["executor"]="eager"
        data["measurement_contract_id"]=measurement_contract_id(data["measurement_contract"])
        with self.assertRaises(MeasurementContractError):
            self.profile(data)

    def test_labels_remain_conditional_and_chain_is_not_a_search_input(self):
        data=case(); profile=self.profile(data)
        self.assertEqual(set(profile.matches),{"silu","mul","fused"})
        self.assertTrue(all(m.diagnostic and m.label.admission=="conditional_dev" for m in profile.matches.values()))
        changed=copy.deepcopy(data); changed["statistics"]["chain"]["median_us"]=.0001
        a,_=search_fixture(self.graph,profile,0,self.registry,"exact")
        b,_=search_fixture(self.graph,self.profile(changed),0,self.registry,"exact")
        self.assertEqual(a["candidates"],b["candidates"])

    def test_full_duration_has_no_extra_launch_and_is_low_confidence(self):
        profile=self.profile(case(fused=3.))
        result,plan=search_fixture(self.graph,profile,0,self.registry,"exact")
        self.assertAlmostEqual(result["candidates"][0]["predicted_cost_s"],3e-6)
        group=next(g for g in analyze_plan_structure(self.graph,plan,2,registry=self.registry).groups if len(g.node_ids)>1)
        cost=profile.estimate_group(self.graph,group)
        self.assertEqual(cost.launch_time,0.)
        self.assertLessEqual(cost.confidence,.15)
        self.assertIn("downstream_execution_contract_unconfirmed",cost.ood_reasons)

    def test_synthetic_slow_fusion_changes_mcts_output(self):
        # This exercises a branch absent from the empirical four graph cases.
        prior,_=search_fixture(self.graph,AnalyticHardwareProfile(self.hardware),0,self.registry,"mcts")
        fast,_=search_fixture(self.graph,self.profile(case(fused=1.)),0,self.registry,"mcts")
        slow,_=search_fixture(self.graph,self.profile(case(fused=100.)),0,self.registry,"mcts")
        self.assertEqual(prior["candidates"][0]["choice"],"fused")
        self.assertEqual(fast["candidates"][0]["choice"],"fused")
        self.assertEqual(slow["candidates"][0]["choice"],"unfused")

    def test_shape_and_accepted_promotion_are_rejected(self):
        data=case(); data["shape"]=[1,256,3072]
        with self.assertRaises(ValueError):
            self.profile(data)
        data=case(); data["admission"]="accepted"
        with self.assertRaises(MeasurementContractError):
            self.profile(data)

    def test_preference_rule_keeps_small_and_mixed_signs_inconclusive(self):
        small={i:dict(fused=10.05,chain=10.) for i in range(8)}
        mixed={i:dict(fused=12. if i else 8.,chain=10.) for i in range(8)}
        slow={i:dict(fused=12.,chain=10.) for i in range(8)}
        self.assertEqual(paired_preference(small),"inconclusive_or_small_difference")
        self.assertEqual(paired_preference(mixed),"inconclusive_or_small_difference")
        self.assertEqual(paired_preference(slow),"unfused_preferred")


if __name__=="__main__":
    unittest.main()
