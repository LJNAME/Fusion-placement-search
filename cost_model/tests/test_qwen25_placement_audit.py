"""CPU bookkeeping tests for placement auditing, not performance evidence."""
from types import SimpleNamespace

import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile,GroupCost
from cost_model.search_evaluation import qwen25_3b_spec
from integration.flagscale.kerv_add_rmsnorm import adapter_registry
from simulator import FusionGroup,Plan
from simulator.hardware import DeviceSpec,HardwareSpec
from simulator.model_graphs import build_model_graph
from tools.audit_qwen25_placement import PermutedProfile,overlap_statistics,relabel_plan,three_plans


def test_layer_split_preserves_all_groups_and_one_real_boundary():
    graph=build_model_graph(qwen25_3b_spec());plans=three_plans(graph,adapter_registry())
    b=plans['B_layers_0_17_gpu0_18_35_gpu1']
    devices={n:g.device for g in b.groups for n in g.node_ids}
    assert len(devices)==687 and len(b.groups)==651
    assert [sum(g.device==d and len(g.node_ids)>1 for g in b.groups) for d in (0,1)]==[18,18]
    crossing=[e for e in graph.edges if devices[e.src]!=devices[e.dst]]
    assert [(e.src,e.dst,e.bytes) for e in crossing]==[('L017.residual_add_2','L018.input',1048576.)]
    assert devices['embedding']==0 and devices['final_norm']==devices['lm_head']==1


def test_parallel_busy_time_is_not_balanced_work_in_sequential_stages():
    sequential=overlap_statistics([{'start':0.,'end':.002},{'start':.002,'end':.005}])
    assert sequential['multi_device_compute_overlap_ms']==0
    parallel=overlap_statistics([{'start':0.,'end':.003},{'start':.001,'end':.005}])
    assert parallel['multi_device_compute_overlap_ms']==pytest.approx(2.)
    assert parallel['union_compute_busy_ms']==pytest.approx(5.)
    assert parallel['peak_simultaneous_compute_devices']==2


def test_relabel_maps_costs_capacities_links_and_constraints():
    class Fixture(AnalyticHardwareProfile):
        qualifications={};reuse={};native_reuse={};declaration={'scope':'unit fixture'}
        def estimate_group(self,graph,group):
            return GroupCost(group.device+1.,group.device+1.,0.,0.,1.,'fixture')
        def plan_blockers(self,plan): return ['fixture'] if plan.groups[0].device==0 else []
    base=Fixture(HardwareSpec([DeviceSpec(100.,100.,1000.),DeviceSpec(50.,50.,2000.)],
                              link_matrix=[[None,10.],[20.,None]],comm_latency=.01))
    swapped=PermutedProfile(base)
    group=FusionGroup(['a'],0,{'pattern_id':'unfused'})
    assert swapped.estimate_group(None,group).duration==2.
    assert swapped.max_memory(0)==2000. and swapped.hardware.devices[0].flops==50.
    assert swapped.estimate_transfer(0,1,100.).duration==base.estimate_transfer(1,0,100.).duration
    assert swapped.hardware.link_matrix[0][1]==20.
    assert not swapped.plan_blockers(Plan(groups=[group]))
    assert base.hardware.devices[0].flops==100. and group.device==0


def test_relabel_is_an_involution_and_does_not_mutate_original():
    plan=Plan(groups=[FusionGroup(['a'],0,{'pattern_id':'unfused'}),FusionGroup(['b'],1,{'pattern_id':'unfused'})])
    swapped=relabel_plan(plan,(1,0))
    assert [g.device for g in swapped.groups]==[1,0]
    assert [g.device for g in plan.groups]==[0,1]
    assert relabel_plan(swapped,(1,0))==plan
