"""One real reviewed snapshot, exact queries and rejection tests; CPU only."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from cost_model.kerv_add_rmsnorm_profile import load_profile, review
from cost_model.measurement_labels import MeasurementContractError

DATA = Path(__file__).resolve().parents[2]/"data/kerv_add_rmsnorm_timing_01"


def test_real_cost_is_accepted_and_not_charged_twice():
    profile,group=load_profile(DATA)
    cost=profile.estimate_group(profile.keys.graph,group)
    assert cost.source=="measured_group_kernel"
    assert cost.duration==pytest.approx(5.353559851646423e-6)
    assert cost.compute_time==cost.duration
    assert cost.memory_time==0 and cost.launch_time==0


def test_wrong_contract_does_not_fall_back():
    with pytest.raises(MeasurementContractError,match="contract mismatch"):
        load_profile(DATA,contract_id="different-eager-contract")


@pytest.mark.parametrize("change",["device","group","implementation","layout"])
def test_other_conditions_cannot_hit_the_measured_cost(change):
    profile,group=load_profile(DATA)
    if change=="device":
        group=replace(group,device=1)
    elif change=="group":
        group=replace(group,node_ids=["L001.residual_add_1","L001.pre_mlp_norm"])
    elif change=="implementation":
        profile.keys.bindings["add_rmsnorm"]["version"]="different-implementation"
    else:
        profile.keys.graph.nodes["L000.self_attention.out_proj"].extra["layout"]="unknown"
    with pytest.raises((KeyError,MeasurementContractError)):
        profile.estimate_group(profile.keys.graph,group)


def test_changed_raw_receipt_is_rejected(tmp_path):
    directory=tmp_path/"receipt";shutil.copytree(DATA,directory)
    with (directory/"raw/round_0.json").open("a") as handle:handle.write(" ")
    with pytest.raises(MeasurementContractError,match="receipt changed"):
        load_profile(directory)


def test_conditional_label_is_not_implicitly_promoted(tmp_path):
    directory=tmp_path/"receipt";shutil.copytree(DATA,directory)
    path=directory/"review/profile.json";data=json.loads(path.read_text())
    data["latency"]["admission"]="conditional_dev";path.write_text(json.dumps(data))
    with pytest.raises(MeasurementContractError,match="not accepted"):
        load_profile(directory)


def test_source_mismatch_cannot_be_admitted(tmp_path):
    directory=tmp_path/"receipt";shutil.copytree(DATA,directory)
    path=directory/"raw/plan.json";data=json.loads(path.read_text())
    data["collector_sources"]["calibration/kerv_add_rmsnorm_timing.py"]="0"*64
    path.write_text(json.dumps(data))
    with pytest.raises(MeasurementContractError,match="committed source"):
        review(directory)
