"""CPU measurement-contract checks; no timing values used as GPU evidence."""
import copy
import json

import pytest

from calibration import kerv_add_rmsnorm_timing as timing
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from calibration.kerv_add_rmsnorm_probe import prepare_adapter


def rows(values):
    return [{"round":i,"checks":dict(timing.CHECKS),"timing":{
        "elapsed_ms_per_50_calls":v,"error":None,
        "event_boundary":{"device":"cuda:0","start_device":"cuda:0","end_device":"cuda:0",
                          "stream":17,"calls_per_replay":50}}} for i,v in enumerate(values)]


def test_round_statistics_and_amortization():
    result=timing.summarize(rows([.5]*7))
    assert result["status"]=="quality_passed"
    assert result["statistics"]["median_us"]==10
    assert result["statistics"]["n_calls"]==350


@pytest.mark.parametrize("bad",["count","order","stream","device"])
def test_round_or_boundary_changes_are_rejected(bad):
    data=rows([.5]*7)
    if bad=="count":data.pop()
    elif bad=="order":data.reverse()
    elif bad=="stream":data[0]["timing"]["event_boundary"]["stream"]=18
    else:data[0]["timing"]["event_boundary"]["end_device"]="cuda:1"
    with pytest.raises(ValueError):timing.summarize(data)


def test_failed_event_does_not_disappear_or_get_replaced():
    data=rows([.5]*7);data[0]["timing"]["error"]="Event read failed"
    result=timing.summarize(data)
    assert result["status"]=="quality_failed" and result["statistics"] is None
    assert result["reasons"]==["invalid_timing_round_0"]
    assert len(data)==7


def test_iqr_failure_keeps_all_samples():
    result=timing.summarize(rows([.1,.2,.3,.4,.5,.6,.7]))
    assert result["status"]=="quality_failed" and len(result["samples_us"])==7


def observations():
    sample={"shape":[1,256,2048],"dtype":"bf16","layout":"contiguous","strides":[524288,2048,1],
            "layout_provenance":{"kind":"runtime_tensor","device":"cuda:0","observer":"CPU-test fixture"}}
    result={k:copy.deepcopy(sample) for k in ("x","residual","sum","normalized")}
    result["weight"]={**copy.deepcopy(sample),"shape":[2048],"strides":[1]}
    return result


def test_existing_canonical_key_has_both_outputs_and_ignores_uuid():
    bound,_=prepare_adapter(DEFAULT_FLAGSCALE,"cuda:0")
    runtime={"framework":"torch","framework_version":"test-only","accelerator":"HIP","accelerator_version":"test-only"}
    device={"name":"test-only","architecture":"gfx936","uuid":"first"}
    first=timing.kernel_identity(bound,observations(),runtime,device)
    device["uuid"]="second"
    assert timing.kernel_identity(bound,observations(),runtime,device)==first
    assert first["kernel_key"]["outputs"]==[0,1]
    assert "layout" not in bound.placed.graph.nodes[bound.x_node].extra
    assert isinstance(first["kernel_key"]["operators"][0]["attributes"]["shape"],dict)
    bad=observations();bad["x"]["layout"]="unknown"
    with pytest.raises(ValueError):timing.kernel_identity(bound,bad,runtime,device)


def test_no_gpu_leaves_failure_without_timing_or_profile(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    output=tmp_path/"result"
    with pytest.raises(RuntimeError,match="no CPU/eager fallback"):
        timing.main(["run","--flagscale-root",str(DEFAULT_FLAGSCALE),"--output",str(output)])
    assert json.loads((output/"failure.json").read_text())["accepted_profiles_added"]==0
    assert not (output/"result.json").exists()
