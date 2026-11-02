"""Recorded export and synthetic identity mutations, with no GPU observation."""
import copy
from pathlib import Path
import pytest

from cost_model.ffn_profile_export import FFNExportStore, observed_key, read, ROOT
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError


def bundle(): return read(ROOT/"data/ffn_b4_cpu_preparation_v1/profiles.json")


def test_exact_inspection_remains_distinct_from_cost_admission():
    data = bundle(); store = FFNExportStore(data)
    inspected = 0
    for record in data["records"]:
        if not record["identity_gaps"]:
            result = store.inspect_exact(record["kernel_key"],record["measurement_contract_id"],usage=record["usage"])
            assert result["record_id"] == record["record_id"] and not result["cost_use_open"]
            inspected += 1
        with pytest.raises(MeasurementContractError):
            store.query_cost(record["kernel_key"],record["measurement_contract_id"],usage=record["usage"])
    assert inspected == 8


@pytest.mark.parametrize("field",["shape","dtype","layout","version","uuid","flags"])
def test_key_mutations_never_hit_an_existing_record(field):
    data = bundle(); record = next(r for r in data["records"] if r["row_id"]=="silu_mul/fused@gpu1")
    key = copy.deepcopy(record["kernel_key"])
    if field == "shape": key["external_inputs"][0]["shape"][1] = 128
    if field == "dtype": key["external_inputs"][0]["dtype"] = "fp16"
    if field == "layout": key["external_inputs"][0]["strides"][-1] = 2
    if field == "version": key["implementation"]["version"] = "different"
    if field == "uuid": key["hardware"]["source"]["uuid"] = "other-card"
    if field == "flags": key["runtime"]["flags"]["grad_enabled"] = True
    with pytest.raises(KeyError): FFNExportStore(data).inspect_exact(key,record["measurement_contract_id"],usage=record["usage"])


def test_native_chain_cannot_replace_singleton_and_copy_cannot_claim_p2p():
    data = bundle(); store=FFNExportStore(data)
    for kind, wrong in [("native_chain","singleton_identity_inspection"),("transfer","direct_p2p_cost")]:
        record=next(r for r in data["records"] if r["kind"]==kind)
        with pytest.raises(MeasurementContractError): store.inspect_exact(record["kernel_key"],record["measurement_contract_id"],usage=wrong)
    record=data["records"][0]
    with pytest.raises(KeyError): store.inspect_exact(record["kernel_key"],"different-contract",usage=record["usage"])


def test_rehashing_accepted_status_does_not_create_admission_authority():
    data=bundle(); record=data["records"][0];record["admission"]="accepted"
    record["record_id"]=key_id({k:v for k,v in record.items() if k!="record_id"})
    data["export_id"]=key_id({k:v for k,v in data.items() if k!="export_id"})
    with pytest.raises(MeasurementContractError): FFNExportStore(data)


def synthetic_observation(role="fused"):
    # Pure metadata fixture; never written as a measurement or queried as cost.
    def meta(ptr): return {"shape":[1,256,3072],"dtype":"torch.bfloat16","strides":[786432,3072,1],
                           "device":"cuda:1","data_ptr":ptr,"layout_source":"runtime Tensor observation"}
    gates=[meta(10000000+i*1572864) for i in range(50)]
    row="silu_mul/fused@gpu1" if role=="fused" else "gated_mul@gpu1"
    observed={"name":"fixture","total_memory":1024,"uuid":"fixture-GPU1","architecture":"fixture-arch"}
    env={"torch":"fixture","hip":"fixture","triton":"fixture","driver_version":"fixture", "device_uuid":observed["uuid"],"device_architecture":observed["architecture"]}
    record={"row_id":row,"identity":{"implementation":{"id":"fixture", "version":"sha256:"+"f"*64 if role=="fused" else "fixture","collector_source_sha256":"c"*64},
        "actual_torch_build":{"torch_module_version":"fixture","torch_extension_sha256":"e"*64,"git_revision":"fixture","collector_source_sha256":"c"*64,
                              "native_binding":{"functional_source_sha256":"f"*64},"grad_enabled":False,"matmul_fp32_precision":"none","allow_bf16_reduced_precision_reduction":True},
        "devices":{"1":{"observed":observed,"environment":env}},"loaded_kerv_source":{"module_sha256":"f"*64,"git_revision":"fixture"},
        "inputs":{"gates":gates,"up":meta(900000000)},
        "output_layouts":copy.deepcopy(gates) if role=="fused" else [meta(1000000000+i*1572864) for i in range(50)]}}
    placed={"graph":read(ROOT/"data/ffn_placement_b1/identity.json")["subgraph"],
            "hardware":read(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json")}
    return record,placed


def test_addresses_do_not_enter_canonical_key_but_alias_is_enforced():
    record,placed=synthetic_observation(); first,gaps,_=observed_key(record,placed)
    assert not gaps
    for meta in record["identity"]["inputs"]["gates"]+record["identity"]["output_layouts"]: meta["data_ptr"]+=4000000000
    second,_,_=observed_key(record,placed)
    assert key_id(first)==key_id(second)
    record["identity"]["output_layouts"][0]["data_ptr"]+=1
    with pytest.raises(ValueError,match="alias"): observed_key(record,placed)


def test_mul_unrecorded_left_operand_stays_unknown():
    record,placed=synthetic_observation("mul"); key,gaps,_=observed_key(record,placed)
    assert "mul_left.runtime_tensor_metadata_missing" in gaps
    assert key["external_inputs"][0]["layout"] is None


def test_unknown_runtime_stride_is_rejected_not_filled_from_shape():
    record,placed=synthetic_observation()
    record["identity"]["inputs"]["gates"][0]["strides"]=None
    with pytest.raises(ValueError): observed_key(record,placed)
