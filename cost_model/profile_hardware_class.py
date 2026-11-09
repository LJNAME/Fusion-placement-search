"""Hardware-class migration evidence; original device-bound labels stay intact."""
import copy
import json

from calibration.kerv_add_rmsnorm_probe import ROOT,sha
from cost_model.fusion_kernel_profile import key_id

EVIDENCE=ROOT/"data/gpu1_cost_completion_20261106_01"


def hardware_conditions(key):
    """Only physical identity is excluded, never architecture or runtime flags."""
    key=copy.deepcopy(key)
    hw=key["hardware"]
    for field in ("device_index","physical_index"): hw.pop(field,None)
    if isinstance(hw.get("source"),dict):
        for field in ("execution_device","uuid"): hw["source"].pop(field,None)
    return key


def hardware_class_audit(hardware,fused):
    review=json.loads((EVIDENCE/"review/evaluation.json").read_text(encoding="utf-8"))
    snapshots={}
    for name in ("runtime.json","fused_runtime.json"):
        path=EVIDENCE/"raw"/name
        if sha(path)!=review["raw_sha256"][name]: raise ValueError("hardware evidence receipt changed: "+name)
        snapshots[name]=json.loads(path.read_text(encoding="utf-8"))
    shared=snapshots["runtime.json"];gpu1=snapshots["fused_runtime.json"]
    measured=fused["kernel_key"];rt=measured["runtime"];source=measured["hardware"]["source"]
    stack_map={"torch_module_version":"framework_version","torch_git_revision":"torch_git_revision",
               "torch_extension_sha256":"torch_extension_sha256","torch_build_sha256":"torch_build_sha256",
               "accelerator":"accelerator","hip":"accelerator_version","driver":"driver_version",
               "grad_enabled":"grad_enabled","autocast_enabled":"autocast_enabled"}
    stack_ok=all(shared["runtime"].get(a)==rt.get(b) for a,b in stack_map.items())
    env=fused["measurement_contract"]["environment"]
    stack_ok=stack_ok and all(gpu1["environment"].get(k)==env.get(k)
                              for k in ("torch","hip","triton","driver_version"))
    stack_ok=stack_ok and all(gpu1["runtime"].get(k)==rt.get(k) for k in
        ("framework","framework_version","torch_git_revision","torch_extension_sha256","torch_build_sha256",
         "accelerator","accelerator_version","driver_version"))
    reference={"class":measured["hardware"]["class"],"name":measured["hardware"]["name"],
               "architecture":source["architecture"],"total_memory":source["total_memory"]}
    rows={}
    for i,device in enumerate(hardware.devices):
        observed=shared["devices"].get(str(i),{})
        fingerprint={"class":device.extra.get("class"),"name":observed.get("name"),
                     "architecture":observed.get("architecture"),"total_memory":int(device.max_mem)}
        compatible=(stack_ok and fingerprint==reference and device.name==fingerprint["name"])
        if i==1:
            compatible=compatible and all(gpu1["device"].get(k)==fingerprint[k]
                                           for k in ("name","architecture","total_memory"))
        rows[str(i)]={"fingerprint":fingerprint,"hardware_class_key":key_id(fingerprint),
                      "class_and_stack_match":bool(compatible),"measured_on":0,"reuse_on":i,
                      "reuse_kind":"same_device" if i==0 else "homogeneous_hardware_class_transfer",
                      "source":{"kind":"runtime_observation","execution_device":f"cuda:{i}",
                                **{k:fingerprint[k] for k in ("name","architecture","total_memory")}},
                      "not_a_target_device_measurement":i!=0}
    return {"policy":"same complete hardware class and software stack; ordinals/UUID are provenance",
            "measured_on":0,"devices":rows,"software_stack_match":bool(stack_ok),
            "source_sha256":{str((EVIDENCE/"raw"/n).relative_to(ROOT)):sha(EVIDENCE/"raw"/n) for n in snapshots},
            "new_gpu_work":False,"migration_accuracy_calibrated":False,
            "performance_state":"clock/power/temperature/load differences not calibrated by this migration"}
