"""CPU metadata/label fixtures only; no GPU or admission evidence."""
import copy
import json
from pathlib import Path

import pytest

from cost_model.fusion_kernel_profile import KernelLatency, key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract, selected_group_profile
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from cost_model.search_evaluation import build_evaluation_graph
from integration.lightop.catalog import catalog
from integration.lightop.silu_mul import RECIPE, SHAPE, adapter_registry, declared_binding
from simulator import enumerate_fusion_candidates, load_hardware, make_reference_plan
from simulator.placement_artifacts import PlacedOpGraph

ROOT = Path(__file__).resolve().parents[2]


def fixture_case(device=0):
    graph, registry = build_evaluation_graph("small"), adapter_registry()
    candidate = next(c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                     if c.pattern_id == "silu_mul")
    plan = make_reference_plan(graph, registry=registry)
    plan.groups = [g for g in plan.groups if g.node_ids[0] not in candidate.node_ids]
    group = candidate.to_group(device)
    group.extra["implementation"] = declared_binding()
    plan.groups.insert(-1, group)
    index = plan.groups.index(group)
    hardware = load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    placed = PlacedOpGraph(graph, hardware, plan)
    target = f"cuda:{device}"

    def tensor(shape):
        stride, strides = 1, []
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        return {"shape": list(shape), "dtype": "bf16", "layout": "contiguous", "strides": strides,
                "layout_provenance": {"kind": "runtime_tensor", "device": target, "observer": "CPU test fixture"}}

    observation = {"implementation": declared_binding(), "recipe": copy.deepcopy(RECIPE),
                   "tensors": {name: tensor(SHAPE) for name in ("gate", "up", "output")},
                   "storage_overlap": {"gate_up": False, "output_gate": False, "output_up": False}}
    observation["tensors"]["packed"] = tensor((256, 6144))
    runtime = {"framework": "torch", "framework_version": "fixture-torch", "torch_distribution": "fixture-torch",
               "torch_git_revision": "fixture-git", "torch_extension_sha256": "fixture-extension",
               "torch_build_sha256": "fixture-build", "accelerator": "HIP", "accelerator_version": "fixture-HIP",
               "driver_version": 123, "execution_device": target,
               "flags": {"grad_enabled": False, "cuda_autocast_enabled": False, "deterministic_algorithms": False}}
    device_info = {"name": "fixture-BW1000", "architecture": "fixture-arch", "physical_index": device,
                   "total_memory": 32 * 1024**3, "uuid": "fixture-UUID-A"}
    env = {"torch": "fixture-torch", "hip": "fixture-HIP", "triton": None, "device_model": device_info["name"],
           "device_architecture": device_info["architecture"], "runtime_device": target,
           "driver_version": 123, "torch_build_sha256": "fixture-build"}
    actual = {"implementation": declared_binding(), "runtime": runtime, "device": device_info, "environment": env,
              "loaded_source": {"installed": {"version": declared_binding()["provider_version"]},
                                "source_sha256": {"package": catalog()["init_sha256"],
                                                  "native": declared_binding()["op_so_sha256"],
                                                  "adapter": declared_binding()["version"].removeprefix("sha256:")},
                                "paths": {"native": "/fixture/package/op.so"}},
              "provenance": {"device_uuid": "fixture-UUID-A"}}
    return placed, index, observation, actual


def make_keys(case):
    return LightopKernelKeys(*case)


def identifier(keys, case):
    key, gaps = keys.describe(case[0].plan.groups[case[1]])
    return key_id(key), gaps


def label(keys, case, **changes):
    contract = measurement_contract(keys.environment, purpose="calibration")
    data = dict(duration_s=10e-6, kind="measured", provenance="CPU fixture only; not real acceptance",
                measurement_contract_id=measurement_contract_id(contract), admission="accepted")
    data.update(changes)
    return contract, {identifier(keys, case)[0]: KernelLatency(**data)}


def test_key_roundtrip_order_and_graph_immutability():
    case = fixture_case()
    original = copy.deepcopy(case[0].graph)
    keys = make_keys(case)
    key, gaps = keys.describe(case[0].plan.groups[case[1]])
    assert not gaps and key_id(key) == key_id(json.loads(json.dumps(key)))
    case[0].graph.edges.reverse()
    assert identifier(make_keys(case), case)[0] == key_id(key)
    assert all(case[0].graph.nodes[n].extra == original.nodes[n].extra for n in original.nodes)


def test_smoke_full_reference_graph_accepts_observed_metadata_without_guessing_other_layouts():
    from calibration.lightop_profile_smoke import placed_case
    placed, index = placed_case(0)
    _, _, observation, runtime = fixture_case()
    keys = LightopKernelKeys(placed, index, observation, runtime)
    key, gaps = keys.describe(placed.plan.groups[index])
    assert not gaps and key["pattern_id"] == "silu_mul"
    selected = set(placed.plan.groups[index].node_ids)
    assert all(keys.graph.nodes[n].extra == placed.graph.nodes[n].extra
               for n in placed.graph.nodes if n not in selected and
               n not in group_inputs_for_test(placed, index))


def group_inputs_for_test(placed, index):
    from integration.lightop.silu_mul import group_inputs
    return group_inputs(placed.graph, placed.plan.groups[index].node_ids)


def test_uuid_and_install_path_are_provenance_only_but_device_is_not():
    case = fixture_case()
    before = identifier(make_keys(case), case)[0]
    case[3]["device"]["uuid"] = case[3]["provenance"]["device_uuid"] = "fixture-UUID-after-reboot"
    case[3]["loaded_source"]["paths"]["native"] = "/other/fixture/op.so"
    assert identifier(make_keys(case), case)[0] == before
    other = fixture_case(1)
    assert identifier(make_keys(other), other)[0] != before


@pytest.mark.parametrize("change", ["runtime_build", "runtime_flags", "alias", "stride"])
def test_implementation_conditions_do_not_reuse_the_key(change):
    case = fixture_case()
    before = identifier(make_keys(case), case)[0]
    if change == "runtime_build":
        case[3]["runtime"]["torch_extension_sha256"] = "other-extension"
    elif change == "runtime_flags":
        case[3]["runtime"]["flags"]["deterministic_algorithms"] = True
    elif change == "alias":
        case[2]["storage_overlap"]["gate_up"] = True
    else:
        # A unit dimension can have a different stride and still be contiguous.
        case[2]["tensors"]["gate"]["strides"][0] += 1
    assert identifier(make_keys(case), case)[0] != before


@pytest.mark.parametrize("change", ["native_hash", "package_hash", "adapter_version", "shape", "dtype",
                                    "layout", "cpu_provenance", "stride", "alias", "recipe", "environment"])
def test_inconsistent_or_unobserved_identity_is_rejected(change):
    case = fixture_case()
    if change in ("native_hash", "package_hash"):
        case[3]["loaded_source"]["source_sha256"][change.removesuffix("_hash")] = "other"
    elif change == "adapter_version":
        case[3]["implementation"]["version"] = "other"
    elif change in ("shape", "dtype", "layout", "stride", "cpu_provenance"):
        tensor = case[2]["tensors"]["gate"]
        if change == "cpu_provenance":
            tensor["layout_provenance"]["device"] = "cpu"
        else:
            field = "strides" if change == "stride" else change
            tensor[field] = {"shape": [1, 128, 3072], "dtype": "fp32", "layout": "unknown", "strides": [1, 1, 1]}[field]
    elif change == "alias":
        case[2]["storage_overlap"]["output_gate"] = True
    elif change == "recipe":
        case[2]["recipe"]["timed_body"] = ["kernel only"]
    else:
        case[3]["environment"]["runtime_device"] = "cuda:1"
    with pytest.raises(MeasurementContractError):
        make_keys(case)


def test_accepted_fixture_is_charged_once_with_no_additional_pack_or_launch():
    case = fixture_case()
    keys = make_keys(case)
    contract, labels = label(keys, case)
    profile = selected_group_profile(keys, contract, labels)
    cost = profile.estimate_group(keys.graph, case[0].plan.groups[case[1]])
    assert cost.duration == cost.compute_time == 10e-6
    assert cost.launch_time == cost.memory_time == 0 and cost.source == "measured_group_kernel"
    with pytest.raises(MeasurementContractError, match="only the observed"):
        profile.estimate_group(keys.graph, case[0].plan.groups[0])


@pytest.mark.parametrize("status", ["conditional_dev", "legacy_unbound"])
def test_unaccepted_fixture_never_falls_back(status):
    case = fixture_case()
    keys = make_keys(case)
    contract, labels = label(keys, case, admission=status)
    profile = selected_group_profile(keys, contract, labels)
    with pytest.raises(MeasurementContractError):
        profile.estimate_group(keys.graph, case[0].plan.groups[case[1]])


@pytest.mark.parametrize("change", ["smoke", "wrong_contract", "missing", "synthetic", "missing_driver", "missing_arch"])
def test_bad_cost_source_cannot_price_selected_group(change):
    case = fixture_case()
    if change == "missing_driver":
        case[3]["runtime"]["driver_version"] = case[3]["environment"]["driver_version"] = None
    if change == "missing_arch":
        case[3]["device"]["architecture"] = case[3]["environment"]["device_architecture"] = None
    keys = make_keys(case)
    options = {"measurement_contract_id": "sha256:other"} if change == "wrong_contract" else {}
    if change == "synthetic":
        options["kind"] = "synthetic"
    contract, labels = label(keys, case, **options)
    if change == "smoke":
        contract = measurement_contract(keys.environment)
    if change == "missing":
        labels = {}
    with pytest.raises((MeasurementContractError, KeyError, ValueError)):
        profile = selected_group_profile(keys, contract, labels)
        profile.estimate_group(keys.graph, case[0].plan.groups[case[1]])
