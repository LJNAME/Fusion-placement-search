"""Observe loaded LightOP/Torch code and prepare exact selected-group identity."""
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path

import torch

from .catalog import catalog, check_installed
from .silu_mul import PreparedLightopCall, declared_binding
from integration.flagscale.kerv_profile_runtime import driver_version


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loaded_implementation(adapter):
    """Check actual imported paths, not just distribution metadata on disk."""
    installed = check_installed()
    expected = catalog()
    distribution = importlib.metadata.distribution("lightop")
    package = importlib.import_module("lightop")
    native = package.op
    paths = {"package": Path(package.__file__).resolve(),
             "native": Path(native.__file__).resolve(),
             "adapter": Path(inspect.getsourcefile(type(adapter))).resolve()}
    if (paths["package"] != Path(distribution.locate_file("lightop/__init__.py")).resolve()
            or paths["native"] != Path(distribution.locate_file(expected["op_so_path"])).resolve()):
        raise ValueError("loaded LightOP module is outside the verified distribution")
    if adapter.operator is not native.silu_and_mul_opt:
        raise ValueError("adapter callable differs from the loaded LightOP operator")
    if Path(PreparedLightopCall.__call__.__code__.co_filename).resolve() != paths["adapter"]:
        raise ValueError("loaded adapter call comes from a different source file")
    hashes = {name: digest(path) for name, path in paths.items()}
    if (hashes["package"] != expected["init_sha256"]
            or hashes["native"] != expected["op_so_sha256"]
            or "sha256:" + hashes["adapter"] != adapter.implementation["version"]
            or adapter.implementation != declared_binding()):
        raise ValueError("loaded source identity differs from the pinned adapter/provider")
    return {"installed": installed, "source_sha256": hashes,
            "paths": {name: str(path) for name, path in paths.items()}}


def observe_runtime(adapter, device):
    """No timed calls; missing driver/architecture stay missing for strict keys."""
    device = torch.device(device)
    if device.type != "cuda" or device.index is None or not torch.version.hip:
        raise RuntimeError("LightOP runtime identity requires an explicit HIP GPU")
    implementation = loaded_implementation(adapter)
    properties = torch.cuda.get_device_properties(device)
    extension = Path(torch._C.__file__).resolve()
    # The adapter executes under no_grad. Observe the same execution context.
    with torch.no_grad():
        flags = {"grad_enabled": torch.is_grad_enabled(),
                 "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    runtime = {"framework": "torch", "framework_version": str(torch.__version__),
               "torch_distribution": importlib.metadata.version("torch"),
               "torch_git_revision": torch.version.git_version,
               "torch_extension_sha256": digest(extension),
               "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
               "accelerator": "HIP", "accelerator_version": torch.version.hip,
               "driver_version": driver_version(), "flags": flags,
               "execution_device": str(device)}
    observed = {"name": properties.name, "total_memory": properties.total_memory,
                "architecture": getattr(properties, "gcnArchName", None),
                "physical_index": device.index,
                "uuid": str(getattr(properties, "uuid", "unavailable"))}
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None  # This native LightOP API is not declared a Triton kernel.
    environment = {"torch": runtime["torch_distribution"], "hip": torch.version.hip,
                   "triton": triton_version, "device_model": observed["name"],
                   "device_architecture": observed["architecture"], "runtime_device": str(device),
                   "driver_version": runtime["driver_version"],
                   "torch_build_sha256": runtime["torch_build_sha256"]}
    return {"implementation": copy.deepcopy(adapter.implementation),
            "loaded_source": implementation, "runtime": runtime, "device": observed,
            "environment": environment,
            "provenance": {"torch_extension_path": str(extension), "device_uuid": observed["uuid"]}}
