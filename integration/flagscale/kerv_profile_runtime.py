"""Observe the loaded KERV implementation and GPU environment for profiling."""

import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from calibration.kerv_silu_mul import CORE


def loaded_source(module, flagscale_root):
    path = Path(module.__file__).resolve()
    if path != (Path(flagscale_root) / CORE).resolve():
        raise ValueError("loaded KERV module is outside the inspected checkout")
    revision = subprocess.run(["git", "-C", str(flagscale_root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    code = module._silu_mul_kernel.fn.__code__
    if Path(code.co_filename).resolve() != path:
        raise ValueError("loaded JIT function is from a different source file")
    return {"module_path": str(path), "jit_code_file": code.co_filename,
            "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "git_revision": revision}


def driver_version():
    """Read the HIP driver directly; unavailable information stays unknown."""
    if not torch.version.hip:
        return None
    try:
        library = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        hip = ctypes.CDLL(library)
        value = ctypes.c_int()
        if hip.hipDriverGetVersion(ctypes.byref(value)) == 0:
            return str(value.value)
    except (OSError, AttributeError):
        pass
    return None


def observe_environment(module, device):
    props = torch.cuda.get_device_properties(device)
    architecture = getattr(props, "gcnArchName", None)
    if not architecture and not torch.version.hip:
        architecture = "sm_" + "".join(map(str, torch.cuda.get_device_capability(device)))
    observed = {"name": props.name, "total_memory": props.total_memory,
                "uuid": str(getattr(props, "uuid", "unavailable")), "architecture": architecture}
    environment = {"torch": importlib.metadata.version("torch"), "torch_module_version": str(torch.__version__),
                   "hip": torch.version.hip, "cuda": torch.version.cuda,
                   "triton": importlib.metadata.version("triton"),
                   "triton_module_version": str(module.triton.__version__),
                   "driver_version": driver_version(), "device_model": props.name,
                   "device_uuid": observed["uuid"], "device_architecture": architecture}
    return observed, environment


def native_binding():
    path = Path(inspect.getsourcefile(F.silu)).resolve()
    return {"id": "torch.native.silu_then_mul", "version": importlib.metadata.version("torch"),
            "public_operator": ["aten::silu", "aten::mul"], "backend": "torch_native",
            "functional_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime_flags": {"inplace": False, "output_alias_input": None}, "compile_flags": {}}
