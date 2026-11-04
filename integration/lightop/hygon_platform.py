"""Select vLLM's existing ROCm platform for the HIP/HyHAL benchmark environment."""

ENTRY_POINT = "fusion_hygon_rocm"
PLATFORM = "vllm.platforms.rocm.RocmPlatform"


def register():
    # This is a vllm.platform_plugins callback, not a new backend implementation.
    import torch
    if not torch.version.hip:
        return None
    import amdsmi
    amdsmi.amdsmi_init()
    try:
        if not amdsmi.amdsmi_get_processor_handles():
            return None
    finally:
        amdsmi.amdsmi_shut_down()
    return PLATFORM
