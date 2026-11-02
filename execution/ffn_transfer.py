"""Declared preallocated FFN transfer carrier; GPU binding is deferred to B4.

This is one copy operation, not a graph collector or whole-FFN executor.
"""
from __future__ import annotations

import math

import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
ELEMENT_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
IMPLEMENTATION = "torch.Tensor.copy_.preallocated_cross_gpu.v1"


def declared_carrier(shape, dtype):
    """An explicit construction contract, never an observation of a producer."""
    if dtype not in DTYPES or not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid transfer shape/dtype")
    strides, stride = [], 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return {"shape": list(shape), "dtype": dtype, "layout": "contiguous", "strides": strides,
            "tensor_bytes": math.prod(shape) * ELEMENT_BYTES[dtype],
            "layout_provenance": {"kind": "constructor_declaration", "constructor": "torch.empty_strided",
                                  "source": "execution/ffn_transfer.py:allocate_carrier"},
            "producer_layout_observed": False}


def validate_carrier(value, spec):
    if spec != declared_carrier(spec["shape"], spec["dtype"]):
        raise ValueError("carrier declaration is inconsistent")
    if type(value) is not torch.Tensor:
        raise ValueError("transfer requires an ordinary Torch Tensor")
    if (list(value.shape) != spec["shape"] or value.dtype != DTYPES[spec["dtype"]]
            or list(value.stride()) != spec["strides"] or not value.is_contiguous()):
        raise ValueError("transfer Tensor differs from the declared shape/dtype/strides; no implicit packing")


def allocate_carrier(spec, device):
    """Allocate outside capture/timing; CPU is for metadata/numeric tests only."""
    expected = declared_carrier(spec["shape"], spec["dtype"])
    if spec != expected:
        raise ValueError("carrier declaration is inconsistent")
    return torch.empty_strided(spec["shape"], spec["strides"], dtype=DTYPES[spec["dtype"]], device=device)


class BoundTransfer:
    """Bind explicit GPU endpoints and one preallocated destination stream.

    Caller establishes producer completion outside an isolated timed replay,
    retains the source/destination, and enqueues consumers on this same stream.
    B4 must separately prove Graph capture, route, trace and event completion.
    """
    def __init__(self, source, destination, spec, source_gpu, destination_gpu, stream):
        if (type(source_gpu) is not int or type(destination_gpu) is not int
                or source_gpu < 0 or destination_gpu < 0 or source_gpu == destination_gpu):
            raise ValueError("transfer needs distinct explicit GPU ordinals")
        validate_carrier(source, spec)
        validate_carrier(destination, spec)
        expected_source, expected_destination = torch.device("cuda", source_gpu), torch.device("cuda", destination_gpu)
        if source.device != expected_source or destination.device != expected_destination:
            raise ValueError("transfer Tensor device mismatch; CPU/eager relocation fallback is forbidden")
        if stream is None or torch.device(stream.device) != expected_destination:
            raise ValueError("transfer stream must belong to the destination GPU")
        if not torch.cuda.can_device_access_peer(source_gpu, destination_gpu):
            raise ValueError("peer capability missing; no host-staging fallback is authorized")
        self.source, self.destination, self.stream = source, destination, stream
        self.spec = spec
        self.destination_device = expected_destination

    def __call__(self):
        # No allocations, implicit contiguous conversions, stream creation, or
        # host synchronization inside the captured body. Route proof is B4 work.
        with torch.cuda.device(self.destination_device), torch.cuda.stream(self.stream):
            self.destination.copy_(self.source, non_blocking=True)
        return self.destination
