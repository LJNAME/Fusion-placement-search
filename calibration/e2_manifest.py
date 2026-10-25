"""Build the E2 profiling manifest from logical Transformer graphs.

The manifest separates a measurement object (one complete kernel key) from
its graph usages.  This is the boundary needed to prevent the same kernel
shape from being placed in both a training and an unseen-shape split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulator.model_graphs import TransformerModelSpec, build_model_graph
from simulator.schema import OpNode


MANIFEST_FORMAT = "fusion-placement-search.e2-shape-manifest"
MANIFEST_VERSION = "1.0"
KERNEL_KEY_VERSION = "1.0"
DEFAULT_SEQUENCE_LENGTHS = (128, 256, 512, 1024)


def canonical_json(value: Any) -> str:
    """Return the only serialization used for E2 content identifiers."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_id(prefix: str, value: Any) -> str:
    payload = value if prefix == "measurement-object" else {"domain": prefix, "value": value}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def measurement_object_id(kernel_key: Mapping[str, Any]) -> str:
    return content_id("measurement-object", kernel_key)


def _shape_tuple(node: OpNode) -> tuple[int, ...]:
    return tuple(int(value) for value in node.extra.get("output_shape", ()))


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _implementation(op_type: str, local_name: str) -> tuple[str, str, bool, str | None]:
    if op_type == "matmul":
        if local_name.endswith("qk_matmul") or local_name.endswith("av_matmul"):
            return "torch.bmm.reference", "e2-v1", True, None
        return "torch.mm.reference", "e2-v1", True, None
    if op_type == "rmsnorm":
        return "torch.rms_norm.reference", "e2-v1", True, None
    if op_type == "rope":
        return "fusion.rope.reference", "e2-v1", True, None
    if op_type == "softmax":
        return "torch.softmax.fp32_internal.reference", "e2-v1", True, None
    if op_type == "silu":
        return "torch.silu.reference", "e2-v1", True, None
    if op_type == "mul":
        return "torch.mul.reference", "e2-v1", True, None
    if op_type == "add":
        return "torch.add.reference", "e2-v1", True, None
    if op_type in {"input", "embedding"}:
        return "logical.no_kernel", "e2-v1", False, "logical node has no timed kernel"
    return "unsupported", "e2-v1", False, f"unsupported op_type: {op_type}"


def _physical_shape(node: OpNode, spec: TransformerModelSpec) -> dict[str, Any]:
    local_name = str(node.extra.get("local_name", node.id))
    output_shape = _shape_tuple(node)
    if node.op_type == "matmul" and local_name.endswith("qk_matmul"):
        return {
            "batch": spec.micro_batch_size * spec.num_attention_heads,
            "m": spec.seq_length,
            "n": spec.seq_length,
            "k": spec.head_dim,
        }
    if node.op_type == "matmul" and local_name.endswith("av_matmul"):
        return {
            "batch": spec.micro_batch_size * spec.num_attention_heads,
            "m": spec.seq_length,
            "n": spec.head_dim,
            "k": spec.seq_length,
        }
    if node.op_type == "matmul":
        return {key: int(value) for key, value in sorted(node.extra["shape"].items())}
    if node.op_type == "rmsnorm":
        return {"input": list(output_shape), "normalized_size": int(output_shape[-1])}
    if node.op_type in {"rope", "softmax", "silu", "mul", "add", "input", "embedding"}:
        return {"input": list(output_shape), "numel": _numel(output_shape)}
    return {"output": list(output_shape)}


def _measurement_recipe(implementation_id: str) -> dict[str, Any]:
    return {
        "input_initialization": {"distribution": "normal", "seed": 1234},
        "warmup": 10,
        "iterations_per_sample": 50,
        "samples": 7,
        "timing": "accelerator_event",
        "synchronize_before": True,
        "synchronize_after": True,
        "implementation_id": implementation_id,
    }


def _estimated_working_set_bytes(kernel_key: Mapping[str, Any]) -> int:
    shape = kernel_key["shape"]
    input_bytes = 4 if kernel_key["dtype"]["input"] == "fp32" else 2
    output_bytes = 4 if kernel_key["dtype"]["output"] == "fp32" else 2
    compute_bytes = 4 if kernel_key["dtype"]["compute"] == "fp32" else input_bytes
    implementation_id = kernel_key["implementation"]["id"]
    if implementation_id == "torch.mm.reference":
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        return m * k * input_bytes + k * n * input_bytes + m * n * output_bytes
    if implementation_id == "torch.bmm.reference":
        batch = int(shape["batch"])
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        return batch * (
            m * k * input_bytes + k * n * input_bytes + m * n * output_bytes
        )
    numel = int(shape.get("numel", _numel(shape.get("input", ()))))
    if kernel_key["operator"] == "softmax":
        return numel * (input_bytes + compute_bytes + output_bytes)
    input_count = 2 if kernel_key["operator"] in {"mul", "add"} else 1
    return numel * (input_count * input_bytes + output_bytes)


def kernel_key_for_node(
    node: OpNode,
    spec: TransformerModelSpec,
    *,
    hardware_class: str,
    framework_version: str,
    accelerator_version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a complete key and measurement metadata for one logical node."""

    local_name = str(node.extra.get("local_name", node.id))
    impl_id, impl_version, enabled, skip_reason = _implementation(node.op_type, local_name)
    compute_dtype = str(node.extra.get("compute_dtype", node.extra.get("dtype", spec.dtype)))
    output_dtype = str(node.extra.get("dtype", spec.dtype))

    layout: dict[str, Any] = {"contiguity": "contiguous", "order": "row_major"}
    runtime_flags: dict[str, Any] = {"phase": "forward"}
    if impl_id == "torch.bmm.reference":
        layout["rank"] = 3
        runtime_flags["gqa_repeat_factor"] = int(node.extra.get("gqa_repeat_factor", 1))
    elif node.op_type == "matmul":
        layout["rank"] = 2
    elif node.op_type == "rope":
        layout["logical_axes"] = ["batch", "sequence", "heads", "head_dim"]
        runtime_flags.update({"rotary_base": 1_000_000, "pairing": "even_odd"})
    elif node.op_type == "softmax":
        runtime_flags["dim"] = -1

    key = {
        "key_version": KERNEL_KEY_VERSION,
        "hardware": {"class": hardware_class, "scope": "single_device"},
        "backend": {
            "framework": "pytorch",
            "framework_version": framework_version,
            "accelerator": "hip",
            "accelerator_version": accelerator_version,
        },
        "implementation": {"id": impl_id, "version": impl_version},
        "operator": node.op_type,
        "dtype": {
            "input": spec.dtype,
            "compute": compute_dtype,
            "output": output_dtype,
        },
        "layout": layout,
        "shape": _physical_shape(node, spec),
        "compile_flags": {},
        "runtime_flags": runtime_flags,
    }
    measurement = {
        "enabled": enabled,
        "skip_reason": skip_reason,
        "estimated_output_bytes": int(node.out_bytes),
        "estimated_parameter_bytes": int(node.param_bytes),
        "estimated_memory_bytes": _estimated_working_set_bytes(key),
        "recipe": _measurement_recipe(impl_id),
    }
    return key, measurement


def default_model_specs(
    sequence_lengths: Iterable[int] = DEFAULT_SEQUENCE_LENGTHS,
) -> tuple[TransformerModelSpec, ...]:
    result: list[TransformerModelSpec] = []
    bases = (
        TransformerModelSpec(
            model_type="qwen3",
            num_layers=28,
            hidden_size=1024,
            ffn_hidden_size=3072,
            num_attention_heads=16,
            num_query_groups=8,
            head_dim=128,
            seq_length=256,
            micro_batch_size=1,
            vocab_size=151936,
            dtype="bf16",
            qk_layernorm=True,
            attention_softmax_in_fp32=True,
        ),
        TransformerModelSpec(
            model_type="qwen2.5-3b",
            num_layers=36,
            hidden_size=2048,
            ffn_hidden_size=11008,
            num_attention_heads=16,
            num_query_groups=2,
            head_dim=128,
            seq_length=256,
            micro_batch_size=1,
            vocab_size=151936,
            dtype="bf16",
            qk_layernorm=False,
            attention_softmax_in_fp32=True,
        ),
    )
    for base in bases:
        for sequence_length in sequence_lengths:
            result.append(replace(base, seq_length=int(sequence_length)))
    return tuple(result)


def _model_family(spec: TransformerModelSpec) -> str:
    return "qwen3-0.6b" if spec.model_type == "qwen3" else spec.model_type


def _model_id(spec: TransformerModelSpec) -> str:
    return f"{_model_family(spec)}-s{spec.seq_length}-b{spec.micro_batch_size}"


def build_shape_manifest(
    specs: Iterable[TransformerModelSpec] | None = None,
    *,
    hardware_class: str = "BW1000",
    framework_version: str = "2.9.0",
    accelerator_version: str = "6.3.26093",
) -> dict[str, Any]:
    """Build a deterministic, deduplicated E2 shape manifest."""

    selected_specs = tuple(specs or default_model_specs())
    objects: dict[str, dict[str, Any]] = {}
    usage_counts: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    models: list[dict[str, Any]] = []

    for spec in selected_specs:
        model_id = _model_id(spec)
        models.append(
            {
                "model_id": model_id,
                "model_family": _model_family(spec),
                "spec": asdict(spec),
            }
        )
        graph = build_model_graph(spec)
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            kernel_key, measurement = kernel_key_for_node(
                node,
                spec,
                hardware_class=hardware_class,
                framework_version=framework_version,
                accelerator_version=accelerator_version,
            )
            object_id = measurement_object_id(kernel_key)
            existing = objects.get(object_id)
            if existing is not None and existing["kernel_key"] != kernel_key:
                raise AssertionError(f"content identifier collision: {object_id}")
            if existing is None:
                objects[object_id] = {
                    "object_id": object_id,
                    "kernel_key": kernel_key,
                    "measurement": measurement,
                    "usages": [],
                }

            local_name = str(node.extra.get("local_name", node_id))
            usage_key = (model_id, local_name)
            usage = usage_counts[object_id].get(usage_key)
            if usage is None:
                usage = {
                    "model_id": model_id,
                    "model_family": _model_family(spec),
                    "sequence_length": spec.seq_length,
                    "micro_batch_size": spec.micro_batch_size,
                    "local_name": local_name,
                    "occurrence_count": 0,
                    "example_node_id": node_id,
                }
                usage_counts[object_id][usage_key] = usage
            usage["occurrence_count"] += 1

    for object_id, item in objects.items():
        item["usages"] = sorted(
            usage_counts[object_id].values(),
            key=lambda usage: (usage["model_id"], usage["local_name"]),
        )

    ordered_objects = [objects[key] for key in sorted(objects)]
    ordered_models = sorted(models, key=lambda model: model["model_id"])
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "identity_rule": "sha256(canonical_json(kernel_key))",
        "environment_contract": {
            "hardware_class": hardware_class,
            "framework": "pytorch",
            "framework_version": framework_version,
            "accelerator": "hip",
            "accelerator_version": accelerator_version,
        },
        "models": ordered_models,
        "objects": ordered_objects,
    }
    manifest["manifest_id"] = content_id("manifest", manifest)
    return manifest


def validate_shape_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("unexpected E2 manifest format")
    objects = list(manifest.get("objects", ()))
    object_ids = [item["object_id"] for item in objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("duplicate measurement object IDs")
    if object_ids != sorted(object_ids):
        raise ValueError("measurement objects must be sorted by object ID")
    unhashed = dict(manifest)
    claimed_manifest_id = unhashed.pop("manifest_id", None)
    if claimed_manifest_id != content_id("manifest", unhashed):
        raise ValueError("manifest content hash mismatch")
    for item in objects:
        expected = measurement_object_id(item["kernel_key"])
        if item["object_id"] != expected:
            raise ValueError(f"kernel key hash mismatch: {item['object_id']}")
        if not item.get("usages"):
            raise ValueError(f"measurement object has no graph usages: {item['object_id']}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", default="128,256,512,1024")
    parser.add_argument("--hardware-class", default="BW1000")
    parser.add_argument("--framework-version", default="2.9.0")
    parser.add_argument("--accelerator-version", default="6.3.26093")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sequences = tuple(int(value) for value in args.sequences.split(",") if value.strip())
    manifest = build_shape_manifest(
        default_model_specs(sequences),
        hardware_class=args.hardware_class,
        framework_version=args.framework_version,
        accelerator_version=args.accelerator_version,
    )
    validate_shape_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    enabled = sum(bool(item["measurement"]["enabled"]) for item in manifest["objects"])
    print(f"models: {len(manifest['models'])}")
    print(f"measurement_objects: {len(manifest['objects'])}")
    print(f"enabled: {enabled}")
    print(f"disabled: {len(manifest['objects']) - enabled}")
    print(f"manifest_id: {manifest['manifest_id']}")
    print(f"output: {args.output}")
    print("Fusion E2a shape manifest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
