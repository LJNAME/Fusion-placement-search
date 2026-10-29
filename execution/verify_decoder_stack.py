"""A1 numerical checks against a direct, independent continuous-layer formula.

    CPU is the default. Explicit --devices 0,1 runs correctness on GPUs with no
    fallback and no performance claim. This follows the existing logical graph's
    unscaled, unmasked attention, not a pretrained Qwen implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from simulator import DeviceSpec, HardwareSpec, PlacedOpGraph, make_reference_plan
from simulator.model_graphs.spec import TransformerModelSpec
from simulator.model_graphs.decoder_stack import build_decoder_stack
from execution.audit_gpu_holdout import write_json
from execution.decoder_reference import ContinuousDecoderExecutor
from execution.gpu_holdout import CanonicalReferenceExecutor, compare_outputs


def small_spec(layers=2, dtype="fp32", qk_layernorm=True):
    return TransformerModelSpec(model_type="qwen3" if qk_layernorm else "qwen2.5",
        num_layers=layers, hidden_size=16, ffn_hidden_size=32, num_attention_heads=4,
        num_query_groups=2, head_dim=4, seq_length=5, micro_batch_size=2, vocab_size=32,
        dtype=dtype, qk_layernorm=qk_layernorm)


def canonical_tensor(shape, dtype, seed, salt):
    generator = torch.Generator(device="cpu").manual_seed(seed + salt)
    return torch.randn(shape, dtype=dtype, generator=generator) * .02


def independent_decoder_forward(spec, value, seed=1234, *, first_layer_delta=None):
    """Direct layers, no OpGraph, plan traversal or executor operator helpers."""
    value = value.cpu()

    def norm(x):
        return F.rms_norm(x, (x.shape[-1],), torch.ones(x.shape[-1], dtype=x.dtype), eps=1e-6)

    def rope(x):
        frequencies = 1000000. ** (-torch.arange(0, spec.head_dim, 2).float() / spec.head_dim)
        angles = torch.arange(spec.seq_length).float()[:, None] * frequencies
        cosine, sine = angles.cos().to(x.dtype)[None, :, None, :], angles.sin().to(x.dtype)[None, :, None, :]
        rotated = torch.empty_like(x)
        rotated[..., ::2] = x[..., ::2] * cosine - x[..., 1::2] * sine
        rotated[..., 1::2] = x[..., 1::2] * cosine + x[..., ::2] * sine
        return rotated

    for layer in range(spec.num_layers):
        def linear(x, local_name, width):
            node_id = f"L{layer:03d}.{local_name}"
            weight = canonical_tensor((x.shape[-1], width), x.dtype, seed,
                                      1000 + sum(map(ord, node_id)))
            return F.linear(x, weight.t())

        normalized = norm(value)
        q = linear(normalized, "self_attention.q_proj", spec.q_projection_size).reshape(
            spec.micro_batch_size, spec.seq_length, spec.num_attention_heads, spec.head_dim)
        k = linear(normalized, "self_attention.k_proj", spec.kv_projection_size).reshape(
            spec.micro_batch_size, spec.seq_length, spec.num_query_groups, spec.head_dim)
        v = linear(normalized, "self_attention.v_proj", spec.kv_projection_size).reshape_as(k)
        if spec.qk_layernorm:
            q, k = norm(q), norm(k)
        if spec.use_rotary_position_embeddings:
            q, k = rope(q), rope(k)
        repeat = spec.num_attention_heads // spec.num_query_groups
        scores = q.transpose(1, 2) @ k.repeat_interleave(repeat, 2).permute(0, 2, 3, 1)
        probabilities = F.softmax(scores.float() if spec.attention_softmax_in_fp32 else scores, -1).to(value.dtype)
        context = (probabilities @ v.repeat_interleave(repeat, 2).transpose(1, 2)).transpose(1, 2)
        residual = value + linear(context.reshape(spec.micro_batch_size, spec.seq_length,
                                                  spec.q_projection_size), "self_attention.out_proj", spec.hidden_size)
        normalized = norm(residual)
        gated = F.silu(linear(normalized, "mlp.gate_proj", spec.ffn_hidden_size))
        gated = gated * linear(normalized, "mlp.up_proj", spec.ffn_hidden_size)
        value = residual + linear(gated, "mlp.down_proj", spec.hidden_size)
        if layer == 0 and first_layer_delta is not None:
            value = value + first_layer_delta
    return value


def placed_stack(spec, placement="uniform0"):
    graph = build_decoder_stack(spec)
    plan = make_reference_plan(graph)
    if placement not in {"uniform0", "uniform1", "alternate_layers", "boundary_inputs"}:
        raise ValueError("unknown placement")
    for group in plan.groups:
        node = graph.nodes[group.node_ids[0]]
        layer = node.extra["layer"]
        if placement.startswith("uniform"):
            group.device = int(placement[-1])
        elif placement == "alternate_layers":
            group.device = layer % 2
        else:
            group.device = (layer + int(node.op_type == "input")) % 2
    hardware = HardwareSpec([DeviceSpec(1e12, 1e11, 1e9, name=f"logical{i}") for i in range(2)],
                            name="A1-numerical-fixture")
    return PlacedOpGraph(graph=graph, hardware=hardware, plan=plan)


def verify(devices=("cpu", "cpu")):
    results = []
    with torch.no_grad():
        for layers in (2, 4):
            for dtype in ("fp32", "bf16"):
                for qk_norm in (False, True):
                    spec = small_spec(layers, dtype, qk_norm)
                    tensor_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
                    initial = canonical_tensor((2, 5, 16), tensor_dtype, 1234, 11)
                    reference = independent_decoder_forward(spec, initial)
                    rtol, atol = (1e-5, 1e-7) if dtype == "fp32" else (.01, .001)
                    for placement in ("uniform0", "uniform1", "alternate_layers", "boundary_inputs"):
                        placed = placed_stack(spec, placement)
                        executor = ContinuousDecoderExecutor(placed, devices)
                        actual, *_ = executor.run_once(False)
                        accuracy = compare_outputs(actual, reference, rtol=rtol, atol=atol)
                        perturbed = initial + .1
                        changed, *_ = ContinuousDecoderExecutor(placed, devices, input_tensor=perturbed).run_once(False)
                        expected_changed = independent_decoder_forward(spec, perturbed)
                        perturb_accuracy = compare_outputs(changed, expected_changed, rtol=rtol, atol=atol)
                        propagated = not torch.allclose(actual.cpu(), changed.cpu(), rtol=rtol, atol=atol)
                        legacy, *_ = CanonicalReferenceExecutor(placed, devices, 1234, {}).run_once(False)
                        legacy_rejected = not compare_outputs(legacy, reference, rtol=rtol, atol=atol)["pass"]
                        row = {"layers": layers, "dtype": dtype, "qk_layernorm": qk_norm,
                            "placement": placement, "nodes": len(placed.graph.nodes),
                            "edges": len(placed.graph.edges), "accuracy": accuracy,
                            "perturbed_accuracy": perturb_accuracy, "input_perturbation_propagates": propagated,
                            "legacy_layer_reset_rejected": legacy_rejected,
                            "root_input_cache_entries": len(executor._inputs)}
                        if not (accuracy["pass"] and perturb_accuracy["pass"] and propagated and
                                legacy_rejected and len(executor._inputs) == 1):
                            raise AssertionError(json.dumps(row))
                        results.append(row)
    root = Path(__file__).resolve().parents[1]
    sources = ("execution/verify_decoder_stack.py", "execution/decoder_reference.py",
               "simulator/model_graphs/decoder_stack.py", "execution/reference.py", "execution/gpu_holdout.py",
               "simulator/model_graphs/dense_layer.py", "simulator/model_graphs/spec.py")
    return {"format": "fusion-placement-search.a1-decoder-correctness", "cases": results,
        "devices": list(devices), "all_pass": True,
        "fixture_dimensions": {"batch": 2, "sequence": 5, "hidden": 16, "ffn": 32,
                               "attention_heads": 4, "query_groups": 2, "head_dim": 4},
        "environment": {"torch": torch.__version__, "hip": torch.version.hip},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "scope": "Synthetic decoder-body correctness only; no embedding/head, causal mask, attention scaling, training or performance claim.",
        "gpu_copy_validated": len(set(devices)) == 2 and all(str(d).startswith("cuda:") for d in devices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cpu", help="cpu or two physical GPU indices, e.g. 0,1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    devices = ("cpu", "cpu")
    if args.devices != "cpu":
        indices = tuple(int(value) for value in args.devices.split(","))
        if len(indices) != 2 or len(set(indices)) != 2 or min(indices) < 0:
            raise ValueError("two distinct non-negative GPU indices are required")
        if not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count():
            raise RuntimeError("requested GPU runtime unavailable; no CPU fallback")
        devices = tuple(f"cuda:{index}" for index in indices)
    report = verify(devices)
    if args.output:
        write_json(args.output, report)
    print(f"A1 cases={len(report['cases'])} all_pass={report['all_pass']} gpu_copy_validated={report['gpu_copy_validated']}")


if __name__ == "__main__":
    main()
