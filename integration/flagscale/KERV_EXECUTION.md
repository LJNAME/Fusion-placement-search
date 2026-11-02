# PlacedOpGraph to KERV group execution

The first adapter handles one Qwen3-0.6B `silu_mul` group. The reference
model has 28 layers, hidden size 1024 and FFN intermediate width 3072.
Boundary values are supplied by the caller. The validation command generates
synthetic boundary values with the selected reference node's shape; it does
not run the projections, the other groups, or the complete model.

## Capability and runtime

`load_kerv_registry(capability_path, flagscale_root)` verifies the existing
source-pinned capability before exposing an opt-in `implemented` pattern.
The global default registry and historical development registry are unchanged.
The v1 artifact field `backend=template` remains for compatibility. The actual
operator binding is the capability's
`torch.ops.flagos_embodied.kerv_silu_mul`, including its implementation hash.

The implemented registry checks the existing shape/dtype/forward/liveness
guards, explicit same-layer metadata, and that the internal SiLU is not an
exported graph output. Unknown contiguity remains a runtime requirement.
`implemented` means the adapter has code to call this implementation; inspect
the execution evidence separately before claiming GPU validation.

`KervSiluMulAdapter(...).bind(placed, group_index, device_map)` binds only the
selected group. `device_map` explicitly maps its logical device to `cuda:N`
(also the Torch device API on HIP). The caller supplies gate/up on that device.
The adapter resolves gate/up by graph edges, not input dictionary order.
It verifies actual shapes, dtypes, strides, contiguity, placement and forward-only
inputs. It rejects storage aliases between gate and any other live tensor.
There is no implicit device transfer, contiguous conversion, or input clone.

Calling the bound group mutates gate through the public KERV operator, removes
the gate node from the live tensor map and publishes only the Mul output.
The SiLU intermediate is not materialized. The caller relinquishes gate's old
value and must not retain unreported aliases. A failed device call aborts that
invocation; do not reuse possibly mutated inputs after a runtime failure.
KERV configuration is process-global; use the adapter sequentially.

```python
adapter = KervSiluMulAdapter(capability_path, flagscale_root)
placed = load_placed_op_graph(artifact_path, registry=adapter.registry)
bound = adapter.bind(placed, group_index, {0: "cuda:0"})
live = {bound.gate_node: gate_tensor, bound.up_node: up_tensor}
call = bound(live, observe_dispatch=True)
output = live[call.output_node]  # aliases gate_tensor
```

The pinned wrapper calls the registered Torch operator. Explicit Triton mode,
enabled operator, GPU tensors and width <=8192 are required. `observe_dispatch`
checks exactly one JIT launch through the public call. Validation additionally
requires a synchronized GPU trace containing exactly one `silu_mul_kernel`;
an operator hit log alone does not pass.

## GPU validation

Run from the Fusion repository at `c7a4885` (or a compatible newer revision)
after applying the supplied source overlay.
The FlagScale checkout must still be the inspected `f8e9a62` revision with the
capability's source hashes. Set `FLAGSCALE_ROOT` to that checkout and select the
Python environment containing the corresponding Torch/HIP/Triton runtime.

```bash
export FLAGSCALE_ROOT=/absolute/path/to/FlagScale
export FUSION_PYTHON=/absolute/path/to/python
bash tools/run_kerv_adapter_validation.sh /absolute/path/to/fresh-result-directory
```

Optional `FUSION_KERV_DEVICE=cuda:1` selects the runtime GPU; the artifact's
logical device remains 0 and the mapping is recorded explicitly. This run
checks the adapter, not automatic matching of the hardware snapshot to a GPU.

The equivalent commands are:

```bash
python -m calibration.validate_kerv_adapter prepare \
  --flagscale-root "$FLAGSCALE_ROOT" --out-dir /fresh/output/prepare
python -m calibration.validate_kerv_adapter run \
  --flagscale-root "$FLAGSCALE_ROOT" \
  --placed /fresh/output/prepare/placed.json \
  --device cuda:0 --rounds 7 --calls 50 --warmup 10 \
  --out-dir /fresh/output/gpu
```

The collector checks gate aliasing, unchanged up, and numerical agreement with
both native SiLU-then-Mul and the FP32 fused expression at the pinned tolerance.
It saves the JIT dispatch evidence and `dispatch_trace.json`, then captures a
batch with 50 independent gate buffers. Gate restore, compilation, allocation,
numerical checks and profiling are outside timed replay. Each of 7 timed rounds
checks all 50 outputs; Event time includes one graph replay submission boundary.
The metric is **graph-batch-amortized seconds per call**, not eager operator
latency or isolated device instruction time. No speedup comparison is made.

Return the complete result directory: `prepare/placed.json`, `prepare/prepare.json`,
`gpu/dispatch_trace.json`, `gpu/samples.json`, and `gpu/result.json` or failure
record. CPU fallback, trace failures, incorrect output or invalid Event times
fail the command. CPU tests use mocked dispatch and never count as GPU evidence.

The report carries graph/plan/input/source identities, observed hardware,
runtime versions and the explicit measurement contract. Admission remains
`conditional_dev`; no Canonical Profile is exported or existing label promoted.
The next stage binds this evidence to Canonical Kernel Keys and validates the
cost-model lookup. Downstream deployment and complete-model execution remain
separate stages.
