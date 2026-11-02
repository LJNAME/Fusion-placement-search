"""GPU-only B4 graph cases. No GPU execution occurs on module import."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from calibration.probe_kerv_graph import memory_snapshot, POOL
from calibration.validate_kerv_adapter import prepare
from calibration.ffn_b4_policy import diagnostic_time
from integration.flagscale.kerv_execution import KervSiluMulAdapter
from integration.flagscale.kerv_profile_runtime import loaded_source, native_binding, observe_environment
from cost_model.kerv_profile import digest
from simulator.placement_artifacts import load_placed_op_graph
from execution.ffn_transfer import BoundTransfer, declared_carrier

E, H, CALLS = (1, 256, 3072), (1, 256, 1024), 50


class PathViolation(RuntimeError):
    """Observed execution contradicts the declared path, not just tracing gaps."""


def bytes_equal(left, right):
    return bool(torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def tensor_meta(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.stride()),
            "device": str(value.device), "data_ptr": value.data_ptr(), "layout_source": "runtime Tensor observation",
            "address_scope": "alias evidence only; not a canonical measurement-key field"}


def sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


class GraphCase:
    def __init__(self, row_id, device, devices, body, reset, check, identity, stream=None):
        self.row_id, self.device, self.devices = row_id, device, devices
        self.body, self.reset, self.check, self.identity = body, reset, check, identity
        self.stream = stream or torch.cuda.Stream(device=device)
        self.graph, self.outputs = None, None

    def capture(self):
        self.reset(0)
        sync(self.devices)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.body()  # Compile/library initialization before capture.
        sync(self.devices)
        self.reset(0)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self.device), torch.cuda.graph(self.graph, stream=self.stream, pool=None):
            self.outputs = self.body()
        current.wait_stream(self.stream)
        sync(self.devices)
        self.reset(0)
        self.identity["output_layouts"] = [tensor_meta(value) for value in self.outputs]

    def replay(self, index, *, diagnostic=False):
        self.reset(index)
        sync(self.devices)
        timing_error, elapsed = None, None
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            try:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(self.stream)
            except Exception as error:
                if not diagnostic:
                    raise
                timing_error, start, end = error, None, None
            self.graph.replay()
            if end is not None:
                try:
                    end.record(self.stream)
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                except Exception as error:
                    if not diagnostic:
                        raise
                    timing_error = error
        sync(self.devices)
        invariants = self.check(self.outputs, index)
        if not all(v is True for v in invariants.values()):
            raise AssertionError("graph invariant failed: " + repr(invariants))
        return diagnostic_time(elapsed, timing_error), invariants

    def trace(self, output):
        self.reset(0)
        sync(self.devices)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                self.graph.replay()
            sync(self.devices)
        profile.export_chrome_trace(str(output))
        trace = json.loads(Path(output).read_text(encoding="utf-8"))
        kernels = [e.get("name", "") for e in trace["traceEvents"] if e.get("cat") == "kernel"]
        host_staging = [e.get("name", "") for e in trace["traceEvents"] if any(
            text in e.get("name", "").lower() for text in ("htod", "dtoh", "host to device", "device to host"))]
        if host_staging:
            raise PathViolation("host staging observed inside captured replay: " + repr(host_staging))
        if "silu_mul/fused" in self.row_id:
            if any("silu_mul_kernel" not in name for name in kernels):
                raise PathViolation("unexpected device kernels on the fused replay")
            if len(kernels) != 50:
                raise ValueError("profiler did not expose all 50 fused graph kernels")
        d2d = any(any(word in e.get("name", "").lower() for word in ("dtod", "ptop", "device to device", "device -> device")) for e in trace["traceEvents"])
        return {"path_trace_written": True, "kernel_names": kernels, "host_staging_detected": False,
                "device_copy_trace_observed": d2d,
                "peer_route_certified": False, "scope": "replay trace; capability alone is not a peer-route certificate"}


class Runtime:
    def __init__(self, output, recipe, flagscale_root=DEFAULT_FLAGSCALE):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not torch.version.hip:
            raise RuntimeError("B4 requires two HIP GPUs; CPU/CUDA fallback is forbidden")
        self.output, self.recipe = Path(output), recipe
        self.context = {"pool_config": dict(POOL), "stage": "identity"}
        self.adapter = KervSiluMulAdapter(DEFAULT_BUNDLE / "capability.json", flagscale_root)
        self.loaded = loaded_source(self.adapter.module, flagscale_root)
        self.observations = {gpu: observe_environment(self.adapter.module, torch.device(f"cuda:{gpu}")) for gpu in (0, 1)}
        if any(not obs[0].get("architecture") or obs[0].get("uuid") in (None, "unavailable") for obs in self.observations.values()):
            raise RuntimeError("actual GPU architecture/UUID unavailable")
        self.peer = {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a, b in ((0, 1), (1, 0))}
        self.bound = {}
        self.build_identity = {"torch_module_version": str(torch.__version__), "git_revision": torch.version.git_version,
            "torch_extension_sha256": digest(Path(torch._C.__file__)), "native_binding": native_binding(),
            "collector_source_sha256": digest(Path(__file__)), "grad_enabled": torch.is_grad_enabled(),
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        for gpu in (0, 1):
            directory = self.output / f"placed_gpu{gpu}"
            directory.mkdir()
            index = prepare(directory, DEFAULT_BUNDLE / "capability.json", flagscale_root,
                            ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json", device=gpu)
            placed = load_placed_op_graph(directory / "placed.json", registry=self.adapter.registry)
            self.bound[gpu] = self.adapter.bind(placed, index, {gpu: f"cuda:{gpu}"})
        self.cpu, self.cache, self.active_group = {}, {}, None
        self.identity_record = {"loaded_source": self.loaded, "observations": self.observations,
                                "peer": self.peer, "torch_config": torch.__config__.show(),
                                "torch_git_revision": torch.version.git_version,
                                "grad_enabled": torch.is_grad_enabled()}

    def tensor(self, name, shape, gpu):
        if name not in self.cpu:
            spec = self.recipe["named_fixtures"][name]
            self.cpu[name] = torch.randn(shape, dtype=torch.bfloat16,
                generator=torch.Generator(device="cpu").manual_seed(spec["seed"])) * spec["scale"]
        return self.cpu[name].to(f"cuda:{gpu}")

    def y(self, gpu):
        if "Y_cpu" not in self.cpu:
            with torch.cuda.device(0):
                value = self.tensor("producer_V", E, 0).reshape(256, 3072)
                weight = self.tensor("W_down", (3072, 1024), 0)
                self.cpu["Y_cpu"] = (value @ weight).reshape(H).cpu()
        return self.cpu["Y_cpu"].to(f"cuda:{gpu}")

    def identity(self, devices, implementation):
        implementation = copy.deepcopy(implementation)
        if isinstance(implementation.get("kerv_binding"), dict):
            implementation.update(implementation["kerv_binding"])
        else:
            operator = implementation.get("operator") or implementation.get("role") or implementation.get("path")
            implementation.update(id="ffn_b4.reference." + str(operator), version=str(torch.__version__), backend="torch_native")
        implementation["collector_source_sha256"] = self.build_identity["collector_source_sha256"]
        return {"devices": {str(g): {"observed": self.observations[g][0], "environment": self.observations[g][1]} for g in devices},
                "implementation": implementation, "loaded_kerv_source": self.loaded, "peer_capabilities": self.peer,
                "actual_torch_build": self.build_identity}

    def release(self, group):
        if self.active_group != group:
            self.cache.clear()
            gc.collect()
            sync((0, 1))
            torch.cuda.empty_cache()
            self.active_group = group

    def activation(self, gpu):
        token = f"activation{gpu}"
        if token in self.cache:
            return self.cache[token]
        original = self.tensor("gate_original", E, gpu)
        up = self.tensor("up", E, gpu)
        up_backup = up.clone()
        gates = original.unsqueeze(0).repeat((50, 1, 1, 1))
        views = list(gates.unbind(0))
        silu_original = F.silu(original)
        left = silu_original.unsqueeze(0).repeat((50, 1, 1, 1))
        expected_native = silu_original * up
        expected_fp32 = (F.silu(original.float()) * up.float()).to(torch.bfloat16)
        device = torch.device(f"cuda:{gpu}")
        bound = self.bound[gpu]

        def fused(gate):
            result = bound({bound.gate_node: gate, bound.up_node: up}, observe_dispatch=False)
            return result.output

        gates.copy_(original.unsqueeze(0).expand_as(gates))
        with torch.cuda.device(device):
            probe = bound({bound.gate_node: views[0], bound.up_node: up}, observe_dispatch=True)
            torch.cuda.synchronize(device)
        if probe.observed_jit_launches != 1 or probe.output.data_ptr() != views[0].data_ptr():
            raise RuntimeError("actual public KERV path or gate alias failed")
        cases = {}
        for role, row_id in (("silu", f"silu@gpu{gpu}"), ("mul", f"gated_mul@gpu{gpu}"),
                             ("native", f"silu_mul/native@gpu{gpu}"), ("fused", f"silu_mul/fused@gpu{gpu}")):
            def reset(_index):
                gates.copy_(original.unsqueeze(0).expand_as(gates))
                sync((gpu,))
                if not bytes_equal(gates, original.unsqueeze(0).expand_as(gates)):
                    raise AssertionError("gate restore is not byte-exact")

            def body(role=role):
                if role == "silu": return [F.silu(g) for g in views]
                if role == "mul": return [g * up for g in left.unbind(0)]
                if role == "native": return [F.silu(g) * up for g in views]
                return [fused(g) for g in views]

            def check(outputs, _index, role=role):
                expected = silu_original if role == "silu" else expected_native
                for i, value in enumerate(outputs):
                    torch.testing.assert_close(value, expected, rtol=.01, atol=.001)
                    if role == "fused":
                        torch.testing.assert_close(value, expected_fp32, rtol=.01, atol=.001)
                        if value.data_ptr() != views[i].data_ptr():
                            raise AssertionError("KERV output does not alias its gate")
                if not bytes_equal(up, up_backup):
                    raise AssertionError("shared up bytes changed")
                if role != "fused" and not bytes_equal(gates, original.unsqueeze(0).expand_as(gates)):
                    raise AssertionError("Native/SiLU/Mul unexpectedly changed gate")
                return {"all_50_outputs": len(outputs) == 50, "up_bytes_unchanged": True,
                        "gate_restore_exact": True, "gate_output_alias": role != "fused" or all(o.data_ptr() == g.data_ptr() for o, g in zip(outputs, views))}

            cases[row_id] = GraphCase(row_id, device, (gpu,), body, reset, check,
                {**self.identity((gpu,), {"role": role, "kerv_binding": self.adapter.manifest["binding"] if role == "fused" else "Torch Reference"}),
                 "inputs": {"gates": [tensor_meta(g) for g in views], "up": tensor_meta(up)}, "observed_jit_probe_launches": 1})
        self.cache[token] = cases
        return cases

    def gemm(self, row_id):
        local, gpu_text = row_id.split("@gpu")
        gpu = int(gpu_text)
        device = torch.device(f"cuda:{gpu}")
        down = local == "down_proj"
        base = self.tensor("producer_V", E, gpu) if down else self.y(gpu)
        weight_name = "W_down" if down else "W_gate" if local == "gate_proj" else "W_up"
        weight = self.tensor(weight_name, (3072, 1024) if down else (1024, 3072), gpu)
        inputs = base.unsqueeze(0).repeat((50, 1, 1, 1))
        expected = (base.float().reshape(256, -1) @ weight.float()).to(torch.bfloat16).reshape(H if down else E)
        before, weight_before = inputs.clone(), weight.clone()
        def body(): return [(value.reshape(256, -1) @ weight).reshape(H if down else E) for value in inputs.unbind(0)]
        def check(outputs, _):
            for value in outputs: torch.testing.assert_close(value, expected, rtol=.01, atol=.001)
            if not bytes_equal(inputs, before) or not bytes_equal(weight, weight_before):
                raise AssertionError("GEMM input/weight bytes changed")
            return {"all_50_outputs": len(outputs) == 50, "inputs_preserved": True}
        return GraphCase(row_id, device, (gpu,), body, lambda _: None, check,
                         {**self.identity((gpu,), {"operator": "torch.matmul", "weight": weight_name}),
                          "inputs": [tensor_meta(v) for v in inputs.unbind(0)], "weight": tensor_meta(weight)})

    def copy(self, row_id):
        _, size_text, direction = row_id.split("/")
        src, dst = map(int, direction.split("->"))
        shape = H if int(size_text) == 524288 else E
        base = self.y(src) if shape == H else self.tensor("gate_original", E, src)
        sources = base.unsqueeze(0).repeat((50, 1, 1, 1))
        destinations = torch.empty_like(sources, device=f"cuda:{dst}")
        source_stream, stream = torch.cuda.Stream(device=src), torch.cuda.Stream(device=dst)
        fork, join = torch.cuda.Event(), torch.cuda.Event()
        bindings = [BoundTransfer(a, b, declared_carrier(list(shape), "bf16"), src, dst, stream)
                    for a, b in zip(sources.unbind(0), destinations.unbind(0))]
        expected = [None]
        def reset(index):
            # Outside replay: make stale-output detection possible in smoke.
            current = base + index
            sources.copy_(current.unsqueeze(0).expand_as(sources))
            expected[0] = current.to(f"cuda:{dst}")
            destinations.fill_(float("nan"))
            sync((src, dst))
        def body():
            # Torch's cross-device copy may enqueue work on the source current
            # stream as well. Both sides must use joined non-default streams.
            fork.record(stream)
            source_stream.wait_event(fork)
            with torch.cuda.stream(source_stream):
                outputs = [call() for call in bindings]
                join.record(source_stream)
            stream.wait_event(join)
            return outputs
        def check(outputs, index):
            for value in outputs: torch.testing.assert_close(value, expected[0], rtol=0, atol=0)
            if not bytes_equal(sources, (base+index).unsqueeze(0).expand_as(sources)):
                raise AssertionError("copy source bytes changed")
            return {"all_50_outputs": len(outputs) == 50, "source_preserved": True, "stale_output_excluded": True}
        return GraphCase(row_id, torch.device(f"cuda:{dst}"), (src, dst), body, reset, check,
                         {**self.identity((src, dst), {"operator": "preallocated Tensor.copy_", "size_bytes": int(size_text)}),
                          "inputs": [tensor_meta(v) for v in sources.unbind(0)], "destinations": [tensor_meta(v) for v in destinations.unbind(0)]}, stream)

    def chain(self, row_id):
        # Attempt the declared multi-GPU capture. Unsupported HIP capture fails
        # this cell; no serialized eager substitute is returned.
        v = self.tensor("producer_V", E, 0).unsqueeze(0).repeat((50, 1, 1, 1))
        wd, wg = self.tensor("W_down", (3072, 1024), 0), self.tensor("W_gate", (1024, 3072), 1)
        targets = torch.empty((50, *H), device="cuda:1", dtype=torch.bfloat16)
        expected = (self.y(1).float().reshape(256,1024) @ wg.float()).to(torch.bfloat16).reshape(E)
        pstream, cstream = torch.cuda.Stream(device=0), torch.cuda.Stream(device=1)
        start_fence = torch.cuda.Event()
        done = [torch.cuda.Event() for _ in range(50)]
        consumer_done = [torch.cuda.Event() for _ in range(50)]
        produced = []
        def body():
            produced.clear()
            start_fence.record(cstream)
            pstream.wait_event(start_fence)
            results = []
            for i, value in enumerate(v.unbind(0)):
                if i:
                    pstream.wait_event(consumer_done[i-1])
                with torch.cuda.device(0), torch.cuda.stream(pstream):
                    tensor = (value.reshape(256,3072) @ wd).reshape(H)
                    produced.append(tensor)
                    done[i].record(pstream)
                cstream.wait_event(done[i])
                with torch.cuda.stream(pstream):
                    with torch.cuda.device(1), torch.cuda.stream(cstream):
                        targets[i].copy_(tensor, non_blocking=True)
                        results.append((targets[i].reshape(256,1024) @ wg).reshape(E))
                        consumer_done[i].record(cstream)
            return results
        def check(outputs, _):
            for value in outputs: torch.testing.assert_close(value, expected, rtol=.01, atol=.001)
            return {"all_50_outputs": len(outputs) == 50, "consumer_after_copy": True}
        if not self.peer["0->1"]:
            raise RuntimeError("chain peer path unavailable")
        return GraphCase(row_id, torch.device("cuda:1"), (0, 1), body, lambda _: None, check,
                         {**self.identity((0, 1), {"path": "Down0 -> copy 0.5MiB -> Gate1"}),
                          "inputs": [tensor_meta(t) for t in v.unbind(0)], "weights": [tensor_meta(wd), tensor_meta(wg)]}, cstream)

    def make(self, row_id):
        self.context.update(row_id=row_id, stage="prepare_case")
        if row_id.startswith("copy/"): return self.copy(row_id)
        if row_id.startswith("diagnostic/"): return self.chain(row_id)
        gpu = int(row_id.rsplit("gpu", 1)[-1])
        if row_id.startswith(("silu@", "gated_mul@", "silu_mul/")):
            return self.activation(gpu)[row_id]
        return self.gemm(row_id)

    def anchor(self, gpu):
        source = self.tensor("anchor", E, gpu)
        expected = F.silu(source)
        def body(): return [F.silu(source) for _ in range(50)]
        def check(outputs, _):
            for value in outputs: torch.testing.assert_close(value, expected, rtol=.01, atol=.001)
            return {"all_50_outputs": len(outputs) == 50}
        return GraphCase(f"anchor@gpu{gpu}", torch.device(f"cuda:{gpu}"), (gpu,), body, lambda _: None, check,
                         self.identity((gpu,), {"operator": "F.silu", "role": "anchor_not_profile"}))

    def snapshots(self):
        return {str(g): memory_snapshot(torch, g) for g in (0, 1)}


ROOT = Path(__file__).resolve().parents[1]
