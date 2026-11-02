"""Single-device pilot runtime; the source-frozen B4 v3 collector stays intact."""
from pathlib import Path

import torch
import torch.nn.functional as F

from calibration.ffn_b4_runtime import (
    Runtime, GraphCase, E, ROOT, bytes_equal, sync, tensor_meta,
)
from calibration.kerv_silu_mul import DEFAULT_BUNDLE
from calibration.probe_kerv_graph import POOL, memory_snapshot
from calibration.validate_kerv_adapter import prepare
from cost_model.kerv_profile import digest
from integration.flagscale.kerv_execution import KervSiluMulAdapter
from integration.flagscale.kerv_profile_runtime import loaded_source, observe_environment, native_binding
from simulator.placement_artifacts import load_placed_op_graph


class ControlRuntime(Runtime):
    def __init__(self, output, recipe, flagscale_root):
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("anchor pilot requires HIP GPU0; no CPU/CUDA fallback")
        self.output, self.recipe = Path(output), recipe
        self.context = {"pool_config": dict(POOL), "stage": "identity", "active_devices": [0]}
        torch.cuda.set_device(0)
        self.adapter = KervSiluMulAdapter(DEFAULT_BUNDLE / "capability.json", flagscale_root)
        self.loaded = loaded_source(self.adapter.module, flagscale_root)
        self.observations = {0: observe_environment(self.adapter.module, torch.device("cuda:0"))}
        observed = self.observations[0][0]
        if not observed.get("architecture") or observed.get("uuid") in (None, "unavailable"):
            raise RuntimeError("actual GPU0 architecture/UUID unavailable")
        self.peer = {}  # No peer queries or GPU1 work in this pilot.
        self.build_identity = {
            "torch_module_version": str(torch.__version__), "git_revision": torch.version.git_version,
            "torch_extension_sha256": digest(Path(torch._C.__file__)), "native_binding": native_binding(),
            "collector_source_sha256": digest(Path(__file__)), "grad_enabled": torch.is_grad_enabled(),
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        directory = self.output / "placed_gpu0"
        directory.mkdir()
        index = prepare(directory, DEFAULT_BUNDLE / "capability.json", flagscale_root,
                        ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json", device=0)
        placed = load_placed_op_graph(directory / "placed.json", registry=self.adapter.registry)
        self.bound = {0: self.adapter.bind(placed, index, {0: "cuda:0"})}
        self.cpu, self.cache, self.active_group = {}, {}, None
        self.identity_record = self.identity((0,), {"operator": "anchor pilot"})

    def snapshots(self):
        return {"0": memory_snapshot(torch, 0)}

    def make(self, row_id):
        if row_id not in {"down_proj@gpu0", "up_proj@gpu0"}:
            raise ValueError("only the two frozen projection workloads are open")
        return super().make(row_id)

    def activation(self, gpu, *, mul_smoke_only=False):
        """Future single-device cases record operands at their actual use site.

        Not opened by make()/the pilot matrix. This separate implementation
        preserves the v3 source and its archived source/contract identities.
        """
        if gpu != 0:
            raise ValueError("control runtime is GPU0 only")
        token = "mul_metadata_smoke0" if mul_smoke_only else "activation0"
        if token in self.cache:
            return self.cache[token]
        original, up = self.tensor("gate_original", E, gpu), self.tensor("up", E, gpu)
        up_backup = up.clone()
        gates = original.unsqueeze(0).repeat((50, 1, 1, 1))
        views = list(gates.unbind(0))
        silu_original = F.silu(original)
        left = silu_original.unsqueeze(0).repeat((50, 1, 1, 1))
        left_views, left_backup = list(left.unbind(0)), left.clone()
        expected_native = silu_original * up
        expected_fp32 = (F.silu(original.float()) * up.float()).to(torch.bfloat16)
        device, bound = torch.device("cuda:0"), self.bound[0]

        def fused(gate):
            return bound({bound.gate_node: gate, bound.up_node: up}, observe_dispatch=False).output

        probe_launches = 0
        if not mul_smoke_only:
            with torch.cuda.device(device):
                probe = bound({bound.gate_node: views[0], bound.up_node: up}, observe_dispatch=True)
                sync((0,))
            if probe.observed_jit_launches != 1 or probe.output.data_ptr() != views[0].data_ptr():
                raise RuntimeError("actual public KERV path or gate alias failed")
            probe_launches = 1
        cases = {}
        for role, row_id in (("silu", "silu@gpu0"), ("mul", "gated_mul@gpu0"),
                             ("native", "silu_mul/native@gpu0"), ("fused", "silu_mul/fused@gpu0")):
            if mul_smoke_only and role != "mul":
                continue
            def reset(_):
                gates.copy_(original.unsqueeze(0).expand_as(gates))
                sync((0,))
                if not bytes_equal(gates, original.unsqueeze(0).expand_as(gates)):
                    raise AssertionError("gate restore is not byte-exact")

            def body(role=role):
                if role == "silu": return [F.silu(g) for g in views]
                if role == "mul": return [g * up for g in left_views]
                if role == "native": return [F.silu(g) * up for g in views]
                return [fused(g) for g in views]

            def check(outputs, _, role=role):
                for i, value in enumerate(outputs):
                    torch.testing.assert_close(value, silu_original if role == "silu" else expected_native, rtol=.01, atol=.001)
                    if role == "fused":
                        torch.testing.assert_close(value, expected_fp32, rtol=.01, atol=.001)
                        if value.data_ptr() != views[i].data_ptr():
                            raise AssertionError("KERV output does not alias gate")
                if not bytes_equal(up, up_backup) or not bytes_equal(left, left_backup):
                    raise AssertionError("read-only up or actual Mul left bytes changed")
                if role != "fused" and not bytes_equal(gates, original.unsqueeze(0).expand_as(gates)):
                    raise AssertionError("out-of-place operator changed gate")
                return {"all_50_outputs": len(outputs) == 50, "up_bytes_unchanged": True,
                        "mul_left_bytes_unchanged": True, "gate_restore_exact": True,
                        "gate_output_alias": role != "fused" or all(o.data_ptr() == g.data_ptr() for o, g in zip(outputs, views))}

            inputs = {"gates": [tensor_meta(g) for g in views], "up": tensor_meta(up)}
            if role == "mul":
                inputs["left"] = [tensor_meta(g) for g in left_views]
            cases[row_id] = GraphCase(row_id, device, (0,), body, reset, check,
                {**self.identity((0,), {"role": role, "kerv_binding": self.adapter.manifest["binding"] if role == "fused" else "Torch Reference"}),
                 "inputs": inputs, "observed_jit_probe_launches": probe_launches,
                 "operand_metadata_version": "actual-operands.v1"})
        self.cache[token] = cases
        return cases
