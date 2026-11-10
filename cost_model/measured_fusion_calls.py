"""Opt-in complete-call measurements for fixed LightOP execution recipes.

No per-shape fitting, timing decomposition, or new global Profile admission.
Unknown layouts remain target runtime requirements, never observations.
"""
import hashlib
import json
import math
from pathlib import Path

from calibration.fusion_timing_expansion import CASES, PreparedCall, summarize
from cost_model.fusion_kernel_profile import key_id
from cost_model.hardware_profile import GroupCost
from cost_model.fusion_library import JointQKRmsRopePattern
from integration.lightop.catalog import catalog, resolve
from simulator.fusion_rules import enumerate_fusion_candidates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEASUREMENTS = ROOT / "data/fusion_timing_expansion_20261109_01"
PATTERNS = {"lightop_silu_mul": "silu_mul_3b", "lightop_add_rmsnorm": "add_rmsnorm_3b",
            "lightop_qk_rmsnorm_rope": "rmsnorm_rope_qwen3"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MeasuredFusionCalls:
    def __init__(self, base, library, directory=DEFAULT_MEASUREMENTS):
        self.base, self.library, self.directory = base, library, Path(directory).resolve()
        self.frozen = read(self.directory / "plan.json")
        receipt = read(self.directory.with_name(self.directory.name + "_review") / "verification.json")
        if not receipt["passed"]:
            raise ValueError("measurement review did not pass")
        for name, digest in receipt["raw_file_sha256"].items():
            if sha(self.directory / name) != digest:
                raise ValueError("measurement evidence changed: " + name)
        parent = read(self.directory / "summary.json")
        self.records = {}
        for pid, case_id in PATTERNS.items():
            result = read(self.directory / case_id / "result.json")
            rows = [read(self.directory / case_id / f"round_{i}.json") for i in range(7)]
            quality = summarize(rows, 1 if case_id == "silu_mul_3b" else 2)
            if (parent["cases"][case_id]["returncode"] != 0 or quality["status"] != "quality_passed"
                    or result["quality"] != quality or key_id(result["identity"]) != result["execution_key_id"]
                    or result["identity"]["case"] != self.frozen["cases"][case_id]
                    or result["contract"] != self.frozen["contract"]):
                raise ValueError("invalid measured call: " + case_id)
            if result["identity"]["case"]["recipe"] != CASES[case_id]["recipe"]:
                raise ValueError("unknown execution recipe")
            if result["identity"]["runtime"]["native_sha256"] != catalog()["op_so_sha256"]:
                raise ValueError("measurement uses a different provider build")
            self.records[pid] = result
        # Existing GPU1 evidence verifies the two cards' class AND software stack.
        evidence = ROOT / "data/gpu1_cost_completion_20261106_01"
        saved = read(evidence / "review/evaluation.json")
        self.hardware_evidence = evidence / "raw/runtime.json"
        if sha(self.hardware_evidence) != saved["raw_sha256"]["runtime.json"]:
            raise ValueError("hardware evidence changed")
        observed = read(self.hardware_evidence)
        self.device_match = {}
        for i, device in enumerate(self.hardware.devices):
            matches = []
            for result in self.records.values():
                rt = result["identity"]["runtime"]
                matches.append(device.extra.get("class") == "BW1000" and
                    device.name == observed["devices"].get(str(i), {}).get("name") == rt["device"]["name"] and
                    observed["devices"].get(str(i), {}).get("architecture") == rt["device"]["architecture"] and
                    int(device.max_mem) == rt["device"]["total_memory"] and
                    all(observed["runtime"].get(a) == rt.get(b) for a, b in (
                        ("torch_module_version", "torch"), ("torch_git_revision", "torch_git_revision"),
                        ("torch_extension_sha256", "torch_extension_sha256"), ("torch_build_sha256", "torch_build_sha256"),
                        ("hip", "hip"), ("driver", "driver"))))
            self.device_match[i] = all(matches)

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def match(self, graph, group):
        pid = group.extra.get("pattern_id") if hasattr(group, "extra") else group.pattern_id
        if pid not in self.records or not self.device_match.get(group.device, False):
            return None
        record = self.records[pid]
        ids = tuple(group.node_ids)
        pattern = next(e.pattern for e in self.library.entries if e.fusion_id == pid)
        if not pattern.matches(graph, ids):
            return None
        spec = record["identity"]["case"]
        if any(graph.nodes[n].extra.get("dtype") != "bf16" or
               graph.nodes[n].extra.get("phase") != "forward" or graph.nodes[n].extra.get("layer") is None for n in ids):
            return None
        if pid == "lightop_silu_mul":
            silu, mul = ids
            gate = list(graph.predecessors(silu))
            up = [n for n in graph.predecessors(mul) if n != silu]
            if len(gate) != 1 or len(up) != 1 or gate == up or len(graph.in_edges(mul)) != 2:
                return None
            roles = {"gate": gate[0], "up": up[0]}
        elif pid == "lightop_add_rmsnorm":
            add, norm = ids
            parents = list(graph.predecessors(add))
            if len(parents) != 2 or len(set(parents)) != 2 or list(graph.predecessors(norm)) != [add]:
                return None
            roles = dict(zip(("x", "residual"), parents))
            if graph.nodes[norm].param_bytes != spec["shape"][-1] * 2:
                return None
        else:
            roles = {"q": next(iter(graph.predecessors(ids[0]))), "k": next(iter(graph.predecessors(ids[2])))}
        for name, nid in roles.items():
            observed = record["identity"]["inputs"][name]
            node = graph.nodes[nid]
            if (tuple(node.extra.get("output_shape", ())) != tuple(observed["shape"])
                    or node.extra.get("dtype") != observed["dtype"]):
                return None
        for nid in (*roles.values(), *ids):
            node = graph.nodes[nid]
            expected = spec.get("key_shape") if pid.endswith("rope") and nid in (roles.get("k"), *ids[2:]) else spec["shape"]
            if tuple(node.extra.get("output_shape", ())) != tuple(expected):
                return None
            strides = node.extra.get("strides")
            contiguous = tuple(math.prod(expected[j + 1:]) for j in range(len(expected)))
            if strides is not None and tuple(strides) != contiguous:
                return None
            layout = node.extra.get("layout")
            if isinstance(layout, dict):
                if layout.get("order") not in (None, "row_major"):
                    return None
                layout = layout.get("contiguity")
            if layout not in (None, "unknown", "contiguous"):
                return None
            if node.extra.get("eps", 1e-6) != 1e-6:
                return None
        return {"execution_key_id": record["execution_key_id"], "case_id": PATTERNS[pid],
                "recipe_id": spec["recipe"], "implementation": spec["implementation"],
                "measurement_contract_id": key_id(record["contract"]),
                "median_us": record["quality"]["statistics"]["median_us"],
                "measured_on": 0, "reuse_on": group.device,
                "reuse_kind": "same_hardware_class" if group.device == 0 else "homogeneous_hardware_class_transfer",
                "target_independently_measured": group.device == 0,
                "input_nodes": roles, "required_layout": "contiguous",
                "layout_source": "target execution contract; unknown graph layout not rewritten",
                "eps": 1e-6, "scope": "local measured complete-call cost; not global accepted admission",
                "binding": "calibration.fusion_timing_expansion.PreparedCall",
                "workspace_bytes": math.prod(spec["shape"]) * 4 if pid == "lightop_silu_mul" else 0}

    def annotate_plan(self, graph, plan):
        for group in plan.groups:
            binding = self.match(graph, group)
            if binding:
                group.extra["measured_complete_call"] = binding
                group.extra["workspace_bytes"] = binding["workspace_bytes"]
        return plan

    def prepare(self, graph, group, inputs):
        """Use the exact collector call, including copies/packing. No fallback."""
        import torch
        binding = self.match(graph, group)
        if binding is None:
            raise ValueError("group does not satisfy a measured execution recipe")
        record = self.records[group.extra["pattern_id"]]
        for name, expected in record["identity"]["inputs"].items():
            value = inputs[name]
            if (list(value.shape) != expected["shape"] or str(value.dtype).removeprefix("torch.") not in
                    ({"bfloat16"} if expected["dtype"] == "bf16" else {expected["dtype"]}) or
                    not value.is_contiguous() or value.requires_grad or not value.is_cuda):
                raise ValueError("runtime tensor violates measured call contract: " + name)
        if not torch.version.hip or len({t.device for t in inputs.values()}) != 1:
            raise ValueError("one HIP device required")
        case = binding["case_id"]
        return PreparedCall(torch, case, inputs, resolve(CASES[case]["capability"]))

    def estimate_group(self, graph, group):
        binding = self.match(graph, group)
        if binding is None:
            return self.base.estimate_group(graph, group)
        source = "measured_complete_call" if group.device == 0 else "transferred_complete_call"
        return GroupCost(binding["median_us"] * 1e-6, 0., 0., 0., .9, source)

    def audit_graph(self, graph, registry):
        candidates = [c for c in enumerate_fusion_candidates(graph, registry, True) if len(c.node_ids) > 1]
        rows = [{"pattern_id": c.pattern_id, "nodes": list(c.node_ids), "device": device,
                 "measurement": self.match(graph, c.to_group(device))}
                for c in candidates for device in range(self.n_devs)]
        joint = next(e.pattern for e in self.library.entries if isinstance(e.pattern, JointQKRmsRopePattern))
        structural = list(joint.candidate_node_ids(graph))
        return {"registered": list(PATTERNS), "device_class_and_stack_match": self.device_match,
                "candidate_costs": rows, "joint_qk_structural_groups": len(structural),
                "joint_qk_matched_groups": sum(joint.matches(graph, ids) for ids in structural),
                "joint_qk_blocked_reason": "graph rotary/normalization semantic declaration missing or incompatible"
                    if structural and not any(joint.matches(graph, ids) for ids in structural) else None,
                "hardware_evidence_sha256": sha(self.hardware_evidence),
                "measurement_files": {p.relative_to(ROOT).as_posix(): sha(p) for p in self.directory.rglob('*.json')}}
