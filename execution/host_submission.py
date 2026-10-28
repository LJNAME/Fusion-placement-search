"""Small training-task probes for host dispatch steps and GPU stream windows."""

from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import statistics
import time

import torch

from calibration.e2_manifest import content_id, default_model_specs, kernel_key_for_node, measurement_object_id
from execution.gpu_holdout import CanonicalReferenceExecutor
from execution.reference import _sync
from simulator import make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.model_graphs import build_decoder_layer


CPU_PERTURBATION_TARGET = .15
BASELINE_SHIFT_TARGET = .10


def training_cases(freeze, manifest, sequences=(256, 1024)):
    if not sequences or len(set(sequences)) != len(sequences) or any(value not in (128, 256, 512, 1024) for value in sequences):
        raise ValueError("host calibration is restricted to existing training sequences 128,256,512,1024")
    known = set(freeze["train_object_ids"])
    contract = manifest["environment_contract"]
    result = []
    for spec in default_model_specs(sequences):
        graph = build_decoder_layer(spec, layer_id=0)
        family = "qwen3-0.6b" if spec.model_type == "qwen3" else "qwen2.5-3b"
        model_id = f"{family}-s{spec.seq_length}-b1"
        node_objects = {}
        for node in graph.nodes.values():
            key, recipe = kernel_key_for_node(node, spec, **{name: contract[name] for name in
                ("hardware_class", "framework_version", "accelerator_version")})
            object_id = measurement_object_id(key)
            if recipe["enabled"] and object_id not in known:
                raise ValueError(f"training graph introduces an unmeasured object: {object_id}")
            node_objects[node.id] = object_id if recipe["enabled"] else None
        reference = make_reference_plan(graph, device=0)
        for name in ("uniform0", "uniform1", "key_branch", "mlp_up"):
            plan = copy.deepcopy(reference)
            for group in plan.groups:
                local = graph.nodes[group.node_ids[0]].extra["local_name"]
                group.device = int(name == "uniform1" or
                    (name == "key_branch" and local in {"self_attention.k_proj", "self_attention.k_norm", "self_attention.k_rope"}) or
                    (name == "mlp_up" and local == "mlp.up_proj"))
            item = {"model_id": model_id, "spec": asdict(spec), "plan_name": name,
                "graph": graph_to_dict(graph), "plan": plan_to_dict(plan), "node_training_object_ids": node_objects}
            result.append({"case_id": content_id("reference-host-case", item), **item})
    return result


class SubmissionProbeExecutor(CanonicalReferenceExecutor):
    """Mirror run_once(False), without a profiler, dispatch mode or node sync.

    Timed calls append small tuples in the loop. Shape/layout serialization and
    event queries happen after the forward has finished. Baselines still use
    the original executor and are checked around the probe blocks.
    """

    def __init__(self, *args, device_windows=False, environment_id="", **kwargs):
        self.device_windows = device_windows
        self.environment_id = environment_id
        self.body_source_sha256 = hashlib.sha256(Path(__file__).with_name("reference.py").read_bytes()).hexdigest()
        self.catalog = {}
        self.events = {}
        super().__init__(*args, **kwargs)

    def event_pair(self, key, device):
        if not self.device_windows or device.type != "cuda":
            return None
        if key not in self.events:
            self.events[key] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        return self.events[key]

    def run_probe(self):
        outputs = {}
        remaining = {node_id: len(self.graph.out_edges(node_id)) for node_id in self.graph.nodes}
        raw = []
        _sync(self.devices)
        total_start = time.perf_counter()
        for group in self.placed.plan.groups:
            node_id = group.node_ids[0]
            target = self.devices[group.device]
            predecessors = self.graph.predecessors(node_id)
            # Preserve the original executor's bookkeeping even when its
            # instrumented transfer timer is disabled.
            _cross_device = any(outputs[item].device != target for item in predecessors)
            inputs = []
            for predecessor in predecessors:
                value = outputs[predecessor]
                if value.device == target:
                    inputs.append(value)
                    continue
                key = f"copy:{predecessor}->{node_id}:{len(inputs)}"
                pair = self.event_pair(key, value.device)
                if pair:
                    pair[0].record(torch.cuda.current_stream(value.device))
                started = time.perf_counter()
                copied = value.to(target)
                ended = time.perf_counter()
                if pair:
                    pair[1].record(torch.cuda.current_stream(value.device))
                raw.append((key, started, ended, pair))
                inputs.append(copied)
                if key not in self.catalog:
                    self.catalog[key] = {"kind": "copy", "source_node": predecessor, "node_id": node_id,
                        "source": str(value.device), "destination": str(target),
                        "payload_bytes": value.numel() * value.element_size()}
            key = f"node:{node_id}"
            pair = self.event_pair(key, target) if self.graph.nodes[node_id].op_type != "input" else None
            if pair:
                pair[0].record(torch.cuda.current_stream(target))
            started = time.perf_counter()
            outputs[node_id] = self._execute_node(node_id, inputs, target)
            ended = time.perf_counter()
            if pair:
                pair[1].record(torch.cuda.current_stream(target))
            raw.append((key, started, ended, pair))
            if key not in self.catalog:
                node = self.graph.nodes[node_id]
                body = {"measurement_kind": "reference_body_stream_window_v1", "operator": node.op_type,
                    "executor_source_sha256": self.body_source_sha256, "environment_id": self.environment_id,
                    "torch": torch.__version__, "hip": torch.version.hip,
                    "local_name": node.extra.get("local_name", node_id), "node_metadata": node.extra,
                    "inputs": [{"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)} for value in inputs],
                    "output_shape": list(outputs[node_id].shape), "output_dtype": str(outputs[node_id].dtype)}
                self.catalog[key] = {"kind": "node", "node_id": node_id, "operator": node.op_type,
                    "local_name": node.extra.get("local_name", node_id), "body_key": body,
                    "body_id": content_id("reference-body-window", body)}
            for predecessor in predecessors:
                remaining[predecessor] -= 1
                if remaining[predecessor] == 0:
                    del outputs[predecessor]
        body_end = time.perf_counter()
        _sync(self.devices)
        total_end = time.perf_counter()
        # All devices are already idle: estimate API overhead, not a pending
        # GPU wait. This is never taken from the first, busy synchronization.
        idle_start = time.perf_counter()
        _sync(self.devices)
        idle_end = time.perf_counter()
        operations = []
        for index, (key, started, ended, pair) in enumerate(raw):
            next_start = raw[index + 1][1] if index + 1 < len(raw) else body_end
            operations.append({"key": key, "start_s": started - total_start,
                "api_call_s": ended - started, "dispatch_step_s": next_start - started,
                "bookkeeping_after_s": next_start - ended,
                "gpu_stream_window_s": pair[0].elapsed_time(pair[1]) / 1000 if pair else None})
        output_node = str(self.graph.extra.get("output_node") or self.placed.plan.groups[-1].node_ids[-1])
        return outputs[output_node], {"total_s": total_end - total_start,
            "start_s": raw[0][1] - total_start, "idle_sync_s": idle_end - idle_start,
            "busy_final_sync_s_not_a_host_parameter": total_end - body_end, "operations": operations}


def summarize_blocks(records):
    strata = defaultdict(dict)
    for row in records:
        if row["mode"] in strata[row["case_id"]]:
            raise ValueError("duplicate host calibration block")
        strata[row["case_id"]][row["mode"]] = row
    estimates, quality, body_windows = defaultdict(list), [], defaultdict(list)
    expected = {"baseline_before", "cpu_probe", "window_probe", "baseline_after"}
    for case_id, blocks in strata.items():
        if set(blocks) != expected:
            raise ValueError("incomplete host calibration case")
        medians = {mode: statistics.median(blocks[mode]["samples_s"]) for mode in expected}
        before, after = medians["baseline_before"], medians["baseline_after"]
        shift = after / before - 1
        row = {"case_id": case_id, "model_id": blocks["cpu_probe"]["model_id"],
            "plan_name": blocks["cpu_probe"]["plan_name"], "baseline_before_s": before, "baseline_after_s": after,
            "baseline_shift_fraction": shift}
        for mode in ("cpu_probe", "window_probe"):
            ratios = [medians[mode] / before, medians[mode] / after]
            row[mode] = {"median_s": medians[mode], "over_before_ratio": ratios[0], "over_after_ratio": ratios[1],
                "target_pass": all(abs(value - 1) <= CPU_PERTURBATION_TARGET for value in ratios) and abs(shift) <= BASELINE_SHIFT_TARGET}
        quality.append(row)
        cpu = blocks["cpu_probe"]
        per_key = defaultdict(list)
        for sample in cpu["probe_samples"]:
            estimates["start_s:" + case_id].append(sample["start_s"])
            estimates["finish_s:" + case_id].append(sample["idle_sync_s"])
            for operation in sample["operations"]:
                item = cpu["catalog"][operation["key"]]
                if item["kind"] == "copy":
                    per_key["copy_submit_s"].append(operation["dispatch_step_s"])
                else:
                    per_key["op:" + item["operator"]].append(operation["dispatch_step_s"])
                    per_key["local:" + item["local_name"]].append(operation["dispatch_step_s"])
        # First take one median per case: many repeated GEMMs do not give one
        # workload greater weight in the cross-case host parameter estimate.
        for key, values in per_key.items():
            estimates[key].append(statistics.median(values))
        window = blocks["window_probe"]
        row["window_data_complete"] = all(
            operation["gpu_stream_window_s"] is not None
            for sample in window["probe_samples"] for operation in sample["operations"]
            if window["catalog"][operation["key"]].get("operator") != "input")
        for sample in window["probe_samples"]:
            for operation in sample["operations"]:
                item = window["catalog"][operation["key"]]
                if item["kind"] == "node" and operation["gpu_stream_window_s"] is not None:
                    body_windows[item["body_id"], case_id].append(operation["gpu_stream_window_s"])
    by_operator = {key[3:]: statistics.median(values) for key, values in estimates.items() if key.startswith("op:")}
    by_operator.update({key: statistics.median(values) for key, values in estimates.items() if key.startswith("local:")})
    starts = [statistics.median(values) for key, values in estimates.items() if key.startswith("start_s:")]
    finishes = [statistics.median(values) for key, values in estimates.items() if key.startswith("finish_s:")]
    passed = bool(quality) and bool(estimates["copy_submit_s"]) and all(row["cpu_probe"]["target_pass"] for row in quality)
    return {"host_parameters": {"start_s": statistics.median(starts), "finish_s": statistics.median(finishes),
        "node_submit_s": statistics.median(value for key, value in by_operator.items() if not key.startswith("local:")),
        "copy_submit_s": statistics.median(estimates["copy_submit_s"]) if estimates["copy_submit_s"] else 0.0,
        "operator_submit_s": by_operator, "calibrated": passed},
        "cases": quality, "all_cpu_probe_targets_pass": passed,
        "all_window_probe_targets_pass": all(row["window_probe"]["target_pass"] for row in quality),
        "all_window_data_complete": all(row["window_data_complete"] for row in quality),
        "body_window_candidates": [{"body_id": key[0], "case_id": key[1], "samples": len(values),
            "median_s": statistics.median(values)} for key, values in sorted(body_windows.items())],
        "protocol": {"probe_deviation_target": CPU_PERTURBATION_TARGET, "baseline_shift_target": BASELINE_SHIFT_TARGET,
            "aggregation": "median within each case, then median across cases; operator and local-body fallbacks",
            "host_target": "API-start to next API-start, including following executor bookkeeping; not GPU completion time",
            "finish_target": "already-idle synchronization API only; busy final GPU wait is excluded"}}
