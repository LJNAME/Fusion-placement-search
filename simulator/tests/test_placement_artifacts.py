"""SearchResult and PlacedOpGraph artifact tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from simulator import (
    DeviceSpec,
    FusionGroup,
    HardwareSpec,
    OpGraph,
    OpNode,
    Plan,
    PlacedOpGraph,
    SearchCandidate,
    SearchResult,
    TensorEdge,
    graph_fingerprint,
    hardware_fingerprint,
    hardware_from_dict,
    hardware_to_dict,
    load_placed_op_graph,
    load_search_result,
    placed_from_search_result,
    save_placed_op_graph,
    save_search_result,
)


def _node(node_id: str, op_type: str) -> OpNode:
    return OpNode(
        id=node_id,
        op_type=op_type,
        out_bytes=16.0,
        extra={
            "output_shape": (8,),
            "dtype": "bf16",
            "phase": "forward",
            "layer": 0,
        },
    )


def _graph(reverse_insertion: bool = False) -> OpGraph:
    graph = OpGraph(name="artifact-swiglu")
    nodes = [
        _node("gate", "matmul"),
        _node("up", "matmul"),
        _node("silu", "silu"),
        _node("mul", "mul"),
    ]
    for node in reversed(nodes) if reverse_insertion else nodes:
        graph.add_node(node)
    edges = [
        TensorEdge("gate", "silu", 16.0),
        TensorEdge("silu", "mul", 16.0),
        TensorEdge("up", "mul", 16.0),
    ]
    for edge in reversed(edges) if reverse_insertion else edges:
        graph.add_edge(edge)
    return graph


def _hardware(max_mem: float = 1024.0) -> HardwareSpec:
    return HardwareSpec(
        name="two-devices",
        devices=[
            DeviceSpec(1e12, 1e11, max_mem, name="bw0"),
            DeviceSpec(8e11, 8e10, max_mem, name="bw1"),
        ],
        comm_bw=1e10,
        comm_latency=1e-6,
    )


def _fused_plan() -> Plan:
    return Plan(
        name="fused",
        extra={"search_space": "legal_fusion_v1"},
        groups=[
            FusionGroup(
                ["gate"],
                0,
                {"pattern_id": "unfused", "backend": "reference"},
            ),
            FusionGroup(
                ["up"],
                1,
                {"pattern_id": "unfused", "backend": "reference"},
            ),
            FusionGroup(
                ["silu", "mul"],
                1,
                {
                    "pattern_id": "silu_mul",
                    "backend": "template",
                    "implementation_status": "planned",
                    "workspace_bytes": 32.0,
                },
            ),
        ],
    )


def _reference_plan() -> Plan:
    return Plan(
        name="reference",
        groups=[
            FusionGroup(
                [node_id],
                index % 2,
                {"pattern_id": "unfused", "backend": "reference"},
            )
            for index, node_id in enumerate(["gate", "up", "silu", "mul"])
        ],
    )


def _result(max_mem: float = 1024.0) -> SearchResult:
    return SearchResult(
        graph=_graph(),
        hardware=_hardware(max_mem),
        objective={"metric": "predicted_makespan", "direction": "minimize"},
        constraints={},
        search_method="beam",
        budget={"beam": 8, "rollouts": 3},
        seed=7,
        candidates=[
            SearchCandidate(1, _fused_plan(), 1.0, (128.0, 256.0)),
            SearchCandidate(2, _reference_plan(), 1.2, (128.0, 192.0)),
        ],
        baselines=[],
    )


class TestFingerprints(unittest.TestCase):
    def test_graph_hash_ignores_insertion_order(self):
        self.assertEqual(graph_fingerprint(_graph()), graph_fingerprint(_graph(True)))

    def test_hardware_hash_survives_json_numeric_round_trip(self):
        hardware = _hardware(max_mem=16 * 1024**3)
        restored = hardware_from_dict(hardware_to_dict(hardware))
        self.assertEqual(
            hardware_fingerprint(hardware),
            hardware_fingerprint(restored),
        )


class TestSearchResultArtifact(unittest.TestCase):
    def test_round_trip_is_deterministic_and_preserves_plan_extra(self):
        result = _result()
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            save_search_result(first, result)
            restored = load_search_result(first)
            save_search_result(second, restored)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(restored.seed, 7)
        self.assertEqual(restored.candidates[0].rank, 1)
        self.assertEqual(
            restored.candidates[0].plan.extra["search_space"],
            "legal_fusion_v1",
        )

    def test_tampered_graph_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "search.json"
            save_search_result(path, _result())
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["inputs"]["graph"]["nodes"][0]["flops"] = 1.0
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "graph_hash"):
                load_search_result(path)

    def test_memory_overflow_is_rejected(self):
        result = _result(max_mem=200.0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "search.json"
            with self.assertRaisesRegex(ValueError, "exceeds device memory"):
                save_search_result(path, result)

    def test_non_contiguous_ranks_are_rejected(self):
        result = _result()
        result.candidates[1].rank = 3
        with TemporaryDirectory() as directory:
            path = Path(directory) / "search.json"
            with self.assertRaisesRegex(ValueError, "candidate ranks"):
                save_search_result(path, result)


class TestPlacedOpGraphArtifact(unittest.TestCase):
    def test_node_annotations_and_round_trip(self):
        result = _result()
        placed = placed_from_search_result(result, rank=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "placed.json"
            save_placed_op_graph(path, placed)
            raw = json.loads(path.read_text(encoding="utf-8"))
            restored = load_placed_op_graph(path)

        node_by_id = {node["id"]: node for node in raw["graph"]["nodes"]}
        self.assertEqual(node_by_id["silu"]["fusion_group_id"], "fg0002")
        self.assertEqual(node_by_id["mul"]["device_id"], 1)
        self.assertEqual(raw["fusion_groups"][2]["pattern_id"], "silu_mul")
        self.assertEqual(restored.candidate_id, placed.candidate_id)
        self.assertEqual(restored.plan.extra, placed.plan.extra)

    def test_tampered_node_annotation_is_rejected(self):
        placed = PlacedOpGraph(_graph(), _hardware(), _fused_plan())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "placed.json"
            save_placed_op_graph(path, placed)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["graph"]["nodes"][0]["device_id"] = 99
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "annotation mismatch"):
                load_placed_op_graph(path)


if __name__ == "__main__":
    unittest.main()
