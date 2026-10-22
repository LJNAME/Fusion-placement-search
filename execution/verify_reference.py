"""CPU verification entry point for the M5 reference execution loop."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from simulator import (
    DeviceSpec,
    FusionGroup,
    HardwareSpec,
    OpGraph,
    OpNode,
    PlacedOpGraph,
    Plan,
    TensorEdge,
    load_placed_op_graph,
    make_reference_plan,
    save_placed_op_graph,
)

from .reference import execute_reference_plan, save_execution_report


def _node(
    node_id: str,
    op_type: str,
    output_shape: tuple[int, ...],
    **extra: object,
) -> OpNode:
    metadata = {"output_shape": output_shape, "dtype": "fp32", "layer": 0, **extra}
    return OpNode(node_id, op_type=op_type, out_bytes=4.0, extra=metadata)


def _placed() -> PlacedOpGraph:
    graph = OpGraph(name="m5-reference-smoke", extra={"output_node": "add"})
    graph.add_node(_node("input", "input", (2, 4)))
    graph.add_node(_node("matmul", "matmul", (2, 4), shape={"m": 2, "n": 4, "k": 4}))
    graph.add_node(_node("silu", "silu", (2, 4)))
    graph.add_node(_node("add", "add", (2, 4)))
    graph.add_edge(TensorEdge("input", "matmul"))
    graph.add_edge(TensorEdge("matmul", "silu"))
    graph.add_edge(TensorEdge("silu", "add"))
    graph.add_edge(TensorEdge("input", "add"))
    hardware = HardwareSpec(
        [
            DeviceSpec(1e12, 1e11, 1e9, name="cpu0"),
            DeviceSpec(1e12, 1e11, 1e9, name="cpu1"),
        ],
        name="m5-cpu",
    )
    plan = make_reference_plan(graph)
    for index, group in enumerate(plan.groups):
        group.device = index % 2
    return PlacedOpGraph(graph=graph, hardware=hardware, plan=plan)


def _verify_planned_rejection() -> None:
    graph = OpGraph(name="planned-rejection")
    graph.add_node(_node("silu", "silu", (2, 4)))
    graph.add_node(_node("mul", "mul", (2, 4)))
    graph.add_edge(TensorEdge("silu", "mul"))
    plan = Plan(
        [
            FusionGroup(
                ["silu", "mul"],
                0,
                extra={
                    "pattern_id": "silu_mul",
                    "backend": "template",
                    "implementation_status": "planned",
                },
            )
        ]
    )
    placed = PlacedOpGraph(
        graph,
        HardwareSpec([DeviceSpec(1e12, 1e11, 1e9)]),
        plan,
    )
    try:
        execute_reference_plan(placed, ["cpu"], warmup_iterations=0, measured_iterations=1)
    except ValueError as exc:
        if "no executable backend" not in str(exc):
            raise
    else:
        raise AssertionError("planned fusion pattern was accepted by the executor")


def verify_reference() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        placed_path = Path(directory) / "placed.json"
        report_path = Path(directory) / "report.json"
        save_placed_op_graph(placed_path, _placed())
        placed = load_placed_op_graph(placed_path)
        report = execute_reference_plan(
            placed,
            ["cpu", "cpu"],
            warmup_iterations=1,
            measured_iterations=2,
        )
        save_execution_report(report_path, report)
        if not report_path.is_file():
            raise AssertionError("execution report was not written")
    _verify_planned_rejection()
    if report.output_shape != (2, 4) or len(report.groups) != 4:
        raise AssertionError(f"unexpected report: {report}")
    if any(group.transfer_median_s != 0.0 for group in report.groups):
        raise AssertionError("same-device CPU execution reported a transfer")
    return {
        "groups": len(report.groups),
        "devices": ",".join(report.devices),
        "output_shape": report.output_shape,
        "planned_pattern_rejected": True,
    }


def main() -> None:
    for key, value in verify_reference().items():
        print(f"{key}: {value}")
    print("Fusion M5 reference execution verification: OK")


if __name__ == "__main__":
    main()
