"""Run a placed operator graph through the Torch reference backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulator import load_placed_op_graph

from .reference import execute_reference_plan, save_execution_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--devices", default="0,1", help="physical CUDA/HIP ids, or cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _runtime_devices(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("--devices must not be empty")
    return tuple("cpu" if item == "cpu" else f"cuda:{int(item)}" for item in items)


def main() -> None:
    args = _parse_args()
    placed = load_placed_op_graph(args.artifact)
    report = execute_reference_plan(
        placed,
        _runtime_devices(args.devices),
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
        seed=args.seed,
    )
    save_execution_report(args.output, report)
    print(f"scope: {report.scope}")
    print(f"groups: {len(report.groups)}")
    print(f"predicted_makespan_ms: {None if report.predicted_makespan_s is None else report.predicted_makespan_s * 1e3}")
    print(f"total_median_ms: {report.total_median_s * 1e3:.6f}")
    print(f"peak_memory_bytes: {list(report.peak_memory_bytes)}")
    print(f"output_shape: {report.output_shape}")
    print(f"report: {args.output}")
    print("Fusion M5 reference execution: OK")


if __name__ == "__main__":
    main()
