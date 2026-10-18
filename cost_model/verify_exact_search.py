"""CPU verification entry point for M4b MCTS tuning against the exact oracle."""

from __future__ import annotations

from cost_model import (
    GroupCost,
    StructuralCostModel,
    TransferCost,
    exact_legal_search,
    legal_beam_search,
    legal_mcts_search,
    measure_optimality_gap,
)
from simulator import OpGraph, OpNode, TensorEdge


class _VerificationProfile:
    n_devs = 2
    overlap_send = True
    interference = 0.0

    def max_memory(self, device: int) -> float:
        return 1e9

    def estimate_group(self, graph, group) -> GroupCost:
        device_factor = 1.0 if group.device == 0 else 1.35
        fusion_factor = 0.65 if len(group.node_ids) > 1 else 1.0
        duration = device_factor * fusion_factor
        return GroupCost(duration, duration, 0.0, 0.0, 1.0, "verification")

    def estimate_transfer(self, src_device, dst_device, nbytes) -> TransferCost:
        return TransferCost(0.4, 1.0, "verification")

    def link_key(self, src_device, dst_device):
        return (0, 0)


def _node(node_id: str, op_type: str) -> OpNode:
    return OpNode(
        id=node_id,
        op_type=op_type,
        flops=100.0,
        out_bytes=64.0,
        extra={
            "output_shape": (32,),
            "dtype": "bf16",
            "phase": "forward",
            "layer": 0,
        },
    )


def _eight_node_graph() -> OpGraph:
    graph = OpGraph(name="m4a-eight-node")
    op_types = ("matmul", "silu", "mul", "add", "mul", "add", "mul", "matmul")
    for index, op_type in enumerate(op_types):
        graph.add_node(_node(f"n{index}", op_type))
        if index:
            graph.add_edge(TensorEdge(f"n{index - 1}", f"n{index}", 64.0))
    return graph


def verify_exact_search() -> dict[str, object]:
    graph = _eight_node_graph()
    model = StructuralCostModel()
    profile = _VerificationProfile()
    scorer = lambda plan: model.predict(graph, plan, profile)

    exact = exact_legal_search(
        graph,
        scorer,
        n_devs=2,
        top_k=5,
        max_nodes=15,
        max_evaluations=1_000_000,
    )
    beam = legal_beam_search(
        graph,
        lambda plan: scorer(plan).makespan,
        n_devs=2,
        beam=8,
        rollouts=3,
        fusion_probability=0.7,
        seed=23,
    )
    mcts_gaps = []
    for seed in (0, 1, 7, 23):
        mcts = legal_mcts_search(
            graph,
            lambda plan: scorer(plan).makespan,
            n_devs=2,
            iterations=100,
            rollouts=2,
            fusion_probability=0.7,
            seed=seed,
            n_best=8,
        )
        mcts_gaps.append(measure_optimality_gap(exact, mcts, scorer))
    beam_gap = measure_optimality_gap(exact, beam, scorer)

    if exact.evaluated_plans != exact.search_space_size:
        raise AssertionError("exact search did not evaluate the complete space")
    if beam_gap.candidate_score < exact.optimal_score:
        raise AssertionError("beam score is better than the exact optimum")
    if any(gap.candidate_score < exact.optimal_score for gap in mcts_gaps):
        raise AssertionError("MCTS score is better than the exact optimum")
    if any(gap.relative_gap > 1e-12 for gap in mcts_gaps):
        raise AssertionError("guided MCTS did not close the exact gap for all seeds")

    return {
        "nodes": len(graph.nodes),
        "search_space": exact.search_space_size,
        "evaluated": exact.evaluated_plans,
        "feasible": exact.feasible_plans,
        "optimal_score": f"{exact.optimal_score:.6f}",
        "optimal_groups": len(exact.best_plan.groups),
        "beam_gap": f"{beam_gap.relative_gap:.6f}",
        "mcts_seeds": "0,1,7,23",
        "mcts_max_gap": f"{max(gap.relative_gap for gap in mcts_gaps):.6f}",
    }


def main() -> None:
    result = verify_exact_search()
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Fusion M4b MCTS exact-gap verification: OK")


if __name__ == "__main__":
    main()
