"""自动并行策略仿真器（方向 1 · 模块二）。

包含：统一输入图格式（schema）、硬件参数（hardware）、
roofline + 事件队列执行模型（execution）、随机/启发式方案生成（generators）、
训练标签生成（dataset）。
"""

from .schema import OpNode, TensorEdge, OpGraph, FusionGroup, Plan
from .hardware import (
    DeviceSpec,
    HardwareSpec,
    hardware_from_dict,
    hardware_to_dict,
    load_hardware,
    save_hardware,
)
from .validate import validate_plan
from .execution import simulate, SimResult, GroupStats, TransferStats
from .shape_model import compute_efficiency, group_compute_time, resolve_shape_model
from .generators import (
    random_plan,
    heuristic_plan,
    alternating_plan,
    random_topological_order,
    greedy_topological_order,
)
from .dataset import (
    graph_to_dict,
    graph_from_dict,
    plan_to_dict,
    plan_from_dict,
    build_label_records,
    generate_label_dataset,
    load_labels,
)
from .graphs import (
    make_chain_graph,
    make_diamond_graph,
    make_mlp_block,
    make_random_dag,
    make_transformer_block,
    make_dataset_graphs,
    make_extrapolation_graphs,
)
from .model_graphs import (
    TransformerModelSpec,
    build_decoder_layer,
    build_model_graph,
    build_pipeline_stages,
    build_stage_graph,
    slice_pipeline_stage,
)

__all__ = [
    "OpNode",
    "TensorEdge",
    "OpGraph",
    "FusionGroup",
    "Plan",
    "DeviceSpec",
    "HardwareSpec",
    "hardware_from_dict",
    "hardware_to_dict",
    "load_hardware",
    "save_hardware",
    "validate_plan",
    "simulate",
    "SimResult",
    "GroupStats",
    "TransferStats",
    "compute_efficiency",
    "group_compute_time",
    "resolve_shape_model",
    "random_plan",
    "heuristic_plan",
    "alternating_plan",
    "random_topological_order",
    "greedy_topological_order",
    "graph_to_dict",
    "graph_from_dict",
    "plan_to_dict",
    "plan_from_dict",
    "build_label_records",
    "generate_label_dataset",
    "load_labels",
    "make_chain_graph",
    "make_diamond_graph",
    "make_mlp_block",
    "make_random_dag",
    "make_transformer_block",
    "make_dataset_graphs",
    "make_extrapolation_graphs",
    "TransformerModelSpec",
    "build_decoder_layer",
    "build_model_graph",
    "build_pipeline_stages",
    "build_stage_graph",
    "slice_pipeline_stage",
]
