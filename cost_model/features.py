"""手写特征：图 + 方案 -> 定长特征向量（不跑仿真器，供基线模型用）。"""

from __future__ import annotations

import numpy as np

from simulator.hardware import HardwareSpec
from simulator.schema import OpGraph, Plan, edge_bytes, group_boundary_io


def compute_plan_features(graph: OpGraph, plan: Plan, hw: HardwareSpec) -> np.ndarray:
    """把 (图, 方案, 硬件) 编码为定长特征向量。

    覆盖：规模、计算量、参数/张量字节、融合组边界 IO、跨设备通信量、
    融合程度、组大小分布、设备负载均衡、硬件能力、开销总量。
    """

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    n_groups = len(plan.groups)
    total_flops = sum(n.flops for n in graph.nodes.values())
    total_param = sum(n.param_bytes for n in graph.nodes.values())
    total_out = sum(n.out_bytes for n in graph.nodes.values())

    in_io = 0.0
    out_io = 0.0
    group_sizes: list[int] = []
    group_devs: list[int] = []
    group_est: list[float] = []
    total_overhead = 0.0
    for g in plan.groups:
        ib, ob = group_boundary_io(graph, g.node_ids)
        in_io += ib
        out_io += ob
        flops_g = sum(graph.nodes[x].flops for x in g.node_ids)
        dev = hw.devices[g.device]
        est = max(flops_g / dev.flops, (ib + ob) / dev.mem_bw)
        launch = hw.launch_overhead + hw.fusion_overhead * max(0, len(g.node_ids) - 1)
        est += launch
        group_sizes.append(len(g.node_ids))
        group_devs.append(g.device)
        group_est.append(est)
        total_overhead += launch

    comm_bytes = 0.0
    n_transfers = 0
    for e in graph.edges:
        sg = plan.group_of(e.src)
        dg = plan.group_of(e.dst)
        if (
            sg is not None
            and dg is not None
            and sg != dg
            and plan.groups[sg].device != plan.groups[dg].device
        ):
            comm_bytes += edge_bytes(e, graph)
            n_transfers += 1

    dev_load = [0.0] * hw.n_devs
    for est, d in zip(group_est, group_devs):
        dev_load[d] += est
    mean_load = float(np.mean(dev_load))
    max_load = float(np.max(dev_load))
    imbalance = max_load / (mean_load + 1e-12)

    avg_dev_flops = float(np.mean([d.flops for d in hw.devices]))
    avg_dev_bw = float(np.mean([d.mem_bw for d in hw.devices]))
    max_group = max(group_sizes) if group_sizes else 0
    mean_group = float(np.mean(group_sizes)) if group_sizes else 0.0
    total_io = in_io + out_io + 1e-12

    feats = [
        float(np.log1p(n_nodes)),
        float(np.log1p(n_edges)),
        float(np.log1p(n_groups)),
        n_groups / max(n_nodes, 1),
        float(np.log1p(total_flops)),
        float(np.log1p(total_param)),
        float(np.log1p(total_out)),
        float(np.log1p(in_io)),
        float(np.log1p(out_io)),
        float(np.log1p(comm_bytes)),
        float(np.log1p(n_transfers)),
        max_group / max(n_nodes, 1),
        mean_group,
        imbalance,
        hw.n_devs,
        float(np.log1p(avg_dev_flops)),
        float(np.log1p(avg_dev_bw)),
        float(np.log1p(total_overhead)),
        float(np.log1p(max(group_est) if group_est else 0.0)),
        comm_bytes / total_io,
    ]
    return np.asarray(feats, dtype=float)


def make_targets(record: dict) -> np.ndarray:
    """从标签记录提取对数空间目标：[log makespan, log 峰值显存]。"""

    makespan = float(record["makespan"])
    peak = max(float(v) for v in record["peak_mem"].values())
    return np.asarray([np.log(makespan), np.log(peak)], dtype=float)
