"""Legacy log-space roofline baseline kept for historical experiments.

makespan ≈ max(每设备组时长累计) + 总线传输总耗时
峰值显存 ≈ max(每设备：参数 + 组边界输出字节)

忽略调度/竞争细节，是"传统解析式代价模型"的代表；
New code should use ``StructuralCostModel`` and its physical-unit
``CostEstimate`` output.  This class retains the original numpy log-vector API
so historical datasets and comparison scripts remain reproducible.
"""

from __future__ import annotations

import numpy as np

from simulator.schema import edge_bytes, group_boundary_io


class RooflineCostModel:
    def __init__(self, hw):
        self.hw = hw

    def predict(self, graph, plan) -> np.ndarray:
        """返回 log 空间预测：[log makespan, log 峰值显存]。"""

        dev_load = [0.0] * self.hw.n_devs
        per_dev_mem = [0.0] * self.hw.n_devs
        for g in plan.groups:
            ib, ob = group_boundary_io(graph, g.node_ids)
            flops = sum(graph.nodes[x].flops for x in g.node_ids)
            params = sum(graph.nodes[x].param_bytes for x in g.node_ids)
            dev = self.hw.devices[g.device]
            dur = (
                max(flops / dev.flops, (ib + ob) / dev.mem_bw)
                + self.hw.launch_overhead
                + self.hw.fusion_overhead * max(0, len(g.node_ids) - 1)
            )
            dev_load[g.device] += dur
            per_dev_mem[g.device] += params + ob

        bus = 0.0
        for e in graph.edges:
            sg = plan.group_of(e.src)
            dg = plan.group_of(e.dst)
            if (
                sg is not None
                and dg is not None
                and sg != dg
                and plan.groups[sg].device != plan.groups[dg].device
            ):
                bus += edge_bytes(e, graph) / self.hw.comm_bw + self.hw.comm_latency

        makespan = max(dev_load) + bus
        peak = max(per_dev_mem)
        return np.asarray(
            [np.log(max(makespan, 1e-12)), np.log(max(peak, 1e-12))], dtype=float
        )
