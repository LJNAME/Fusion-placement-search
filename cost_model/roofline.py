"""解析式 roofline 代价模型（对比基线 A：不学习，纯公式）。

makespan ≈ max(每设备组时长累计) + 总线传输总耗时
峰值显存 ≈ max(每设备：参数 + 组边界输出字节)

忽略调度/竞争细节，是"传统解析式代价模型"的代表；
与可学习模型对比可回答"结构感知的代价模型值多少钱"。
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
