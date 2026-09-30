"""模块一：数据层 —— 统一输入图格式与并行方案表示。

约定：
- 节点（算子）属性固定：id / op_type / flops / out_bytes / param_bytes，
  预留 extra 扩展字段；
- 边（张量依赖）属性固定：src / dst / bytes（None 时取 src.out_bytes），
  预留 extra 扩展字段；
- 并行方案 = 一组融合组（FusionGroup），每个融合组是拓扑序的连续片段，
  且整体拼接必须构成图的一个拓扑序。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OpNode:
    """计算图节点（算子）。"""

    id: str
    op_type: str = "generic"          # 算子类型（matmul / layernorm / ...）
    flops: float = 0.0                # 计算量（FLOPs）
    out_bytes: float = 0.0            # 输出张量大小（字节），0 表示无输出
    param_bytes: float = 0.0          # 参数/权重大小（字节），常驻设备显存
    extra: dict = field(default_factory=dict)  # 扩展字段（如算子名称、精度等）


@dataclass
class TensorEdge:
    """数据依赖边（张量）。"""

    src: str
    dst: str
    bytes: Optional[float] = None     # 通信/传输字节数，None -> src.out_bytes
    extra: dict = field(default_factory=dict)


def edge_bytes(edge: TensorEdge, graph: "OpGraph") -> float:
    """边实际传输的字节数。"""

    if edge.bytes is not None:
        return float(edge.bytes)
    return float(graph.nodes[edge.src].out_bytes)


@dataclass
class OpGraph:
    """统一计算图。"""

    nodes: Dict[str, OpNode] = field(default_factory=dict)
    edges: List[TensorEdge] = field(default_factory=list)
    name: str = "graph"
    extra: dict = field(default_factory=dict)

    def add_node(self, node: OpNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"重复节点: {node.id}")
        self.nodes[node.id] = node

    def add_edge(self, edge: TensorEdge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise ValueError(f"边引用了不存在的节点: {edge.src} -> {edge.dst}")
        if edge.src == edge.dst:
            raise ValueError(f"不允许自环: {edge.src}")
        self.edges.append(edge)

    def successors(self, node_id: str) -> List[str]:
        return [e.dst for e in self.edges if e.src == node_id]

    def predecessors(self, node_id: str) -> List[str]:
        return [e.src for e in self.edges if e.dst == node_id]

    def in_edges(self, node_id: str) -> List[TensorEdge]:
        return [e for e in self.edges if e.dst == node_id]

    def out_edges(self, node_id: str) -> List[TensorEdge]:
        return [e for e in self.edges if e.src == node_id]

    def topological_order(self) -> List[str]:
        """确定性拓扑序（Kahn + 节点 id 字典序）。"""

        indeg = {nid: len(self.in_edges(nid)) for nid in self.nodes}
        heap = [nid for nid in self.nodes if indeg[nid] == 0]
        heapq.heapify(heap)
        order: List[str] = []
        while heap:
            nid = heapq.heappop(heap)
            order.append(nid)
            for e in self.out_edges(nid):
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    heapq.heappush(heap, e.dst)
        if len(order) != len(self.nodes):
            raise ValueError("计算图存在环，无法得到拓扑序")
        return order


@dataclass
class FusionGroup:
    """融合组：一组按拓扑序排列、在同一设备上作为一个 kernel 执行的算子。"""

    node_ids: List[str]
    device: int                       # 设备索引（0..n_devs-1）
    extra: dict = field(default_factory=dict)


@dataclass
class Plan:
    """完整并行方案：融合组列表（按拓扑序拼接）。"""

    groups: List[FusionGroup]
    name: str = "plan"
    extra: dict = field(default_factory=dict)

    def nodes(self) -> List[str]:
        """全部节点（按执行顺序，即各融合组拼接）。"""

        order: List[str] = []
        for g in self.groups:
            order.extend(g.node_ids)
        return order

    def group_of(self, node_id: str) -> Optional[int]:
        for gi, g in enumerate(self.groups):
            if node_id in g.node_ids:
                return gi
        return None


def group_boundary_io(graph: OpGraph, node_ids: List[str]) -> tuple[float, float]:
    """融合组的边界 IO（字节）。

    - 输入：组内算子的参数 + 来自组外生产者的边张量；
    - 输出：组内产生的、被组外消费者使用的张量，以及图输出
      （无消费者的节点，写全局内存后保留到仿真结束）。
    """

    ids = set(node_ids)
    in_bytes = 0.0
    for nid in node_ids:
        in_bytes += graph.nodes[nid].param_bytes
    for e in graph.edges:
        if e.dst in ids and e.src not in ids:
            in_bytes += edge_bytes(e, graph)

    out_tensors = set()
    for e in graph.edges:
        if e.src in ids and e.dst not in ids:
            out_tensors.add(e.src)
    # 图输出（没有后继边的节点）同样写一次全局内存
    for nid in ids:
        if not any(e.src == nid for e in graph.edges):
            out_tensors.add(nid)
    out_bytes = sum(graph.nodes[nid].out_bytes for nid in out_tensors)
    return in_bytes, out_bytes
