"""训练标签生成（打标签）：随机 + 启发式方案 -> 仿真器 -> (图, 方案, makespan, 显存)。"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .execution import simulate
from .generators import generate_plans
from .hardware import HardwareSpec
from .schema import FusionGroup, OpGraph, OpNode, Plan, TensorEdge


def graph_to_dict(graph: OpGraph) -> Dict:
    return {
        "name": graph.name,
        "extra": graph.extra,
        "nodes": [
            {
                "id": n.id,
                "op_type": n.op_type,
                "flops": n.flops,
                "out_bytes": n.out_bytes,
                "param_bytes": n.param_bytes,
                "extra": n.extra,
            }
            for n in graph.nodes.values()
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "bytes": e.bytes, "extra": e.extra}
            for e in graph.edges
        ],
    }


def plan_to_dict(plan: Plan) -> Dict:
    return {
        "name": plan.name,
        "groups": [
            {"device": g.device, "node_ids": list(g.node_ids), "extra": g.extra}
            for g in plan.groups
        ],
    }


def graph_from_dict(data: Dict) -> OpGraph:
    """从 graph_to_dict 的字典还原 OpGraph。"""

    g = OpGraph(name=data.get("name", "graph"), extra=data.get("extra", {}))
    for nd in data["nodes"]:
        g.add_node(
            OpNode(
                nd["id"],
                op_type=nd.get("op_type", "generic"),
                flops=nd.get("flops", 0.0),
                out_bytes=nd.get("out_bytes", 0.0),
                param_bytes=nd.get("param_bytes", 0.0),
                extra=nd.get("extra", {}),
            )
        )
    for ed in data["edges"]:
        g.add_edge(
            TensorEdge(
                ed["src"],
                ed["dst"],
                bytes=ed.get("bytes"),
                extra=ed.get("extra", {}),
            )
        )
    return g


def plan_from_dict(data: Dict) -> Plan:
    """从 plan_to_dict 的字典还原 Plan。"""

    groups = [
        FusionGroup(list(gd["node_ids"]), gd["device"], extra=gd.get("extra", {}))
        for gd in data["groups"]
    ]
    return Plan(groups, name=data.get("name", "plan"))


def build_label_records(
    graphs: List[OpGraph],
    hw: HardwareSpec,
    plans_per_graph: int = 8,
    random_ratio: float = 0.75,
    random_merge_prob: float = 0.5,
    heuristic_modes: List[str] = ("threshold", "chain"),
    threshold_bytes: float = 0.0,
    seed: int = 0,
    graph_id_offset: int = 0,
) -> List[Dict]:
    """对每个图生成随机 + 启发式方案，跑仿真器，返回带标签的记录列表。"""

    records: List[Dict] = []
    for gi, graph in enumerate(graphs):
        n_random = max(0, int(round(plans_per_graph * random_ratio)))
        plans = generate_plans(
            graph,
            hw,
            n_random=n_random,
            random_merge_prob=random_merge_prob,
            heuristic_modes=heuristic_modes,
            threshold_bytes=threshold_bytes,
            seed=seed + gi,
        )
        for plan in plans:
            res = simulate(graph, plan, hw, collect_timeline=False)
            records.append(
                {
                    "graph_id": gi + graph_id_offset,
                    "graph": graph_to_dict(graph),
                    "plan": plan_to_dict(plan),
                    "makespan": res.makespan,
                    "peak_mem": {str(d): v for d, v in res.peak_mem.items()},
                    "mem_overflow": res.mem_overflow,
                    "overflow_devices": res.overflow_devices,
                    "num_groups": len(plan.groups),
                    "num_nodes": len(graph.nodes),
                }
            )
    return records


def generate_label_dataset(
    graphs: List[OpGraph],
    hw: HardwareSpec,
    out_jsonl: str,
    **kwargs,
) -> List[Dict]:
    """生成标签并写入 JSONL 文件，返回记录列表。"""

    records = build_label_records(graphs, hw, **kwargs)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_labels(path: str) -> List[Dict]:
    """从 JSONL 读取标签。"""

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
