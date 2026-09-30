"""合成算子图生成器：为代价模型 Stage 0 提供多样化的训练图。

覆盖链式 / MLP / Transformer 块 / 随机 DAG 四类结构，
节点属性按真实计算量估算（FLOPs、张量字节、参数字节）。
"""

from __future__ import annotations

import random
from typing import List, Optional

from .schema import OpGraph, OpNode, TensorEdge


def make_chain_graph(
    n_ops: int = 8,
    flops_range: tuple[float, float] = (1e8, 1e10),
    out_bytes_range: tuple[float, float] = (1e6, 2e8),
    seed: int = 0,
) -> OpGraph:
    """链式图：A -> B -> C -> ...（最朴素的融合场景）。"""

    rng = random.Random(seed)
    g = OpGraph(name=f"chain-{n_ops}")
    prev = None
    for i in range(n_ops):
        nid = f"op{i}"
        g.add_node(
            OpNode(
                nid,
                op_type="generic",
                flops=rng.uniform(*flops_range),
                out_bytes=rng.uniform(*out_bytes_range),
            )
        )
        if prev is not None:
            g.add_edge(TensorEdge(prev, nid, bytes=g.nodes[prev].out_bytes))
        prev = nid
    return g


def make_mlp_block(
    hidden_sizes: tuple[int, ...] = (1024, 4096, 1024),
    batch: int = 64,
    seed: int = 0,
) -> OpGraph:
    """MLP 块：Linear -> GELU -> Linear -> GELU ...，带输入节点。"""

    rng = random.Random(seed)
    g = OpGraph(name="mlp")
    g.add_node(
        OpNode(
            "input",
            op_type="input",
            flops=0.0,
            out_bytes=batch * hidden_sizes[0] * 4,
            extra={"shape": {"n": batch * hidden_sizes[0]}},
        )
    )
    cur = "input"
    prev = hidden_sizes[0]
    for i, h in enumerate(hidden_sizes[1:]):
        lin = f"linear{i}"
        g.add_node(
            OpNode(
                lin,
                op_type="matmul",
                flops=2.0 * batch * prev * h,
                out_bytes=batch * h * 4,
                param_bytes=(prev * h + h) * 4,
                extra={"shape": {"m": batch, "n": h, "k": prev}},
            )
        )
        g.add_edge(TensorEdge(cur, lin, bytes=batch * prev * 4))
        cur = lin

        act = f"gelu{i}"
        g.add_node(
            OpNode(
                act,
                op_type="gelu",
                flops=5.0 * batch * h,
                out_bytes=batch * h * 4,
                extra={"shape": {"n": batch * h}},
            )
        )
        g.add_edge(TensorEdge(cur, act, bytes=batch * h * 4))
        cur = act
        prev = h
    return g


def make_transformer_block(
    d_model: int = 1024,
    seq_len: int = 1024,
    batch: int = 2,
    ffn_dim: int = 4096,
    heads: int = 16,
    seed: int = 0,
) -> OpGraph:
    """Transformer 块：LN -> 并行 Q/K/V -> Attention -> 输出投影 -> FFN，
    含残差连接（add 有两个输入边），结构上存在可并行分支。
    """

    rng = random.Random(seed)
    g = OpGraph(name="transformer-block")
    b = batch
    s = seq_len
    d = d_model
    f = ffn_dim
    h = heads
    f32 = 4

    def add_op(nid, op_type, flops, out_bytes, param_bytes=0.0, shape=None):
        g.add_node(
            OpNode(
                nid,
                op_type=op_type,
                flops=flops,
                out_bytes=out_bytes,
                param_bytes=param_bytes,
                extra={"shape": shape} if shape else {},
            )
        )

    add_op("input", "input", 0.0, b * s * d * f32, shape={"n": b * s * d})
    add_op("ln1", "layernorm", 5.0 * b * s * d, b * s * d * f32, 2 * d * f32, {"n": b * s * d})
    add_op("q_proj", "matmul", 2.0 * b * s * d * d, b * s * d * f32, d * d * f32, {"m": b * s, "n": d, "k": d})
    add_op("k_proj", "matmul", 2.0 * b * s * d * d, b * s * d * f32, d * d * f32, {"m": b * s, "n": d, "k": d})
    add_op("v_proj", "matmul", 2.0 * b * s * d * d, b * s * d * f32, d * d * f32, {"m": b * s, "n": d, "k": d})
    add_op("qk_matmul", "matmul", 2.0 * b * s * s * d, b * s * s * f32, shape={"m": b * s, "n": s, "k": d})
    add_op("softmax", "softmax", 5.0 * b * s * s, b * s * s * f32, shape={"n": b * s * s})
    add_op("av_matmul", "matmul", 2.0 * b * s * s * d, b * s * d * f32, shape={"m": b * s, "n": d, "k": s})
    add_op("out_proj", "matmul", 2.0 * b * s * d * d, b * s * d * f32, d * d * f32, {"m": b * s, "n": d, "k": d})
    add_op("add1", "add", 2.0 * b * s * d, b * s * d * f32, shape={"n": b * s * d})
    add_op("ffn1", "matmul", 2.0 * b * s * d * f, b * s * f * f32, (d * f + f) * f32, {"m": b * s, "n": f, "k": d})
    add_op("ffn_gelu", "gelu", 5.0 * b * s * f, b * s * f * f32, shape={"n": b * s * f})
    add_op("ffn2", "matmul", 2.0 * b * s * f * d, b * s * d * f32, (f * d + d) * f32, {"m": b * s, "n": d, "k": f})
    add_op("add2", "add", 2.0 * b * s * d, b * s * d * f32, shape={"n": b * s * d})
    add_op("ln2", "layernorm", 5.0 * b * s * d, b * s * d * f32, 2 * d * f32, {"n": b * s * d})

    def edge(src, dst, bytes_=None):
        g.add_edge(TensorEdge(src, dst, bytes=bytes_ if bytes_ is not None else g.nodes[src].out_bytes))

    edge("input", "ln1")
    edge("ln1", "q_proj")
    edge("ln1", "k_proj")
    edge("ln1", "v_proj")
    edge("q_proj", "qk_matmul")
    edge("k_proj", "qk_matmul")
    edge("qk_matmul", "softmax")
    edge("softmax", "av_matmul")
    edge("v_proj", "av_matmul")
    edge("av_matmul", "out_proj")
    edge("out_proj", "add1")
    edge("input", "add1")  # 残差
    edge("add1", "ffn1")
    edge("ffn1", "ffn_gelu")
    edge("ffn_gelu", "ffn2")
    edge("ffn2", "add2")
    edge("input", "add2")  # 残差
    edge("add2", "ln2")
    return g


def make_random_dag(
    n_nodes: int = 14,
    edge_prob: float = 0.3,
    flops_range: tuple[float, float] = (1e8, 1e10),
    out_bytes_range: tuple[float, float] = (1e6, 2e8),
    seed: int = 0,
) -> OpGraph:
    """随机 DAG：边只允许从小编号指向大编号（天然无环），
    且保证每个节点至少有一个前驱（输入节点除外）。
    """

    rng = random.Random(seed)
    g = OpGraph(name=f"dag-{n_nodes}")
    types = ["matmul", "gelu", "add", "layernorm", "softmax"]
    for i in range(n_nodes):
        t = rng.choice(types)
        f = rng.uniform(*flops_range)
        ob = rng.uniform(*out_bytes_range)
        if t == "matmul":
            mn = max(ob / 4.0, 1.0)
            shape = {"m": mn**0.5, "n": mn**0.5, "k": 2.0 * f / ob}
        else:
            shape = {"n": ob / 4.0}
        g.add_node(
            OpNode(
                f"n{i}",
                op_type=t,
                flops=f,
                out_bytes=ob,
                extra={"shape": shape},
            )
        )
    incoming = {i: 0 for i in range(n_nodes)}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                g.add_edge(TensorEdge(f"n{i}", f"n{j}", bytes=g.nodes[f"n{i}"].out_bytes))
                incoming[j] += 1
    for j in range(1, n_nodes):
        if incoming[j] == 0:
            i = rng.randrange(j)
            g.add_edge(TensorEdge(f"n{i}", f"n{j}", bytes=g.nodes[f"n{i}"].out_bytes))
    return g


def make_diamond_graph(
    stages: int = 4,
    branch_len: int = 3,
    flops_range: tuple[float, float] = (1e8, 1e10),
    out_bytes_range: tuple[float, float] = (1e6, 2e8),
    seed: int = 0,
) -> OpGraph:
    """菱形/分叉汇聚图：每级 split -> 两条并行链 -> join。

    制造关键路径与负载均衡的差异（straggler 场景）：两条分支可以并行，
    但 join 要等慢的那条；不同放置（同卡/异卡、链路是否共享）会改变排名，
    而 roofline 的 max(设备累计负载) + 总线总和 无法区分这类调度结构。
    """

    rng = random.Random(seed)
    g = OpGraph(name=f"diamond-s{stages}-b{branch_len}")
    prev: List[str] = []
    for st in range(stages):
        split = f"s{st}"
        g.add_node(
            OpNode(
                split,
                op_type="generic",
                flops=rng.uniform(*flops_range) * 0.1,
                out_bytes=rng.uniform(*out_bytes_range),
            )
        )
        for p in prev:
            g.add_edge(TensorEdge(p, split, bytes=g.nodes[p].out_bytes))
        prev = [split]
        for br in range(2):
            cur = split
            for k in range(branch_len):
                nid = f"b{st}_{br}_{k}"
                g.add_node(
                    OpNode(
                        nid,
                        op_type="generic",
                        flops=rng.uniform(*flops_range),
                        out_bytes=rng.uniform(*out_bytes_range),
                    )
                )
                g.add_edge(TensorEdge(cur, nid, bytes=g.nodes[cur].out_bytes))
                cur = nid
            prev.append(cur)
        join = f"j{st}"
        g.add_node(
            OpNode(
                join,
                op_type="generic",
                flops=rng.uniform(*flops_range) * 0.1,
                out_bytes=rng.uniform(*out_bytes_range),
            )
        )
        for p in prev:
            g.add_edge(TensorEdge(p, join, bytes=g.nodes[p].out_bytes))
        prev = [join]
    return g


def make_dataset_graphs(
    n_per_type: int = 20,
    seed: int = 0,
    with_transformer: bool = True,
) -> List[OpGraph]:
    """生成一组训练图：每类 n_per_type 个，尺寸随机但可复现。"""

    rng = random.Random(seed)
    graphs: List[OpGraph] = []
    s = seed
    for i in range(n_per_type):
        graphs.append(
            make_chain_graph(
                n_ops=rng.randint(6, 16),
                seed=s + i * 7,
            )
        )
        graphs.append(
            make_random_dag(
                n_nodes=rng.randint(10, 18),
                edge_prob=rng.uniform(0.2, 0.45),
                seed=s + i * 11 + 3,
            )
        )
        graphs.append(
            make_mlp_block(
                hidden_sizes=(rng.choice([512, 1024, 2048]), rng.choice([2048, 4096]), rng.choice([512, 1024, 2048])),
                batch=rng.choice([32, 64, 128]),
                seed=s + i * 13 + 5,
            )
        )
        if with_transformer:
            graphs.append(
                make_transformer_block(
                    d_model=rng.choice([512, 1024]),
                    seq_len=rng.choice([256, 512, 1024]),
                    batch=rng.choice([1, 2, 4]),
                    ffn_dim=rng.choice([2048, 4096]),
                    heads=16,
                    seed=s + i * 17 + 7,
                )
            )
    return graphs


def make_extrapolation_graphs(
    n_per_type: int = 10,
    seed: int = 0,
) -> List[OpGraph]:
    """外推测试图：结构类型与训练相同，但尺寸显著更大（更长/更深/更宽）。

    用于检验代价模型的结构外推能力（训练集从未见过这个规模）。
    """

    rng = random.Random(seed)
    graphs: List[OpGraph] = []
    s = seed
    for i in range(n_per_type):
        graphs.append(
            make_chain_graph(
                n_ops=rng.randint(24, 40),
                flops_range=(1e9, 5e10),
                out_bytes_range=(5e7, 1e9),
                seed=s + i * 23 + 11,
            )
        )
        graphs.append(
            make_random_dag(
                n_nodes=rng.randint(24, 36),
                edge_prob=rng.uniform(0.2, 0.4),
                flops_range=(1e9, 5e10),
                out_bytes_range=(5e7, 1e9),
                seed=s + i * 29 + 13,
            )
        )
        graphs.append(
            make_mlp_block(
                hidden_sizes=(4096, 16384, 4096),
                batch=128,
                seed=s + i * 31 + 17,
            )
        )
        graphs.append(
            make_transformer_block(
                d_model=2048,
                seq_len=rng.choice([2048, 4096]),
                batch=4,
                ffn_dim=8192,
                heads=32,
                seed=s + i * 37 + 19,
            )
        )
    return graphs
