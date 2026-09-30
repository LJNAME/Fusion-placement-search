"""算子形状相关的效率模型（仿真器 1.1：形状感知 roofline）。

动机
----
朴素 roofline 只用一个峰值算力和一个峰值带宽，隐含假定任意 kernel 都能打满
峰值。真实硬件里，算子能达到的"有效算力"取决于算子类型和输入形状，其中最
典型的是 GEMM（matmul）：

- 规约维 K 越大，算术强度越高，越接近峰值算力；
- 小 K 的 GEMM 访存受限，有效算力显著低于峰值。

于是仿真器不再与"峰值 roofline"同构：解析式 roofline 是理论上界，形状感知
模型刻画真实可达性能，两者会系统性分离。

文献依据
--------
- Roofline: Williams, Waterman, Patterson, CACM 2009（理论上界）；
- nn-Meter: MobiSys 2021（kernel 级形状相关延迟预测，优于解析式）；
- Habitat: USENIX ATC 2021（跨 GPU 的逐迭代运行时预测）；
- IEEE TPDS 2024 综述（解析式 vs 学习式性能建模）。

当前只对 matmul 的 K 做饱和曲线，参数是占位假设；拿到国产卡后按
configs/README.md 的微基准流程逐算子校准，代码不用改。其他算子类型和
访存侧的形状效应预留为后续校准项。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .hardware import HardwareSpec
from .schema import OpGraph


# 默认形状效率曲线。config 顶层 "shape_model" 会浅合并覆盖这里。
DEFAULT_SHAPE_MODEL: Dict[str, Dict] = {
    "matmul": {
        "compute": {"curve": "saturate_k", "eta_max": 0.9, "k_half": 512.0},
    },
}


def _clamp_eta(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _saturate(x: float, x_half: float, eta_max: float) -> float:
    x = max(float(x), 0.0)
    x_half = max(float(x_half), 1e-12)
    return _clamp_eta(eta_max) * x / (x + x_half)


def resolve_shape_model(hw: HardwareSpec) -> Dict[str, Dict]:
    """合并默认曲线与 config 里的 shape_model（浅合并各算子的 compute）。"""

    cfg = (hw.extra or {}).get("shape_model") or {}
    out: Dict[str, Dict] = {}
    for key in set(DEFAULT_SHAPE_MODEL) | set(cfg):
        base = DEFAULT_SHAPE_MODEL.get(key, {})
        over = cfg.get(key, {})
        if not isinstance(over, dict):
            over = {}
        out[key] = {
            "compute": {**base.get("compute", {}), **(over.get("compute") or {})},
        }
    return out


def _shape_of(node) -> Dict:
    return (node.extra or {}).get("shape") or {}


def _matmul_k(node) -> float:
    """取 matmul 的规约维 K；缺省时由 flops/out_bytes 反推（fp32）。"""

    k = _shape_of(node).get("k")
    if k is not None and float(k) > 0:
        return float(k)
    ob = float(node.out_bytes)
    if ob > 0:
        # flops = 2*m*n*k, out_bytes = m*n*4 -> k = 2*flops/out_bytes
        return 2.0 * float(node.flops) / ob
    return 1.0


def compute_efficiency(hw: HardwareSpec, node) -> float:
    """节点的有效算力系数 eta ∈ (0, 1]，默认 1.0（无形状修正）。"""

    model = resolve_shape_model(hw)
    entry = model.get(node.op_type) or model.get("default") or {}
    spec = entry.get("compute") or {}
    curve = spec.get("curve", "const")
    if curve == "saturate_k":
        return _saturate(
            _matmul_k(node),
            spec.get("k_half", 512.0),
            spec.get("eta_max", 0.9),
        )
    if curve == "saturate_bytes":
        return _saturate(
            float(node.out_bytes),
            spec.get("bytes_half", 1e6),
            spec.get("eta_max", 0.9),
        )
    if curve == "table":
        # 实测数据点（真机校准产物）：[[x, eta], ...]，x 升序，线性插值。
        pts = sorted((float(p[0]), _clamp_eta(p[1])) for p in (spec.get("points") or []))
        if not pts:
            return _clamp_eta(spec.get("eta", 1.0))
        x = _matmul_k(node) if node.op_type == "matmul" else float(node.out_bytes)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return _clamp_eta(float(np.interp(x, xs, ys)))
    return _clamp_eta(spec.get("eta", 1.0))


def group_compute_time(graph: OpGraph, node_ids, dev, hw: HardwareSpec) -> float:
    """形状感知的融合组计算时间 = sum(每个算子 flops / (峰值算力 * eta))。"""

    if dev.flops <= 0:
        return 0.0
    total = 0.0
    for nid in node_ids:
        node = graph.nodes[nid]
        rate = dev.flops * compute_efficiency(hw, node)
        total += float(node.flops) / rate if rate > 0 else 0.0
    return total
