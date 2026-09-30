"""硬件参数清单（对应笔记中的参数清单）。

算力、显存带宽、启动开销、融合开销、n_devs、max_mem，
以及通信模型：默认共享总线（单 FIFO），可选逐链路拓扑
（ASTRA-sim 参考：链路矩阵 + 逐链路 FIFO），以及计算-通信干扰系数
（Echo 参考：设备同时收发与计算时互相降速）。
设备支持异构（每台设备独立规格）。

支持从 JSON 配置文件加载（configs/ 目录），参数来源见 configs/README.md：
官方标称规格只作起点，正式使用需用微基准实测校准（综述盲区 4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceSpec:
    """单台设备规格（支持异构）。"""

    flops: float                      # 算力（FLOPs/s）
    mem_bw: float                     # 全局显存带宽（Bytes/s）
    max_mem: float                    # 显存上限（Bytes）
    name: str = ""                    # 设备名，如 "K100"、"M300"
    extra: dict = field(default_factory=dict)  # 扩展字段（厂商/精度/来源/校准标记等）


@dataclass
class HardwareSpec:
    """集群硬件规格。"""

    devices: List[DeviceSpec]
    comm_bw: float = 1e12             # 共享总线带宽（Bytes/s）
    comm_latency: float = 1e-6        # 每次传输延迟（s）
    launch_overhead: float = 0.0      # 每个 kernel（融合组）启动开销（s）
    fusion_overhead: float = 0.0      # 融合组内每多一个算子的额外开销（s）
    link_bw: Optional[float] = None   # 默认链路带宽；None 时逐链路退回 comm_bw
    link_matrix: Optional[List[List[Optional[float]]]] = None  # n×n 逐对链路带宽
    interference: float = 0.0         # 计算-通信干扰系数：设备有在途发送时计算降速比例
    overlap_send: bool = True         # 组级发送-计算重叠：True=发送与后续组计算重叠（流水），False=设备等自己的发送完成（主机同步拷贝/流序串行）
    name: str = ""                    # 集群名
    extra: dict = field(default_factory=dict)  # 扩展字段（说明/数据来源等）

    @property
    def n_devs(self) -> int:
        return len(self.devices)

    @property
    def has_links(self) -> bool:
        """是否启用逐链路拓扑（缺省时退回单条共享总线语义）。"""

        return self.link_bw is not None or self.link_matrix is not None

    def link_bandwidth(self, i: int, j: int) -> float:
        """设备 i -> j 的链路带宽：链路矩阵 > 默认链路 > 共享总线。"""

        if self.link_matrix is not None:
            try:
                v = self.link_matrix[i][j]
            except IndexError:
                v = None
            if v is not None and v > 0:
                return float(v)
        if self.link_bw is not None and self.link_bw > 0:
            return float(self.link_bw)
        return float(self.comm_bw)

    def __post_init__(self) -> None:
        if self.n_devs == 0:
            raise ValueError("至少需要一台设备")
        for d in self.devices:
            if d.flops <= 0:
                raise ValueError(f"设备 {d.name or '?'} 算力必须为正")
            if d.mem_bw <= 0:
                raise ValueError(f"设备 {d.name or '?'} 显存带宽必须为正")
            if d.max_mem <= 0:
                raise ValueError(f"设备 {d.name or '?'} 显存上限必须为正")


def hardware_from_dict(data: dict) -> HardwareSpec:
    """从配置字典构造 HardwareSpec。

    格式见 configs/template.json：
      - 顶层：name / comm_bw / comm_latency / launch_overhead /
        fusion_overhead / comm_efficiency / devices / 其他任意字段（存入 extra）；
      - 设备：name / flops / mem_bw / max_mem / efficiency / 其他任意字段（存入 extra）。

    效率系数：没有真卡校准时，先用"标称值 × 效率系数"作为有效值进入仿真器。
    标称值（nominal）与效率系数会保留在 extra 中，便于后期替换为实测校准值。
    """

    devs = data.get("devices")
    if not devs:
        raise ValueError("配置缺少 devices 列表")
    device_known = {"name", "flops", "mem_bw", "max_mem"}
    devices = []
    for d in devs:
        eff = d.get("efficiency", {})
        flops_nominal = float(d["flops"])
        mem_bw_nominal = float(d["mem_bw"])
        flops_eff = flops_nominal * float(eff.get("flops", 1.0))
        mem_bw_eff = mem_bw_nominal * float(eff.get("mem_bw", 1.0))
        extra = {k: v for k, v in d.items() if k not in device_known}
        extra.setdefault(
            "nominal", {"flops": flops_nominal, "mem_bw": mem_bw_nominal}
        )
        devices.append(
            DeviceSpec(
                name=d.get("name", ""),
                flops=flops_eff,
                mem_bw=mem_bw_eff,
                max_mem=float(d["max_mem"]),
                extra=extra,
            )
        )

    hw_known = {
        "name",
        "devices",
        "comm_bw",
        "comm_latency",
        "launch_overhead",
        "fusion_overhead",
        "link_bw",
        "link_matrix",
        "interference",
        "overlap_send",
    }
    hw_extra = {k: v for k, v in data.items() if k not in hw_known}
    comm_nominal = float(data.get("comm_bw", 1e12))
    comm_efficiency = float(data.get("comm_efficiency", 1.0))
    hw_extra.setdefault("comm_nominal", comm_nominal)
    link_bw_nominal = data.get("link_bw")
    link_bw_eff = None
    if link_bw_nominal is not None:
        link_bw_eff = float(link_bw_nominal) * comm_efficiency
        hw_extra.setdefault("link_bw_nominal", link_bw_nominal)
    link_matrix_nominal = data.get("link_matrix")
    link_matrix_eff = None
    if link_matrix_nominal is not None:
        link_matrix_eff = [
            [None if v is None else float(v) * comm_efficiency for v in row]
            for row in link_matrix_nominal
        ]
        hw_extra.setdefault("link_matrix_nominal", link_matrix_nominal)
    return HardwareSpec(
        devices=devices,
        comm_bw=comm_nominal * comm_efficiency,
        comm_latency=float(data.get("comm_latency", 1e-6)),
        launch_overhead=float(data.get("launch_overhead", 0.0)),
        fusion_overhead=float(data.get("fusion_overhead", 0.0)),
        link_bw=link_bw_eff,
        link_matrix=link_matrix_eff,
        interference=float(data.get("interference", 0.0)),
        overlap_send=bool(data.get("overlap_send", True)),
        name=data.get("name", ""),
        extra=hw_extra,
    )


def hardware_to_dict(hw: HardwareSpec) -> dict:
    """把 HardwareSpec 序列化为配置字典（与 hardware_from_dict 互逆）。"""

    data = {
        "name": hw.name,
        "comm_bw": hw.extra.get("comm_nominal", hw.comm_bw),
        "comm_latency": hw.comm_latency,
        "launch_overhead": hw.launch_overhead,
        "fusion_overhead": hw.fusion_overhead,
        "interference": hw.interference,
        "overlap_send": hw.overlap_send,
    }
    if "link_bw_nominal" in hw.extra:
        data["link_bw"] = hw.extra["link_bw_nominal"]
    elif hw.link_bw is not None:
        data["link_bw"] = hw.link_bw
    if "link_matrix_nominal" in hw.extra:
        data["link_matrix"] = hw.extra["link_matrix_nominal"]
    elif hw.link_matrix is not None:
        data["link_matrix"] = hw.link_matrix
    devices = []
    for d in hw.devices:
        nominal = d.extra.get("nominal")
        dev = {"name": d.name, "max_mem": d.max_mem}
        if nominal is not None:
            dev["flops"] = nominal.get("flops", d.flops)
            dev["mem_bw"] = nominal.get("mem_bw", d.mem_bw)
        else:
            dev["flops"] = d.flops
            dev["mem_bw"] = d.mem_bw
        dev_extra = {k: v for k, v in d.extra.items() if k != "nominal"}
        if nominal is None:
            # 手工构造的有效值，不输出效率系数，避免重载时二次缩放
            dev_extra.pop("efficiency", None)
        dev.update(dev_extra)
        devices.append(dev)
    data["devices"] = devices

    hw_extra = {
        k: v
        for k, v in hw.extra.items()
        if k not in ("comm_nominal", "link_bw_nominal", "link_matrix_nominal")
    }
    if "comm_nominal" not in hw.extra:
        hw_extra.pop("comm_efficiency", None)
    data.update(hw_extra)
    return data


def load_hardware(path) -> HardwareSpec:
    """从 JSON 配置文件加载 HardwareSpec。"""

    import json

    with open(path, "r", encoding="utf-8") as f:
        return hardware_from_dict(json.load(f))


def save_hardware(hw: HardwareSpec, path) -> None:
    """把 HardwareSpec 保存为 JSON 配置文件。"""

    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(hardware_to_dict(hw), f, ensure_ascii=False, indent=2)
        f.write("\n")
