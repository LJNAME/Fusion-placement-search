"""FlagScale / Megatron-LM-FL 侧上下文适配层。

设计约束（见笔记 Flagos集成架构 第 2.4 节）：
  - 本目录是项目中**唯一**允许 import megatron / flagscale 的地方；
  - 对外只暴露本项目自己的数据结构 ``HeteroExecutionContext``；
  - 拿不到 Megatron 时必须能退化到 :func:`synthetic`，以便离线开发与单卡验证。

接口依据（2026-10-04 在 BW1000 上 dump 的 megatron-core 0.17.1 真实 API）：

    ParallelContext.get_current_process_mesh()
    ParallelContext.get_{tensor_model,pipeline_model,data,context,expert_model}_parallel_{rank,world_size}()
    ParallelContext.is_pipeline_first_stage() / is_pipeline_last_stage()
    ParallelContext.get_global_group_ranks(token, is_expert=False, check_initialized=False)
    ProcessMesh.logical_coords_to_physical_ranks(coords, is_expert=False)
    ProcessMesh.get_process_group_ranks(token, is_expert=False, check_initialized=False)
    RankMapper.to_physical_ranks(logical_ranks) / to_logical_ranks(physical_ranks)

已核实的接口语义（2026-10-04 读 megatron-core 0.17.1 源码）：

  - ``get_current_process_mesh()`` 返回 **ProcessMesh 对象**（源码 ``return self._process_meshes[idx]``）；
    mesh 下标在私有属性 ``_current_process_mesh_index``，全 mesh 列表在 ``_process_meshes``。
  - ~~ProcessMesh 自带各并行度字段~~ **实测（0.18.2）：这些构造参数不保存为公开属性，读出来全是 None**。
    `ProcessMesh` 的 public attrs 只有 13 个：`build_all_process_groups` / `build_process_group` /
    `create_gloo_process_groups` / `get_all_process_group_ranks` / `get_ddp_config` / `get_optimizer_config` /
    `get_parallel_size` / `get_process_group` / `get_process_group_ranks` / `get_process_group_size` /
    `get_transformer_config` / `logical_coords_to_physical_ranks` / `order`（值是 None）。
  - **并行度入口按可靠性排序**：`get_parallel_size(axis)` > `get_process_group_size(axis)` > `pc.get_*_world_size()`。
    `get_process_group_size("ep")` 在 ep=1 时会 `KeyError: 'ep'`（没建 expert 组）；
    `get_parallel_size` 对 5 个轴都返回正常值。
  - **进程组 token 只能是单轴名**（"tp"/"cp"/"ep"/"dp"/"pp"）：全名 `"tp-cp-ep-dp-pp"` 会
    `ValueError: Invalid token`，`get_process_group_ranks` 对未知 token 返回 `None` 而不是抛错。
    ``expert_model_parallel_size`` / ``data_parallel_size`` / ``pipeline_model_parallel_size`` /
    ``virtual_pipeline_model_parallel_size`` / ``offset``。
  - ``token`` **不是数字拼串，而是轴名串**：``ProcessMesh.order`` 默认 ``'tp-cp-ep-dp-pp'``；
    ``get_group_name(token, is_expert)`` 非 expert 原样返回、expert 加 ``exp_`` 前缀。
  - ``logical_coords_to_physical_ranks(coords, is_expert)``：``coords`` 是 **4 元组列表**，
    内部按 prefix-product 算逻辑 rank、加 ``offset``，再经 ``RankMapper`` 查表转物理 rank。

仍未确认：

  - ``coords`` 的轴序到底是 ``tp-cp-dp-pp`` 还是 ``tp-cp-ep-pp`` —— 源码注释说前者，
    但跳过逻辑是 ``idx != 3``（后者才自洽）。**ep=1 时两者数值等价**，ep>1 时必须实测。
    因此 :func:`physical_ranks_of` 目前只建议在 ep=1 时使用。
  - ``ProcessMesh`` 的私有属性名以 0.17.1 为准，换版本要复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 本项目自己的数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeteroExecutionContext:
    """一个进程看到的异构上下文。纯 Python，不依赖 torch / megatron。

    字段来源见模块 docstring；标 (算) 的是适配层自己推出来的，不在 Megatron 接口里。
    """

    # ---- 身份 ----
    mesh_id: int
    device_type: str
    local_rank: int = 0
    global_rank: int = 0

    # ---- 并行度（该 mesh 的） ----
    tp: int = 1
    cp: int = 1
    ep: int = 1
    dp: int = 1
    pp: int = 1

    # ---- 本进程在各并行维里的位置 ----
    tp_rank: int = 0
    cp_rank: int = 0
    ep_rank: int = 0
    dp_rank: int = 0
    pp_rank: int = 0

    # ---- 流水线位置 ----
    global_stage_id: int = 0          # (算) 全局 stage 索引
    is_first_stage: bool = False
    is_last_stage: bool = False

    # ---- 层范围（本 stage 负责的层，左闭右开） ----
    layer_start: int = 0              # (算)
    layer_end: int = 0                # (算)

    # ---- rank 集合 ----
    mesh_ranks: List[int] = field(default_factory=list)
    global_ranks: List[int] = field(default_factory=list)

    # ---- 拓扑 ----
    hostnames: List[str] = field(default_factory=list)
    mesh_device_types: List[str] = field(default_factory=list)

    # ---- 元信息 ----
    backend: str = "unknown"          # "megatron" | "synthetic"
    extra: dict = field(default_factory=dict)

    # ---- 派生量 ----

    @property
    def n_devices(self) -> int:
        """该 mesh 的卡数 = TP*CP*EP*DP*PP。

        依据 hetero_searcher._get_valid_mesh_configs: total_gpus = tp*cp*ep*pp*dp。
        """
        return self.tp * self.cp * self.ep * self.dp * self.pp

    @property
    def n_meshes(self) -> int:
        return len(self.mesh_device_types) if self.mesh_device_types else 1

    @property
    def is_hetero(self) -> bool:
        """mesh 数 > 1 即算异构。

        注意：官方 functional test 的 device_types 是 ["A800","A800","A800"]，
        即**同型号多 mesh 也走异构路径**，所以判据是 mesh 数而非型号种类数。
        """
        return self.n_meshes > 1

    @property
    def span(self) -> int:
        """本 stage 负责的层数。"""
        return max(0, self.layer_end - self.layer_start)

    def as_placement_seed(self) -> dict:
        """给 Placement 用的初值（见 Flagos集成架构 第 2.3 节）。"""
        return {
            "mesh_id": self.mesh_id,
            "device_type": self.device_type,
            "device_ids": list(self.mesh_ranks),
            "global_ranks": list(self.global_ranks),
            "hostnames": list(self.hostnames),
            "global_pp_stage": self.global_stage_id,
            "tp": self.tp,
            "cp": self.cp,
            "ep": self.ep,
            "dp": self.dp,
        }

    def describe(self) -> str:
        return (
            f"[{self.backend}] mesh={self.mesh_id}/{self.n_meshes} "
            f"type={self.device_type} "
            f"TP={self.tp} CP={self.cp} EP={self.ep} DP={self.dp} PP={self.pp} "
            f"(n_devices={self.n_devices}) "
            f"stage={self.global_stage_id} layer=[{self.layer_start},{self.layer_end}) "
            f"ranks={self.mesh_ranks} hetero={self.is_hetero}"
        )


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


#: Megatron 里 mesh 的轴序（**注意：这不是合法的进程组 token**）
DEFAULT_MESH_ORDER = "tp-cp-ep-dp-pp"

#: 五个并行轴的短名（进程组 token 用短名，见 ProcessMesh.build_process_group("dp", ...)）
MESH_AXES = ("tp", "cp", "ep", "dp", "pp")


def layer_range_of(stage_index: int, layer_split: Sequence[int]) -> Tuple[int, int]:
    """全局 stage 索引 -> (layer_start, layer_end)。

    ``layer_split`` 就是 config 里的 ``hetero_pipeline_layer_split``，
    它的长度等于**全局 PP size**（= 各 mesh 的 PP 之和）。
    """
    if stage_index < 0 or stage_index >= len(layer_split):
        raise IndexError(f"stage_index {stage_index} 越界（layer_split 长度 {len(layer_split)}）")
    start = int(sum(layer_split[:stage_index]))
    return start, start + int(layer_split[stage_index])


def mesh_of_stage(stage_index: int, mesh_pp_sizes: Sequence[int]) -> int:
    """全局 stage 索引 -> 它属于第几个 mesh。

    对应 Megatron 侧 hetero_theoretical_memory._get_mesh_params_for_stage 的累加逻辑。
    """
    offset = 0
    for mesh_idx, pp in enumerate(mesh_pp_sizes):
        if offset <= stage_index < offset + pp:
            return mesh_idx
        offset += pp
    raise IndexError(f"stage_index {stage_index} 超出所有 mesh（总 PP={sum(mesh_pp_sizes)}）")


# --------------------------------------------------------------------------- #
# 适配器：真实 Megatron 上下文
# --------------------------------------------------------------------------- #


def mesh_pp_sizes_of(parallel_context) -> Optional[List[int]]:
    """取各 mesh 的 PP 列表（算全局 stage 索引要用）。

    公开 API 没有暴露"全部 mesh"的列表，只能用私有属性 ``_process_meshes``。换版本要复核。
    """
    meshes = getattr(parallel_context, "_process_meshes", None)
    if not meshes:
        return None
    sizes = []
    for m in meshes:
        try:
            sizes.append(int(m.get_process_group_size("pp", is_expert=False, gloo=False)))
        except Exception:  # noqa: BLE001
            return None
    return sizes


def physical_ranks_of(mesh, coords) -> List[int]:
    """逻辑坐标 -> 物理 rank。

    ``coords`` 是 4 元组的列表（轴序见 ``ProcessMesh.order``，默认 'tp-cp-ep-dp-pp'）。
    ``ProcessMesh.logical_coords_to_physical_ranks`` 内部会跳过 expert / cp 轴，
    再走 ``RankMapper`` 查表 —— **这就是 Placement.device_ids 的来源**。

    注意：ep>1 时轴序语义尚未实测确认，见模块 docstring。
    """
    return [int(r) for r in mesh.logical_coords_to_physical_ranks(coords)]


def from_megatron(
    args=None,
    token: Optional[str] = None,
    layer_split: Optional[Sequence[int]] = None,
    mesh_pp_sizes: Optional[Sequence[int]] = None,
    hostnames: Optional[Sequence[str]] = None,
) -> HeteroExecutionContext:
    """从运行中的 FlagScale/Megatron 进程构造上下文。

    只在真跑在 Megatron 里时调用；任何一步拿不到就抛异常，
    由调用方决定是否退化到 :func:`synthetic`。

    args:
        训练参数对象（含 ``hetero_current_device_type`` / ``hetero_device_types``）。
        取当前进程卡型号见 arguments_fs.py:226。
    token:
        并行组名（轴名串）。None 时取 ``mesh.order``（默认 'tp-cp-ep-dp-pp'）。
    layer_split:
        config 里的 ``hetero_pipeline_layer_split``，用于算 layer 范围。
    mesh_pp_sizes:
        各 mesh 的 PP 列表；None 时尝试从 ``pc._process_meshes`` 推。
    hostnames:
        该 mesh 的节点名，从 hostfile 解析（parallel_context 不提供）。
    """
    from megatron.plugin.hetero.parallel_context import get_parallel_context

    pc = get_parallel_context()
    if pc is None:
        raise RuntimeError("parallel context 未初始化（set_parallel_context 是否被调用？）")

    mesh = pc.get_current_process_mesh()
    if mesh is None:
        raise RuntimeError("get_current_process_mesh() 返回 None")

    mesh_id = int(getattr(pc, "_current_process_mesh_index", 0))

    def _size(axis: str, getter: str) -> int:
        """取该 mesh 在某个轴上的并行度。

        实测（2026-10-05，megatron-core 0.18.2）：ProcessMesh 的构造参数
        （tensor_model_parallel_size 等）**不作为公开属性保存**，读出来全是 None；
        正确入口是 ``get_process_group_size(token)``，token 是**单轴名**
        （"tp"/"cp"/"ep"/"dp"/"pp"）。
        """
        # 1) get_parallel_size —— 实测对全部 5 个轴都可用（含 ep=1）
        try:
            value = mesh.get_parallel_size(axis, is_expert=False)
            if value is not None:
                return int(value)
        except Exception:  # noqa: BLE001
            pass
        # 2) get_process_group_size —— 注意 ep 轴在没建 expert 组时会 KeyError
        try:
            value = mesh.get_process_group_size(axis, is_expert=False, gloo=False)
            if value is not None:
                return int(value)
        except Exception:  # noqa: BLE001
            pass
        # 3) ParallelContext
        try:
            return int(getattr(pc, getter)())
        except Exception:  # noqa: BLE001
            return 1

    tp = _size("tp", "get_tensor_model_parallel_world_size")
    cp = _size("cp", "get_context_parallel_world_size")
    ep = _size("ep", "get_expert_model_parallel_world_size")
    dp = _size("dp", "get_data_parallel_world_size")
    pp = _size("pp", "get_pipeline_model_parallel_world_size")

    tp_rank = int(pc.get_tensor_model_parallel_rank())
    cp_rank = int(pc.get_context_parallel_rank())
    ep_rank = int(pc.get_expert_model_parallel_rank())
    dp_rank = int(pc.get_data_parallel_rank())
    pp_rank = int(pc.get_pipeline_model_parallel_rank())

    # token 只能是**单轴名**（"tp"/"cp"/"ep"/"dp"/"pp"）；
    # 全名 "tp-cp-ep-dp-pp" 会 ValueError: Invalid token（实测 0.18.2）。
    if token is None:
        token = "dp"

    # mesh 内的全部 rank：用公开接口 logical_coords_to_physical_ranks 枚举坐标。
    # 注意：该函数的 4 个坐标轴在 ep=1 或 dp=1 时两种可能解释等价（见模块 docstring）。
    mesh_ranks: List[int] = []
    try:
        import itertools

        coords = list(itertools.product(*[range(max(1, v)) for v in (tp, cp, dp, pp)]))
        mesh_ranks = physical_ranks_of(mesh, coords)
    except Exception:  # noqa: BLE001
        mesh_ranks = []
    if not mesh_ranks:
        try:
            ranks = mesh.get_process_group_ranks(token, is_expert=False, check_initialized=False)
            mesh_ranks = [int(r) for r in ranks] if ranks else []
        except Exception:  # noqa: BLE001
            mesh_ranks = []

    global_ranks: List[int] = []
    try:
        global_ranks = [int(r) for r in pc.get_global_group_ranks(token, is_expert=False, check_initialized=True)]
    except Exception:  # noqa: BLE001
        pass

    # 本进程的 rank
    global_rank, local_rank = 0, 0
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            global_rank = int(dist.get_rank())
            if mesh_ranks and global_rank in mesh_ranks:
                local_rank = mesh_ranks.index(global_rank)
    except Exception:  # noqa: BLE001
        pass

    device_type = "unknown"
    if args is not None:
        device_type = getattr(args, "hetero_current_device_type", "unknown")
    mesh_device_types = (
        list(getattr(args, "hetero_device_types", []) or []) if args is not None else []
    )

    # 全局 stage 索引 = 前面各 mesh 的 PP 之和 + 本 mesh 内的 pp_rank
    if mesh_pp_sizes is None:
        mesh_pp_sizes = mesh_pp_sizes_of(pc)
    if mesh_pp_sizes is None:
        mesh_pp_sizes = [pp] * (mesh_id + 1)   # 退化：只在单 mesh 时正确
    global_stage = int(sum(mesh_pp_sizes[:mesh_id])) + pp_rank

    layer_start, layer_end = 0, 0
    if layer_split is not None:
        layer_start, layer_end = layer_range_of(global_stage, layer_split)

    return HeteroExecutionContext(
        mesh_id=mesh_id,
        device_type=device_type,
        local_rank=local_rank,
        global_rank=global_rank,
        tp=tp, cp=cp, ep=ep, dp=dp, pp=pp,
        tp_rank=tp_rank, cp_rank=cp_rank, ep_rank=ep_rank, dp_rank=dp_rank, pp_rank=pp_rank,
        global_stage_id=global_stage,
        is_first_stage=bool(pc.is_pipeline_first_stage()),
        is_last_stage=bool(pc.is_pipeline_last_stage()),
        layer_start=layer_start,
        layer_end=layer_end,
        mesh_ranks=mesh_ranks,
        global_ranks=global_ranks,
        hostnames=list(hostnames or []),
        mesh_device_types=mesh_device_types,
        backend="megatron",
    )

# --------------------------------------------------------------------------- #
# 回退：无 Megatron 时的合成上下文（离线开发 / 单卡验证）
# --------------------------------------------------------------------------- #


def synthetic(
    device_type: str = "SIM",
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    dp: int = 1,
    pp: int = 1,
    num_layers: int = 8,
    layer_split: Optional[Sequence[int]] = None,
    mesh_device_types: Optional[Sequence[str]] = None,
    hostnames: Optional[Sequence[str]] = None,
    mesh_id: int = 0,
) -> HeteroExecutionContext:
    """合成一个上下文，用于没有 Megatron 的环境（本地开发、BW 单卡首验）。

    单卡默认：tp=cp=ep=dp=pp=1 -> n_devices=1、mesh 数=1、is_hetero=False。
    """
    types = list(mesh_device_types) if mesh_device_types else [device_type]
    n_meshes = len(types)
    global_pp = pp * n_meshes          # 全局 PP = 各 mesh 的 PP 之和

    if layer_split is None:
        if num_layers % global_pp != 0:
            raise ValueError(
                f"num_layers({num_layers}) 不能被全局 PP({global_pp}) 整除，请显式传 layer_split"
            )
        layer_split = [num_layers // global_pp] * global_pp
    else:
        layer_split = list(layer_split)

    # 一致性校验：对应 hetero_pruner._check_architectural_validity 的两条硬约束
    if len(layer_split) != global_pp:
        raise ValueError(f"len(layer_split)={len(layer_split)} != 全局 PP={global_pp}")
    if sum(layer_split) != num_layers:
        raise ValueError(f"sum(layer_split)={sum(layer_split)} != num_layers={num_layers}")
    if any(c < pp for c in layer_split):
        raise ValueError(f"每个 stage 的层数必须 >= 本 mesh 的 PP(={pp})：layer_split={layer_split}")

    pp_list = [pp] * n_meshes
    global_stage = sum(pp_list[:mesh_id]) + 0
    layer_start, layer_end = layer_range_of(global_stage, layer_split)

    n_ranks = tp * cp * ep * dp * pp
    return HeteroExecutionContext(
        mesh_id=mesh_id,
        device_type=device_type,
        local_rank=0,
        global_rank=0,
        tp=tp, cp=cp, ep=ep, dp=dp, pp=pp,
        global_stage_id=global_stage,
        is_first_stage=(global_stage == 0),
        is_last_stage=(global_stage == len(layer_split) - 1),
        layer_start=layer_start,
        layer_end=layer_end,
        mesh_ranks=list(range(n_ranks)),
        global_ranks=list(range(n_ranks)),
        hostnames=list(hostnames or [f"node{mesh_id}"]),
        mesh_device_types=types,
        backend="synthetic",
    )


def current(
    args=None,
    token: Optional[str] = None,
    layer_split: Optional[Sequence[int]] = None,
    hostnames: Optional[Sequence[str]] = None,
    allow_synthetic: bool = True,
) -> HeteroExecutionContext:
    """统一入口：能拿到真实 Megatron 上下文就用真的，否则退化到合成上下文。"""
    try:
        return from_megatron(args=args, token=token, layer_split=layer_split, hostnames=hostnames)
    except Exception as exc:  # noqa: BLE001
        if not allow_synthetic:
            raise
        ctx = synthetic(layer_split=layer_split, hostnames=hostnames)
        return HeteroExecutionContext(**{**ctx.__dict__, "extra": {"fallback_reason": repr(exc)}})


# --------------------------------------------------------------------------- #
# 自检（不需要 torch / megatron）
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    a = synthetic(device_type="BW1000")
    print("单卡 :", a.describe())
    assert a.n_devices == 1 and not a.is_hetero

    b = synthetic(device_type="BW1000", tp=2, pp=2, num_layers=8, layer_split=[4, 4])
    print("单mesh:", b.describe())
    assert b.n_devices == 4

    c = synthetic(
        device_type="BW1000", tp=2, pp=2, num_layers=8, layer_split=[2, 2, 2, 2],
        mesh_device_types=["BW1000", "K100"], mesh_id=1,
    )
    print("双mesh:", c.describe())
    assert c.is_hetero and c.mesh_id == 1

    assert layer_range_of(0, [2, 4, 2]) == (0, 2)
    assert layer_range_of(2, [2, 4, 2]) == (6, 8)
    assert mesh_of_stage(0, [1, 1, 1]) == 0 and mesh_of_stage(2, [1, 1, 1]) == 2
    print("placement seed:", c.as_placement_seed())
    print("自检通过")