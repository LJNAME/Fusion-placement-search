"""HeteroExecutionContext 的纯 Python 测试（不需要 torch / megatron）。

运行：
    pytest integration/tests/test_context.py -q
    python integration/tests/test_context.py          # 也支持直接跑
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integration.flagscale.context import (  # noqa: E402
    HeteroExecutionContext,
    layer_range_of,
    mesh_of_stage,
    synthetic,
)


def test_single_card_is_not_hetero():
    ctx = synthetic(device_type="BW1000")
    assert ctx.n_devices == 1
    assert ctx.n_meshes == 1
    assert ctx.is_hetero is False
    assert ctx.backend == "synthetic"


def test_single_mesh_tp2_pp2():
    # 1 个 mesh + PP=2 -> 全局 PP 只能是 2，所以 layer_split 长度必须是 2
    ctx = synthetic(device_type="BW1000", tp=2, pp=2, num_layers=8, layer_split=[4, 4])
    assert ctx.n_devices == 4
    assert ctx.is_hetero is False
    assert ctx.global_stage_id == 0
    assert (ctx.layer_start, ctx.layer_end) == (0, 4)


def test_two_meshes_is_hetero():
    ctx = synthetic(
        device_type="BW1000", tp=2, pp=2, num_layers=8, layer_split=[2, 2, 2, 2],
        mesh_device_types=["BW1000", "K100"], mesh_id=1,
    )
    assert ctx.is_hetero is True
    assert ctx.mesh_id == 1
    assert ctx.global_stage_id == 2
    assert (ctx.layer_start, ctx.layer_end) == (4, 6)


def test_same_vendor_multi_mesh_counts_as_hetero():
    """官方 functional test 的 device_types 全是 "A800"，同型号多 mesh 也算异构路径。"""
    ctx = synthetic(mesh_device_types=["A800", "A800", "A800"], num_layers=6)
    assert ctx.is_hetero is True


def test_layer_range_of():
    assert layer_range_of(0, [2, 4, 2]) == (0, 2)
    assert layer_range_of(1, [2, 4, 2]) == (2, 6)
    assert layer_range_of(2, [2, 4, 2]) == (6, 8)


def test_mesh_of_stage():
    assert mesh_of_stage(0, [1, 1, 1]) == 0
    assert mesh_of_stage(1, [1, 1, 1]) == 1
    assert mesh_of_stage(2, [1, 1, 1]) == 2
    assert mesh_of_stage(3, [2, 2, 2]) == 1


def test_placement_seed_shape():
    ctx = synthetic(device_type="BW1000", tp=2, pp=2, mesh_device_types=["BW1000", "K100"], mesh_id=1)
    seed = ctx.as_placement_seed()
    for key in ("mesh_id", "device_type", "device_ids", "hostnames", "global_pp_stage", "tp", "cp", "ep", "dp"):
        assert key in seed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")