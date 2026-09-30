"""形状感知效率模型的回归测试（纯 Python + numpy，无 torch，本地可跑）。

运行：python -m unittest discover -s simulator/tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from cost_model.roofline import RooflineCostModel
from simulator.execution import simulate
from simulator.hardware import hardware_from_dict
from simulator.schema import FusionGroup, OpGraph, OpNode, Plan
from simulator.shape_model import compute_efficiency, group_compute_time


def _hw():
    return hardware_from_dict(
        {
            "name": "t",
            "comm_bw": 1e11,
            "devices": [
                {"name": "d0", "flops": 1e12, "mem_bw": 1e12, "max_mem": 1e12},
            ],
        }
    )


def _matmul_graph(k: float) -> OpGraph:
    g = OpGraph(name=f"mm-k{k}")
    g.add_node(
        OpNode(
            "a",
            op_type="matmul",
            flops=2.0 * k,
            out_bytes=4.0,
            extra={"shape": {"m": 1, "n": 1, "k": k}},
        )
    )
    return g


class TestShapeModel(unittest.TestCase):
    def test_matmul_efficiency_saturates_with_k(self):
        hw = _hw()
        e_small = compute_efficiency(hw, _matmul_graph(16).nodes["a"])
        e_big = compute_efficiency(hw, _matmul_graph(4096).nodes["a"])
        self.assertLess(e_small, e_big)
        self.assertLessEqual(e_big, 1.0)

    def test_shape_aware_compute_ge_naive(self):
        hw = _hw()
        g = _matmul_graph(64)
        dev = hw.devices[0]
        naive = g.nodes["a"].flops / dev.flops
        shaped = group_compute_time(g, ["a"], dev, hw)
        self.assertGreater(shaped, naive)

    def test_roofline_diverges_from_simulator(self):
        hw = _hw()
        g = _matmul_graph(64)
        plan = Plan([FusionGroup(["a"], 0)])
        res = simulate(g, plan, hw, collect_timeline=False)
        rm = RooflineCostModel(hw)
        naive_log = float(rm.predict(g, plan)[0])
        self.assertLess(naive_log, float(np.log(res.makespan)))


if __name__ == "__main__":
    unittest.main()
