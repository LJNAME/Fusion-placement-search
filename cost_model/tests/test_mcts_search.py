"""MCTS 主流程单元测试（无 torch 依赖，本地可跑）。

运行：python -m unittest discover -s cost_model/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cost_model.roofline import RooflineCostModel  # noqa: E402
from cost_model.search import beam_search, mcts_search  # noqa: E402
from simulator import load_hardware  # noqa: E402
from simulator.graphs import make_mlp_block  # noqa: E402
from simulator.validate import validate_plan  # noqa: E402


def _scorer_for(graph, hw):
    rm = RooflineCostModel(hw)
    return lambda plan: float(np.exp(rm.predict(graph, plan)[0]))


def _plan_key(plan):
    return tuple((tuple(g.node_ids), g.device) for g in plan.groups)


class TestMCTSSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hw = load_hardware(str(REPO / "configs" / "hygon_4cards.example.json"))
        cls.graph = make_mlp_block(hidden_sizes=(1024, 1024, 512), batch=8, seed=1)

    def _scorer(self):
        # 注意：不能把函数存成类属性再通过 self 访问，否则会被绑定成方法
        return _scorer_for(self.graph, self.hw)

    def test_returns_valid_complete_plans(self):
        scorer = self._scorer()
        plans = mcts_search(
            self.graph, scorer, n_devs=self.hw.n_devs,
            iterations=40, rollouts=3, seed=0, n_best=4,
        )
        self.assertTrue(1 <= len(plans) <= 4)
        for p in plans:
            validate_plan(self.graph, p, self.hw.n_devs)  # 非法会抛 ValueError

    def test_deterministic_with_seed(self):
        scorer = self._scorer()
        a = mcts_search(
            self.graph, scorer, n_devs=self.hw.n_devs,
            iterations=30, rollouts=2, seed=7, n_best=3,
        )
        b = mcts_search(
            self.graph, scorer, n_devs=self.hw.n_devs,
            iterations=30, rollouts=2, seed=7, n_best=3,
        )
        self.assertEqual(
            [_plan_key(p) for p in a], [_plan_key(p) for p in b]
        )

    def test_small_budget_smoke(self):
        # 预算为 1 轮也应返回合法方案（退化但不崩溃）
        scorer = self._scorer()
        plans = mcts_search(
            self.graph, scorer, n_devs=self.hw.n_devs,
            iterations=1, rollouts=1, seed=0, n_best=2,
        )
        self.assertTrue(1 <= len(plans) <= 2)
        validate_plan(self.graph, plans[0], self.hw.n_devs)

    def test_mcts_not_worse_than_beam_order_of_magnitude(self):
        # 软性冒烟：MCTS 最优预测不劣于 beam 最优一个数量级以上
        scorer = self._scorer()
        beam_plans = beam_search(
            self.graph, scorer, n_devs=self.hw.n_devs,
            beam=8, rollouts=3, seed=0,
        )
        mcts_plans = mcts_search(
            self.graph, scorer, n_devs=self.hw.n_devs,
            iterations=80, rollouts=3, seed=0, n_best=8,
        )
        b = min(scorer(p) for p in beam_plans)
        m = min(scorer(p) for p in mcts_plans)
        self.assertLessEqual(m, b * 10.0)


if __name__ == "__main__":
    unittest.main()
