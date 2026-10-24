from __future__ import annotations

import unittest

from simulator.model_graphs import validate_model_graph

from cost_model.search_evaluation import build_evaluation_graph, qwen25_3b_spec
from cost_model.verify_search_evaluation import verify_search_evaluation


class TestSearchEvaluation(unittest.TestCase):
    def test_small_search_evaluation(self) -> None:
        result = verify_search_evaluation()
        self.assertEqual(result["nodes"], 5)
        self.assertEqual(result["searches"], 9)
        self.assertEqual(result["joint_space"], 48)

    def test_qwen25_3b_scale_uses_real_model_dimensions(self) -> None:
        spec = qwen25_3b_spec()
        graph = build_evaluation_graph("model_3b")
        summary = validate_model_graph(graph, spec)
        self.assertEqual(summary.parameter_elements, 3_085_846_528)
        self.assertEqual(len(graph.nodes), 687)
        self.assertEqual(graph.extra["num_layers"], 36)


if __name__ == "__main__":
    unittest.main()
