from __future__ import annotations

import unittest

from cost_model.verify_search_evaluation import verify_search_evaluation


class TestSearchEvaluation(unittest.TestCase):
    def test_small_search_evaluation(self) -> None:
        result = verify_search_evaluation()
        self.assertEqual(result["nodes"], 5)
        self.assertEqual(result["searches"], 9)
        self.assertEqual(result["joint_space"], 48)


if __name__ == "__main__":
    unittest.main()
