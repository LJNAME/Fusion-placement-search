from __future__ import annotations

import unittest

from cost_model.verify_search_stability import verify_search_stability


class TestSearchStability(unittest.TestCase):
    def test_repeated_search_report(self) -> None:
        result = verify_search_stability()
        self.assertEqual(result["records"], 12)
        self.assertEqual(result["aggregates"], 6)


if __name__ == "__main__":
    unittest.main()
