from __future__ import annotations

import unittest

from execution.verify_reference import verify_reference


class TestReferenceExecution(unittest.TestCase):
    def test_cpu_reference_loop(self) -> None:
        result = verify_reference()
        self.assertEqual(result["groups"], 4)
        self.assertTrue(result["planned_pattern_rejected"])


if __name__ == "__main__":
    unittest.main()
