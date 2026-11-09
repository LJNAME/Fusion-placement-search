"""Regression for the fused-vs-unfused benefit validation on existing measured pairs."""
import tempfile
import unittest
from pathlib import Path

from calibration.fusion_benefit_validation import NATIVE_ADDRMS, run


@unittest.skipUnless(NATIVE_ADDRMS.exists(), "native AddRMS receipt lives outside the repo")
class FusionBenefitValidation(unittest.TestCase):
    def test_reproduces_recorded_findings(self):
        with tempfile.TemporaryDirectory() as d:
            report = run(Path(d) / "out")
        rows = {r["case_id"]: r for r in report["cases"]}
        self.assertEqual(report["summary"]["uncalibrated_contract_aligned"]["sign_agreement"], "5/6")
        harmful = rows["lightop_silu_mul_split_gpu0_seq256"]
        self.assertEqual(harmful["measured_preference"], "unfused")
        self.assertEqual(harmful["eager_launch_preference"], "fused")  # the one sign error
        seq1 = rows["kerv_silu_mul_gpu0_seq1"]
        self.assertLess(abs(seq1["graph_launch_ratio_error"]), .05)
        seq256 = rows["kerv_silu_mul_gpu0_seq256"]
        self.assertLess(seq256["graph_launch_ratio"], seq256["measured_ratio"] - .1)  # benefit overstated


if __name__ == "__main__":
    unittest.main()
