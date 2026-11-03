import unittest
from tools.review_lightop_vendor_suite import classify, timings
from integration.lightop import capabilities


class VendorReviewTests(unittest.TestCase):
    def test_zero_exit_and_success_text_do_not_hide_false(self):
        status, _, lines = classify("test_scaled_masked_softmax_dropout", 0, "True\nFalse\nSuccess\n", "")
        self.assertEqual(status, "vendor_checks_failed")
        self.assertEqual(lines, [2])

    def test_missing_baseline_is_not_a_kernel_failure(self):
        status, reason, _ = classify("test_act_and_mul_silu", 1, "", "ModuleNotFoundError: No module named 'vllm'")
        self.assertEqual((status, reason), ("missing_dependency", "vllm"))

    def test_gemma_tolerances_remain_distinct(self):
        text = "allclose(atol=1e-2, rtol=1e-2): True\nallclose(atol=1e-3, rtol=1e-3): False\n"
        self.assertEqual(classify("test_gemma_fused_add_rmsnorm", 0, text, "")[0], "tolerance_dependent")
        text = text.replace("1e-2): True", "1e-2): False")
        self.assertEqual(classify("test_gemma_fused_add_rmsnorm", 0, text, "")[0], "vendor_checks_failed")

    def test_nonzero_exit_overrides_partial_passes(self):
        self.assertEqual(classify("x", 1, "PASS\nTrue\n", "ValueError: incomplete")[0], "execution_failed")

    def test_false_configuration_value_is_not_a_failed_check(self):
        status = classify("test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant", 0, "Shape=(8, 32), Res=False\nAll passed.\n", "")[0]
        self.assertEqual(status, "vendor_checks_passed_correctness_only")

    def test_quantization_needs_all_zero_differences(self):
        text = "\u5dee\u5f02: 0.000000\n" * 8
        self.assertEqual(classify("test_quantization_cuda", 0, text, "")[0], "timed_zero_difference_reported")
        self.assertEqual(classify("test_quantization_cuda", 0, text + "\u5dee\u5f02: 3.0\n", "")[0], "requires_review")

    def test_preserve_loop_seconds_and_table_units(self):
        result = timings("x", "===fuse: 2.4\n(128, 1, 8) False 0.461 0.356 277581\n128 | 4096 | FP16 | PASS | 6.36\n")
        self.assertEqual([r["unit"] for r in result], ["s/990_iterations", "ms", "us"])
        self.assertEqual(result[1]["values"], [.461, .356])
        self.assertEqual(result[2]["case"]["dtype"], "FP16")

    def test_overlay_records_evidence_without_promoting_graph_execution(self):
        records = {r["id"]: r for r in capabilities()}
        self.assertTrue(records["fuse_silu_and_mul"]["vendor_suite01"]["gpu_timing_and_checks_available"])
        self.assertFalse(records["FusedScaleMaskSoftmaxDropout"]["vendor_suite01"]["gpu_timing_and_checks_available"])
        self.assertTrue(all(not r["placed_graph_adapter"] for r in records.values()))


if __name__ == "__main__":
    unittest.main()
