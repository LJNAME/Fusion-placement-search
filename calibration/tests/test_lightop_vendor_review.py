import unittest
from pathlib import Path
import tempfile
import zipfile
from tools.review_lightop_vendor_suite import classify, timings, selected_jobs, bmm_workbook_timings
from integration.lightop import capabilities


class VendorReviewTests(unittest.TestCase):
    def test_selected_subset_is_complete_and_ordered(self):
        matrix = {"jobs": [{"id": n} for n in ("a", "b", "c")]}
        plan = {"selected_jobs": ["a", "c"]}
        self.assertEqual(list(selected_jobs(plan, matrix, [{"id": "a"}, {"id": "c"}])), ["a", "b", "c"])
        for rows in ([{"id": "a"}], [{"id": "c"}, {"id": "a"}], [{"id": "a"}, {"id": "a"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                selected_jobs(plan, matrix, rows)

    def test_zero_exit_and_success_text_do_not_hide_false(self):
        status, _, lines = classify("test_scaled_masked_softmax_dropout", 0, "True\nFalse\nSuccess\n", "")
        self.assertEqual(status, "vendor_checks_failed")
        self.assertEqual(lines, [2])

    def test_missing_baseline_is_not_a_kernel_failure(self):
        status, reason, _ = classify("test_act_and_mul_silu", 1, "", "ModuleNotFoundError: No module named 'vllm'")
        self.assertEqual((status, reason), ("missing_dependency", "vllm"))

    def test_followup_tables_keep_baseline_units_and_cases(self):
        cases = [
            ("test_act_and_mul_silu", "| 32 | 4096 | Pass | 6.99 | 4.28 | 1.63 x |", "us", [6.99, 4.28]),
            ("test_fuse_groupgemm_permute_quant", "32 | Pass | 0.225 | 0.077 | 2.92x", "ms", [.225, .077]),
            ("test_reshape_cache_cuda", "8 1 7.525 5.101 1.48 1.09 1.61 PASS", "us", [7.525, 5.101]),
        ]
        for job, line, unit, values in cases:
            with self.subTest(job=job):
                self.assertEqual(classify(job, 0, line, "")[0], "vendor_checks_passed_timed")
                result = timings(job, line)
                self.assertEqual(len(result), 1)
                self.assertEqual((result[0]["unit"], result[0]["values"]), (unit, values))
                self.assertEqual(classify(job, 0, line.replace("Pass", "FAIL").replace("PASS", "FAIL"), "")[0], "vendor_checks_failed")

    def test_zero_reports_and_bmm_are_not_timing_evidence(self):
        line = "diff_inputs_embeds_max: 0.0, diff_previous_hidden_states_max: 0.0\n"
        job = "test_fuse_fill_rms_x2_concat"
        self.assertEqual(classify(job, 0, line * 10, "")[0], "zero_difference_reported_correctness_only")
        for text in (line * 9, line * 9 + line.replace("0.0", "1.0")):
            self.assertEqual(classify(job, 0, text, "")[0], "requires_review")
        bmm = "config: {'MODE': 0}, status: False\n\u901a\u8fc7\u7387: 128/128 = 100.00%\n"
        self.assertEqual(classify("test_fused_trans_bmm_trans_benchmark", 0, bmm, "")[0], "vendor_checks_passed_correctness_only")
        self.assertEqual(timings("test_fused_trans_bmm_trans_benchmark", bmm), [])
        self.assertEqual(classify("test_fused_trans_bmm_trans_benchmark", 0, bmm + "torch check kernel \u7cbe\u5ea6\u68c0\u67e5\u4e0d\u5408\u683c!!!", "")[0], "vendor_checks_failed")

    def test_bmm_workbook_evidence_preserves_boundary_and_rejects_failed_rows(self):
        columns = ["B", "N", "L", "P", "Passed", "Torch_bf16_bmm_time_us", "Q_quant_time_us",
                   "Fused_fp8_bmm_Time_us", "Q_quant+Fused_fp8_bmm_Time_us", "Fused_fp8_bmm_TFlops", "Fused_fp8_bmm_GB/s"]
        header = "<row>" + "".join('<c t="inlineStr"><is><t>' + name + "</t></is></c>" for name in columns) + "</row>"
        body = '<row><c><v>1</v></c><c><v>16</v></c><c><v>128</v></c><c><v>512</v></c><c t="b"><v>1</v></c>'
        body += "".join("<c><v>" + str(n) + "</v></c>" for n in [36, 0, 9.28, 9.28, .2, 107]) + "</row>"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bmm.xlsx"
            def write(text):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + header + text + "</sheetData></worksheet>")
            write(body * 128)
            result = bmm_workbook_timings(path)
            self.assertEqual(len(result), 128)
            self.assertEqual(result[0]["values"], [36, 0, 9.28, 9.28])
            self.assertEqual(result[0]["input_dtype"], "int8")
            self.assertIn("quantization cost to zero", result[0]["boundary"])
            for bad in (body * 127, body * 127 + body.replace('t="b"><v>1', 't="b"><v>0'), body.replace('<v>36</v>', '<f>1+1</f><v>36</v>') * 128):
                write(bad)
                with self.assertRaises(ValueError):
                    bmm_workbook_timings(path)

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

    def test_overlay_keeps_vendor_evidence_separate_from_adapter_validation(self):
        records = {r["id"]: r for r in capabilities()}
        self.assertTrue(records["fuse_silu_and_mul"]["vendor_suite01"]["gpu_timing_and_checks_available"])
        self.assertFalse(records["FusedScaleMaskSoftmaxDropout"]["vendor_suite01"]["gpu_timing_and_checks_available"])
        self.assertEqual([r["id"] for r in records.values() if r["placed_graph_adapter"]],
                         ["op.silu_and_mul_opt"])
        adapter = records["op.silu_and_mul_opt"]["project_adapter"]
        self.assertEqual(adapter["validation"], "cpu_contract_only_gpu_pending")
        self.assertFalse(adapter["gpu_execution_verified"])
        self.assertEqual(adapter["accepted_profiles"], 0)
        latest = "vendor_20261104_02"
        self.assertTrue(records["op.silu_and_mul_opt"][latest]["gpu_timing_and_checks_available"])
        self.assertTrue(records["fused_bmm"][latest]["gpu_timing_and_checks_available"])
        self.assertFalse(records["rms_norm_per_token_fp8_quant"][latest]["gpu_timing_and_checks_available"])


if __name__ == "__main__":
    unittest.main()
