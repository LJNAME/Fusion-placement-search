from __future__ import annotations

import unittest

from calibration.calibrate_bw1000 import (
    MIB,
    build_hardware_config,
    fit_link_model,
    parse_int_csv,
    summarize,
)
def _timing(value: float) -> dict:
    return {"median": value, "mean": value, "std": 0.0, "p10": value, "p90": value, "samples": [value]}


class TestBW1000Calibration(unittest.TestCase):
    def test_parse_and_summary(self):
        self.assertEqual(parse_int_csv("0, 1"), [0, 1])
        self.assertEqual(summarize([1.0, 2.0, 3.0])["median"], 2.0)

    def test_link_fit_recovers_linear_model(self):
        bandwidth = 20e9
        latency = 8e-6
        points = []
        for size in (MIB, 16 * MIB, 64 * MIB):
            duration = latency + size / bandwidth
            points.append(
                {
                    "size_bytes": size,
                    "duration_s": _timing(duration),
                    "bandwidth_bytes_per_s": size / duration,
                }
            )
        fitted = fit_link_model(points)
        self.assertAlmostEqual(fitted["bandwidth_bytes_per_s"], bandwidth, delta=bandwidth * 1e-9)
        self.assertAlmostEqual(fitted["latency_s"], latency, delta=1e-12)

    def test_generated_profile_loads_as_hardware_spec(self):
        devices = []
        for index in (0, 1):
            devices.append(
                {
                    "index": index,
                    "name": "BW1000",
                    "total_memory_bytes": 64 * 1024**3,
                    "copy": {"bandwidth_bytes_per_s": 800e9},
                    "launch": {"duration_s": _timing(5e-6)},
                    "matmul": [
                        {"shape": {"k": 1024}, "throughput_tflops": 100.0},
                        {"shape": {"k": 3072}, "throughput_tflops": 120.0},
                    ],
                }
            )
        links = [
            {"src": 0, "dst": 1, "fit": {"bandwidth_bytes_per_s": 24e9, "latency_s": 9e-6}},
            {"src": 1, "dst": 0, "fit": {"bandwidth_bytes_per_s": 23e9, "latency_s": 10e-6}},
        ]
        raw = {
            "devices": devices,
            "links": links,
            "dtype": "bf16",
            "source_id": "unit-test",
        }
        config = build_hardware_config(raw)
        self.assertEqual(len(config["devices"]), 2)
        self.assertAlmostEqual(config["link_matrix"][0][1], 24e9)
        self.assertAlmostEqual(config["link_matrix"][1][0], 23e9)
        self.assertEqual(config["devices"][0]["dtype"], "bf16")
        self.assertTrue(config["devices"][0]["calibrated"])
        self.assertFalse(config["calibration_scope"]["real_fused_kernels"])


if __name__ == "__main__":
    unittest.main()
