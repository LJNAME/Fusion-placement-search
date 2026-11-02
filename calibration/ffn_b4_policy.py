"""Separate pipeline smoke from fixed-grid formal data collection."""
from __future__ import annotations

import math
import traceback

from cost_model.fusion_kernel_profile import key_id

POLICY = {
    "version": "ffn-b4.execution.v1", "superseded_fused_by": "c4fc415",
    "smoke": {"minimum_rows": ["down_proj@gpu0", "silu_mul/native@gpu0", "silu_mul/fused@gpu0", "copy/524288/0->1"],
              "calls": 50, "replays": 3, "failure": "fail_fast", "timing_is_pass_criterion": False,
              "required_booleans": ["capture_ok", "numeric_ok", "invariants_ok", "identity_written", "contract_written"]},
    "formal": {"attempts": 7, "calls": 50, "warmup": 10, "extra_attempts": 0,
               "replacement": False, "failure": "mark_cell_quality_failed_and_continue_grid",
               "preserve_all_attempts": True, "automatic_admission": False},
}


def active_recipe(base):
    result = {k: v for k, v in base.items() if k != "recipe_id"}
    result.update(max_attempts=7, primary_selection="all seven chronological attempts; any invalid attempt fails that cell",
                  dispersion_failure="quality_failed; no retry or replacement; continue other cells",
                  formal_policy_id=key_id(POLICY["formal"]), base_b3_recipe_id=base["recipe_id"],
                  uncaptured_initialization_bodies=1, path_trace_replays_before_formal=1)
    result["recipe_id"] = key_id(result)
    return result


def diagnostic_time(value=None, error=None):
    return {"elapsed_ms": value if isinstance(value, (int, float)) and math.isfinite(value) else None,
            "raw": repr(value), "error": repr(error) if error else None, "used_for_pass_fail": False}


def smoke_pass(report):
    # Fast/slow, IQR, anchors and speedup deliberately do not participate.
    return all(report.get(name) is True for name in POLICY["smoke"]["required_booleans"])


def formal_quality_review(attempts, roles, devices):
    from cost_model.ffn_measurement_b3 import quality_review
    if len(attempts) != POLICY["formal"]["attempts"]:
        raise ValueError("formal cell must contain exactly seven attempts; no supplements")
    result = quality_review(attempts, roles, devices)
    if result["status"] == "needs_more_attempts":
        result["status"] = "quality_failed_invalid_round"
    result["extra_attempts_allowed"] = 0
    return result


def collect_grid(row_ids, run_cell):
    """Never lose the rest of a formal grid to one failed cell; never retry."""
    result = {}
    for row_id in row_ids:
        try:
            row = run_cell(row_id)
            if row.get("status") not in {"quality_passed", "quality_failed"}:
                raise ValueError("formal collector returned an invalid cell status")
            result[row_id] = row
        except Exception as error:
            result[row_id] = {"row_id": row_id, "status": "quality_failed", "error": repr(error),
                              "traceback": traceback.format_exc(), "automatic_admission": False}
    return result
