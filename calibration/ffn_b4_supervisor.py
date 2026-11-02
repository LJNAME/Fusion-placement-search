"""Run each fixed round group once; native worker crashes cannot erase the grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from calibration.ffn_b4_policy import smoke_pass
from cost_model.fusion_kernel_profile import key_id


def supervise(frozen, output, flagscale_root, run_worker=subprocess.run):
    from calibration.ffn_b4 import save
    output = Path(output)
    groups, cells = [], {}
    for group in frozen["matrix"]["round_groups"]:
        directory = output/"groups"/group["group_id"].replace("/", "_").replace(">", "to")
        directory.mkdir(parents=True, exist_ok=False)
        save(directory/"parent_plan.json", frozen)
        command = [sys.executable, "-X", "faulthandler", "-m", "calibration.ffn_b4_supervisor",
                   "--worker", str(directory), "--group", group["group_id"],
                   "--flagscale-root", str(flagscale_root)]
        record = {"group_id": group["group_id"], "command": command, "attempted_workers": 1, "retries": 0}
        print(json.dumps({"phase": "formal_group_start", "group_id": group["group_id"]}), flush=True)
        try:
            with (directory/"worker.log").open("w", encoding="utf-8") as log:
                completed = run_worker(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            record["returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"worker exited {completed.returncode}; see worker.log and last_stage.json")
            result = json.loads((directory/"formal/grid.json").read_text(encoding="utf-8"))
            if set(result["cells"]) != set(group["members"]):
                raise ValueError("worker grid differs from its fixed round group")
            for row_id, row in result["cells"].items():
                if row.get("status") not in ("quality_passed", "quality_failed"):
                    raise ValueError("invalid worker cell status")
                cells[row_id] = {**row, "worker_evidence": str(directory.relative_to(output))}
            record["status"] = "completed"
        except Exception as error:
            record.update(status="quality_failed", error=repr(error))
            for row_id in group["members"]:
                cells[row_id] = {"row_id": row_id, "status": "quality_failed", "error": repr(error),
                                 "worker_evidence": str(directory.relative_to(output)),
                                 "worker_returncode": record.get("returncode"),
                                 "raw_files_preserved": True, "retries": 0, "automatic_admission": False}
        groups.append(record)
        save(directory/"worker_status.json", record)
        # Persist after each group, including a worker killed by SIGSEGV.
        save(output/"progress.json", {"groups": groups, "cells": cells, "complete": False})
    result = {"phase": "formal", "cells": cells, "groups": groups, "expected_cells": 19,
              "recorded_cells": len(cells), "complete": len(cells) == 19,
              "quality_failed": sum(r["status"] == "quality_failed" for r in cells.values()),
              "retries": 0, "automatic_admission": False,
              "scope": "raw evidence; timing boundaries, routes, keys and admission require review"}
    save(output/"grid.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--flagscale-root", type=Path, required=True)
    args = parser.parse_args()
    from calibration.ffn_b4 import execute_formal, failure, plan, save
    from calibration.ffn_b4_runtime import Runtime
    frozen = json.loads((args.worker/"parent_plan.json").read_text(encoding="utf-8"))
    # The parent must have completed the current minimum smoke before spawning.
    smoke = json.loads((args.worker.parents[2]/"smoke/result.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or len(smoke["cells"]) != 4 or not all(smoke_pass(c) for c in smoke["cells"]):
        raise ValueError("parent smoke has not passed")
    if key_id(plan(args.flagscale_root)) != key_id(frozen):
        raise ValueError("worker source/plan differs from parent")
    runtime = None
    try:
        import torch
        with torch.no_grad():
            runtime = Runtime(args.worker, frozen["active_recipe"], args.flagscale_root)
            save(args.worker/"runtime_identity.json", runtime.identity_record)
            execute_formal(runtime, frozen, args.worker/"formal", group_id=args.group)
    except Exception as error:
        save(args.worker/"failure.json", failure(error, runtime, group_id=args.group))
        raise


if __name__ == "__main__":
    main()
