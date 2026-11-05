"""Diagnostic only: replay-index effect for the batch-01 Copy graphs."""
import sys; sys.path.insert(0, '/root/private_data/LJ/Fusion-placement-search')
import json, sys, time
from pathlib import Path
import torch
from calibration import stage_size_calibration as ssc
out = Path(sys.argv[1])
torch.cuda.set_device(0)
rows = []
with torch.no_grad():
    entries, _, warm = ssc.copy_graphs(torch, {"seed": 20261107}, out, {})
    ids = list(entries)
    for rnd in range(7):
        order = ids if rnd % 2 == 0 else ids[::-1]
        for cid in order:
            e = entries[cid]
            for k in range(6):
                e["invalidate"]()
                for d in e["devices"]: torch.cuda.synchronize(d)
                t = ssc.timed(torch, e)
                e["verify"]()
                rows.append({"round": rnd, "case": cid, "replay_in_round": k,
                             "us": t["elapsed_ms_per_50_calls"] * 1000 / 50 if t["error"] is None else None})
(out / "probe.json").write_text(json.dumps({"scope": "diagnostic, not calibration", "warmup": warm, "rows": rows}, indent=1))
import statistics as st
for cid in ids:
    print(cid)
    for k in range(6):
        v = [r["us"] for r in rows if r["case"] == cid and r["replay_in_round"] == k]
        print(f"  replay{k}: med={st.median(v):8.2f} max={max(v):8.2f}  {[round(x,1) for x in v]}")
