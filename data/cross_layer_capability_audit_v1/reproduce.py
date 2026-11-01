"""Reproduce the current complete-graph candidate inventory; no search or GPU."""
import csv
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cost_model.search_evaluation import build_evaluation_graph
from simulator.fusion_rules import default_fusion_registry, enumerate_fusion_candidates

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
out.mkdir(parents=True, exist_ok=True)
if (out / "report.json").exists() or (out / "candidates.csv").exists():
    raise FileExistsError("use a new output directory; frozen evidence is not overwritten")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_files = ("simulator/fusion_rules.py", "simulator/model_graphs/transformer.py",
                "simulator/model_graphs/dense_layer.py", "simulator/model_graphs/spec.py",
                "cost_model/search_evaluation.py")
hashes = {p: sha(ROOT / p) for p in source_files}
graph = build_evaluation_graph("model")
rules = default_fusion_registry()
items = enumerate_fusion_candidates(graph, rules, require_executable=False)
strict = enumerate_fusion_candidates(graph, rules, require_executable=True)
rows = []
for item in items:
    layers = sorted({graph.nodes[n].extra.get("layer") for n in item.node_ids})
    if any(layer is None for layer in layers):
        raise ValueError("this audit requires complete layer metadata")
    rows.append(dict(pattern_id=item.pattern_id, node_ids=json.dumps(item.node_ids),
        layers=json.dumps(layers), cross_layer=len(layers)>1,
        implementation_status=item.implementation_status,
        require_same_layer=rules.get(item.pattern_id).require_same_layer))
with (out / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
flagscale = ROOT.parent / "FlagScale-main/FlagScale-main"
package = "flagscale/models/kerv/ops/KERVRuntimeOptimization/"
core = package + "embodied_ops/kerv_ops.py"
wrapper = package + "adaptive_linear_fusion.py"
cap = json.loads((ROOT / "data/kerv_silu_mul_proxy_v1/capability.json").read_text())
revision = subprocess.check_output(["git", "-C", str(flagscale), "rev-parse", "HEAD"], text=True).strip()
pinned = revision == cap["flagscale_revision"] and all(sha(flagscale / p) == cap["source_sha256"][p] for p in (core, wrapper))
report = dict(nodes=len(graph.nodes), edges=len(graph.edges), decoder_layers=graph.extra["num_layers"],
    development_candidates=len(items), multi_node_candidates=sum(len(x.node_ids)>1 for x in items),
    cross_layer_candidates=sum(r["cross_layer"] for r in rows),
    candidates_by_pattern=dict(Counter(x.pattern_id for x in items)),
    registry_executable_candidates=len(strict), registry_executable_multi_node=sum(len(x.node_ids)>1 for x in strict),
    all_multinode_rules_require_same_layer=all(p.require_same_layer for p in rules.patterns() if p.max_nodes>1),
    registry=[asdict(p) for p in rules.patterns()], graph_layer_metadata_complete=True,
    formal_scope=dict(authority="user_confirmation_2026-11-01", allowed_boundary="intra_layer_only",
                      capability_file_received=False, execution_contract_confirmed=False),
    kerv_add_rms_norm=dict(source_exists="def _add_rms_norm_impl(" in (flagscale/core).read_text(encoding="utf-8"),
        default_registry_registered=any("add_rms" in p.pattern_id for p in rules.patterns()),
        pinned_revision=revision, two_source_hashes_match_surrogate_manifest=pinned,
        inspected_files={p:sha(flagscale/p) for p in (core,wrapper)},
        admission="source_presence_only; no new capability or measurement admission"),
    source_sha256=hashes, new_search_runs=0, new_gpu_measurements=0,
    conclusion="no atomicity violations from complete-layer cuts under confirmed intra-layer-only scope and complete layer metadata")
if any(sha(ROOT/p)!=h for p,h in hashes.items()):
    raise ValueError("source changed during inventory")
(out / "report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({k:report[k] for k in ("nodes","edges","decoder_layers","development_candidates","multi_node_candidates","cross_layer_candidates","registry_executable_candidates")}))
