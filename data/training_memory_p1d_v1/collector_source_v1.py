"""P1-D: validate the LLMem-based training-memory estimator on BW1000.

Phases (same plan / run / review pattern as the other collectors):

  plan    CPU only. Freeze config + per-placement predictions into plan.json.
  run     GPU. One subprocess per placement (clean allocator per case);
          writes measured_<placement>.json.
  review  CPU only. Per-term error table -> review.json; with --write-baseline
          also a calibrated M_base usable as MemoryBaseline.

Harness semantics (stated, not FlagScale itself): HF Qwen2 decoder layers with
eager (unfused) attention, random weights, layers placed by hand across
cuda:0/1 with explicit .to() transfers. Megatron-style mixed precision is
emulated: BF16 weights, FP32 main params, FP32 main_grad filled by
post-accumulate hooks (BF16 .grad freed per parameter), FP32 AdamW states.
So the state is 18 B/param like FlagScale, but per-parameter BF16 wgrad
buffers exist (gradient_accumulation_fusion=False in the prediction), and
HF's loss upcasts logits to FP32.

Steps per case: context init -> weights -> training state -> warm-up step ->
measured step (forward / backward / optimizer windows) -> GEMM workspace probe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "training_memory_validation_v1"
NUM_LAYERS = 36
# Split = first layer placed on cuda:1. embedding / lm_head (tied) stay on 0.
DEFAULT_PLACEMENTS = {"all_gpu0": None, "split_18": 18, "split_22": 22}


def device_of(node_id: str, layer: int, split):
    if split is None or node_id in ("embedding", "lm_head"):
        return 0
    return 0 if layer < split else 1


# ---------------------------------------------------------------- plan (CPU)

def _graph(seq: int, mbs: int):
    from dataclasses import replace

    from cost_model.search_evaluation import qwen25_3b_spec
    from simulator.model_graphs.transformer import build_model_graph

    spec = replace(qwen25_3b_spec(), seq_length=seq, micro_batch_size=mbs)
    return spec, build_model_graph(spec)


def predict(seq: int, mbs: int, placements: dict, loss_factor: float) -> dict:
    from cost_model import StructuralCostModel
    from cost_model.training_memory import (
        TrainingLifetimeConfig,
        TrainingMemoryConfig,
        evaluate_training_candidate,
    )
    from simulator import FusionGroup, Plan, hardware_from_dict

    spec, graph = _graph(seq, mbs)
    hw = hardware_from_dict({
        "name": "event-order-only", "comm_bw": 5e10, "comm_latency": 1e-5, "link_bw": 5e10,
        "devices": [{"name": f"d{i}", "flops": 1e14, "mem_bw": 1e12, "max_mem": 1e15,
                     "calibrated": True} for i in range(2)],
    })
    life = TrainingLifetimeConfig(loss_fp32_logits_factor=loss_factor,
                                  gradient_accumulation_fusion=False)
    out = {}
    for name, split in placements.items():
        plan = Plan(name=name, groups=[
            FusionGroup([nid], device_of(nid, int(node.extra.get("layer", -1)), split),
                        {"pattern_id": "unfused"})
            for nid, node in graph.nodes.items()
        ])
        ev = evaluate_training_candidate(
            graph, plan, hw, (1e15, 1e15), config=TrainingMemoryConfig(),
            lifetime_model=StructuralCostModel(strict_fusion=False), lifetime_config=life,
        )
        b = ev.breakdown
        params = b["params_per_device"]
        out[name] = {
            "split": split,
            "params_per_device": params,
            "weight_bytes": [p * 2 for p in params],
            "training_state_bytes": b["training_state_bytes"],
            "dynamic_peak_bytes": b["lifetime_bytes_excluding_weights"],
            "retained_at_forward_end": b["retained_at_forward_end"],
            "m_lm_bytes": b["m_lm_bytes"],
            "components_at_peak": b["lifetime_components_at_peak"],
            "cross_device_bytes": b["forward_cross_device_bytes"] + b["backward_cross_device_bytes"],
            "peak_excluding_m_base": list(ev.peak_per_device.values()),
        }
    return {"spec": {"seq_length": spec.seq_length, "micro_batch_size": spec.micro_batch_size,
                     "num_layers": spec.num_layers, "dtype": spec.dtype},
            "lifetime_config": {"loss_fp32_logits_factor": loss_factor,
                                "gradient_accumulation_fusion": False},
            "placements": out}


# ----------------------------------------------------------------- run (GPU)

def _snapshot(torch, dev):
    free, total = torch.cuda.mem_get_info(dev)
    return {"allocated": torch.cuda.memory_allocated(dev),
            "reserved": torch.cuda.memory_reserved(dev),
            "device_used": total - free, "device_total": total}


def run_case(name: str, split, seq: int, mbs: int, seed: int = 0) -> dict:
    import torch
    from transformers import Qwen2Config
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
    )

    if torch.cuda.device_count() < 2:
        raise RuntimeError("needs two visible GPUs")
    devs = [torch.device("cuda", i) for i in range(2)]
    rec = {"placement": name, "split": split, "torch": torch.__version__,
           "hip": getattr(torch.version, "hip", None),
           "device_names": [torch.cuda.get_device_name(d) for d in devs]}

    # 1. context only
    for d in devs:
        torch.zeros(1, device=d)
        torch.cuda.synchronize(d)
    rec["context"] = [_snapshot(torch, d) for d in devs]

    cfg = Qwen2Config(
        vocab_size=151936, hidden_size=2048, intermediate_size=11008, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=16, num_key_value_heads=2, max_position_embeddings=32768,
        rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=True, torch_dtype=torch.bfloat16,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    dev_layer = [devs[device_of(f"L{i:03d}", i, split)] for i in range(NUM_LAYERS)]
    dev_norm = devs[device_of("final_norm", NUM_LAYERS, split)]
    d0 = devs[0]

    # 2. BF16 weights, built directly on their device
    with torch.device(d0):
        embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    layers = []
    for i in range(NUM_LAYERS):
        with torch.device(dev_layer[i]):
            layers.append(Qwen2DecoderLayer(cfg, i).to(torch.bfloat16))
    with torch.device(dev_norm):
        norm = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.bfloat16)
    modules = [embed, *layers, norm]
    params = [p for m in modules for p in m.parameters()]
    with torch.no_grad():
        for p in params:
            if p.dim() > 1:
                p.normal_(0.0, 0.02)
    rec["param_count_per_device"] = [sum(p.numel() for p in params if p.device == d) for d in devs]
    rotary = {d: Qwen2RotaryEmbedding(cfg, device=d) for d in set(dev_layer)}
    for d in devs:
        torch.cuda.synchronize(d)
    rec["weights"] = [_snapshot(torch, d) for d in devs]

    # 3. Megatron-style state: FP32 main params, FP32 main_grad, FP32 Adam m/v
    main = [p.detach().float() for p in params]
    main_grad = [torch.zeros_like(m) for m in main]
    for p, m, g in zip(params, main, main_grad):
        m.grad = g

        def hook(param, g=g):
            g.add_(param.grad)      # in-place BF16 -> FP32 accumulate
            param.grad = None       # free the BF16 wgrad buffer right away
        p.register_post_accumulate_grad_hook(hook)
    try:
        opt = torch.optim.AdamW(main, lr=1e-6, fused=True)
        rec["optimizer_impl"] = "fused"
    except (RuntimeError, TypeError) as exc:
        opt = torch.optim.AdamW(main, lr=1e-6, foreach=False)
        rec["optimizer_impl"] = f"single_tensor ({exc.__class__.__name__})"
    for m in main:  # eager state allocation, as Megatron does at init
        st = opt.state[m]
        st["step"] = torch.zeros((), dtype=torch.float32, device=m.device) \
            if rec["optimizer_impl"] == "fused" else torch.tensor(0.0)
        st["exp_avg"] = torch.zeros_like(m)
        st["exp_avg_sq"] = torch.zeros_like(m)
    for d in devs:
        torch.cuda.synchronize(d)
    rec["state"] = [_snapshot(torch, d) for d in devs]

    ids = torch.randint(0, cfg.vocab_size, (mbs, seq), device=d0)
    labels = ids.clone()
    pos = torch.arange(seq, device=d0).unsqueeze(0).expand(mbs, -1)
    transfer_bytes = [0]

    def move(t, d):
        if t.device != d:
            transfer_bytes[0] += t.numel() * t.element_size()
            return t.to(d)
        return t

    def step(measure: bool):
        windows = {}

        def window(label):
            for d in devs:
                torch.cuda.synchronize(d)
            if measure:
                windows[label] = {
                    "peak_allocated": [torch.cuda.max_memory_allocated(d) for d in devs],
                    "peak_reserved": [torch.cuda.max_memory_reserved(d) for d in devs],
                    "end": [_snapshot(torch, d) for d in devs],
                }
                for d in devs:
                    torch.cuda.reset_peak_memory_stats(d)

        for d in devs:
            torch.cuda.reset_peak_memory_stats(d)
        transfer_bytes[0] = 0
        pe, masks = {}, {}
        h = embed(ids)
        for i, layer in enumerate(layers):
            d = dev_layer[i]
            h = move(h, d)
            if d not in pe:
                pe[d] = rotary[d](h, move(pos, d))
                masks[d] = torch.full((seq, seq), float("-inf"), device=d,
                                      dtype=torch.bfloat16).triu(1)[None, None]
            h = layer(h, attention_mask=masks[d], position_embeddings=pe[d])
        h = norm(move(h, dev_norm))
        logits = torch.nn.functional.linear(move(h, d0), embed.weight)  # tied lm_head
        loss = ForCausalLMLoss(logits, labels, cfg.vocab_size)
        window("forward")
        fwd_transfer = transfer_bytes[0]
        loss.backward()
        del logits, loss, h
        window("backward")
        opt.step()
        for g in main_grad:
            g.zero_()
        with torch.no_grad():
            for p, m in zip(params, main):
                p.copy_(m)
        window("optimizer")
        return windows, fwd_transfer

    step(measure=False)                  # warm-up: BLAS handles, workspaces, allocator
    t0 = time.time()
    rec["step"], fwd_bytes = step(measure=True)
    rec["step_seconds"] = time.time() - t0
    # Backward sends the gradient of every forward transfer back (same sizes).
    rec["cross_device_bytes"] = 2 * fwd_bytes if split is not None else 0

    # 4. GEMM workspace probe: peak minus inputs/outputs of an FFN-sized GEMM
    probe = []
    for d in devs:
        x = torch.randn(mbs * seq, cfg.hidden_size, device=d, dtype=torch.bfloat16)
        w = torch.randn(cfg.intermediate_size, cfg.hidden_size, device=d, dtype=torch.bfloat16)
        torch.cuda.synchronize(d)
        base = torch.cuda.memory_allocated(d)
        torch.cuda.reset_peak_memory_stats(d)
        y = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize(d)
        probe.append(max(0, torch.cuda.max_memory_allocated(d) - base - y.numel() * y.element_size()))
        del x, w, y
    rec["gemm_workspace_probe"] = probe
    return rec


# ---------------------------------------------------------------- review (CPU)

def _err(measured, predicted):
    diff = measured - predicted
    rel = diff / measured if measured else (0.0 if not diff else math.inf)
    return {"measured": measured, "predicted": predicted, "error": diff, "rel_error": rel}


def review(plan: dict, measured: dict) -> dict:
    out = {}
    for name, pred in plan["predictions"]["placements"].items():
        if name not in measured:
            out[name] = {"status": "missing_measurement"}
            continue
        m = measured[name]
        per_dev = []
        for d in range(2):
            ctx = m["context"][d]
            w_alloc = m["weights"][d]["allocated"] - ctx["allocated"]
            s_alloc = m["state"][d]["allocated"] - ctx["allocated"]
            fwd, bwd, opt = (m["step"][k] for k in ("forward", "backward", "optimizer"))
            state_alloc_live = m["state"][d]["allocated"]
            dyn_peak = max(fwd["peak_allocated"][d], bwd["peak_allocated"][d]) - state_alloc_live
            retained = fwd["end"][d]["allocated"] - state_alloc_live
            alloc_peak = max(w["peak_allocated"][d] for w in (fwd, bwd, opt))
            reserved_peak = max(w["peak_reserved"][d] for w in (fwd, bwd, opt))
            # M_base: runtime context outside the allocator + allocator slack at peak
            context = ctx["device_used"] - ctx["reserved"]
            m_base = context + (reserved_peak - alloc_peak)
            pred_peak = pred["peak_excluding_m_base"][d]
            per_dev.append({
                "m_base": {"measured": m_base, "context": context,
                           "allocator_slack_at_peak": reserved_peak - alloc_peak,
                           "predicted": 0.0, "note": "uncalibrated in the estimator"},
                "weights": _err(w_alloc, pred["weight_bytes"][d]),
                "training_state": _err(s_alloc, pred["training_state_bytes"][d]),
                "retained_activations": _err(retained, pred["retained_at_forward_end"][d]),
                "dynamic_peak": _err(dyn_peak, pred["dynamic_peak_bytes"][d]),
                "optimizer_step_transient": _err(
                    opt["peak_allocated"][d] - state_alloc_live, 0.0),
                "gemm_workspace_probe": _err(m["gemm_workspace_probe"][d], 0.0),
                "allocated_peak": _err(alloc_peak - ctx["allocated"], pred_peak),
                "device_peak_with_m_base": _err(reserved_peak + context, pred_peak + m_base),
                "param_count": {"measured": m["param_count_per_device"][d],
                                "graph": pred["params_per_device"][d]},
            })
        out[name] = {
            "per_device": per_dev,
            "communication": _err(m["cross_device_bytes"], pred["cross_device_bytes"]),
            "optimizer_impl": m.get("optimizer_impl"),
        }
    return {"scope": SCOPE, "spec": plan["predictions"]["spec"], "placements": out}


def baseline_from_review(rev: dict, path: str) -> dict:
    """Largest measured M_base per device across placements (conservative)."""

    vals = [[p["per_device"][d]["m_base"]["measured"] for p in rev["placements"].values()
             if "per_device" in p] for d in range(2)]
    if not all(vals):
        raise ValueError("no measured placement to calibrate M_base from")
    return {"per_device": [max(v) for v in vals], "calibrated": True,
            "source": f"{SCOPE}:{path}"}


# ---------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("plan", "run", "run-case", "review"))
    ap.add_argument("--output", type=Path, required=True, help="run directory")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--loss-fp32-factor", type=float, default=1.0)
    ap.add_argument("--placement", help="run-case only")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)
    out = args.output
    plan_path = out / "plan.json"

    if args.phase == "plan":
        out.mkdir(parents=True, exist_ok=False)
        plan = {"scope": SCOPE, "placements": DEFAULT_PLACEMENTS,
                "predictions": predict(args.seq, args.mbs, DEFAULT_PLACEMENTS, args.loss_fp32_factor),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        plan_path.write_text(json.dumps(plan, indent=2))
        print(json.dumps({"phase": "plan", "placements": list(DEFAULT_PLACEMENTS)}))
        return 0

    plan = json.loads(plan_path.read_text())
    spec = plan["predictions"]["spec"]
    if args.phase == "run-case":
        rec = run_case(args.placement, plan["placements"][args.placement],
                       spec["seq_length"], spec["micro_batch_size"])
        (out / f"measured_{args.placement}.json").write_text(json.dumps(rec, indent=2))
        return 0

    if args.phase == "run":
        status = {}
        for name in plan["placements"]:
            cmd = [sys.executable, "-m", "calibration.training_memory_validation", "run-case",
                   "--output", str(out), "--placement", name]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": str(ROOT)})
            (out / f"log_{name}.txt").write_text(proc.stdout + proc.stderr)
            status[name] = proc.returncode
            print(json.dumps({"placement": name, "returncode": proc.returncode}), flush=True)
        return 0 if all(v == 0 for v in status.values()) else 1

    measured = {p.stem.removeprefix("measured_"): json.loads(p.read_text())
                for p in sorted(out.glob("measured_*.json"))}
    rev = review(plan, measured)
    (out / "review.json").write_text(json.dumps(rev, indent=2))
    if args.write_baseline:
        (out / "m_base.json").write_text(json.dumps(baseline_from_review(rev, str(out)), indent=2))
    for name, p in rev["placements"].items():
        if "per_device" not in p:
            print(name, p["status"])
            continue
        for d, row in enumerate(p["per_device"]):
            print(name, f"gpu{d}", {k: (round(v["rel_error"] * 100, 2) if "rel_error" in v else
                                        round(v["measured"] / 1e9, 3))
                                    for k, v in row.items() if k not in ("param_count",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
