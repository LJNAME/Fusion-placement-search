"""P1-D follow-up: audit which tensors autograd saves per Qwen2 decoder layer.

GPU, single device. Builds the same HF Qwen2 eager decoder layer as
training_memory_validation (BF16, seq/mbs from argv) and records:

  saved_tensors: every tensor packed by autograd (saved_tensors_hooks), with
      op (grad_fn name), shape, dtype, bytes, and storage identity so tensors
      saved by several ops are counted once;
  layer_retained: allocated after one layer's forward minus before, with the
      input still alive (the per-layer activation the estimator must match);
  persistent: allocated bytes that stay live after warm-up fwd+bwd+step with
      no activations held (BLAS workspaces etc.);
  optimizer: fused AdamW peak minus pre-step allocated (true step transient).

Writes one JSON; does not modify any plan or prediction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run(seq: int, mbs: int) -> dict:
    import torch
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

    import transformers

    d = torch.device("cuda", 0)
    torch.zeros(1, device=d)
    torch.cuda.synchronize(d)
    cfg = Qwen2Config(
        vocab_size=151936, hidden_size=2048, intermediate_size=11008, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=2, max_position_embeddings=32768,
        rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=True, torch_dtype=torch.bfloat16,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    with torch.device(d):
        layers = [Qwen2DecoderLayer(cfg, i).to(torch.bfloat16) for i in range(2)]
    rotary = Qwen2RotaryEmbedding(cfg, device=d)
    pos = torch.arange(seq, device=d).unsqueeze(0).expand(mbs, -1)
    mask = torch.full((seq, seq), float("-inf"), device=d, dtype=torch.bfloat16).triu(1)[None, None]
    rec = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
           "transformers": transformers.__version__, "seq": seq, "mbs": mbs,
           "blas_backend": str(torch.backends.cuda.preferred_blas_library())}

    def make_input():
        return torch.randn(mbs, seq, cfg.hidden_size, device=d, dtype=torch.bfloat16,
                           requires_grad=True)

    with torch.no_grad():
        pe = rotary(make_input(), pos)

    # Persistent bytes after a warm-up fwd+bwd with nothing held.
    torch.cuda.synchronize(d)
    before_warm = torch.cuda.memory_allocated(d)
    x = make_input()
    layers[0](x, attention_mask=mask, position_embeddings=pe).float().sum().backward()
    for p in layers[0].parameters():
        p.grad = None
    del x
    torch.cuda.synchronize(d)
    rec["persistent_after_warmup"] = torch.cuda.memory_allocated(d) - before_warm

    # Saved tensors of layer 1 (layer 0 above warmed kernels / workspaces).
    saved = []

    def pack(t):
        try:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        except Exception:  # noqa: BLE001
            key = (t.data_ptr(), t.numel() * t.element_size())
        saved.append({"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
                      "bytes": t.numel() * t.element_size(), "storage_ptr": key[0],
                      "storage_bytes": key[1], "is_param": isinstance(t, torch.nn.Parameter),
                      "device": str(t.device)})
        return t

    x = make_input()
    param_ptrs = {p.untyped_storage().data_ptr() for p in layers[1].parameters()}
    torch.cuda.synchronize(d)
    base = torch.cuda.memory_allocated(d)
    torch.cuda.reset_peak_memory_stats(d)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = layers[1](x, attention_mask=mask, position_embeddings=pe)
    torch.cuda.synchronize(d)
    rec["layer_retained"] = torch.cuda.memory_allocated(d) - base
    rec["layer_retained_minus_output"] = rec["layer_retained"] - y.numel() * y.element_size()
    rec["layer_forward_peak_over_base"] = torch.cuda.max_memory_allocated(d) - base

    # Attribute each saved tensor to the grad_fn that saved it.
    owners: dict[int, list[str]] = {}
    seen = set()

    def walk(fn):
        if fn is None or id(fn) in seen:
            return
        seen.add(id(fn))
        for attr in dir(fn):
            if attr.startswith("_saved_"):
                try:
                    v = getattr(fn, attr)
                except Exception:  # noqa: BLE001
                    continue
                vs = v if isinstance(v, (tuple, list)) else (v,)
                for t in vs:
                    if isinstance(t, torch.Tensor):
                        owners.setdefault(t.untyped_storage().data_ptr(), []).append(
                            f"{type(fn).__name__}.{attr[7:]}")
        for nxt, _ in fn.next_functions:
            walk(nxt)

    walk(y.grad_fn)
    uniq = {}
    for s in saved:
        s["owners"] = sorted(set(owners.get(s["storage_ptr"], [])))
        s["is_param"] = s["is_param"] or s["storage_ptr"] in param_ptrs
        uniq.setdefault(s["storage_ptr"], s)
    x_ptr = x.untyped_storage().data_ptr()
    rows = list(uniq.values())
    for s in rows:
        s["role"] = ("parameter" if s["is_param"] else "layer_input" if s["storage_ptr"] == x_ptr
                     else "activation")
        s.pop("storage_ptr")
    rec["saved_tensors"] = sorted(rows, key=lambda s: -s["storage_bytes"])
    act = [s for s in rows if s["role"] == "activation"]
    rec["saved_activation_storage_bytes"] = sum(s["storage_bytes"] for s in act)
    rec["saved_activation_by_dtype"] = {}
    for s in act:
        rec["saved_activation_by_dtype"][s["dtype"]] = (
            rec["saved_activation_by_dtype"].get(s["dtype"], 0) + s["storage_bytes"])
    del y, x

    # Fused AdamW step transient on 18 B/param state for layer 1.
    main = [p.detach().float() for p in layers[1].parameters()]
    for m in main:
        m.grad = torch.zeros_like(m)
    opt = torch.optim.AdamW(main, lr=1e-6, fused=True)
    opt.step()  # lazy state init
    torch.cuda.synchronize(d)
    pre = torch.cuda.memory_allocated(d)
    torch.cuda.reset_peak_memory_stats(d)
    opt.step()
    torch.cuda.synchronize(d)
    rec["optimizer_step_transient"] = torch.cuda.max_memory_allocated(d) - pre
    rec["optimizer_step_residual"] = torch.cuda.memory_allocated(d) - pre
    rec["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="JSON file (must not exist)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--mbs", type=int, default=1)
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; new batches go to new files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rec = run(args.seq, args.mbs)
    args.output.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "saved_tensors"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
