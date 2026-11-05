"""GPU correctness check (no timing) of the LightOP Add+RMSNorm adapter on GPU0 and GPU1.

Reference: s = x + residual; y = s * rsqrt(mean(s^2) + eps) * weight, in FP32
and in BF16 eager. Checks both outputs, that graph inputs are unchanged, and
that repeated calls are deterministic. Writes one JSON receipt.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def reference(x, residual, weight, eps, dtype):
    import torch
    s = x.to(dtype) + residual.to(dtype)
    y = s.float() * torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return s, y


def main(output):
    import torch
    from cost_model.search_evaluation import build_evaluation_graph
    from integration.lightop.add_rmsnorm import EPS, SHAPE, bind, declared_binding
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    graph = build_evaluation_graph("layer")
    group = ("L000.residual_add_1", "L000.pre_mlp_norm")
    x_node, res_node = [e.src for e in graph.in_edges(group[0])]
    report = {"scope": "correctness only; no timing, no Profile", "group": list(group),
              "implementation": declared_binding(), "devices": {}}
    for device in (0, 1):
        dev = torch.device(f"cuda:{device}")
        g = torch.Generator(device=dev).manual_seed(20261114 + device)
        x, residual = [torch.randn(SHAPE, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(2)]
        weight = (1 + .1 * torch.randn(SHAPE[-1], device=dev, generator=g)).to(torch.bfloat16)
        snap = [t.clone() for t in (x, residual, weight)]
        call = bind(graph, group, {x_node: x, res_node: residual}, weight)
        add_out, norm_out = call()
        torch.cuda.synchronize(dev)
        first = (add_out.clone(), norm_out.clone())
        call(); torch.cuda.synchronize(dev)
        s32, y32 = reference(x, residual, weight, EPS, torch.float32)
        s16, y16 = reference(x, residual, weight, EPS, torch.bfloat16)
        err = lambda a, b: float((a.float() - b.float()).abs().max())
        row = {"inputs_unchanged": all(torch.equal(a, b) for a, b in zip((x, residual, weight), snap)),
               "deterministic_repeat": torch.equal(first[0], add_out) and torch.equal(first[1], norm_out),
               "add_max_abs_err_vs_fp32": err(add_out, s32), "add_max_abs_err_vs_bf16": err(add_out, s16),
               "norm_max_abs_err_vs_fp32": err(norm_out, y32), "norm_max_abs_err_vs_bf16_eager": err(norm_out, y16)}
        try:
            torch.testing.assert_close(add_out.float(), s32, rtol=.01, atol=.01)
            torch.testing.assert_close(norm_out.float(), y32, rtol=.02, atol=.02)
            row["within_bf16_tolerance"] = True
        except AssertionError as error:
            row["within_bf16_tolerance"], row["tolerance_failure"] = False, str(error)[:500]
        row["tolerance"] = {"add": {"rtol": .01, "atol": .01}, "rmsnorm": {"rtol": .02, "atol": .02}}
        row["observation"] = call.observation["tensors"]
        report["devices"][str(device)] = row
    report["passed"] = all(r["inputs_unchanged"] and r["deterministic_repeat"] and r["within_bf16_tolerance"]
                           for r in report["devices"].values())
    (output / "numerics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({d: {k: v for k, v in r.items() if k not in ("observation",)}
                      for d, r in report["devices"].items()} | {"passed": report["passed"]}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
