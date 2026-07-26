"""fp32-scale activation quantization for the MoE path (M2-I13 part 3).

MPK's grouped MoE GEMMs consume E4M3 values plus a FLOAT32 [rows, K/128] scale
-- `quantize_fp8_layer(..., scale_ue8m0=False)`, i.e.
per_token_group_quantize_fp8_task_impl<..., SCALE_UE8M0 = false>. The in-tree
quantize test (sm100_quantize_fp8/test_quantize_fp8.py) covers only the packed
UE8M0 variant and only 2D inputs; the w2 expert input is 3D
[batch, topk, moe_intermediate], which flattens to rows = batch*topk. This
test pins the fp32-scale variant at both MoE shapes against the contract in
docs/qwen35/vllm-graph.md 3.4 / v1-architecture.md 6.1:

    absmax = max(max|x| over the 128-group, 1e-10)
    scale  = absmax / 448
    q      = clamp(x / scale, -448, 448)   <- DIVISION, clamp BEFORE the cast
    out    = RN-even e4m3 cast of q

Bit-exactness is the bar, not a tolerance. p10b showed that two INDEPENDENT
implementations of this primitive (vLLM CUTLASS vs HF Triton) disagree at
0.5-3% of positions, always by exactly 1 e4m3 ULP, plus a rare sub-LSB
group-scale nudge (probes/fp8/p10b_activation_quant_disagreement.json). That
bound applies between different implementations; here the reference computes
the identical fp32 expression, so any disagreement at all would be a real
defect -- and the test still reports the ULP histogram so a future divergence
is classified, not just flagged.

Run:  python test_quantize_fp8_f32scale_moe.py
"""

import json
import os
import sys

import torch

import runtime_kernel_blackwell_fp8_moe_qwen35 as moe

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10


def _true_div(t, denom):
    """`t / <python float>` is NOT the same thing as fp32 division in PyTorch:
    the scalar overload is lowered to a RECIPROCAL MULTIPLY, which is 1 ULP off
    for divisors like 448 whose reciprocal is inexact (measured: the same bits
    as `t * (1/448)`, one ULP from `torch.div(t, tensor(448))`). The kernel
    divides, so the reference must divide -- this is the reference-side face of
    vllm-graph.md 3.4 item 3's "division, not reciprocal"."""
    return torch.div(t, torch.tensor(denom, dtype=torch.float32, device=t.device))


def reference_quantize(x_bf16):
    shape = x_bf16.shape
    k = shape[-1]
    xf = x_bf16.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = _true_div(absmax, FP8_MAX)
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (
        q.reshape(shape).to(torch.float8_e4m3fn).contiguous(),
        scale.reshape(-1, k // BLOCK).float().contiguous(),
    )


def ulp_histogram(a_fp8, b_fp8):
    """e4m3 codes are monotone in magnitude within a sign, so the code-space
    distance IS the ULP distance for same-signed values."""
    a = a_fp8.view(torch.uint8).to(torch.int16).flatten()
    b = b_fp8.view(torch.uint8).to(torch.int16).flatten()
    diff = (a - b).abs()
    hist = {}
    for d in diff.unique().tolist():
        hist[int(d)] = int((diff == d).sum())
    return hist


def run_case(label, shape, seed):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16, device=dev, generator=g)
    # Widen the dynamic range so some groups clamp and some are tiny: the
    # eps/clamp branches must be exercised, not just the generic path.
    x = x * torch.exp2(
        torch.randint(-8, 9, (*shape[:-1], 1), device=dev, generator=g).bfloat16()
    )
    hidden = shape[-1]
    rows = 1
    for d in shape[:-1]:
        rows *= d

    q = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.zeros((rows, hidden // BLOCK), dtype=torch.float32, device=dev)
    moe.quantize_fp8_f32scale_sm100(x.contiguous(), q, s)

    ref_q, ref_s = reference_quantize(x)
    hist = ulp_histogram(q, ref_q)
    scale_exact = torch.equal(s, ref_s)
    print(
        f"  {label:<26} shape={tuple(shape)} rows={rows} "
        f"scale_bit_exact={scale_exact} value_ulp_histogram={hist}"
    )
    assert scale_exact, (
        f"{label}: fp32 group scales differ from absmax/448 "
        f"(max |delta| {(s - ref_s).abs().max().item():.3e})"
    )
    assert set(hist) == {0}, (
        f"{label}: quantized values differ from the reference primitive "
        f"(ULP histogram {hist}); p10b's 1-ULP bound covers INDEPENDENT "
        f"implementations, not the identical fp32 expression"
    )
    # A zero group must take the eps branch, not divide by zero.
    return {"case": label, "rows": rows, "hidden": hidden,
            "scale_bit_exact": scale_exact, "ulp_histogram": hist}


def run_zero_group_case():
    """absmax = 0 must clamp to eps = 1e-10, giving scale = 1e-10/448 and an
    all-zero output -- never a NaN."""
    dev = "cuda"
    x = torch.zeros((4, 512), dtype=torch.bfloat16, device=dev)
    q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
    s = torch.zeros((4, 4), dtype=torch.float32, device=dev)
    moe.quantize_fp8_f32scale_sm100(x, q, s)
    ref_q, ref_s = reference_quantize(x)
    assert torch.equal(s, ref_s), (s[0, 0].item(), ref_s[0, 0].item())
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert not torch.isnan(s).any()
    print(f"  {'all-zero group':<26} scale={s[0, 0].item():.6e} "
          f"(eps branch, no NaN)")


def main():
    print("=== fp32-scale MoE activation quantization ===")
    results = []
    # w13 expert input: [tokens, hidden_size]
    for rows in (1, 2, 4, 8, 16):
        results.append(run_case(f"w13 input rows={rows}", (rows, 2048), 900 + rows))
    # w2 expert input: [tokens, topk, moe_intermediate] -> rows = tokens*topk
    for tokens in (1, 2, 8, 16):
        results.append(
            run_case(f"w2 input tokens={tokens}", (tokens, 8, 512), 1900 + tokens)
        )
    run_zero_group_case()

    out_path = os.environ.get("P2_ACTIVATION_JSON")
    if out_path:
        payload = {
            "primitive": "per_token_group_quantize_fp8_task_impl "
            "<SCALE_UE8M0=false> == quantize_fp8_layer(scale_ue8m0=False)",
            "contract": "vllm-graph.md 3.4 / v1-architecture.md 6.1: group 128, "
            "absmax = max(max|x|, 1e-10), scale = absmax/448, x/scale "
            "(division), clamp to +-448 BEFORE the RN-even e4m3 cast",
            "shapes_covered": [
                {"kind": "w13 activation [tokens, 2048]", "rows": [1, 2, 4, 8, 16]},
                {"kind": "w2 activation [tokens, 8, 512]",
                 "rows": [8, 16, 64, 128]},
            ],
            "standalone_build": {
                "all_scales_bit_exact_vs_true_division": all(
                    r["scale_bit_exact"] for r in results
                ),
                "all_values_bit_exact": all(
                    set(r["ulp_histogram"]) == {0} for r in results
                ),
                "zero_group_eps_branch_ok": True,
            },
            "reference_pitfall": "`t / 448.0` with a PYTHON scalar is lowered "
            "by PyTorch to a reciprocal multiply and is 1 fp32 ULP from true "
            "division; a torch reference must use torch.div with a 0-dim "
            "tensor or it mis-reports the kernel as wrong.",
            "cases": results,
        }
        prior = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                prior = json.load(f)
        prior.update(payload)
        with open(out_path, "w") as f:
            json.dump(prior, f, indent=1)
        print(f"WROTE {out_path}")
    print("ALL FP32-SCALE MOE QUANTIZE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
