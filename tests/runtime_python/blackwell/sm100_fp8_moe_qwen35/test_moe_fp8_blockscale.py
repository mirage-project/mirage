"""Synthetic correctness for the fp32-block-scale grouped FP8 MoE GEMM.

The kernel (include/mirage/persistent_kernel/tasks/blackwell/
moe_fp8_blockscale_sm100.cuh) is the routed-expert mirror of M2-I12's dense
preserved-block-scale GEMM, so this mirrors that issue's synthetic test:

  1. FLOOR TEST -- against an fp32 contraction of the SAME fp8 bytes with the
     SAME fp32 block scales, the kernel's only legitimate deviation is the bf16
     rounding of its own output. The gate is frob_rel / bf16_output_floor <=
     1.6, never an elementwise max (a below-RMS e4m3 element can show a large
     relative delta from rounding alone).
  2. SCALE-CONSUMPTION CONTROL -- multiply one expert's weight block scale by
     1.3, a NON-power-of-two. A kernel that truncates scales to UE8M0 (what the
     shipped grouped GEMM does, probe P2) cannot reproduce a 1.3x output; this
     kernel must, exactly.
  3. ROUTING CONTROL -- (token, slot) pairs that were not routed must be left
     untouched, and each routed slot must carry ITS expert's product, not a
     neighbour's.

Run:  python test_moe_fp8_blockscale.py
"""

import sys

import torch

import runtime_kernel_blackwell_fp8_moe_qwen35 as moe

torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10
NUM_EXPERTS = 256
NUM_TOPK = 8
BATCH = 16
W13_N, W13_K = 1024, 2048
W2_N, W2_K = 2048, 512
FLOOR_RATIO_THRESHOLD = 1.6


def _true_div(t, denom):
    """`t / <python float>` lowers to a reciprocal multiply in PyTorch and is
    1 ULP off from real fp32 division for divisors like 448; MPK's quantizer
    divides. See test_quantize_fp8_f32scale_moe.py for the measurement."""
    return torch.div(t, torch.tensor(denom, dtype=torch.float32, device=t.device))


def quantize_activation(x_bf16):
    shape = x_bf16.shape
    k = shape[-1]
    xf = x_bf16.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = _true_div(absmax, FP8_MAX)
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (
        q.reshape(shape).to(torch.float8_e4m3fn).contiguous(),
        scale.reshape(*shape[:-1], k // BLOCK).float().contiguous(),
    )


def quantize_weight_blocks(w_bf16):
    """The checkpoint's weight format: one float32 scale per 128x128 block."""
    n, k = w_bf16.shape
    wf = w_bf16.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3)).clamp(min=EPS)
    scale = _true_div(absmax, FP8_MAX)
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(n, k).to(torch.float8_e4m3fn), scale.contiguous()


def dequant_groups(q, scale):
    shape = q.shape
    k = shape[-1]
    return (
        q.float().reshape(-1, k // BLOCK, BLOCK) * scale.reshape(-1, k // BLOCK, 1)
    ).reshape(shape)


def dequant_blocks(q, block_scale):
    n, k = q.shape
    wf = q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (wf * block_scale[:, None, :, None]).reshape(n, k)


def make_routing(active_tokens, experts_per_token, device, seed):
    g = torch.Generator().manual_seed(seed)
    routing = torch.zeros(NUM_EXPERTS, BATCH, dtype=torch.int32)
    token_experts = {}
    for t in range(active_tokens):
        picks = torch.randperm(NUM_EXPERTS, generator=g)[:experts_per_token]
        token_experts[t] = [int(e) for e in picks]
        for slot, e in enumerate(token_experts[t]):
            routing[e, t] = slot + 1
    activated = sorted({e for v in token_experts.values() for e in v})
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32)
    for i, e in enumerate(activated):
        mask[i] = e
    mask[NUM_EXPERTS] = len(activated)
    return routing.to(device), mask.to(device), token_experts, activated


def reference(x_q, x_s, w_q, w_s, token_experts, out_n, per_slot):
    out = torch.zeros(
        (BATCH, NUM_TOPK, out_n), dtype=torch.float32, device=x_q.device
    )
    for t, experts in token_experts.items():
        for slot, e in enumerate(experts):
            xq = x_q[t, slot] if per_slot else x_q[t]
            xs = x_s[t, slot] if per_slot else x_s[t]
            xd = dequant_groups(xq.unsqueeze(0), xs.unsqueeze(0))
            wd = dequant_blocks(w_q[e], w_s[e])
            out[t, slot] = (xd @ wd.t()).squeeze(0)
    return out


def check_floor(name, actual, ref):
    a = actual.float()
    err = (a - ref).norm().item() / ref.norm().item()
    floor = (
        (ref.to(torch.bfloat16).float() - ref).norm().item() / ref.norm().item()
    )
    ratio = err / max(floor, 1e-30)
    print(
        f"  {name:<34} frob_rel={err:.3e} bf16_floor={floor:.3e} "
        f"ratio={ratio:.2f} max_abs={((a - ref).abs().max().item()):.3e}"
    )
    assert ratio <= FLOOR_RATIO_THRESHOLD, (
        f"{name}: frob_rel {err:.3e} is {ratio:.2f}x the bf16 output-rounding "
        f"floor {floor:.3e} (limit {FLOOR_RATIO_THRESHOLD})"
    )
    return err, floor


def run_shape(label, out_n, red_k, per_slot, kernel_fn, active_tokens, seed):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    routing, mask, token_experts, activated = make_routing(
        active_tokens, NUM_TOPK, dev, seed
    )

    w = torch.zeros((NUM_EXPERTS, out_n, red_k), dtype=torch.float8_e4m3fn,
                    device=dev)
    w_s = torch.zeros((NUM_EXPERTS, out_n // BLOCK, red_k // BLOCK),
                      dtype=torch.float32, device=dev)
    for e in activated:
        wq, ws = quantize_weight_blocks(
            torch.randn((out_n, red_k), dtype=torch.bfloat16, device=dev,
                        generator=g)
        )
        w[e] = wq
        w_s[e] = ws

    x_shape = (BATCH, NUM_TOPK, red_k) if per_slot else (BATCH, red_k)
    x = torch.randn(x_shape, dtype=torch.bfloat16, device=dev, generator=g)
    x_q, x_s = quantize_activation(x)

    out = torch.zeros((BATCH, NUM_TOPK, out_n), dtype=torch.bfloat16, device=dev)
    kernel_fn(x_q, x_s, w, w_s, routing, mask, out)
    ref = reference(x_q, x_s, w, w_s, token_experts, out_n, per_slot)

    routed = torch.zeros((BATCH, NUM_TOPK), dtype=torch.bool, device=dev)
    for t, experts in token_experts.items():
        for slot in range(len(experts)):
            routed[t, slot] = True

    print(f"[{label}] tokens={active_tokens} experts={len(activated)} "
          f"N={out_n} K={red_k}")
    check_floor("routed slots", out[routed], ref[routed])

    # --- routing control: untouched slots stay exactly zero ---
    assert torch.count_nonzero(out[~routed]) == 0, (
        "the kernel wrote to a (token, slot) pair that was never routed"
    )

    # --- scale-consumption control ---
    # 1.3 is deliberately NOT a power of two: a kernel that truncates scales to
    # UE8M0 would round it away and leave the output unchanged.
    victim = activated[len(activated) // 2]
    w_s_pert = w_s.clone()
    w_s_pert[victim] *= 1.3
    out_pert = torch.zeros_like(out)
    kernel_fn(x_q, x_s, w, w_s_pert, routing, mask, out_pert)

    hit = torch.zeros((BATCH, NUM_TOPK), dtype=torch.bool, device=dev)
    for t, experts in token_experts.items():
        for slot, e in enumerate(experts):
            if e == victim:
                hit[t, slot] = True
    assert hit.any(), "the perturbed expert is not routed anywhere"
    base = out[hit].float()
    pert = out_pert[hit].float()
    keep = base.abs() > 1e-2
    ratio = (pert[keep] / base[keep]).median().item()
    print(f"  scale-consumption control: median out_pert/out = {ratio:.6f} "
          f"(expected 1.300000)")
    assert abs(ratio - 1.3) < 5e-3, (
        f"perturbing a weight block scale by 1.3x moved the output by "
        f"{ratio:.6f}x -- the kernel is not consuming the fp32 scale as given"
    )
    # Rows that do NOT touch the perturbed expert must be bit-identical.
    assert torch.equal(out[~hit], out_pert[~hit]), (
        "perturbing one expert's scale changed another expert's output"
    )
    return True


def main():
    print("=== fp32-block-scale grouped FP8 MoE GEMM: synthetic ===")
    for active_tokens in (1, 4, 8, 16):
        run_shape(
            "w13",
            W13_N,
            W13_K,
            per_slot=False,
            kernel_fn=moe.moe_w13_blockscale_sm100,
            active_tokens=active_tokens,
            seed=20260726 + active_tokens,
        )
        run_shape(
            "w2",
            W2_N,
            W2_K,
            per_slot=True,
            kernel_fn=moe.moe_w2_blockscale_sm100,
            active_tokens=active_tokens,
            seed=20270726 + active_tokens,
        )
    print("ALL SYNTHETIC MOE BLOCKSCALE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
