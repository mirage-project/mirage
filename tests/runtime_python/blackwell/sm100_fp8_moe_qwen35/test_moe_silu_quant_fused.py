#!/usr/bin/env python3
"""M4-I9 -- the fused MoE SwiGLU+quantize task is BYTE-IDENTICAL to the pair.

WHAT IS BEING CLAIMED. `MPK_FUSE_SILU_QUANT=1` replaces two graph tasks

    moe_silu_mul  : mid[.., 2I] -> act[.., I]              (bf16)
    quantize_fp8  : act[.., I]  -> actq[.., I], acts[.., I/128]

with one task that never materialises `act`. The claim is bit-exactness BY
CONSTRUCTION, not "close enough": the fused body evaluates HEAD's own
expressions at HEAD's own cast positions, and the only floating-point reduction
involved is `fmaxf`, which is exact and order-independent. This test is the
falsifier -- if a single fp8 byte or a single fp32 scale differs, the claim is
wrong and the fusion needs an AC-3 justification rather than a construction
argument.

WHY BOTH nvcc LANES. `-use_fast_math` rewrites `expf` and `/` (the megakernel
ships it, persistent_kernel.py). Both arms live in the SAME translation unit, so
the flag hits both equally and the identity must survive it -- but only a run in
that lane proves it. `MOE_TEST_FAST_MATH=1` builds the fast-math arm.

WHY IT IS NOT ENOUGH TO COMPARE AGAINST TORCH. The reference here is the
SHIPPED PAIR, not a mathematical ideal: the pair rounds the SwiGLU to bf16 and
then quantizes that bf16. A torch reference in fp32 would disagree with both
arms and prove nothing about the fusion.

Shapes cover the real Qwen3.5 site (I = moe_intermediate_size = 512, rows =
mbt*topk up to 128) plus two off-shape widths, and both a value range that
saturates E4M3 and one that lands in its denormals.

Build + run (both lanes):
    cd tests/runtime_python/blackwell/sm100_fp8_moe_qwen35
    python setup.py build_ext --inplace && python test_moe_silu_quant_fused.py
    MOE_TEST_FAST_MATH=1 python setup.py build_ext --inplace && \
        MOE_TEST_FAST_MATH=1 python test_moe_silu_quant_fused.py
"""
import os
import sys

import torch

import runtime_kernel_blackwell_fp8_moe_qwen35 as K

SHAPES = [
    # (rows, intermediate) -- rows = tokens * experts_per_tok
    (1, 512),     # the fused task's own tile: one (token, expert-slot)
    (2, 512),
    (8, 512),
    (16, 512),
    (128, 512),   # mbt=16 * topk=8, the whole layer's activation at once
    (1, 256),
    (1, 1024),
    (8, 1024),
]

# Scales chosen so the fp8 clamp, the 1e-10 amax floor and E4M3's denormal
# region are all exercised.
SCALES = [1.0, 1e-3, 60.0, 1e-8]


def one(rows, inter, scale, seed):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    mid = (torch.randn((rows, 2 * inter), generator=g, device=dev,
                       dtype=torch.float32) * scale).to(torch.bfloat16)

    # ---- reference: the SHIPPED pair, run as two kernels -----------------
    act = torch.empty((rows, inter), device=dev, dtype=torch.bfloat16)
    K.moe_silu_mul_sm100(mid, act)
    ref_q = torch.empty((rows, inter), device=dev, dtype=torch.float8_e4m3fn)
    ref_s = torch.empty((rows, inter // 128), device=dev, dtype=torch.float32)
    K.quantize_fp8_f32scale_sm100(act, ref_q, ref_s)

    # ---- arm F: one fused kernel ----------------------------------------
    got_q = torch.empty((rows, inter), device=dev, dtype=torch.float8_e4m3fn)
    got_s = torch.empty((rows, inter // 128), device=dev, dtype=torch.float32)
    K.moe_silu_mul_quantize_fp8_sm100(mid, got_q, got_s)

    q_bad = int((ref_q.view(torch.uint8) != got_q.view(torch.uint8)).sum())
    s_bad = int((ref_s.view(torch.int32) != got_s.view(torch.int32)).sum())
    return q_bad, s_bad, ref_q.numel(), ref_s.numel()


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    fast = os.environ.get("MOE_TEST_FAST_MATH") == "1"
    print(f"lane: {'fastmath' if fast else 'nofastmath'}  "
          f"(MOE_TEST_FAST_MATH={os.environ.get('MOE_TEST_FAST_MATH', '0')})")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'rows':>5s}{'inter':>7s}{'scale':>9s}{'fp8 diff':>10s}"
          f"{'scale diff':>12s}{'n_fp8':>8s}{'n_scale':>9s}  verdict")
    total_bad = 0
    ncase = 0
    seed = 20260730
    for rows, inter in SHAPES:
        for sc in SCALES:
            seed += 1
            ncase += 1
            q_bad, s_bad, nq, ns = one(rows, inter, sc, seed)
            total_bad += q_bad + s_bad
            v = "IDENTICAL" if (q_bad == 0 and s_bad == 0) else "DIFFERS"
            print(f"{rows:5d}{inter:7d}{sc:9.0e}{q_bad:10d}{s_bad:12d}"
                  f"{nq:8d}{ns:9d}  {v}")
    print()
    if total_bad == 0:
        print(f"PASS: fused == silu+quantize, byte-identical on all {ncase} "
              f"cases (fp8 bytes AND fp32 scales)")
        return 0
    print(f"FAIL: {total_bad} differing elements over {ncase} cases -- the "
          f"bit-exactness-by-construction claim is refuted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
