#!/usr/bin/env python3
"""M4-I9 flag A — the fused RMS-norm+quantize task is BYTE-IDENTICAL to the pair.

WHAT IS BEING CLAIMED. `MPK_FUSE_NORM_QUANT=1` replaces two graph tasks

    rmsnorm_hopper : h[.., H] -> nrm[.., H]                     (bf16)
    quantize_fp8   : nrm[.., H] -> xq[.., H], xs[.., H/128]

with one task. Where the bf16 norm still has another consumer (GDN layers feed
the `ba` projection from it) the fused task keeps it as a third output; where the
quantize was its only consumer (attention layers) it is dropped and the value
never leaves shared memory. Both forms are tested.

The bit-exactness claim is STRUCTURAL, not arithmetic: the norm half is
`rms_norm_hopper_impl`'s code unchanged, and the quantize half calls
`per_token_group_quantize_fp8_task_impl` — the same function the standalone task
calls — with its input pointer redirected from global to the shared staging
buffer the norm already fills. So the only difference between the arms is the
address space of one load, and this test is the falsifier for that.

WHY 256 THREADS. That is the megakernel's real block size
(`WORKER_NUM_THREADS = 256`), and `rms_norm_hopper_impl` requires its NUM_THREADS
template parameter to equal blockDim: the cp.async warm-up covers exactly
NUM_THREADS * CHUNK elements with no loop, so a mismatch would silently leave
part of the tile unloaded. The silu-fusion test above runs at 128 because its
impl indexes by `blockDim.x`; this one cannot.

WHY BOTH nvcc LANES. `-use_fast_math` rewrites `rsqrt` and the two divisions
(`sum/HIDDEN` and `orig/y_scale`), and the megakernel ships it.

Build + run (both lanes):
    cd tests/runtime_python/blackwell/sm100_fp8_moe_qwen35
    python setup.py build_ext --inplace && python test_rmsnorm_quant_fused.py
    MOE_TEST_FAST_MATH=1 python setup.py build_ext --inplace && \
        MOE_TEST_FAST_MATH=1 python test_rmsnorm_quant_fused.py
"""
import os
import sys

import torch

import runtime_kernel_blackwell_fp8_moe_qwen35 as K

# (rows, hidden). hidden=2048 is Qwen3.5's hidden_size, i.e. the real pre-norm
# site; rows=1 is the real per-task tile (grid (mbt,1,1) splits dim 0).
SHAPES = [
    (1, 2048),
    (2, 2048),
    (4, 2048),
    (16, 2048),
    (1, 512),
    (1, 1024),
    (4, 1024),
]

# Value scales spanning the fp8 clamp, the 1e-10 amax floor, and inputs small
# enough that the 1e-6 norm eps dominates the reciprocal square root.
SCALES = [1.0, 1e-3, 60.0, 1e-8]


def one(rows, hidden, scale, seed, write_norm):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    h = (torch.randn((rows, hidden), generator=g, device=dev,
                     dtype=torch.float32) * scale).to(torch.bfloat16)
    w = (torch.randn((hidden,), generator=g, device=dev,
                     dtype=torch.float32) * 0.5 + 1.0).to(torch.bfloat16)

    # ---- reference: the SHIPPED pair, run as two kernels ------------------
    ref_n = torch.empty((rows, hidden), device=dev, dtype=torch.bfloat16)
    K.rmsnorm_sm100(h, w, ref_n)
    ref_q = torch.empty((rows, hidden), device=dev, dtype=torch.float8_e4m3fn)
    ref_s = torch.empty((rows, hidden // 128), device=dev, dtype=torch.float32)
    K.quantize_fp8_f32scale_sm100(ref_n, ref_q, ref_s)

    # ---- arm N: one fused kernel ----------------------------------------
    got_n = torch.zeros((rows, hidden), device=dev, dtype=torch.bfloat16)
    got_q = torch.empty((rows, hidden), device=dev, dtype=torch.float8_e4m3fn)
    got_s = torch.empty((rows, hidden // 128), device=dev, dtype=torch.float32)
    K.rmsnorm_quantize_fp8_sm100(h, w, got_n, got_q, got_s, write_norm)

    q_bad = int((ref_q.view(torch.uint8) != got_q.view(torch.uint8)).sum())
    s_bad = int((ref_s.view(torch.int32) != got_s.view(torch.int32)).sum())
    n_bad = (int((ref_n.view(torch.int16) != got_n.view(torch.int16)).sum())
             if write_norm else -1)
    return q_bad, s_bad, n_bad, ref_q.numel(), ref_s.numel()


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    fast = os.environ.get("MOE_TEST_FAST_MATH") == "1"
    print(f"lane: {'fastmath' if fast else 'nofastmath'}  "
          f"(MOE_TEST_FAST_MATH={os.environ.get('MOE_TEST_FAST_MATH', '0')})")
    print(f"device: {torch.cuda.get_device_name(0)}   block: 256 threads "
          f"(= WORKER_NUM_THREADS, the megakernel's own)")
    print(f"{'rows':>5s}{'hidden':>8s}{'scale':>9s}{'norm?':>7s}"
          f"{'fp8 diff':>10s}{'scale diff':>12s}{'bf16 diff':>11s}  verdict")
    total_bad = 0
    ncase = 0
    seed = 20260731
    for rows, hidden in SHAPES:
        for sc in SCALES:
            for wn in (True, False):
                seed += 1
                ncase += 1
                q_bad, s_bad, n_bad, nq, ns = one(rows, hidden, sc, seed, wn)
                bad = q_bad + s_bad + max(n_bad, 0)
                total_bad += bad
                v = "IDENTICAL" if bad == 0 else "DIFFERS"
                print(f"{rows:5d}{hidden:8d}{sc:9.0e}{str(wn):>7s}"
                      f"{q_bad:10d}{s_bad:12d}"
                      f"{(n_bad if n_bad >= 0 else 0):11d}  {v}")
    print()
    if total_bad == 0:
        print(f"PASS: fused == rmsnorm+quantize, byte-identical on all {ncase} "
              f"cases (fp8 bytes, fp32 scales, and the bf16 norm where kept)")
        return 0
    print(f"FAIL: {total_bad} differing elements over {ncase} cases -- the "
          f"bit-exactness-by-construction claim is refuted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
