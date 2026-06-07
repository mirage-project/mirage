"""Test (D3): linear_fp8_bmm_dense_fp8out_sm100 via PersistentKernel test_mode.

Validates the FP8-out flavor of the per-head dense BMM — the kernel behind
MPK_DSV3_FUSE_EPILOGUE_QUANT (with MPK_DSV3_BMM_DENSE). The epilogue fuses the
float32-scale per-token-group quantize that previously ran as a standalone task
after DSv3 decode BMM2, feeding the o_proj dense GEMM. Each CTA computes one
head's (M, N=128) output = exactly one 128-K-group of the o_proj input row
[M, H*128], so the per-head row max IS that group's scale.

This goes through the FULL MPK compile pipeline (TMA descriptor creation, task
registration with the (4 in, 2 out) tuple, megakernel codegen, nvcc) — the same
path the DeepSeek V3 demo uses.

Compares against pytorch_reference.reference_fp8out (unfused BMM-dense -> float32
quantize):
  - FP8 output bytes:  cosine(out_fp8, ref_fp8) > 0.99 and no zero heads
  - float32 scale:     close to ref_scale (the per-head rowmax/448)

NO GPU was used to author this file (designed under a no-GPU constraint). Run it
later on a clean, idle GPU:

    python tests/runtime_python/blackwell/sm100_linear_fp8_bmm_dense/test_bmm_dense_fp8out_testmode.py
"""

import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

from pytorch_reference import (
    quantize_act_1x128,
    quantize_wt_128x128,
    reference_fp8out,
    cosine_sim,
    FP8_MAX,
)


def run_pk_testmode(M: int, H: int, K: int = 512, N: int = 128, seed: int = 0):
    """Run linear_fp8_bmm_dense_fp8out_sm100 in test_mode, one head per CTA.

    Shapes (DSv3 decode BMM2 → o_proj):
      input_fp8    [M, H, K]   activation (attn_out_fp8), FP8
      input_scale  [M, H, K/128] float32 1x128-group activation scale
      weight_fp8   [H, N, K]   per-head kv_b_v, FP8
      weight_scale [H, 1, K/128] float32 128x128-block weight scale
    Fused outputs (o_proj input layout):
      output_fp8   [M, H*N]    FP8 (per head N=128 -> one 128-K-group/head)
      output_scale [M, H]      float32 (one scale per head)
    """
    label = f"M={M}, H={H}, K={K}, N={N}"
    print(f"\n{'=' * 70}\nPK test_mode fp8out BMM-dense: {label}\n{'=' * 70}")

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((M, H, K), device=device, dtype=torch.bfloat16,
                         generator=g)
    b_f32 = torch.randn((H, N, K), device=device, generator=g) * 0.1

    a_fp8, sa = quantize_act_1x128(a_bf16)
    b_fp8, sb = quantize_wt_128x128(b_f32)

    ref_fp8, ref_scale = reference_fp8out(a_fp8, sa, b_fp8, sb)  # [M,H*N], [M,H]
    print(f"  a_fp8 {tuple(a_fp8.shape)}  sa {tuple(sa.shape)}  "
          f"b_fp8 {tuple(b_fp8.shape)}  sb {tuple(sb.shape)}")

    # Output buffers (start non-zero so a no-op write is detectable).
    out_fp8 = torch.zeros((M, H * N), device=device, dtype=torch.float8_e4m3fn)
    out_scale = torch.full((M, H), -1.0, device=device, dtype=torch.float32)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = M
    params["max_num_batched_requests"] = M
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sb_dt = pk.attach_input(sb, name="sb")
    out_fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    out_scale_dt = pk.attach_input(out_scale, name="out_scale")

    # grid = (1, H, 1): grid.x must be 1 (per-head D_out=128=BN); grid.y = H
    # (one head per CTA). block = (256,1,1) on SM100.
    pk.linear_fp8_bmm_dense_fp8out_sm100_layer(
        input_fp8=a_dt,
        input_scale=sa_dt,
        weight_fp8=b_dt,
        weight_scale=sb_dt,
        output_fp8=out_fp8_dt,
        output_scale=out_scale_dt,
        grid_dim=(1, H, 1),
        block_dim=(256, 1, 1),
    )

    compile_dir = os.path.join(THIS_DIR, f"pk_fp8out_{M}_{H}_{K}_{N}")
    os.makedirs(compile_dir, exist_ok=True)
    print("  Compiling...")
    pk.compile(output_dir=compile_dir)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    # --- FP8 output comparison ---
    zero_heads = (out_fp8.float().reshape(M, H, N).abs().sum(dim=(0, 2)) == 0
                  ).sum().item()
    cos_fp8 = cosine_sim(out_fp8, ref_fp8)
    md_fp8 = (out_fp8.float() - ref_fp8.float()).abs().max().item()

    # --- float32 scale comparison ---
    # The kernel's y_scale = rowmax(acc)/448 over the FP32 accumulator; the
    # reference uses the same formula over the FP32 BMM, so they should match to
    # FP8/accumulator rounding. Compare relative error of the per-head scales.
    sc_rel = ((out_scale - ref_scale).abs()
              / ref_scale.abs().clamp(min=1e-30)).max().item()
    sc_cos = cosine_sim(out_scale, ref_scale)

    print(f"  out_fp8[0,:6]={out_fp8.float()[0,:6].tolist()}")
    print(f"  ref_fp8[0,:6]={ref_fp8.float()[0,:6].tolist()}")
    print(f"  out_scale[0,:4]={out_scale[0,:4].tolist()}")
    print(f"  ref_scale[0,:4]={ref_scale[0,:4].tolist()}")
    print(f"  FP8 cosine={cos_fp8:.6f}  max_abs_diff={md_fp8:.4f}  "
          f"zero_heads={zero_heads}")
    print(f"  scale cosine={sc_cos:.6f}  max_rel_err={sc_rel:.4f}")

    # FP8 rounding + the (divide-by-scale vs reference) path are matched, so the
    # tolerance is tight: cosine > 0.99, no zero heads, scale within ~5% (one
    # FP8-rounding step can shift a per-head rowmax to a neighboring quantum).
    passed = (cos_fp8 > 0.99) and (zero_heads == 0) and (sc_rel < 0.05)
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")

    pk.finalize()
    return passed, cos_fp8, sc_rel


def main():
    results = {}
    # DSv3 BMM2 real shape: K=KV_LORA=512, N=V_HEAD_DIM=128.
    # H = num_local_q_heads (TP8=16, TP4=32, TP1=128). M = decode tokens (<=16).
    for H in (16, 32, 128):
        for M in (1, 4):
            p, cf, sr = run_pk_testmode(M=M, H=H, seed=H * 100 + M)
            results[f"M={M},H={H}"] = (p, cf, sr)

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    all_ok = True
    for label, (p, cf, sr) in results.items():
        print(f"  {label}: {'PASS' if p else 'FAIL'}  fp8_cos={cf:.4f}  "
              f"scale_rel_err={sr:.4f}")
        all_ok = all_ok and p
    print("\nAll cases PASSED" if all_ok else "\nSome cases FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
