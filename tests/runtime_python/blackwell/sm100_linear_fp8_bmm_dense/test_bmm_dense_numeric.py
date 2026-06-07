"""Standalone numeric correctness test for the per-head dense FP8 BMM
(linear_fp8_bmm_dense_sm100_task_impl<128,3,2>), the kernel behind
MPK_DSV3_BMM_DENSE for DSv3 decode BMM2 (o-down un-absorption).

Builds per-head FP8 inputs with the SAME scale granularity the
demo/builder produce:
  - activation A [M, H, K]   : FP8 + float32 1x128-group scale sa [M, H, K/128]
                               (== quantize_fp8_layer(scale_ue8m0=False) layout)
  - weight     B [H, N, K]   : FP8 + float32 128x128-block scale sb [H, 1, K/128]
                               (== _quantize_f32_to_checkpoint_fp8 reshaped per head)
Runs the kernel (grid=(1,H,1), one head/CTA) and compares to a per-head
torch FP32 dequant + matmul reference.  Same math as the swapAB BMM, so the
kernel output must match the reference to high cosine (only FP8 rounding).

Run:  python tests/runtime_python/blackwell/sm100_linear_fp8_bmm_dense/test_bmm_dense_numeric.py
"""
import os
import sys
import subprocess

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FP8_MAX = 448.0


def quantize_act_1x128(a_bf16):
    """A [M,H,K] -> FP8 [M,H,K] + float32 1x128-group scale [M,H,K/128].
    Matches quantize_fp8_layer(scale_ue8m0=False): per (row, 128-K-group) max."""
    M, H, K = a_bf16.shape
    nk = K // 128
    a_f32 = a_bf16.float()
    blk = a_f32.reshape(M, H, nk, 128)
    amax = blk.abs().amax(dim=-1).clamp(min=1e-10)          # [M,H,nk]
    scale = amax / FP8_MAX                                   # 1x128 group scale
    q = (blk / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    a_fp8 = q.reshape(M, H, K).to(torch.float8_e4m3fn)
    return a_fp8, scale.contiguous()                         # sa [M,H,nk]


def quantize_wt_128x128(b_f32):
    """B [H,N,K] -> FP8 [H,N,K] + float32 128x128-block scale [H,N/128,K/128].
    Matches _quantize_f32_to_checkpoint_fp8 (block max over 128x128).
    For N=128 -> [H,1,nk]."""
    H, N, K = b_f32.shape
    bN, nk = N // 128, K // 128
    blk = b_f32.reshape(H, bN, 128, nk, 128)
    amax = blk.abs().amax(dim=(2, 4)).clamp(min=1e-12)       # [H,bN,nk]
    scale = amax / FP8_MAX
    q = (blk / scale.unsqueeze(2).unsqueeze(4)).clamp(-FP8_MAX, FP8_MAX)
    b_fp8 = q.reshape(H, N, K).to(torch.float8_e4m3fn)
    return b_fp8, scale.contiguous()                         # sb [H,bN,nk]


def reference(a_fp8, sa, b_fp8, sb):
    """Per-head C[m,h,:] = (A_dq[m,h,:]) @ (B_dq[h,:,:])^T, FP32 -> bf16."""
    M, H, K = a_fp8.shape
    N = b_fp8.shape[1]
    nk = K // 128
    a_dq = (a_fp8.float().reshape(M, H, nk, 128) *
            sa.unsqueeze(-1)).reshape(M, H, K)               # [M,H,K]
    b_dq = (b_fp8.float().reshape(H, N // 128, 128, nk, 128) *
            sb.unsqueeze(2).unsqueeze(4)).reshape(H, N, K)   # [H,N,K]
    # einsum: out[m,h,n] = sum_k a_dq[m,h,k] * b_dq[h,n,k]
    out = torch.einsum("mhk,hnk->mhn", a_dq, b_dq)
    return out.to(torch.bfloat16)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def run_case(mod, M, H, K, N=128, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    a_bf16 = torch.randn(M, H, K, device=dev, dtype=torch.bfloat16)
    b_f32 = torch.randn(H, N, K, device=dev) * 0.1
    a_fp8, sa = quantize_act_1x128(a_bf16)
    b_fp8, sb = quantize_wt_128x128(b_f32)
    ref = reference(a_fp8, sa, b_fp8, sb)
    out = torch.zeros(M, H, N, device=dev, dtype=torch.bfloat16)
    mod.linear_fp8_bmm_dense_launch(a_fp8, b_fp8, sa, sb, out)
    torch.cuda.synchronize()
    c = cos(out, ref)
    md = (out.float() - ref.float()).abs().max().item()
    nz = (out.float().abs().sum(dim=(0, 2)) == 0).sum().item()  # zero heads
    ok = c > 0.99 and nz == 0
    print(f"  M={M:<3d} H={H:<3d} K={K:<4d} N={N:<3d}  cos={c:.6f}  "
          f"max_abs_diff={md:.4f}  zero_heads={nz}  {'PASS' if ok else 'FAIL'}")
    # spot check a couple of (m,h) rows
    if not ok:
        print(f"    out[0,0,:6]={out[0,0,:6].tolist()}")
        print(f"    ref[0,0,:6]={ref[0,0,:6].tolist()}")
    return ok


def main():
    so = os.path.join(THIS_DIR,
                      "runtime_kernel_blackwell_bmm_dense"
                      ".cpython-311-x86_64-linux-gnu.so")
    cu = os.path.join(THIS_DIR, "runtime_kernel_wrapper_sm100.cu")
    if (not os.path.exists(so)) or os.path.getmtime(cu) > os.path.getmtime(so):
        print("Building C++ extension...")
        bd = os.path.join(THIS_DIR, "build")
        if os.path.exists(bd):
            import shutil
            shutil.rmtree(bd)
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=THIS_DIR)
    sys.path.insert(0, THIS_DIR)
    import runtime_kernel_blackwell_bmm_dense as mod

    print("=== Per-head dense FP8 BMM numeric test (DSv3 BMM2 kv_b_v) ===")
    print("Layouts: A[M,H,512] 1x128 act scale; B[H,128,512] 128x128 blk scale")
    results = []
    # DSv3 BMM2 real shape: K=KV_LORA=512, N=V_HEAD_DIM=128. H = num_local_q_heads
    # (TP1=128, TP2=64, TP4=32, TP8=16). M = decode active tokens (<=16).
    for H in (16, 32, 128):
        for M in (1, 4, 16):
            results.append(run_case(mod, M, H, K=512, N=128, seed=H * 100 + M))
    ok = all(results)
    print(f"\nVERDICT: {'PASS (all cos>0.99)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
