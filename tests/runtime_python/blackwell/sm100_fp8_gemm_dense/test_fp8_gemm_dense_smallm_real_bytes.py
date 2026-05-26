"""Standalone repro of the QKV-a fusion bug using REAL MPK bytes.

Loads:
  - layer0_input_norm.pt              from a fused MPK dump (BF16 (128, 7168))
  - qkv_a_proj.weight                 from the MPK weight cache (FP8 (2176, 7168))
  - qkv_a_proj.weight_scale_inv       from the weight cache (FP32 (17, 56))

Quantizes the BF16 input to FP8 + per-row-128-block FP32 scale (same algo
as MPK's quantize_fp8_f32scale_sm100 kernel), then calls the same
`fp8_gemm_dense_smallm_sm100_task_impl<128, 3>` kernel with grid =
(num_workers=128, 1, 1) — identical to the production launch.

Compares output to a PyTorch FP32 dequant + matmul reference.

Discriminator:
  - If this test FAILS (rows 1..71 zero) → the kernel has a data-dependent
    bug triggered by the actual weight/input bytes. Should hand to kernel
    team for fix.
  - If this test PASSES → the bug is in the MPK persistent-kernel runtime
    *context* (concurrent task interference, TMEM/SMEM state leakage, or
    the launch wrapper), not in the kernel logic itself. Should debug the
    MPK runtime path.
"""

import os
import sys

import torch
import safetensors.torch as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


FP8_MAX = 448.0  # e4m3 max representable


def quantize_a_from_bf16(a_bf16: torch.Tensor):
    """Quantize A [M, K] to FP8 e4m3 + float32 scale [M, K/128]."""
    M, K = a_bf16.shape
    assert K % 128 == 0
    nk = K // 128
    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.zeros((M, nk), dtype=torch.float32, device=a_bf16.device)
    a_f32 = a_bf16.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = (abs_max / FP8_MAX) if abs_max != 0.0 else 1.0
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX)
                .to(torch.float8_e4m3fn))
    return a_fp8, sa


def reference_gemm(a_fp8, sa, b_fp8, sb):
    """Dequant A and B, then C = A @ B.T in float32, return bfloat16."""
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    nk = K // 128
    a_f32 = a_fp8.float()
    b_f32 = b_fp8.float()
    a_dq = torch.empty(M, K, dtype=torch.float32, device=a_fp8.device)
    for m in range(M):
        for ki in range(nk):
            a_dq[m, ki * 128:(ki + 1) * 128] = (
                a_f32[m, ki * 128:(ki + 1) * 128] * sa[m, ki])
    b_dq = torch.empty(N, K, dtype=torch.float32, device=b_fp8.device)
    nb = N // 128
    for bi in range(nb):
        for ki in range(nk):
            b_dq[bi * 128:(bi + 1) * 128,
                 ki * 128:(ki + 1) * 128] = (
                b_f32[bi * 128:(bi + 1) * 128,
                      ki * 128:(ki + 1) * 128] * sb[bi, ki])
    return torch.matmul(a_dq, b_dq.t()).to(torch.bfloat16)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def compress_ranges(zr):
    if zr.numel() == 0:
        return ''
    zl = zr.tolist()
    ranges = []
    i = 0
    while i < len(zl):
        j = i
        while j + 1 < len(zl) and zl[j + 1] == zl[j] + 1:
            j += 1
        ranges.append((zl[i], zl[j]))
        i = j + 1
    return ', '.join(f'{a}' if a == b else f'{a}..{b}' for a, b in ranges)


def main():
    import importlib.util
    import subprocess

    so_name = "runtime_kernel_blackwell_fp8_gemm_dense"
    so_path = os.path.join(THIS_DIR, f"{so_name}.cpython-311-x86_64-linux-gnu.so")
    wrapper_cu = os.path.join(THIS_DIR, "runtime_kernel_wrapper_sm100.cu")
    needs_rebuild = (
        not os.path.exists(so_path)
        or os.path.getmtime(wrapper_cu) > os.path.getmtime(so_path))
    if needs_rebuild:
        print("Building C++ extension...")
        build_dir = os.path.join(THIS_DIR, "build")
        import shutil
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        if os.path.exists(so_path):
            os.remove(so_path)
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=THIS_DIR)
    import runtime_kernel_blackwell_fp8_gemm_dense as kernel_mod

    device = "cuda"

    # === Load real MPK bytes ===
    dump_dir = os.environ.get("MPK_DSV3_DUMP_DIR")
    if not dump_dir or not os.path.isdir(dump_dir):
        import pytest
        pytest.skip(
            "set MPK_DSV3_DUMP_DIR to a directory containing "
            "layer0_input_norm.pt to run this test")
    input_norm_path = os.path.join(dump_dir, "layer0_input_norm.pt")
    print(f"Loading input from: {input_norm_path}")
    a_bf16 = torch.load(input_norm_path, weights_only=True).to(device)
    print(f"  a_bf16 shape={tuple(a_bf16.shape)} dtype={a_bf16.dtype}")

    cache_root = os.environ.get(
        "MPK_DSV3_WEIGHT_CACHE",
        "/tmp/dpskv3_v8_weight_cache_qkva_fused_2176")
    if not os.path.isdir(cache_root):
        import pytest
        pytest.skip(
            "set MPK_DSV3_WEIGHT_CACHE to the dpskv3 qkva-fused weight cache "
            "directory to run this test")
    weight_file = None
    for sub in sorted(os.listdir(cache_root)):
        sp = os.path.join(cache_root, sub)
        if not os.path.isdir(sp):
            continue
        for fn in sorted(os.listdir(sp)):
            if "rank0" in fn:
                weight_file = os.path.join(sp, fn)
                break
        if weight_file:
            break
    if weight_file is None:
        print("ERROR: rank0.safetensors not found in weight cache.")
        return 1
    print(f"Loading weight from: {weight_file}")
    d = st.load_file(weight_file)
    w_key = "model.layers.0.self_attn.qkv_a_proj.weight"
    s_key = "model.layers.0.self_attn.qkv_a_proj.weight_scale_inv"
    b_fp8 = d[w_key].to(device).contiguous()
    sb = d[s_key].to(device).contiguous()
    print(f"  b_fp8 shape={tuple(b_fp8.shape)} dtype={b_fp8.dtype}")
    print(f"  sb    shape={tuple(sb.shape)}    dtype={sb.dtype}")

    # === Quantize input bf16 → fp8 + scale ===
    print("Quantizing input...")
    a_fp8, sa = quantize_a_from_bf16(a_bf16)

    # === PyTorch reference ===
    print("Computing PyTorch FP32 dequant + matmul reference...")
    ref = reference_gemm(a_fp8, sa, b_fp8, sb)
    print(f"  ref shape={tuple(ref.shape)}  ref[row 1, 0:8]={ref[1,:8].tolist()}")

    # === Call kernel ===
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    num_workers = 128
    output = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
    print(f"Calling kernel with M={M}, N={N}, K={K}, num_workers={num_workers}")
    kernel_mod.fp8_gemm_dense_smallm_multi_cta_launch(
        a_fp8, b_fp8, sa, sb, output, num_workers)
    torch.cuda.synchronize()
    print(f"  out[row 1, 0:8] = {output[1,:8].tolist()}")

    # === Zero-row check + cos similarity ===
    row_norms = output.float().abs().sum(dim=1)
    zr = (row_norms == 0).nonzero(as_tuple=True)[0]
    if zr.numel() > 0:
        rng = compress_ranges(zr)
        print(f"  Zero rows (total {zr.numel()}): {rng}")
    else:
        print("  No zero rows in output.")

    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = cos_sim(output, ref)
    print(f"  max_abs_diff: {max_diff:.4f}")
    print(f"  cos:          {cos:.6f}")

    # Also check the slices used by MPK
    print()
    print("=== Per-slice analysis (matching MPK layout) ===")
    for label, slc in [("q_a   [:1536]",    output[:, :1536]),
                       ("c_lat [1536:2048]", output[:, 1536:2048]),
                       ("k_pe  [2048:2112]", output[:, 2048:2112])]:
        zr = (slc.float().abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        ref_slc = (ref[:, :1536] if "q_a" in label else
                   ref[:, 1536:2048] if "c_lat" in label else
                   ref[:, 2048:2112])
        cos = cos_sim(slc, ref_slc)
        print(f"  {label:>22s}  zero_rows={zr.numel():>3d} (range={compress_ranges(zr)})  cos={cos:.4f}")

    passed = (cos > 0.99) and ((output.float().abs().sum(dim=1) == 0).sum().item() == 0)
    print()
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
