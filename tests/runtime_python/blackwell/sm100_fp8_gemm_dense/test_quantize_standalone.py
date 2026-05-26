"""Standalone quantize-only test.

Tests MPK's per_token_group_quantize_fp8 kernel in isolation (no GEMM, no
megakernel context). Compares against PyTorch reference + MPK's in-megakernel
output to localise the QKV-a fusion bug.

Three test cases:
  1. Real MPK bytes (rmsnorm_out from a layer-0 fused-mode dump): does
     standalone match MPK's in-megakernel quantize?
  2. Synthetic random data with similar magnitude: does the kernel handle
     normal FP8-quantization-friendly data correctly?
  3. Synthetic data with all rows zero: sanity check for the fallback path.

Build:
  cd tests/runtime_python/blackwell/sm100_fp8_gemm_dense
  pip install -e .
"""
import os
import sys
import subprocess

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def build_module():
    """Build the C++ extension if needed."""
    print("Building runtime_kernel_blackwell_fp8_gemm_dense ...")
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=THIS_DIR,
    )


def pytorch_quantize(a_bf16, group_size=128, eps=1e-10, max_8bit=448.0):
    """PyTorch reference: matches MPK kernel's group_max / 448 logic."""
    M, K = a_bf16.shape
    num_groups = K // group_size
    a_f = a_bf16.float()
    g = a_f.reshape(M, num_groups, group_size)
    group_max = g.abs().amax(dim=2)  # (M, num_groups)
    group_max = torch.clamp(group_max, min=eps)
    scale = group_max / max_8bit  # (M, num_groups)
    # Quantize: fp8_val = clamp(orig / scale, -max_8bit, max_8bit)
    scale_expanded = scale.repeat_interleave(group_size, dim=1)  # (M, K)
    quant_f = torch.clamp(a_f / scale_expanded, -max_8bit, max_8bit)
    fp8 = quant_f.to(torch.float8_e4m3fn)
    return fp8, scale


def cmp(name, mpk_scale, ref_scale, mpk_fp8=None, ref_fp8=None,
        a_bf16=None, tol=1e-5):
    """Print comparison stats."""
    print(f"\n=== {name} ===")
    # Scale
    rel_diff = (mpk_scale.float() - ref_scale.float()).abs() / (ref_scale.float().abs() + 1e-12)
    n_match = (rel_diff < 0.01).sum().item()
    n_total = ref_scale.numel()
    print(f"  scale match (within 1% rel): {n_match}/{n_total} ({100*n_match/n_total:.1f}%)")
    abs_diff = (mpk_scale.float() - ref_scale.float()).abs()
    print(f"  scale abs_diff: max={abs_diff.max():.3e}  mean={abs_diff.mean():.3e}")
    # Show a few row 0 cells
    print(f"  row 0 scales (first 5):")
    print(f"    mpk: {mpk_scale[0, :5].tolist()}")
    print(f"    ref: {ref_scale[0, :5].tolist()}")

    if mpk_fp8 is not None and ref_fp8 is not None:
        # FP8 bytes diff
        mpk_u8 = mpk_fp8.view(torch.uint8) if mpk_fp8.dtype != torch.uint8 else mpk_fp8
        ref_u8 = ref_fp8.view(torch.uint8) if ref_fp8.dtype != torch.uint8 else ref_fp8
        byte_match = (mpk_u8 == ref_u8).float().mean().item()
        print(f"  fp8 byte match: {100*byte_match:.2f}%")

    if a_bf16 is not None:
        # Dequant back and compare to original
        scale_expanded = mpk_scale.float().repeat_interleave(128, dim=1)
        dq = mpk_fp8.float() * scale_expanded
        cos = torch.nn.functional.cosine_similarity(
            dq.flatten().unsqueeze(0), a_bf16.float().flatten().unsqueeze(0))
        max_diff = (dq - a_bf16.float()).abs().max().item()
        print(f"  dequant cos vs original: {cos.item():.6f}  max_diff: {max_diff:.4f}")


def main():
    build_module()
    sys.path.insert(0, THIS_DIR)
    import runtime_kernel_blackwell_fp8_gemm_dense as krn

    device = "cuda"

    # =========================================================================
    # Test 1: MPK real bytes
    # =========================================================================
    mpk_dump_dir = os.environ.get("MPK_DSV3_DUMP_DIR")
    if not mpk_dump_dir or not os.path.isdir(mpk_dump_dir):
        import pytest
        pytest.skip(
            "set MPK_DSV3_DUMP_DIR to a directory containing "
            "layer0_input_norm.pt to run this test")
    rmsnorm_out = torch.load(
        f"{mpk_dump_dir}/layer0_input_norm.pt", weights_only=True
    ).to(device).contiguous()
    print(f"\nLoaded rmsnorm_out shape={tuple(rmsnorm_out.shape)} "
          f"dtype={rmsnorm_out.dtype}")
    print(f"  max_abs={rmsnorm_out.float().abs().max().item():.4f}")
    print(f"  row 1 norm={rmsnorm_out[1].float().norm().item():.4f}")
    print(f"  row 71 norm={rmsnorm_out[71].float().norm().item():.4f}")

    # PyTorch ref
    ref_fp8, ref_scale = pytorch_quantize(rmsnorm_out)

    # MPK standalone kernel
    out_fp8 = torch.zeros((128, 7168), dtype=torch.float8_e4m3fn, device=device)
    out_scale = torch.zeros((128, 56), dtype=torch.float32, device=device)
    krn.quantize_fp8_7168_launch(rmsnorm_out, out_fp8, out_scale)
    torch.cuda.synchronize()

    cmp("MPK real bytes — standalone kernel vs PyTorch ref",
        out_scale, ref_scale, out_fp8, ref_fp8, rmsnorm_out)

    # Also compare standalone kernel output vs MPK's IN-MEGAKERNEL output
    mpk_scale_in_kernel = torch.load(
        f"{mpk_dump_dir}/fp8_scale_v2_7168.pt", weights_only=True
    ).to(device).contiguous()
    mpk_fp8_in_kernel = torch.load(
        f"{mpk_dump_dir}/fp8_input_v2_7168.pt", weights_only=True
    ).to(device).contiguous()
    cmp("MPK real bytes — STANDALONE vs IN-MEGAKERNEL output",
        out_scale, mpk_scale_in_kernel, out_fp8, mpk_fp8_in_kernel)

    # =========================================================================
    # Test 2: Synthetic random data, similar magnitude to rmsnorm output
    # =========================================================================
    torch.manual_seed(42)
    synth = (torch.randn(128, 7168, device=device, dtype=torch.bfloat16) * 0.05).contiguous()
    print(f"\nSynth random data: max_abs={synth.float().abs().max().item():.4f}, "
          f"mean_abs={synth.float().abs().mean().item():.6f}")

    ref_fp8_s, ref_scale_s = pytorch_quantize(synth)
    out_fp8_s = torch.zeros((128, 7168), dtype=torch.float8_e4m3fn, device=device)
    out_scale_s = torch.zeros((128, 56), dtype=torch.float32, device=device)
    krn.quantize_fp8_7168_launch(synth, out_fp8_s, out_scale_s)
    torch.cuda.synchronize()

    cmp("SYNTH random — standalone vs PyTorch ref",
        out_scale_s, ref_scale_s, out_fp8_s, ref_fp8_s, synth)

    # =========================================================================
    # Test 3: All-zero input (sanity for fallback path)
    # =========================================================================
    zero = torch.zeros(128, 7168, device=device, dtype=torch.bfloat16).contiguous()
    out_fp8_z = torch.zeros((128, 7168), dtype=torch.float8_e4m3fn, device=device)
    out_scale_z = torch.zeros((128, 56), dtype=torch.float32, device=device)
    krn.quantize_fp8_7168_launch(zero, out_fp8_z, out_scale_z)
    torch.cuda.synchronize()
    print(f"\n=== Zero input ===")
    print(f"  scale row 0 max={out_scale_z[0].max().item():.3e}  "
          f"(expected fallback: {1e-10/448:.3e})")
    print(f"  fp8 row 0 unique values: "
          f"{torch.unique(out_fp8_z[0].view(torch.uint8)).tolist()[:5]}")


if __name__ == "__main__":
    sys.exit(main())
