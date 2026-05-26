"""Standalone test for the H8 OUTPUT_STRIDE fix in per_token_group_quantize_fp8.

Background:
  The QKV-a fusion uses a quantize task that reads a [128, 1536] column slice
  of a wider [128, 2176] BF16 buffer (qkv_a_out) and writes to a smaller
  [128, 1536] FP8 buffer (q_b's input). Before the fix, the kernel used
  `GLOBAL_STRIDE=2176` for BOTH the input AND output addressing, causing the
  output writes to go out of bounds for batch_idx >= 90:

    output_q[batch_idx * 2176 + col]    # uses input stride for output!

  For (128, 1536) output buffer (= 196608 bytes), the writes from batch_idx
  >= 90 land at offsets >= 195840, with most of them past the 196608 buffer
  end, corrupting whatever's adjacent in MPK's buffer pool.

  This script reproduces the bug-free case by setting up a (128, 1536)
  output buffer with a CANARY pad region immediately after it. If the
  canary stays unchanged after the quantize, the fix is correct.

  After the fix (OUTPUT_STRIDE=1536), writes go to `batch_idx * 1536 + col`,
  staying within the buffer.
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

FP8_MAX = 448.0


def reference_quantize(a_bf16_wide):
    """Quantize a_bf16_wide[:, :1536] using 1x128 group abs-max scaling.
    Returns (fp8 (128,1536), scale (128,12))."""
    a_slice = a_bf16_wide[:, :1536].contiguous()
    M, K = a_slice.shape
    assert K == 1536
    nk = K // 128
    a_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=a_bf16_wide.device)
    sa = torch.zeros(M, nk, dtype=torch.float32, device=a_bf16_wide.device)
    a_f32 = a_slice.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = max(abs_max / FP8_MAX, 1e-10)
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn))
    return a_fp8, sa


def main():
    from _build_helper import ensure_extension_built
    ensure_extension_built()
    import runtime_kernel_blackwell_fp8_gemm_dense as kernel_mod

    device = "cuda"
    # Use the actual MPK input bytes for realism (same as test_real_bytes).
    dump_dir = os.environ.get("MPK_DSV3_DUMP_DIR")
    dump_path = (
        os.path.join(dump_dir, "layer0_input_norm.pt")
        if dump_dir else None)
    if dump_path is not None and os.path.exists(dump_path):
        print(f"Loading bf16 input from: {dump_path}")
        a_bf16_orig = torch.load(dump_path, weights_only=True).to(device)
        # Tile / pad to (128, 2176) — the wider qkv_a_out layout
        a_bf16 = torch.zeros(128, 2176, dtype=torch.bfloat16, device=device)
        a_bf16[:, :a_bf16_orig.shape[1]] = a_bf16_orig[:128, :2176]
        if a_bf16_orig.shape[1] < 2176:
            # Wrap from beginning if input is narrower than 2176
            a_bf16[:, a_bf16_orig.shape[1]:] = (
                a_bf16_orig[:128, :2176 - a_bf16_orig.shape[1]])
    else:
        print("Using random BF16 input")
        a_bf16 = torch.randn(128, 2176, dtype=torch.bfloat16, device=device,
                             generator=torch.Generator(device=device).manual_seed(42))

    # Allocate output buffer + canary trailer in a SINGLE contiguous block,
    # so that if the kernel writes past the (128, 1536) region, it lands in
    # the canary (which we can then inspect).
    OUT_SIZE = 128 * 1536  # = 196608 bytes for fp8
    CANARY_SIZE = 128 * 2176 - 128 * 1536  # = 81920 bytes — enough to catch the bug overflow (81280 bytes)
    full_buf_bytes = OUT_SIZE + CANARY_SIZE
    full_buf = torch.empty(full_buf_bytes, dtype=torch.float8_e4m3fn, device=device)
    # Initialize canary region to a sentinel value (0xff = NaN in e4m3) so we can
    # detect ANY write into it.
    canary_sentinel = 0xff
    full_buf.view(torch.uint8).fill_(canary_sentinel)
    # First 128*1536 bytes are the "output" region; rest is the canary.
    out_fp8 = full_buf.narrow(0, 0, OUT_SIZE).view(128, 1536)
    canary = full_buf.narrow(0, OUT_SIZE, CANARY_SIZE).view(torch.uint8)
    # Also need to zero out_fp8 region so we can detect unwritten rows
    out_fp8.view(torch.uint8).fill_(0)

    scale_out = torch.zeros(128, 12, dtype=torch.float32, device=device)

    # === Run the kernel ===
    print(f"Calling quantize_fp8_slice_launch with (128, 2176) input, "
          f"(128, 1536) output, OUTPUT_STRIDE=1536")
    kernel_mod.quantize_fp8_slice_launch(a_bf16, out_fp8, scale_out)
    torch.cuda.synchronize()

    # === Check 1: canary unchanged ===
    canary_bytes = canary.cpu().numpy()
    corrupted = int((canary_bytes != canary_sentinel).sum())
    print(f"Canary bytes corrupted by overflow: {corrupted}/{CANARY_SIZE}")
    if corrupted > 0:
        first_changed = int((canary_bytes != canary_sentinel).nonzero()[0][0])
        print(f"  First corrupted byte offset: {first_changed}")

    # === Check 2: all 128 output rows are written ===
    out_zero_rows = (out_fp8.view(128, 1536).to(torch.float32).abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    print(f"Zero rows in output[128, 1536]: {out_zero_rows.numel()}/128")
    if out_zero_rows.numel() > 0 and out_zero_rows.numel() < 20:
        print(f"  Zero rows: {out_zero_rows.tolist()}")

    # === Check 3: output matches PyTorch reference ===
    a_fp8_ref, sa_ref = reference_quantize(a_bf16)
    # Compare as raw bytes (FP8 has no native diff op)
    out_bytes = out_fp8.view(torch.uint8)
    ref_bytes = a_fp8_ref.view(torch.uint8)
    byte_match = (out_bytes == ref_bytes).all().item()
    byte_mismatches = int((out_bytes != ref_bytes).sum())
    print(f"FP8 byte match vs PyTorch ref: {byte_match} (mismatches: {byte_mismatches}/{out_bytes.numel()})")

    scale_match = torch.allclose(scale_out, sa_ref, rtol=0.05, atol=1e-6)
    scale_max_diff = (scale_out - sa_ref).abs().max().item()
    print(f"Scale match vs PyTorch ref: {scale_match} (max_abs_diff: {scale_max_diff:.6e})")

    # === Verdict ===
    passed = (corrupted == 0) and (out_zero_rows.numel() == 0) and byte_match and scale_match
    print()
    print(f"Result: {'PASS — H8 fix verified' if passed else 'FAIL — bug not fully fixed'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
