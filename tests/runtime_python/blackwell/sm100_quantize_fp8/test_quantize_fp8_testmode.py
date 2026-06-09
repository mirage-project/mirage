"""Test mode wrapper around ``PersistentKernel.quantize_fp8_layer`` at DSV3 shapes.

Builds a one-task graph that runs the SM100 block-wise BF16 -> FP8 quantizer
(``per_token_group_quantize_fp8_task_impl``) end-to-end through the MPK
compile + run pipeline, then compares both the scale tensor and the FP8
output against the pure-PyTorch reference in ``pytorch_reference.py``.

DeepSeek-V3 is a per-element op, so this is a **bs-only** sweep (K is not
TP-sharded for the quantize op itself); we sweep ``bs in {1,2,4,8,16}`` for
TWO scale modes at TWO representative DSV3 K widths:

  * **UE8M0 path** (``scale_ue8m0=True``): output fp8 ``(bs,K)`` + packed
    UE8M0 ``uint32`` scale in the deepgemm column-major ``[packed_k,
    aligned_bs]`` layout the MPK builder allocates (e.g. NEW-MoE input
    quantize, K=HIDDEN=7168). Scale is compared bit-exactly against the
    shared quantizer; fp8 dequant compared on relative mean.
  * **f32 path** (``scale_ue8m0=False``): output fp8 ``(bs,K)`` + float32
    ``(bs, K/128)`` row-major scale (the MoE group-GEMM path, e.g. MoE-input
    K=7168 and silu-output K=2048). Scale compared to ~1e-6; fp8 dequant on
    relative mean.

K values exercise the per-128-block scaling at DSV3 widths: K=7168
(MoE-input HIDDEN) and K=2048 (a routed-MoE silu-output intermediate).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (  # noqa: E402
    BLOCK_K,
    allocate_packed_ue8m0_scale_deepgemm_style,
    dequant_from_packed_ue8m0_deepgemm_style,
)
from pytorch_reference import quantize_fp8_ref  # noqa: E402

# DSV3 facts: per-element quantize -> bs sweep only (K not TP-sharded here).
BATCH_SIZES = [1, 2, 4, 8, 16]
# Two representative DSV3 K widths: MoE-input HIDDEN and a silu-output width.
K_VALUES = [7168, 2048]
SCALE_MODES = [True, False]  # UE8M0, f32

# Relative-mean tolerance on the fp8 dequant vs the (snapped) reference.
# fp8 e4m3 has ~2 mantissa bits -> per-element error up to ~6%, mean ~2-3%.
FP8_REL_MEAN_TOL = 0.05
# float32 scale tolerance (no UE8M0 snapping; both compute max/448 in f32).
F32_SCALE_TOL = 1e-6


def _dequant_f32_scale(out_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize fp8 with a per-group float32 scale ``(bs, K/128)``."""
    bs, K = out_fp8.shape
    num_groups = K // BLOCK_K
    deq = out_fp8.float().view(bs, num_groups, BLOCK_K)
    deq = deq * scale.view(bs, num_groups, 1)
    return deq.reshape(bs, K)


def _rel_mean(out_deq: torch.Tensor, ref_deq: torch.Tensor) -> float:
    """Mean relative error against the reference dequant magnitude."""
    num = (out_deq - ref_deq).abs()
    den = ref_deq.abs().clamp(min=1e-6)
    return (num / den).mean().item()


def _run_case(bs: int, K: int, scale_ue8m0: bool):
    device = "cuda"
    torch.manual_seed(42)
    assert K % BLOCK_K == 0
    mode = "ue8m0" if scale_ue8m0 else "f32"
    tag = f"[bs={bs} K={K} {mode}]"

    x = torch.randn(bs, K, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.zeros(bs, K, dtype=torch.float8_e4m3fn, device=device)
    if scale_ue8m0:
        # Packed UE8M0 uint32, deepgemm column-major [packed_k, aligned_bs].
        out_scale = allocate_packed_ue8m0_scale_deepgemm_style(bs, K, device)
    else:
        # Plain float32 per-group scale, (bs, K/128) row-major.
        out_scale = torch.zeros(bs, K // BLOCK_K, dtype=torch.float32, device=device)
    out_scale.zero_()

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    pk = PersistentKernel(**params)

    assert pk.target_cc == 100, (
        "quantize_fp8_layer requires SM100 (Blackwell); "
        f"current target_cc={pk.target_cc}"
    )

    x_dt = pk.attach_input(x, name="x")
    out_fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    out_scale_dt = pk.attach_input(out_scale, name="out_scale")

    block_dim = (128, 1, 1)  # matches the MPK builder for this layer
    pk.quantize_fp8_layer(
        input=x_dt,
        output_fp8=out_fp8_dt,
        output_scale=out_scale_dt,
        grid_dim=(bs, 1, 1),  # overridden by the wrapper; pass natural shape
        block_dim=block_dim,
        scale_ue8m0=scale_ue8m0,
    )

    print(f"{tag} Compiling test kernel...")
    pk.compile(output_dir=THIS_DIR)
    print(f"{tag} Running test kernel...")
    pk()
    torch.cuda.synchronize()

    if scale_ue8m0:
        ref_fp8, ref_scale = quantize_fp8_ref(
            x, scale_ue8m0=True, layout="deepgemm_col_major"
        )
        # Scale: packed UE8M0 bytes must match bit-for-bit.
        assert out_scale.shape == ref_scale.shape, (
            f"{tag} scale shape mismatch: {out_scale.shape} vs {ref_scale.shape}"
        )
        assert out_scale.stride() == ref_scale.stride(), (
            f"{tag} scale stride mismatch: {out_scale.stride()} vs "
            f"{ref_scale.stride()}"
        )
        torch.testing.assert_close(out_scale, ref_scale, rtol=0, atol=0)
        print(f"{tag} packed UE8M0 scale matches reference bit-exactly.")
        # Dequant both sides through the same UE8M0 decode and compare.
        out_deq = dequant_from_packed_ue8m0_deepgemm_style(out_fp8, out_scale)
        ref_deq = dequant_from_packed_ue8m0_deepgemm_style(ref_fp8, ref_scale)
    else:
        ref_fp8, ref_scale = quantize_fp8_ref(x, scale_ue8m0=False)
        assert out_scale.shape == ref_scale.shape, (
            f"{tag} scale shape mismatch: {out_scale.shape} vs {ref_scale.shape}"
        )
        scale_max_diff = (out_scale - ref_scale).abs().max().item()
        print(f"{tag} f32 scale max abs diff: {scale_max_diff:.3e}")
        torch.testing.assert_close(
            out_scale, ref_scale, rtol=F32_SCALE_TOL, atol=F32_SCALE_TOL
        )
        print(f"{tag} f32 scale matches reference within {F32_SCALE_TOL}.")
        out_deq = _dequant_f32_scale(out_fp8, out_scale)
        ref_deq = _dequant_f32_scale(ref_fp8, ref_scale)

    # FP8 element agreement (e4m3 rounding): compare raw fp8 bytes diff and
    # the dequantized relative-mean error against the reference.
    fp8_max_diff = (out_fp8.float() - ref_fp8.float()).abs().max().item()
    rel_mean = _rel_mean(out_deq, ref_deq)
    print(
        f"{tag} fp8 max-abs-diff(float)={fp8_max_diff:.3f}  "
        f"dequant rel-mean={rel_mean * 100:.3f}%"
    )
    assert rel_mean <= FP8_REL_MEAN_TOL, (
        f"{tag} dequant rel-mean {rel_mean * 100:.3f}% exceeds "
        f"{FP8_REL_MEAN_TOL * 100:.1f}%"
    )

    pk.finalize()
    print(f"{tag} PASSED")


def test_quantize_fp8_testmode():
    """Full DSV3 matrix: bs sweep x {UE8M0, f32} x {K=7168, K=2048}."""
    for scale_ue8m0 in SCALE_MODES:
        for K in K_VALUES:
            for bs in BATCH_SIZES:
                _run_case(bs=bs, K=K, scale_ue8m0=scale_ue8m0)


if __name__ == "__main__":
    failures = []
    for scale_ue8m0 in SCALE_MODES:
        for K in K_VALUES:
            for bs in BATCH_SIZES:
                try:
                    _run_case(bs=bs, K=K, scale_ue8m0=scale_ue8m0)
                except Exception as exc:  # noqa: BLE001 - report & continue
                    mode = "ue8m0" if scale_ue8m0 else "f32"
                    msg = f"[bs={bs} K={K} {mode}] FAILED: {exc}"
                    print(msg)
                    failures.append(msg)
    if failures:
        print("\n==== FAILURES ====")
        for f in failures:
            print(f)
        sys.exit(1)
    print("\nALL CONFIGS PASSED")
