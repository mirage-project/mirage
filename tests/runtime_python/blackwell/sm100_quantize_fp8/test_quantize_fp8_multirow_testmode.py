"""Stress test for the per_token_group_quantize_fp8 multi-row-per-task path.

Drives the wrapper with batch_size > num_workers so the persistent runtime
launches grid_y = num_workers and the kernel internally loops
ROWS_PER_TASK > 1 rows per CTA. Compares both packed UE8M0 scale and FP8
output against the reference, bit-for-bit (scales) and within FP8 tolerance
(values).

Added 2026-05-14 to exercise the dirty multi-row change that the previous
session left unverified.
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
)
from pytorch_reference import quantize_fp8_ref  # noqa: E402


def _run_at(batch_size: int, hidden_dim: int = 7168, seed: int = None):
    device = "cuda"
    torch.manual_seed(42 if seed is None else seed)
    assert hidden_dim % BLOCK_K == 0

    x = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.zeros(
        batch_size, hidden_dim, dtype=torch.float8_e4m3fn, device=device
    )
    out_scale = allocate_packed_ue8m0_scale_deepgemm_style(
        batch_size, hidden_dim, device
    )
    out_scale.zero_()

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    assert pk.target_cc == 100, (
        "quantize_fp8_layer requires SM100 (Blackwell); "
        f"current target_cc={pk.target_cc}"
    )

    x_dt = pk.attach_input(x, name="x")
    out_fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    out_scale_dt = pk.attach_input(out_scale, name="out_scale")

    block_dim = (128, 1, 1)
    # grid_dim is overridden by the wrapper; pass the natural (batch, 1, 1).
    pk.quantize_fp8_layer(
        input=x_dt,
        output_fp8=out_fp8_dt,
        output_scale=out_scale_dt,
        grid_dim=(batch_size, 1, 1),
        block_dim=block_dim,
        scale_ue8m0=True,
    )

    print(f"[batch={batch_size}] Compiling test kernel...")
    pk.compile(output_dir=THIS_DIR)

    print(f"[batch={batch_size}] Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref_fp8, ref_scale = quantize_fp8_ref(
        x, scale_ue8m0=True, layout="deepgemm_col_major"
    )

    assert out_scale.shape == ref_scale.shape, (
        f"scale shape mismatch: got {out_scale.shape} vs ref {ref_scale.shape}"
    )
    assert out_scale.stride() == ref_scale.stride(), (
        f"scale stride mismatch: got {out_scale.stride()} vs ref {ref_scale.stride()}"
    )
    # Diagnostic: show which rows/packs differ and by what byte.
    diff_mask = (out_scale != ref_scale)
    n_diff = int(diff_mask.sum().item())
    if n_diff:
        rows_with_diff = diff_mask.any(dim=1).nonzero(as_tuple=False).flatten().tolist()
        print(f"[batch={batch_size}] DIFF: {n_diff} elements across rows {rows_with_diff[:20]}")
        for row in rows_with_diff[:8]:
            for col in range(out_scale.shape[1]):
                a = int(out_scale[row, col].item()) & 0xFFFFFFFF
                b = int(ref_scale[row, col].item()) & 0xFFFFFFFF
                if a != b:
                    print(f"  (row={row}, pack={col}): "
                          f"mpk=0x{a:08x}  ref=0x{b:08x}  diff=0x{(a ^ b):08x}")
    torch.testing.assert_close(out_scale, ref_scale, rtol=0, atol=0)
    print(f"[batch={batch_size}] Scale tensor matches reference exactly.")

    torch.testing.assert_close(
        out_fp8.float(), ref_fp8.float(), rtol=1e-1, atol=16.0,
    )
    max_diff = (out_fp8.float() - ref_fp8.float()).abs().max().item()
    print(f"[batch={batch_size}] FP8 max abs diff: {max_diff}")

    pk.finalize()
    print(f"[batch={batch_size}] PASSED")


def test_quantize_fp8_batch16():
    _run_at(batch_size=16)


def test_quantize_fp8_batch32():
    _run_at(batch_size=32)


def test_quantize_fp8_batch64():
    _run_at(batch_size=64)


def test_quantize_fp8_at_num_workers():
    # batch_size = 128 = num_workers → grid_y = 128, ROWS_PER_TASK = 1.
    # Should match legacy behavior exactly.
    _run_at(batch_size=128)


def test_quantize_fp8_multirow_per_task():
    # batch_size > num_workers (128) → ROWS_PER_TASK = ceil(512/128) = 4.
    # Exercises the multi-row inner loop, the per-iter __syncthreads, and
    # the OOB guard for the tail.
    _run_at(batch_size=512)


def test_quantize_fp8_non_multiple_of_workers():
    # batch_size = 130 → grid_y = min(130, 128) = 128, ROWS_PER_TASK = 2.
    # Task index 64 covers rows 128, 129 (both valid). Tasks 65..127 would
    # cover rows 130..255 which are OOB; the kernel must early-return.
    _run_at(batch_size=130)


if __name__ == "__main__":
    test_quantize_fp8_at_num_workers()
    test_quantize_fp8_multirow_per_task()
    test_quantize_fp8_non_multiple_of_workers()
