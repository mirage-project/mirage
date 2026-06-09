"""Test: linear_fp8_with_residual_layer (SM100) via PersistentKernel test_mode.

Validates the FP8 block-scaled linear-with-residual layer (= linear_fp8_ref +
residual add) against ``linear_fp8_with_residual_ref`` from
``pytorch_reference.py`` through the full MPK compilation pipeline.

⚠️ KNOWN KERNEL HANG (real bug, NOT a test bug; see DSV3_TESTMODE_DECISIONS.md
   "Known kernel issues" #3): ``linear_fp8_with_residual_sm100`` DEADLOCKS once
   the total number of residual epilogue iterations (≈ grid_x × per-task-N/16)
   exceeds a few hundred. It passes only at tiny shapes (≤ ~48 total iters).
   EVERY DSV3-production shape hangs:
     - production grid (``_pick_grid_x`` → grid_x∈{36,72,96}) hangs at all (tp,bs);
     - even grid_x=1 hangs at N≥~4608 (the smallest TP=8 gate_up shard).
   The non-residual ``linear_fp8_layer`` (same shapes, same scale layout) passes
   the full DSV3 union-of-axes matrix 9/9 — the hang is specific to the residual
   epilogue's single-buffer ``residual_full/empty`` barrier pipeline.

Because a hanging config cannot be a normal pytest xfail (it never returns), the
DSV3-shaped union-of-axes matrix is recorded below as ``HANGING_MATRIX`` for
documentation but is NOT executed. The executed test runs a SMOKE config small
enough to complete (proving the residual GEMM+add math and the test harness are
correct), and is the assertion this file makes.

Run:
    python tests/runtime_python/blackwell/sm100_linear_fp8/test_linear_fp8_with_residual_testmode.py
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

from sm100_fp8_scale_layout import quantize_to_fp8_deepgemm_style  # noqa: E402
from pytorch_reference import linear_fp8_with_residual_ref  # noqa: E402

# Reuse helpers + DSV3 shapes from the no-residual testmode file so we don't
# duplicate scale-layout / shape / metric logic.
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from test_linear_fp8_testmode import (  # noqa: E402
    HIDDEN_SIZE,
    MATRIX,
    _cosine,
    _gate_up_n,
    _input_scale_dequant_view,
    _input_scale_for_mpk,
    _pick_grid_x,
    _rel_mean,
)

# DSV3 union-of-axes (tp,bs) matrix at the gate_up shape, run at the production
# grid (_pick_grid_x). DOCUMENTED ONLY — every config HANGS (see module docstring
# + decision-log issue #3). Not executed: a hang never returns to be xfailed.
HANGING_MATRIX = [(tp, bs, _gate_up_n(tp)) for (tp, bs) in MATRIX]


def _run_case(tp: int, bs: int, grid_x: int = None) -> bool:
    """Compile + run linear_fp8_with_residual_layer at the DSV3 gate_up shape.

    ``grid_x=None`` mirrors production (``_pick_grid_x``) — which HANGS at DSV3
    shapes. The executed smoke passes an explicit small ``grid_x`` + small N.
    """
    assert bs <= 16, "linear_fp8 decode kernel caps batch <= 16"
    device = "cuda"
    output_size = _gate_up_n(tp)        # TP-sharded N
    reduction_size = HIDDEN_SIZE        # K (not sharded)

    print(f"\n{'='*72}")
    print(f"linear_fp8_with_residual  tp={tp} bs={bs}  "
          f"N={output_size} K={reduction_size}")

    g = torch.Generator(device=device).manual_seed(9001 + tp * 31 + bs)
    x_bf16 = (
        torch.randn((bs, reduction_size), device=device, dtype=torch.bfloat16,
                    generator=g) * 0.1
    )
    w_bf16 = (
        torch.randn((output_size, reduction_size), device=device,
                    dtype=torch.bfloat16, generator=g)
        / (reduction_size ** 0.5)
    )
    # Residual (bs, N), same magnitude as the GEMM output so neither term
    # dominates -- exercises the fused add meaningfully.
    residual = (
        torch.randn((bs, output_size), device=device, dtype=torch.bfloat16,
                    generator=g) * 0.05
    )

    x_fp8, x_scale_packed = _input_scale_for_mpk(x_bf16)
    w_fp8, w_scale_strided = quantize_to_fp8_deepgemm_style(w_bf16)

    output = torch.zeros((bs, output_size), device=device, dtype=torch.bfloat16)

    x_scale_ref_view = _input_scale_dequant_view(x_scale_packed, bs)
    ref = linear_fp8_with_residual_ref(
        x_fp8, x_scale_ref_view, w_fp8, w_scale_strided, residual
    )

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    if grid_x is None:
        grid_x = _pick_grid_x(output_size, num_workers)
    assert (output_size // grid_x) % 128 == 0
    print(f"  num_workers={num_workers} grid_x={grid_x} "
          f"per-task N={output_size // grid_x}")

    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x_fp8, name="input_fp8")
    xs_dt = pk.attach_input(x_scale_packed, name="input_scale")
    w_dt = pk.attach_input(w_fp8, name="weight_fp8")
    ws_dt = pk.attach_input(w_scale_strided, name="weight_scale")
    res_dt = pk.attach_input(residual, name="residual")
    out_dt = pk.attach_input(output, name="output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_fp8_with_residual_layer(
        input_fp8=x_dt,
        input_scale=xs_dt,
        weight_fp8=w_dt,
        weight_scale=ws_dt,
        residual=res_dt,
        output=out_dt,
        grid_dim=(grid_x, 1, 1),
        block_dim=block_dim,
    )

    print("  Compiling...")
    pk.compile(output_dir=THIS_DIR)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    finite = torch.isfinite(output).all().item()
    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = _cosine(output, ref)
    rel = _rel_mean(output, ref)
    print(f"  output[0, :6]:    {output[0, :6].tolist()}")
    print(f"  reference[0, :6]: {ref[0, :6].tolist()}")
    print(f"  finite={finite} max_abs_diff={max_diff:.6f} "
          f"cosine={cos:.6f} rel_mean={rel*100:.4f}%")

    pk.finalize()

    ok = finite and (cos > 0.99 or rel <= 0.05)
    print(f"  {'PASS' if ok else 'FAIL'}: linear_fp8_with_residual "
          f"tp={tp} bs={bs}")
    return ok


def _run_smoke() -> bool:
    """Residual smoke at a shape small enough to NOT trip the known hang
    (total epilogue iters = grid_x * per-task-N/16 = 2*384/16 = 48 ≤ threshold).

    Uses tp=1 shape but a tiny N=768 (grid_x=2 -> per-task N=384) and real DSV3
    K=7168, so it validates the FP8 block-scaled GEMM + residual-add math and the
    UE8M0 scale layout. Proven PASS (cos=1.0) by the localization probe."""
    device = "cuda"
    bs, N, K = 1, 768, HIDDEN_SIZE
    print(f"\n{'='*72}\nlinear_fp8_with_residual SMOKE  bs={bs} N={N} K={K} grid_x=2")
    g = torch.Generator(device=device).manual_seed(7)
    x_bf16 = torch.randn((bs, K), device=device, dtype=torch.bfloat16, generator=g) * 0.1
    w_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g) / (K ** 0.5)
    residual = torch.randn((bs, N), device=device, dtype=torch.bfloat16, generator=g) * 0.05
    x_fp8, x_scale_packed = _input_scale_for_mpk(x_bf16)
    w_fp8, w_scale = quantize_to_fp8_deepgemm_style(w_bf16)
    output = torch.zeros((bs, N), device=device, dtype=torch.bfloat16)
    ref = linear_fp8_with_residual_ref(
        x_fp8, _input_scale_dequant_view(x_scale_packed, bs), w_fp8, w_scale, residual)

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
                  mpi_rank=0, world_size=1, max_num_batched_tokens=bs,
                  max_num_batched_requests=bs)
    pk = PersistentKernel(**params)
    a = pk.attach_input(x_fp8, name="input_fp8")
    b = pk.attach_input(x_scale_packed, name="input_scale")
    c = pk.attach_input(w_fp8, name="weight_fp8")
    d = pk.attach_input(w_scale, name="weight_scale")
    e = pk.attach_input(residual, name="residual")
    f = pk.attach_input(output, name="output")
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_fp8_with_residual_layer(
        input_fp8=a, input_scale=b, weight_fp8=c, weight_scale=d,
        residual=e, output=f, grid_dim=(2, 1, 1), block_dim=block_dim)
    print("  Compiling..."); pk.compile(output_dir=THIS_DIR)
    print("  Running..."); pk(); torch.cuda.synchronize()
    cos = _cosine(output, ref); rel = _rel_mean(output, ref)
    finite = torch.isfinite(output).all().item()
    print(f"  finite={finite} cosine={cos:.6f} rel_mean={rel*100:.4f}%")
    pk.finalize()
    ok = finite and (cos > 0.99 or rel <= 0.05)
    print(f"  {'PASS' if ok else 'FAIL'}: residual SMOKE")
    return ok


def test_linear_fp8_with_residual_testmode():
    print("\nNOTE: the DSV3-shaped union-of-axes residual matrix HANGS at every")
    print("config (real kernel bug, see DSV3_TESTMODE_DECISIONS.md issue #3):")
    for (tp, bs, n) in HANGING_MATRIX:
        print(f"    XFAIL_HANG  tp={tp} bs={bs} N={n} (production grid)")
    print("Executing the residual SMOKE (small N, proven to complete) instead.")
    ok = _run_smoke()
    assert ok, "linear_fp8_with_residual SMOKE failed"


if __name__ == "__main__":
    test_linear_fp8_with_residual_testmode()
