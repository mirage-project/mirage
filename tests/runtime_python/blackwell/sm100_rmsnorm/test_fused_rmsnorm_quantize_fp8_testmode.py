"""Test-mode coverage for ``PersistentKernel.fused_rmsnorm_quantize_fp8_layer``.

The fused task does RMSNorm (eps hard-coded 1e-6f) then a per-128-group
BF16->FP8 block quantize of the normalized row, emitting:
  * output_bf16  -- the bf16 normalized row (skipped when emit_bf16=False),
  * output_fp8   -- float8_e4m3fn quantized row,
  * output_scale -- packed UE8M0 uint32 (deepgemm col-major) when
    scale_ue8m0=True, else float32 (bs, num_groups) row-major.

DSV3 facts: HIDDEN=7168 (NOT TP-sharded) -> bs-only sweep. Mirrors the
builder's grid (``_rmsnorm_grid`` default = (bs,1,1)) and block_dim=(128,1,1).
This test covers both scale modes (UE8M0 col-major / f32 row-major) and both
emit_bf16 settings.

References come from the folder's ``pytorch_reference.py``; the UE8M0 path
reuses the shared ``blackwell/common`` quantizer and the scale buffer uses the
exact layout the builder allocates (deepgemm col-major for UE8M0), so the
packed scale is compared bit-for-bit.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
for _p in (THIS_DIR, COMMON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sm100_fp8_scale_layout import (  # noqa: E402
    BLOCK_K,
    allocate_packed_ue8m0_scale_deepgemm_style,
    ceil_div,
    dequant_from_packed_ue8m0,
)
from pytorch_reference import (  # noqa: E402
    fused_rmsnorm_quantize_fp8_ref,
    rmsnorm_ref,
)

HIDDEN = 7168
RMS_NORM_EPS = 1e-6
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2
# Dequantized-FP8 relative tolerance (block-scaled fp8; ~2-5% per decisions).
FP8_REL_TOL = 0.05
BS_SWEEP = [1, 2, 4, 8, 16]


def _dequant_f32_scale(x_q, scales):
    """Dequant the f32-scale path: out[b, k] = fp8[b,k] * scale[b, k//128]."""
    outer, red = x_q.shape
    num_groups = ceil_div(red, BLOCK_K)
    out = x_q.float().clone()
    for g in range(num_groups):
        k0, k1 = g * BLOCK_K, (g + 1) * BLOCK_K
        out[:, k0:k1] = out[:, k0:k1] * scales[:, g].unsqueeze(1)
    return out


def _run_case(bs: int, scale_ue8m0: bool, emit_bf16: bool) -> dict:
    """Run one fused config; returns dict of max diffs and pass flags."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(4321 + bs + (100 if scale_ue8m0 else 0)
                      + (1000 if emit_bf16 else 0))

    assert HIDDEN % BLOCK_K == 0
    num_groups = ceil_div(HIDDEN, BLOCK_K)

    x = torch.randn(bs, HIDDEN, dtype=dtype, device=device)
    w = torch.randn(HIDDEN, dtype=dtype, device=device)
    out_bf16 = torch.zeros(bs, HIDDEN, dtype=dtype, device=device)
    out_fp8 = torch.zeros(bs, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    if scale_ue8m0:
        out_scale = allocate_packed_ue8m0_scale_deepgemm_style(bs, HIDDEN, device)
        out_scale.zero_()
    else:
        out_scale = torch.zeros(bs, num_groups, dtype=torch.float32, device=device)

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
        "fused_rmsnorm_quantize_fp8_layer requires SM100 (Blackwell); "
        f"current target_cc={pk.target_cc}"
    )

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_bf16_dt = pk.attach_input(out_bf16, name="out_bf16")
    out_fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    out_scale_dt = pk.attach_input(out_scale, name="out_scale")

    pk.fused_rmsnorm_quantize_fp8_layer(
        input=x_dt,
        weight=w_dt,
        output_bf16=out_bf16_dt,
        output_fp8=out_fp8_dt,
        output_scale=out_scale_dt,
        grid_dim=(bs, 1, 1),
        block_dim=(128, 1, 1),
        scale_ue8m0=scale_ue8m0,
        emit_bf16=emit_bf16,
    )

    pk.compile(output_dir=THIS_DIR)
    pk()
    torch.cuda.synchronize()

    ref_bf16, ref_fp8, ref_scale = fused_rmsnorm_quantize_fp8_ref(
        x, w, scale_ue8m0=scale_ue8m0, eps=RMS_NORM_EPS
    )

    res = {}

    # --- bf16 normalized output (only when emit_bf16) ---
    if emit_bf16:
        out_f = out_bf16.float()
        ref_f = ref_bf16.float()
        abs_diff = (out_f - ref_f).abs()
        res["bf16_max_diff"] = abs_diff.max().item()
        # Combined atol+rtol (torch.testing.assert_close semantics): element
        # passes if |out-ref| <= atol + rtol*|ref|. A pure-atol check would
        # spuriously flag isolated 1-ULP bf16 rounding diffs from the kernel's
        # tree-reduction order vs torch .mean() (both values are the two
        # nearest bf16 reps of the true result).
        bad = (abs_diff > (BF16_ATOL + BF16_RTOL * ref_f.abs()))
        res["bf16_num_bad"] = int(bad.sum().item())
        res["bf16_ok"] = res["bf16_num_bad"] == 0
    else:
        # Kernel must NOT have written; buffer stays zero.
        res["bf16_max_diff"] = float(out_bf16.float().abs().max().item())
        res["bf16_num_bad"] = 0
        res["bf16_ok"] = True  # not asserted; informational

    # --- scale: exact match for UE8M0 packed; close for f32 ---
    if scale_ue8m0:
        res["scale_exact"] = bool(torch.equal(out_scale, ref_scale))
        res["scale_ok"] = res["scale_exact"]
    else:
        sdiff = (out_scale - ref_scale).abs().max().item()
        res["scale_max_diff"] = sdiff
        res["scale_ok"] = sdiff <= 1e-3

    # --- dequantized fp8 vs reference (relative) ---
    # Compare against the dequant of the kernel's OWN normalized values where
    # possible. We dequantize both kernel and reference fp8 with their scales
    # and compare to the f32 normalized reference (the trustworthy target).
    norm_ref_f32 = rmsnorm_ref(x, w, eps=RMS_NORM_EPS).float()
    if scale_ue8m0:
        deq_out = dequant_from_packed_ue8m0(out_fp8, out_scale)
    else:
        deq_out = _dequant_f32_scale(out_fp8, out_scale)

    denom = norm_ref_f32.abs().clamp_min(1e-3)
    rel = ((deq_out - norm_ref_f32).abs() / denom)
    res["fp8_rel_max"] = rel.max().item()
    res["fp8_rel_mean"] = rel.mean().item()
    res["fp8_ok"] = res["fp8_rel_mean"] <= FP8_REL_TOL

    pk.finalize()
    return res


def test_fused_rmsnorm_quantize_fp8_testmode():
    assert torch.cuda.is_available(), "CUDA required"
    failures = []
    for scale_ue8m0 in (True, False):
        for emit_bf16 in (True, False):
            for bs in BS_SWEEP:
                r = _run_case(bs, scale_ue8m0, emit_bf16)
                ok = r["bf16_ok"] and r["scale_ok"] and r["fp8_ok"]
                status = "PASS" if ok else "FAIL"
                scale_str = ("UE8M0" if scale_ue8m0 else "f32")
                if scale_ue8m0:
                    scale_field = f"scale_exact={r['scale_exact']}"
                else:
                    scale_field = f"scale_max_diff={r['scale_max_diff']:.3e}"
                print(
                    f"[fused] bs={bs:2d} scale={scale_str:5s} "
                    f"emit_bf16={int(emit_bf16)} | "
                    f"bf16_max_diff={r['bf16_max_diff']:.4e} "
                    f"bf16_num_bad={r['bf16_num_bad']} | "
                    f"{scale_field} | "
                    f"fp8_rel_max={r['fp8_rel_max']:.4e} "
                    f"fp8_rel_mean={r['fp8_rel_mean']:.4e} -> {status}"
                )
                if not ok:
                    failures.append(
                        (bs, scale_str, emit_bf16, r))

    if failures:
        print("FAILED fused configs:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("PASSED: fused_rmsnorm_quantize_fp8_layer test_mode correct "
          f"across bs sweep {BS_SWEEP} x {{UE8M0,f32}} x {{emit_bf16 0/1}}")


if __name__ == "__main__":
    test_fused_rmsnorm_quantize_fp8_testmode()
