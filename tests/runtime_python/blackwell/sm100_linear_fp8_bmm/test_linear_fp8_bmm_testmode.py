"""End-to-end test_mode coverage for the MPK FP8 BMM kernel.

Drives the full MPK compilation pipeline (Python layer → register_task →
codegen → nvcc → persistent runtime) for shape configurations representative
of DSv3 MLA Q-absorption (`q_nope @ kv_b_k^T` per head).

BMM contract from the kernel (linear_fp8_bmm_sm100.cuh + register at
src/kernel/task_register.cc:5388):
  - input  [N, H, D_in]
  - weight [H, D_out, D_in]
  - output [N, H, D_out] = input @ weight^T per head
  - grid = (D_out / 128, H, 1) — per-task D_out shard = 128, one head per CTA
  - block = (256, 1, 1)
  - N <= 16; D_in % 128 == 0; D_out % 128 == 0

Run:
  CUDA_VISIBLE_DEVICES=<idle-gpu> \
    python tests/runtime_python/blackwell/sm100_linear_fp8_bmm/test_linear_fp8_bmm_testmode.py
"""
import os
import sys
import torch

# Re-use the FP8/UE8M0 helpers shared across blackwell tests.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from mirage.mpk.models.deepseek_v3 import tasks as dsv3_tasks
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_packed_ue8m0,
    dequant_from_packed_ue8m0,
)

FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_bmm")
os.makedirs(FOLDER, exist_ok=True)


def _make_pk(batch_size: int) -> PersistentKernel:
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    return PersistentKernel(**params)


def _quantize_2d(x_bf16: torch.Tensor):
    """Quantize a 2D BF16 tensor to FP8 + UE8M0-packed scale (row-major)."""
    x_q, packed_scales = quantize_to_fp8_packed_ue8m0(x_bf16)
    return x_q.contiguous(), packed_scales.contiguous()


def _quantize_3d_per_head(x_bf16: torch.Tensor):
    """Quantize a 3D tensor [outer, H, K] or [H, outer, K] by treating the
    H dimension as a leading batch — each head gets its own row-major
    UE8M0 packed scale block of shape [outer, packed_K].

    Returns (fp8 [..., H, K], scale [..., H, packed_K]) flattened along H
    so the final 3D layout matches what the BMM kernel's per-head TMA
    descriptors expect."""
    assert x_bf16.dim() == 3
    outer, H, K = x_bf16.shape
    # Quantize per head — kernel reads weight scale row stride = packed_K
    # within a head and per-head offset = outer * packed_K; the UE8M0
    # packing is done over chunks of 128 along K which is independent
    # across heads.
    fp8_chunks, scale_chunks = [], []
    for h in range(H):
        q, sc = quantize_to_fp8_packed_ue8m0(x_bf16[:, h, :].contiguous())
        fp8_chunks.append(q)
        scale_chunks.append(sc)
    fp8 = torch.stack(fp8_chunks, dim=1).contiguous()       # [outer, H, K]
    scale = torch.stack(scale_chunks, dim=1).contiguous()   # [outer, H, packed_K]
    return fp8, scale


def _quantize_weight(weight_bf16: torch.Tensor):
    """Weight layout is [H, D_out, D_in]. Per-head, quantize the (D_out, D_in)
    slice — UE8M0 packs along D_in (last axis). Output: (H, D_out, D_in) FP8
    + (H, D_out, packed_K) uint32 scale."""
    assert weight_bf16.dim() == 3
    H, D_out, D_in = weight_bf16.shape
    fp8_chunks, scale_chunks = [], []
    for h in range(H):
        q, sc = quantize_to_fp8_packed_ue8m0(weight_bf16[h].contiguous())
        fp8_chunks.append(q)
        scale_chunks.append(sc)
    fp8 = torch.stack(fp8_chunks, dim=0).contiguous()       # [H, D_out, D_in]
    scale = torch.stack(scale_chunks, dim=0).contiguous()   # [H, D_out, packed_K]
    return fp8, scale


def _manual_dequant(fp8: torch.Tensor, packed_scale: torch.Tensor):
    """Layout-detection-free dequant. For a 2D tensor (outer, reduction)
    with UE8M0 packed scale (outer, packed_K), reproduce the kernel's
    exact dequant: every 128-K chunk gets one 8-bit exponent.

    The shared helper `dequant_from_packed_ue8m0` asserts a stride
    pattern that breaks for very small shapes (e.g. outer=1, packed_K=1
    where row-major and col-major look identical). Bypass it here."""
    outer, reduction = fp8.shape
    BLOCK_K = 128
    SCALE_PACK = 4
    logical_k = (reduction + BLOCK_K - 1) // BLOCK_K
    out = torch.empty_like(fp8, dtype=torch.float32)
    fp32 = fp8.float()
    for o in range(outer):
        for sk in range(logical_k):
            packed = int(packed_scale[o, sk // SCALE_PACK].item())
            encoded = (packed >> ((sk % SCALE_PACK) * 8)) & 0xFF
            # UE8M0: encoded == 8-bit exponent (biased 127), value = 2^(exp-127)
            scale = 2.0 ** (encoded - 127)
            k0 = sk * BLOCK_K
            k1 = min(k0 + BLOCK_K, reduction)
            out[o, k0:k1] = fp32[o, k0:k1] * scale
    return out


def _dequant_input(input_fp8, input_scale):
    """Dequant a 3D input [N, H, D_in] back to FP32 per head."""
    N, H, D_in = input_fp8.shape
    out = torch.empty(N, H, D_in, dtype=torch.float32, device=input_fp8.device)
    for h in range(H):
        out[:, h, :] = _manual_dequant(
            input_fp8[:, h, :].contiguous(),
            input_scale[:, h, :].contiguous(),
        )
    return out


def _dequant_weight(weight_fp8, weight_scale):
    """Dequant 3D weight [H, D_out, D_in] per head."""
    H, D_out, D_in = weight_fp8.shape
    out = torch.empty(H, D_out, D_in, dtype=torch.float32,
                      device=weight_fp8.device)
    for h in range(H):
        out[h] = _manual_dequant(weight_fp8[h], weight_scale[h])
    return out


def _run_case(label: str, batch: int, num_heads: int, d_in: int,
              d_out: int, tol: float = 0.05):
    """Compile + run linear_fp8_bmm_layer (swapAB body) end-to-end.

    Verifies output[n, h, :] == input[n, h, :] @ weight[h, :, :]^T (per head)
    against the dequantized FP8 reference.
    """
    assert batch <= 16, f"BMM kernel caps batch ≤ 16, got {batch}"
    assert d_in % 128 == 0, f"d_in={d_in} must be % 128"
    assert d_out % 128 == 0, f"d_out={d_out} must be % 128"
    grid_x = d_out // 128
    grid_y = num_heads

    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  N={batch}  H={num_heads}  D_in={d_in}  D_out={d_out}  "
          f"grid=({grid_x}, {grid_y}, 1)")

    device = "cuda"
    torch.manual_seed(42)
    input_bf16 = (torch.randn(batch, num_heads, d_in,
                              dtype=torch.bfloat16, device=device) * 0.1
                  ).contiguous()
    weight_bf16 = (torch.randn(num_heads, d_out, d_in,
                               dtype=torch.bfloat16, device=device)
                   / (d_in ** 0.5)).contiguous()

    input_fp8, input_scale = _quantize_3d_per_head(input_bf16)
    weight_fp8, weight_scale = _quantize_weight(weight_bf16)
    output = torch.zeros(batch, num_heads, d_out, dtype=torch.bfloat16,
                         device=device)

    # FP8-dequant reference: kernel computes on the quantized operands.
    input_dq = _dequant_input(input_fp8, input_scale)
    weight_dq = _dequant_weight(weight_fp8, weight_scale)
    # ref[n, h, m] = sum_k input_dq[n, h, k] * weight_dq[h, m, k]
    ref = torch.einsum("nhk,hmk->nhm", input_dq, weight_dq).to(torch.bfloat16)

    pk = _make_pk(batch)
    i_fp8 = pk.attach_input(input_fp8, name="bmm_input_fp8")
    i_sc = pk.attach_input(input_scale, name="bmm_input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="bmm_weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="bmm_weight_scale")
    o = pk.attach_input(output, name="bmm_output")

    dsv3_tasks.linear_fp8_bmm_layer(pk, dense=False,
        
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        output=o,
        grid_dim=(grid_x, grid_y, 1),
        block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=FOLDER)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    assert torch.isfinite(output).all(), "BMM output has non-finite values"

    diff = (output.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(f"  output[0,0,:8]: {output[0,0,:8].tolist()}")
    print(f"  ref   [0,0,:8]: {ref[0,0,:8].tolist()}")
    print(f"  max-abs:  {max_abs:.4f}  (tol {tol})")
    print(f"  mean-abs: {mean_abs:.6f}")
    print(f"  cos-sim:  {cos:.6f}")

    pk.finalize()
    ok = max_abs <= tol and cos >= 0.99
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


# ============================================================================
# Test cases — DSv3 MLA Q-absorption shapes.
#
# DSv3 V3:
#   NUM_Q_HEADS = 128, V_HEAD_DIM = 128, KV_LORA_RANK = 512.
#   At decode, the natural BMM application is:
#     q_nope    (N, H=128, D_in=128)
#     kv_b_k    (H=128, D_out=512, D_in=128)   # = kv_b_k.T (the absorption)
#     output    (N, H=128, D_out=512)
# ============================================================================


def test_smoke_b1():
    """Single-token decode (B=1, H=4, D_in=128, D_out=128) — smallest valid."""
    return _run_case("smoke B=1 H=4", batch=1, num_heads=4,
                     d_in=128, d_out=128)


def test_dsv3_decode_b1():
    """DSv3 decode batch=1 shape: H=128, D_in=128, D_out=512."""
    return _run_case("dsv3 q-abs decode B=1",
                     batch=1, num_heads=128, d_in=128, d_out=512)


def test_dsv3_decode_b4():
    """MTP-like batch (spec_length=3+1=4)."""
    return _run_case("dsv3 q-abs decode B=4 (MTP-ish)",
                     batch=4, num_heads=128, d_in=128, d_out=512)


def test_dsv3_decode_b16():
    """Batch=16 — the kernel's max."""
    return _run_case("dsv3 q-abs decode B=16",
                     batch=16, num_heads=128, d_in=128, d_out=512)


def test_dsv3_tp4_decode_b1():
    """TP=4 sharded: H_local = 32, otherwise same shape."""
    return _run_case("dsv3 q-abs TP4 decode B=1",
                     batch=1, num_heads=32, d_in=128, d_out=512)


if __name__ == "__main__":
    results = {}
    for fn in (test_smoke_b1,
               test_dsv3_tp4_decode_b1,
               test_dsv3_decode_b1,
               test_dsv3_decode_b4,
               test_dsv3_decode_b16):
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            print(f"  EXCEPTION in {fn.__name__}: {e}")
            results[fn.__name__] = False
    print("\n" + "=" * 72)
    print("Summary:")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    fail_count = sum(1 for v in results.values() if not v)
    sys.exit(0 if fail_count == 0 else 1)
