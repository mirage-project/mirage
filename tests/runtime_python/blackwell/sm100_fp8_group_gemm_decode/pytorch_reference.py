"""PyTorch reference for the NEW DeepSeek-V3 decode MoE grouped FP8 GEMM
(`fp8_group_gemm_layer` -> largem + COMPACT active-mask dispatch, kernel
`fp8_group_gemm_largem_compact_sm100.cuh`).

The kernel consumes pre-packed UE8M0 block scales (it does NOT re-encode them
at runtime — `tcgen05.mma.kind::mxf8f6f4.block_scale` dequantizes directly from
the stored 8-bit exponents). Therefore the reference dequantizes A and B using
*exactly the same packed UE8M0 bytes* that are fed to the kernel's SFA/SFB TMA
descriptors. This makes the reference self-consistent with the hardware regard-
less of the rounding convention used to encode the scales.

These helpers intentionally re-implement the builder's pack/encode math
(`DeepSeekV3Builder._float_to_ue8m0`, `_pack_moe_scale_ue8m0`) so the bytes are
production-identical, and add the inverse unpack so the reference dequant uses
the byte-for-byte scales the kernel sees.
"""

import torch

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


# ---------------------------------------------------------------------------
# UE8M0 encode / pack — byte-identical to builder.py
# ---------------------------------------------------------------------------
def float_to_ue8m0(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> UE8M0 (8-bit exponent only), CEIL rounding of log2.

    Mirrors DeepSeekV3Builder._float_to_ue8m0 exactly.
    """
    pos = torch.where(t > 0, t, torch.full_like(t, 1e-30))
    p2 = torch.pow(2.0, torch.ceil(torch.log2(pos)))
    bits = p2.view(torch.int32)
    ue = ((bits >> 23) & 0xFF).to(torch.uint8)
    ue = torch.where(t > 0, ue, torch.zeros_like(ue))
    return ue


def pack_moe_scale_ue8m0(scale_per_row: torch.Tensor) -> torch.Tensor:
    """[dim, nk] fp32 -> [num_sf_k, dim] uint32 row-major, UE8M0-packed.

    Mirrors DeepSeekV3Builder._pack_moe_scale_ue8m0 exactly: 4 consecutive
    UE8M0 exponents along the K-block axis are packed little-endian into one
    uint32, and the result is transposed to (num_sf_k, dim).
    """
    dim, nk = scale_per_row.shape
    num_sf_k = (nk + 3) // 4
    ue = float_to_ue8m0(scale_per_row).to(torch.int64)
    out = torch.zeros(num_sf_k, dim, dtype=torch.int64, device=scale_per_row.device)
    zero = torch.zeros(dim, num_sf_k, dtype=torch.int64, device=scale_per_row.device)
    for j in range(4):
        ki = torch.arange(num_sf_k, device=scale_per_row.device) * 4 + j
        valid = ki < nk
        ue_col = torch.where(valid, ue[:, ki.clamp(max=nk - 1)], zero[:, 0:num_sf_k])
        out |= (ue_col.t() & 0xFF) << (j * 8)
    return out.to(torch.uint32).contiguous()


def unpack_ue8m0_scale(packed: torch.Tensor, dim: int, nk: int) -> torch.Tensor:
    """Inverse of pack_moe_scale_ue8m0 -> per-(row, k-block) float scale.

    packed: (num_sf_k, dim) uint32. Returns (dim, nk) float32 dequant scales
    (= 2^(ue - 127), with the UE8M0==0 "zero" sentinel mapped to 0.0).
    """
    num_sf_k = packed.shape[0]
    assert packed.shape[1] == dim
    packed_i64 = packed.to(torch.int64).t()  # (dim, num_sf_k)
    out = torch.zeros(dim, nk, dtype=torch.float32, device=packed.device)
    for j in range(4):
        ki = torch.arange(num_sf_k, device=packed.device) * 4 + j
        valid = ki < nk
        if not valid.any():
            continue
        ue = (packed_i64 >> (j * 8)) & 0xFF  # (dim, num_sf_k)
        scale = torch.where(ue > 0,
                            torch.pow(2.0, (ue.float() - 127.0)),
                            torch.zeros_like(ue.float()))
        dst = ki[valid]
        out[:, dst] = scale[:, valid]
    return out


# ---------------------------------------------------------------------------
# FP8 quantize (block_k=128 along the last dim) — UE8M0 round-trip
# ---------------------------------------------------------------------------
def quantize_fp8_blockk(x: torch.Tensor):
    """Quantize [..., K] to FP8 E4M3 + per-128-K-block UE8M0 scale.

    Returns (x_fp8 [..., K] e4m3, packed_scale [num_sf_k, prod(rows)] uint32,
    deq_scale [rows, nk] float32) where deq_scale is recovered from the packed
    bytes (so the reference matmul uses the exact bytes fed to the kernel).
    """
    assert x.shape[-1] % 128 == 0
    K = x.shape[-1]
    nk = K // 128
    flat = x.reshape(-1, K)
    rows = flat.shape[0]
    xb = flat.reshape(rows, nk, 128)
    amax = xb.abs().amax(dim=2)                       # (rows, nk)
    scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)    # (rows, nk) float
    x_fp8 = (xb / scale.unsqueeze(2)).reshape(rows, K).to(torch.float8_e4m3fn)
    packed = pack_moe_scale_ue8m0(scale)              # (num_sf_k, rows)
    deq_scale = unpack_ue8m0_scale(packed, rows, nk)  # (rows, nk)
    return x_fp8.reshape(x.shape), packed, deq_scale


def dequantize_fp8_blockk(x_fp8: torch.Tensor, deq_scale: torch.Tensor) -> torch.Tensor:
    """[rows, K] fp8 + [rows, nk] scale -> [rows, K] float32."""
    rows, K = x_fp8.shape
    nk = K // 128
    xb = x_fp8.float().reshape(rows, nk, 128)
    return (xb * deq_scale.reshape(rows, nk, 1)).reshape(rows, K)


# ---------------------------------------------------------------------------
# Grouped-GEMM reference (compact decode regime)
# ---------------------------------------------------------------------------
def group_gemm_compact_ref(a_fp8, a_deq_scale, b_fp8, b_deq_scale,
                          m_indices, active_experts, bm_padding):
    """Reference for the COMPACT decode group GEMM.

    For every ACTIVE expert e, computes the full BM=bm_padding output block
    D[e*bm : e*bm+bm, :] = dequant(A_block) @ dequant(B[e]).T using the SAME
    UE8M0-decoded scales the kernel sees. Output blocks for INACTIVE experts
    are returned as NaN — the compact kernel must NOT write them.

    a_fp8        (M_total, K)  fp8 e4m3
    a_deq_scale  (M_total, nk) float32  (per-row, per-128-K-block)
    b_fp8        (E, N, K)     fp8 e4m3
    b_deq_scale  (E*N, nk)     float32  (per-output-row, per-128-K-block)
    m_indices    (M_total,)    int32    (expert id per A row block)
    active_experts: iterable of expert ids that are active
    Returns ref (M_total, N) float32 with NaN on inactive blocks.
    """
    M_total, K = a_fp8.shape
    E, N, _ = b_fp8.shape
    nk = K // 128
    a_deq = dequantize_fp8_blockk(a_fp8, a_deq_scale)          # (M_total, K)
    # b_deq_scale is (E*N, nk) -> reshape per expert
    b_deq_scale_e = b_deq_scale.reshape(E, N, nk)
    ref = torch.full((M_total, N), float("nan"), dtype=torch.float32,
                     device=a_fp8.device)
    for e in active_experts:
        rows = slice(e * bm_padding, e * bm_padding + bm_padding)
        # confirm m_indices maps this block to expert e (production invariant)
        assert int(m_indices[e * bm_padding].item()) == e
        b_e = b_fp8[e].float().reshape(N, nk, 128) * b_deq_scale_e[e].reshape(N, nk, 1)
        b_e = b_e.reshape(N, K)
        ref[rows] = a_deq[rows] @ b_e.T
    return ref
