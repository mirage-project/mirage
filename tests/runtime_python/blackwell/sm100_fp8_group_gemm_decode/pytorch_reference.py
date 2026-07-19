"""PyTorch reference for the permuted grouped FP8 GEMM
(`fp8_group_gemm_layer` → fp8_group_gemm_{smallm,largem}_sm100).

The kernel computes, for each output row r:

    D[r, :] = sum_over_K ( dequant(A[r]) @ dequant(B[expert(r)]).T )

where
  * A is (M_total, K) fp8_e4m3 with a per-(row, 128-K-block) UE8M0 scale,
  * B is (E, N, K) fp8_e4m3 with a per-(N-row, 128-K-block) UE8M0 scale,
  * expert(r) = m_indices[(r // BM) * BM]  — the kernel reads ONE expert id
    per BM=128 row block, so all 128 rows of a block share an expert,
  * the K dimension is split into 128-element blocks, each with its own
    UE8M0 (power-of-two) scale that is applied to the partial product.

Scale layout (SFA / SFB) — TRANSPOSED, UE8M0-packed uint32:
  * SFA: (num_sf_k, M_total) uint32, M_total innermost. Each uint32 packs 4
    consecutive UE8M0 K-block scales (low byte = K-block 4*j+0, ...).
  * SFB: (num_sf_k, E*N) uint32, E*N innermost. Same packing.
  num_sf_k = ceil(ceil(K/128) / 4).

This module REUSES the shared UE8M0 encode/decode helpers from
`blackwell/common/sm100_fp8_scale_layout.py` (no scale-encoding logic is
duplicated). The transposed packing into (num_sf_k, dim) uint32 is the
layer's own contract (identical to builder._pack_moe_scale_ue8m0 /
test_wrapper.pack_sf); it is implemented once here so both the test_mode
test and the kernel-wrapper test can share it.
"""

import os
import sys

import torch

_COMMON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from sm100_fp8_scale_layout import (  # noqa: E402
    BLOCK_K,
    FP8_MAX,
    encode_ue8m0,
    decode_ue8m0,
)

BM = 128  # kernel row-block tile; one expert per block (m_indices[bm*BM]).


# --------------------------------------------------------------------------
# Quantization: bf16 -> fp8_e4m3 with per-(row, 128-K-block) UE8M0 scales.
# Returns (fp8 tensor, per-row decoded scale [dim, nk], per-row encoded
# UE8M0 bytes [dim, nk] uint8). The decoded scale is what the reference
# dequant multiplies by; the encoded bytes feed the transposed packer so the
# kernel consumes the SAME UE8M0 values the reference uses.
# --------------------------------------------------------------------------
def _encode_ue8m0_vec(scale: torch.Tensor) -> torch.Tensor:
    """Vectorized form of sm100_fp8_scale_layout.encode_ue8m0 (CEIL).

    encoded = clamp(ceil(log2(clamp(scale, 1e-30))) + 127, 0, 255).
    Bit-identical to the scalar helper applied element-wise.
    """
    s = scale.float().clamp_min(1e-30)
    enc = torch.ceil(torch.log2(s)).to(torch.int64) + 127
    return enc.clamp(0, 255)


def _decode_ue8m0_vec(enc: torch.Tensor) -> torch.Tensor:
    """Vectorized 2^(enc-127) — matches sm100_fp8_scale_layout.decode_ue8m0."""
    return torch.pow(torch.tensor(2.0, device=enc.device),
                     (enc.to(torch.float32) - 127.0))


def _assert_vec_matches_shared():
    """Guard: the vectorized UE8M0 encode/decode are bit-identical to the
    shared scalar helpers (so we genuinely reuse the shared convention, not a
    re-derived one)."""
    probe = torch.tensor([1e-30, 1e-5, 1.0 / 448.0, 0.5, 3.14, 1e5, 1e10],
                         dtype=torch.float32)
    venc = _encode_ue8m0_vec(probe)
    for i, v in enumerate(probe.tolist()):
        assert int(venc[i].item()) == encode_ue8m0(v), (v, venc[i])
    # Decode: compare on the realistic exponent range (very large e overflow
    # float32 to inf on both sides; skip those — they never arise for real
    # bf16 block maxima divided by FP8_MAX).
    for e in range(0, 200):
        got = _decode_ue8m0_vec(torch.tensor([e])).item()
        want = decode_ue8m0(e)
        assert abs(got - want) <= want * 1e-6, (e, got, want)


_assert_vec_matches_shared()


def quantize_rowblock_ue8m0(x_bf16: torch.Tensor):
    assert x_bf16.dim() == 2
    dim, K = x_bf16.shape
    assert K % BLOCK_K == 0, (K, BLOCK_K)
    nk = K // BLOCK_K
    x_f32 = x_bf16.float()
    blk = x_f32.reshape(dim, nk, BLOCK_K)                       # [dim,nk,128]
    amax = blk.abs().amax(dim=2).clamp_min(1e-10)              # [dim,nk]
    enc = _encode_ue8m0_vec(amax / FP8_MAX)                    # [dim,nk] int64
    dec = _decode_ue8m0_vec(enc)                               # [dim,nk] f32
    x_q = torch.clamp(blk / dec[:, :, None], -FP8_MAX, FP8_MAX) \
        .reshape(dim, K).to(torch.float8_e4m3fn)
    return x_q, dec, enc.to(torch.uint8)


def pack_sf_transposed(enc_bytes: torch.Tensor) -> torch.Tensor:
    """[dim, nk] uint8 UE8M0 -> [num_sf_k, dim] uint32, dim innermost.

    Packs 4 consecutive K-block scales per uint32 (low byte = K-block
    4*j+0). Matches the layer docstring / builder._pack_moe_scale_ue8m0 /
    test_wrapper.pack_sf transposed layout that the kernel TMA reads.
    """
    dim, nk = enc_bytes.shape
    num_sf_k = (nk + 3) // 4
    ue = enc_bytes.to(torch.int64)                                  # [dim,nk]
    out = torch.zeros(num_sf_k, dim, dtype=torch.int64,
                      device=enc_bytes.device)
    for j in range(4):
        ki = torch.arange(num_sf_k, device=enc_bytes.device) * 4 + j
        valid = ki < nk
        cols = ki.clamp(max=nk - 1)
        ue_col = torch.where(valid, ue[:, cols], torch.zeros_like(ue[:, cols]))
        out |= (ue_col.t() & 0xFF) << (j * 8)
    return out.to(torch.uint32).contiguous()


def grouped_gemm_ref(a_fp8, sa_dec, b_fp8, sb_dec, m_indices):
    """Grouped FP8 GEMM reference matching the kernel exactly.

    a_fp8   : (M_total, K) fp8        sa_dec : (M_total, nk) f32 (decoded UE8M0)
    b_fp8   : (E, N, K)  fp8          sb_dec : (E, N, nk)  f32 (decoded UE8M0)
    m_indices : (M_total,) int  — expert per row; kernel reads one per BM block.
    Returns (M_total, N) bf16.
    """
    M_total, K = a_fp8.shape
    E, N, _ = b_fp8.shape
    nk = K // BLOCK_K
    A = a_fp8.float()
    B = b_fp8.float()
    out = torch.zeros(M_total, N, dtype=torch.float32, device=A.device)
    for bm in range(0, M_total, BM):
        be = min(bm + BM, M_total)
        expert = int(m_indices[bm].item())  # one expert id per BM block
        for ki in range(nk):
            a_blk = A[bm:be, ki * BLOCK_K:(ki + 1) * BLOCK_K]       # [bs,128]
            b_blk = B[expert, :, ki * BLOCK_K:(ki + 1) * BLOCK_K]   # [N,128]
            partial = a_blk @ b_blk.T                               # [bs,N]
            sa_col = sa_dec[bm:be, ki:ki + 1]                       # [bs,1]
            sb_row = sb_dec[expert, :, ki]                          # [N]
            out[bm:be] += partial * sa_col * sb_row[None, :]
    return out.to(torch.bfloat16)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 1.0 if af.norm().item() == 0.0 and bf.norm().item() == 0.0 \
            else 0.0
    return (af @ bf / denom).item()


def rel_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    return (af - bf).abs().mean().item() / (bf.abs().mean().item() + 1e-30)
