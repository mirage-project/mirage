"""MXFP4 reference helpers: decoders for e2m1 nibbles + e8m0 scale bytes, and a
matching bf16 reference matmul. Used by test_linear_1d2d_mxfp4.py.

The MXFP4 SF gmem layout matches NVFP4's atom shape byte-for-byte:
[rows/128, K/64, 32, 4, 4] = 512 bytes per (128 rows, 64 K-elements) atom.
The inner '4' (k_inner) has only 2 active scales for MXFP4 vec::2X — positions
0 and 1 hold a scale each (covering K-elements 0..31 and 32..63 respectively),
and positions 2 and 3 are zero padding ignored by the MMA. This matches what
CUTLASS does for any SFVecSize and lets the kernel reuse the NVFP4 SMEM/cp/TMEM
infrastructure verbatim.
"""

import torch

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def decode_e2m1_packed(packed: torch.Tensor) -> torch.Tensor:
    lut = _E2M1_LUT.to(packed.device)
    lo = packed & 0x0f
    hi = (packed >> 4) & 0x0f
    out = torch.stack([lut[lo.long()], lut[hi.long()]], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def decode_e8m0(s: torch.Tensor) -> torch.Tensor:
    return torch.where(
        s == 0,
        torch.zeros_like(s, dtype=torch.float32),
        torch.pow(2.0, s.to(torch.float32) - 127.0),
    )


def interleaved_sf_to_2d(sf_5d: torch.Tensor) -> torch.Tensor:
    """[rows/128, K/64, 32, 4, 4] → [rows, K/32] uint8.
    Slots 0 and 1 of the inner '4' axis hold active MXFP4 scales (positions
    2 and 3 are padding). The output [rows, K/32] tensor stitches the 2 active
    scales per atom into K/32 = 2 per atom.
    """
    rb, kb, _, _, _ = sf_5d.shape
    rows = rb * 128
    # Take only the first 2 k_inner slots (the active scales)
    x = sf_5d[..., :2]                                # [rb, kb, 32, 4, 2]
    x = x.permute(0, 3, 2, 1, 4)                       # [rb, rg(4), within(32), kb, ki(2)]
    return x.reshape(rb, 4 * 32, kb * 2).reshape(rows, kb * 2)


def mxfp4_dequantize(packed_q: torch.Tensor,
                     sf_5d: torch.Tensor,
                     hidden: int) -> torch.Tensor:
    rows = packed_q.shape[0]
    vals = decode_e2m1_packed(packed_q)        # [rows, hidden]
    sf_2d = interleaved_sf_to_2d(sf_5d)[:rows]  # [rows, hidden/32]
    scales = decode_e8m0(sf_2d)
    vals = vals.view(rows, hidden // 32, 32)
    return (vals * scales.unsqueeze(-1)).reshape(rows, hidden)


def mxfp4_reference_matmul(x_q, x_sf, w_q, w_sf, hidden, residual=None):
    x_f = mxfp4_dequantize(x_q, x_sf, hidden)
    w_f = mxfp4_dequantize(w_q, w_sf, hidden)
    out = x_f.to(torch.float32) @ w_f.to(torch.float32).T
    if residual is not None:
        out = out + residual.to(torch.float32)
    return out.to(torch.bfloat16)
