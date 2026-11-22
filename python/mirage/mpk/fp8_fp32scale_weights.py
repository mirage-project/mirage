"""Convert checkpoint weights into pieces for the FP8 task templates.

The checkpoint format uses e4m3 values and one fp32 scale per 128 x 128 block.
scaled_moe_experts and scaled_gemm_tile use the packed layout below.
The pack functions preserve the e4m3 bytes and fp32 scale values.

Layout: the functions cut a weight into pieces of 128 rows x 128 columns. Each piece is row-major (128 B for each row).
W13 piece (expert e, feature tile m, K tile kt) = index e * (MT13 * KT) + m * KT + kt. Its rows 0..63 are the gate rows of
features [64 m, 64 m + 64), and its rows 64..127 are the up rows of the same features. W2 piece (e, output tile ot, K tile kt)
= e * (OT2 * KT2) + ot * KT2 + kt.
The kernel selects a weight scale for each row of a piece.
A W13 piece uses separate gate and up scales. A W2 piece uses one scale.
pack_w2(ue8m0=True), pack_w13(ue8m0=True) and requant_ue8m0 change the bytes and scales: SGLang's requantization to
power-of-two scales (the UE8M0 modes of scaled_gemm_tile and scaled_moe_experts' W2 and W13).
"""
import torch


def piece_tiles(q: torch.Tensor) -> torch.Tensor:
    """Reorder [N, K] bytes into row-major pieces [N / 128, K / 128, 128, 128]."""
    N, K = q.shape
    return q.view(N // 128, 128, K // 128, 128).transpose(1, 2).contiguous()


def gate_up_rows(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Place 64 gate rows before their 64 up rows for each W13 feature tile."""
    IR = gate.shape[0]
    return torch.stack([gate.view(IR // 64, 64, -1), up.view(IR // 64, 64, -1)], 1).reshape(2 * IR, -1)


def pack_w13(gate: torch.Tensor, gate_scale: torch.Tensor, up: torch.Tensor, up_scale: torch.Tensor, ue8m0: bool = False):
    """Pack one expert's gate and up weights for this GPU.

    Each input has [IR, H] e4m3 bytes and [IR / 128, H / 128] fp32 scales.
    Return uint8 pieces [IR / 64, H / 128, 128, 128].
    Also return fp32 scales [IR / 128, H / 128, 2], with gate before up.
    Feature tile m uses scale block row m // 2.
    ue8m0: the gate and up shards requantized first, each per 128 x 128 block (requant_ue8m0: other bytes, power-of-two
    scales; a block of SGLang's fused [2 I, H] w13 is all gate or all up rows, so its requantization of the full weight
    gives these bytes), for scaled_moe_experts' W13 with the scales applied by the MMA
    (scaled_moe_experts_layer(w13_ue8m0=True)).
    """
    if ue8m0:
        gate, gate_scale = requant_ue8m0(gate.view(torch.uint8), gate_scale)
        up, up_scale = requant_ue8m0(up.view(torch.uint8), up_scale)
    return piece_tiles(gate_up_rows(gate, up)), torch.stack([gate_scale, up_scale], -1).float().contiguous()


def dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] (bytes or float8_e4m3fn) with fp32 block scales [ceil(N / 128), K / 128] -> fp32."""
    N, K = w.shape
    scale = s.float().repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    return w.view(torch.float8_e4m3fn).float() * scale


def requant_ue8m0(w: torch.Tensor, s: torch.Tensor):
    """e4m3 [N, K] (N, K multiples of 128) with fp32 block scales -> (e4m3 [N, K] uint8, fp32 scales [N / 128, K / 128],
    powers of two): SGLang's requantization for DeepGEMM on Blackwell (fp8_utils.requant_weight_ue8m0: block_quant_dequant
    to bf16, then per_block_cast_to_fp8: per 128 x 128 block sf = ceil_to_ue8m0(max(amax, 1e-4) / 448) and e4m3(w / sf)).
    Per block, so a TP shard of whole 128 x 128 blocks requantizes to the same bytes as the full weight's."""
    N, K = w.shape
    v = dequant(w, s).to(torch.bfloat16).view(N // 128, 128, K // 128, 128)
    amax = v.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    bits = (amax / 448.0).view(torch.int32)   # ceil_to_ue8m0: the exponent, up by one if the mantissa is not zero
    e = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254)
    sf = (e << 23).view(torch.float32)
    q = (v * (1.0 / sf)).to(torch.float8_e4m3fn)
    return q.view(N, K).view(torch.uint8), sf.view(N // 128, K // 128).contiguous()


def pack_w2(down: torch.Tensor, down_scale: torch.Tensor, ue8m0: bool = False):
    """Pack one expert's down weights for this GPU.

    The input has [H, IR] e4m3 bytes and [H / 128, IR / 128] fp32 scales.
    Return uint8 pieces [H / 128, IR / 128, 128, 128] and the fp32 scales.
    Each piece (ot, kt) uses scale (ot, kt).
    ue8m0: the shard requantized first (requant_ue8m0: other bytes, power-of-two scales), for scaled_moe_experts' W2 with
    the scales applied by the MMA (scaled_moe_experts_layer(w2_ue8m0=True)).
    """
    if ue8m0:
        down, down_scale = requant_ue8m0(down.view(torch.uint8), down_scale)
    return piece_tiles(down), down_scale.float().contiguous()
