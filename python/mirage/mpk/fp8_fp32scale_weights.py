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


def pack_w13(gate: torch.Tensor, gate_scale: torch.Tensor, up: torch.Tensor, up_scale: torch.Tensor):
    """Pack one expert's gate and up weights for this GPU.

    Each input has [IR, H] e4m3 bytes and [IR / 128, H / 128] fp32 scales.
    Return uint8 pieces [IR / 64, H / 128, 128, 128].
    Also return fp32 scales [IR / 128, H / 128, 2], with gate before up.
    Feature tile m uses scale block row m // 2.
    """
    return piece_tiles(gate_up_rows(gate, up)), torch.stack([gate_scale, up_scale], -1).float().contiguous()


def pack_w2(down: torch.Tensor, down_scale: torch.Tensor):
    """Pack one expert's down weights for this GPU.

    The input has [H, IR] e4m3 bytes and [H / 128, IR / 128] fp32 scales.
    Return uint8 pieces [H / 128, IR / 128, 128, 128] and the fp32 scales.
    Each piece (ot, kt) uses scale (ot, kt).
    """
    return piece_tiles(down), down_scale.float().contiguous()
