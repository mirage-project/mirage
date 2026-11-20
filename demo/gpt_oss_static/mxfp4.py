# Copyright 2026 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read the released GPT-OSS MXFP4 checkpoint for the static demo."""

import json
import os

import torch


MXFP4_ZERO = 0x00
MXFP4_NEUTRAL_E8M0 = 127  # E8M0 decodes as 2 ** (exponent - 127), so 127 represents 1.


def deinterleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    """Even rows are gate, odd rows are up. Matches the bf16 loader."""
    return torch.cat((tensor[:, 0::2], tensor[:, 1::2]), dim=1).contiguous()


def convert_expert_weight(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    deinterleave: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Checkpoint expert row to the GEMM layout.

    ``blocks`` is ``[E, rows, K/32, 16]`` or already ``[E, rows, K/2]``,
    uint8. ``scales`` is ``[E, rows, K/32]``, uint8. ``bias`` is bf16 in the
    released checkpoint (fp32 is accepted and cast). Gate/up rows are
    interleaved on dim 1; down_proj is not.
    """
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError(
            "MXFP4 expert blocks and scales must be uint8 "
            f"(got blocks {blocks.dtype}, scales {scales.dtype}). "
            "A floating-point tensor is a dequantized bf16 checkpoint, "
            "not the packed MXFP4 file."
        )
    if blocks.dim() == 4:
        blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    if deinterleave:
        blocks = deinterleave_gate_up(blocks)
        scales = deinterleave_gate_up(scales)
        bias = deinterleave_gate_up(bias)
    if bias.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"MXFP4 expert bias must be bf16 or fp32, got {bias.dtype}"
        )
    if bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    if blocks.shape[-1] * 2 != scales.shape[-1] * 32:
        raise ValueError(
            f"MXFP4 K mismatch: blocks {tuple(blocks.shape)} "
            f"scales {tuple(scales.shape)}"
        )
    return blocks.contiguous(), scales.contiguous(), bias.contiguous()


def prepare_gpt_oss_expert_bank(
    w13_blocks: torch.Tensor,
    w13_scales: torch.Tensor,
    b13: torch.Tensor,
    w2_blocks: torch.Tensor,
    w2_scales: torch.Tensor,
    b2: torch.Tensor,
    padded_width: int = 2944,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deinterleave and pad a GPT-OSS MXFP4 expert bank for the static node.

    Inputs use checkpoint layout: W13 rows interleave gate/up and W2 rows are
    output rows. Packed weights contain two E2M1 nibbles per byte; ``0x00``
    therefore pads values with exact zero. UE8M0 scale byte 127 represents
    ``2 ** (127 - 127) == 1`` and is used for padded zero blocks.

    Returns contiguous TMA-ready tensors. Each MXFP4 weight tile stores 128
    output rows by 64 packed bytes (a 128-value K tile). Each scale tile uses
    the TCGEN05 32-row interleave: scale ``(row, group)`` is stored at
    ``(row % 32) * 16 + (row // 32) * 4 + group``. Tiles are ordered expert,
    output tile, then K tile, matching ``map_wblk``/``map_sf`` and the static MoE task's
    ``expert * (output_tiles * K_tiles) + output_tile * K_tiles + k_tile``
    addressing. Each W13 output tile contains 64 gate rows followed by the 64
    corresponding up rows. Padded intermediate rows are included so every
    weight tile remains a complete 128-row TCGEN05 tile.
    W13 biases remain row-major as padded gate rows followed by padded up rows.
    """
    if w13_blocks.dim() != 4 or w13_scales.dim() != 3:
        raise ValueError("W13 blocks/scales must have shapes [E,2I,K/32,16] and [E,2I,K/32]")
    if w2_blocks.dim() != 4 or w2_scales.dim() != 3:
        raise ValueError("W2 blocks/scales must have shapes [E,H,I/32,16] and [E,H,I/32]")
    experts, w13_rows, w13_groups, packed_bytes = w13_blocks.shape
    e2, hidden, w2_groups, w2_packed_bytes = w2_blocks.shape
    if experts != e2 or packed_bytes != 16 or w2_packed_bytes != 16:
        raise ValueError("expert count must agree and each MXFP4 group must contain 16 packed bytes")
    if w13_scales.shape != (experts, w13_rows, w13_groups):
        raise ValueError("W13 scales must match its expert, row and 32-value group dimensions")
    if w2_scales.shape != (experts, hidden, w2_groups):
        raise ValueError("W2 scales must match its expert, row and 32-value group dimensions")
    if w13_rows % 2 or w13_rows // 2 != hidden:
        raise ValueError("GPT-OSS W13 must have two interleaved rows per W2 output row")
    if w13_groups != w2_groups or hidden % 64 or padded_width < hidden or padded_width % 128:
        raise ValueError("GPT-OSS intermediate must be divisible by 64; padded_width must be >= H and divisible by 128")
    if w13_groups * 32 != hidden:
        raise ValueError("packed W13 K width must equal the W2 output width")
    if w13_blocks.dtype != torch.uint8 or w2_blocks.dtype != torch.uint8:
        raise TypeError("packed MXFP4 blocks must be uint8")
    if w13_scales.dtype != torch.uint8 or w2_scales.dtype != torch.uint8:
        raise TypeError("UE8M0 MXFP4 scales must be uint8")
    if b13.shape != (experts, w13_rows) or b2.shape != (experts, hidden):
        raise ValueError("GPT-OSS biases must match W13 and W2 output rows")
    if b13.dtype not in (torch.bfloat16, torch.float32) or b2.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("GPT-OSS expert biases must be BF16 or FP32")

    padded_groups = padded_width // 32
    # Convert row order and cast biases without modifying checkpoint tensors.
    w13_src, w13_sf_src, b13_src = convert_expert_weight(
        w13_blocks, w13_scales, b13, deinterleave=True
    )
    w2_src, w2_sf_src, b2_src = convert_expert_weight(
        w2_blocks, w2_scales, b2, deinterleave=False
    )

    gate_up = w13_blocks.new_zeros((experts, 2, padded_width, padded_groups, 16))
    gate_up_sf = w13_scales.new_full((experts, 2, padded_width, padded_groups), MXFP4_NEUTRAL_E8M0)
    gate_up_bias = b13_src.new_zeros((experts, 2, padded_width))
    gate_up[:, :, :hidden, :w13_groups].copy_(
        w13_src.view(experts, 2, hidden, w13_groups, 16)
    )
    gate_up_sf[:, :, :hidden, :w13_groups].copy_(
        w13_sf_src.view(experts, 2, hidden, w13_groups)
    )
    gate_up_bias[:, :, :hidden].copy_(b13_src.view(experts, 2, hidden))

    down = w2_blocks.new_zeros((experts, padded_width, padded_groups, 16))
    down_sf = w2_scales.new_full((experts, padded_width, padded_groups), MXFP4_NEUTRAL_E8M0)
    down_bias = b2_src.new_zeros((experts, padded_width))
    down[:, :hidden, :w2_groups].copy_(w2_blocks)
    down_sf[:, :hidden, :w2_groups].copy_(w2_scales)
    down_bias[:, :hidden].copy_(b2_src)

    output_tiles_per_half = padded_width // 128
    k_tiles = padded_width // 128
    # The static expert task uses complete 128-row TCGEN05 tiles (64 gate +
    # 64 up). Include padded output features as zero rows with neutral scales.
    w13_tiles = padded_width // 64
    w13_paired = torch.cat(
        (
            gate_up[:, 0].view(experts, w13_tiles, 64, padded_groups, 16),
            gate_up[:, 1].view(experts, w13_tiles, 64, padded_groups, 16),
        ),
        dim=2,
    )
    w13_tma = w13_paired.view(experts, w13_tiles, 128, k_tiles, 4, 16)
    w13_tma = w13_tma.permute(0, 1, 3, 2, 4, 5).contiguous()
    w13_sf_paired = torch.cat(
        (
            gate_up_sf[:, 0].view(experts, w13_tiles, 64, padded_groups),
            gate_up_sf[:, 1].view(experts, w13_tiles, 64, padded_groups),
        ),
        dim=2,
    )
    w13_sf_tma = w13_sf_paired.view(experts, w13_tiles, 128, k_tiles, 4)
    w13_sf_tma = w13_sf_tma.permute(0, 1, 3, 2, 4).contiguous()
    # tcgen05.mma.block_scale consumes each 512-byte chunk in 32-row bands:
    # rows r, r+32, r+64, r+96 and their four K-group factors are adjacent.
    # TMA and tcgen05.cp copy scale bytes without transposing them.
    w13_sf_tma = (
        w13_sf_tma.view(experts, w13_tiles, k_tiles, 4, 32, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    w2_tma = down.view(experts, output_tiles_per_half, 128, k_tiles, 4, 16)
    w2_tma = w2_tma.permute(0, 1, 3, 2, 4, 5).contiguous()
    w2_sf_tma = down_sf.view(experts, output_tiles_per_half, 128, k_tiles, 4)
    w2_sf_tma = w2_sf_tma.permute(0, 1, 3, 2, 4).contiguous()
    w2_sf_tma = (
        w2_sf_tma.view(experts, output_tiles_per_half, k_tiles, 4, 32, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )

    return (
        w13_tma.reshape(experts, w13_tiles, k_tiles, 8192).contiguous(),
        w13_sf_tma.reshape(experts, w13_tiles, k_tiles, 512).contiguous(),
        w2_tma.reshape(experts, output_tiles_per_half, k_tiles, 8192).contiguous(),
        w2_sf_tma.reshape(experts, output_tiles_per_half, k_tiles, 512).contiguous(),
        gate_up_bias.reshape(experts, 2 * padded_width).contiguous(),
        down_bias.contiguous(),
    )


class Mxfp4Checkpoint:
    """Lazy safetensors reader. One tensor at a time, so a 120B load does not
    hold every shard resident."""

    def __init__(self, model_path: str):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        single = os.path.join(model_path, "model.safetensors")
        self._dir = model_path
        if os.path.isfile(index_path):
            with open(index_path) as f:
                self._weight_map = json.load(f)["weight_map"]
        elif os.path.isfile(single):
            self._weight_map = None
            self._single = single
        else:
            raise FileNotFoundError(
                f"no model.safetensors in {model_path}")

    def __getitem__(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        if self._weight_map is None:
            path = self._single
        else:
            path = os.path.join(self._dir, self._weight_map[key])
        with safe_open(path, framework="pt", device="cpu") as st:
            return st.get_tensor(key)

    def get_expert_tensor(self, key: str, expert: int) -> torch.Tensor:
        """Read one expert row without materializing the full expert tensor."""
        from safetensors import safe_open

        if self._weight_map is None:
            path = self._single
        else:
            path = os.path.join(self._dir, self._weight_map[key])
        with safe_open(path, framework="pt", device="cpu") as st:
            return st.get_slice(key)[expert].contiguous()
