"""Read the released GPT-OSS MXFP4 checkpoint for the static demo."""

import json
import os

import torch


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
