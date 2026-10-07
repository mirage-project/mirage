"""GPT-OSS MXFP4 checkpoint layout.

The released weights store MoE experts as OCP MXFP4 and leave every other
tensor in bf16. ``blocks`` has shape ``[E, rows, K/32, 16]`` (two E2M1 values
per byte, low nibble = even K). ``scales`` has shape ``[E, rows, K/32]`` and
is an E8M0 exponent (``2**(byte - 127)``).

``rows`` is already the GEMM N dimension MPK wants:
  gate_up  [E, 2I, H] with gate and up interleaved on dim 1
  down     [E, H, I]
"""

import json
import os

import torch

# Same table as transformers.integrations.mxfp4.FP4_VALUES.
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def quant_method(config) -> str | None:
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        return None
    if isinstance(quant, dict):
        method = quant.get("quant_method")
    else:
        method = getattr(quant, "quant_method", None)
    if method is None:
        return None
    # str-Enum stringifies as "QuantizationMethod.MXFP4"; the value is "mxfp4".
    value = getattr(method, "value", method)
    return str(value)


def _checkpoint_keys(model_path: str) -> list[str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            return list(json.load(f)["weight_map"])
    if os.path.isfile(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as st:
            return list(st.keys())
    return []


def checkpoint_expert_format(config, model_path: str | None = None) -> str:
    """``mxfp4`` when the checkpoint stores packed expert blocks.

    ``quantization_config.quant_method`` is the primary signal. A directory
    whose tensors are ``*_blocks`` / ``*_scales`` is also MXFP4: falling
    through to ``AutoModelForCausalLM(..., dtype=bfloat16)`` dequantizes those
    weights. ``torch_dtype: bfloat16`` on the config is the non-expert
    tensors, not a reason to unpack the experts.
    """
    if quant_method(config) == "mxfp4":
        return "mxfp4"
    if model_path and any(
        key.endswith("mlp.experts.gate_up_proj_blocks")
        for key in _checkpoint_keys(model_path)
    ):
        return "mxfp4"
    return "bf16"


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
    # tcgen05's 128B swizzle needs a row stride that is a multiple of 128
    # bytes. K=2880 packs to 1440 bytes, so pad the tail with zeros.
    row_bytes = blocks.shape[-1]
    padded = (row_bytes + 127) // 128 * 128
    if padded != row_bytes:
        out = torch.zeros(*blocks.shape[:-1], padded, dtype=torch.uint8)
        out[..., :row_bytes] = blocks
        blocks = out
    return blocks.contiguous(), scales.contiguous(), bias.contiguous()


def unpack_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize to fp32. ``blocks`` is ``[..., K/2]`` or ``[..., K/32, 16]``."""
    if blocks.shape[-1] == 16 and blocks.dim() == scales.dim() + 1:
        blocks = blocks.reshape(*blocks.shape[:-2], -1)
    low = (blocks & 0x0F).to(torch.int64)
    high = ((blocks >> 4) & 0x0F).to(torch.int64)
    values = _E2M1[torch.stack((low, high), dim=-1)]
    values = values.reshape(*values.shape[:-2], -1)
    nblk = scales.shape[-1]
    values = values.reshape(*values.shape[:-1], nblk, 32)
    scale = torch.pow(2.0, scales.to(torch.float32) - 127.0).unsqueeze(-1)
    return (values * scale).reshape(*values.shape[:-2], nblk * 32)


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
