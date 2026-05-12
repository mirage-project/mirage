"""Wrap the official DeepSeek V3 inference model with selective layers
and forward hooks for per-layer hidden state dumping.

The official model lives at /home/muhengl/DeepSeek-V3/inference/model.py.
We import its `Block`, `RMSNorm`, `ParallelEmbedding`, `ColumnParallelLinear`,
`precompute_freqs_cis`, `ModelArgs` and compose them ourselves so we can
build just the layers we want to test.

Why not subclass `Transformer` directly? Because `Transformer.__init__`
hardcodes `n_layers` Blocks and uses `Block.layer_id` to decide MoE vs
MLP. To test layers `[0, 5, 7]` we'd need (a) the Block at position 0
to be dense and Blocks at 1, 2 to be MoE, (b) load weights from HF
indices 0, 5, 7 into our positions 0, 1, 2. Easier to build the list
manually with each Block's original layer_id.
"""

from __future__ import annotations
import sys
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Insert the official DeepSeek-V3 inference dir on sys.path so we can
# import its `model` and `kernel` modules. The clone lives at
# /home/muhengl/DeepSeek-V3 per project_dpskv3_official_ref memory.
_DEEPSEEK_INFERENCE_DIR = Path(
    os.environ.get(
        "DEEPSEEK_V3_INFERENCE_DIR",
        "/home/muhengl/DeepSeek-V3/inference",
    )
)
if not _DEEPSEEK_INFERENCE_DIR.exists():
    raise FileNotFoundError(
        f"DeepSeek-V3 inference dir not found at {_DEEPSEEK_INFERENCE_DIR}. "
        "Clone github.com/deepseek-ai/DeepSeek-V3 to /home/muhengl/DeepSeek-V3 "
        "or set DEEPSEEK_V3_INFERENCE_DIR env var."
    )
if str(_DEEPSEEK_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEPSEEK_INFERENCE_DIR))

# Import the official module. We touch its module-level globals
# (`world_size`, `rank`, `gemm_impl`, `attn_impl`, `Linear.dtype`) to
# configure single-rank BF16 execution.
import model as _official_model  # noqa: E402


def deepseek_v3_args(
    *,
    n_layers: int,
    n_dense_layers: int = 3,
    max_seq_len: int = 16384,
    dtype: str = "bf16",
    max_batch_size: int = 1,
) -> "_official_model.ModelArgs":
    """Build official `ModelArgs` for DeepSeek V3 production config.

    Defaults match the published 671B model (vocab=129280, dim=7168,
    128 attn heads, etc.). `n_layers` is the SUBSET count we'll build;
    `n_dense_layers` stays at 3 (the production split — first 3 layers
    are dense MLP, rest are MoE).

    Use `dtype="fp8"` to keep weights in FP8 (matches HF checkpoint
    natively); `dtype="bf16"` for dequantized BF16 weights (simpler,
    no per-block scale plumbing).
    """
    return _official_model.ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        # DSv3 production constants
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        n_heads=128,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=8,
        n_expert_groups=8,
        n_limited_groups=4,
        score_func="sigmoid",
        route_scale=2.5,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        original_seq_len=4096,
        rope_theta=10000.0,
        rope_factor=40.0,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
    )


def _set_official_globals(
    world_size: int = 1,
    rank: int = 0,
    gemm_impl: str = "bf16",
    attn_impl: str = "absorb",
    linear_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Configure the official `model` module's module-level globals.

    Must be called BEFORE constructing any of the official model's
    classes — they read these on `__init__`.
    """
    _official_model.world_size = world_size
    _official_model.rank = rank
    _official_model.gemm_impl = gemm_impl
    _official_model.attn_impl = attn_impl
    _official_model.Linear.dtype = linear_dtype


class DeepseekV3SubsetModel(nn.Module):
    """Subset of the official Transformer that builds ONLY the
    requested layers.

    Forward signature matches the runner's expectation. Per-layer
    hidden-state recording is controlled by `record_hidden`.
    """

    def __init__(
        self,
        args: "_official_model.ModelArgs",
        layer_indices: list[int],
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        world_size: int = 1,
        rank: int = 0,
    ):
        """Construct the embedding + N requested Blocks + final norm
        + lm_head. Each Block keeps its ORIGINAL layer_id so its MoE-
        vs-MLP choice matches the production model.

        The official model uses module-level globals; we set them
        here before instantiating modules.
        """
        super().__init__()
        # Configure the official module's globals BEFORE constructing
        # any of its classes. dtype="bf16" → `Linear.dtype = bfloat16`
        # so weights are allocated in BF16 (matches our dequantized
        # load flow).
        _set_official_globals(
            world_size=world_size,
            rank=rank,
            gemm_impl="bf16",  # weights stored BF16 after dequantize
            attn_impl="absorb",  # matches MPK's MLA default
            linear_dtype=dtype,
        )

        self.args = args
        self.layer_indices = list(layer_indices)

        # Embedding (full vocab — replicated since world_size=1).
        self.embed = _official_model.ParallelEmbedding(args.vocab_size, args.dim)

        # Blocks — one per requested layer index. Use the ORIGINAL
        # index as `layer_id` so the Block picks MoE vs MLP correctly
        # (Block.__init__: `MLP if layer_id < args.n_dense_layers else MoE`).
        self.layers = nn.ModuleDict()
        for li in self.layer_indices:
            self.layers[str(li)] = _official_model.Block(li, args)

        # Final norm + lm_head (replicated for world_size=1).
        self.norm = _official_model.RMSNorm(args.dim)
        self.head = _official_model.ColumnParallelLinear(
            args.dim, args.vocab_size, dtype=dtype
        )

        # YaRN-scaled freqs_cis — precomputed once for all positions.
        # NOT a buffer so `model.to(dtype=bf16)` doesn't silently cast
        # complex64 → bf16 (which discards the imaginary part). The
        # official model side-steps this by relying on
        # `set_default_dtype`, but we need an explicit `.to()` because
        # callers expect normal module conventions.
        self._freqs_cis_cpu = _official_model.precompute_freqs_cis(args)
        self.freqs_cis = self._freqs_cis_cpu  # will be moved to device below

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        record_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: 1-D LongTensor of shape [T] (single batch).
            positions: 1-D LongTensor of shape [T] giving the absolute
                positions for RoPE. Defaults to `arange(T)` if None.
            record_hidden: if True, return per-layer residual hidden states.

        Returns:
            Dict with keys: argmax (LongTensor [T]), and if
            `record_hidden`: embed, layer_{li}_residual for each
            layer index, final_norm, logits.

        Notes:
            Internally we reshape to [1, T] to match the official
            model's expected 2D shape (batch, seq).
        """
        if input_ids.dim() == 1:
            input_ids_2d = input_ids.unsqueeze(0)
        else:
            input_ids_2d = input_ids
        T = input_ids_2d.size(1)
        if positions is None:
            positions = torch.arange(T, device=input_ids.device, dtype=torch.long)
        # The official Block uses `freqs_cis` indexed by start_pos:start_pos+T.
        # When positions are sparse/non-contiguous we'd need different
        # handling, but for our test cases (prefill from 0, or decode
        # from prompt_length) positions is always arange(start, start+T).
        start_pos = int(positions[0].item())

        out: dict[str, torch.Tensor] = {}

        h = self.embed(input_ids_2d)  # [1, T, H]
        if record_hidden:
            out["embed"] = h.squeeze(0).detach().clone()

        # Lazily move freqs_cis to the input device on first forward.
        if self.freqs_cis.device != input_ids.device:
            self.freqs_cis = self.freqs_cis.to(input_ids.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask = None
        if T > 1:
            mask = torch.full(
                (T, T), float("-inf"), device=input_ids.device
            ).triu_(1)

        for li in self.layer_indices:
            block = self.layers[str(li)]
            h = block(h, start_pos, freqs_cis, mask)
            if record_hidden:
                out[f"layer_{li}_residual"] = h.squeeze(0).detach().clone()

        h_norm = self.norm(h)  # [1, T, H]
        if record_hidden:
            out["final_norm"] = h_norm.squeeze(0).detach().clone()

        logits = self.head(h_norm)  # [1, T, V]
        if record_hidden:
            out["logits"] = logits.squeeze(0).detach().clone()
        argmax = logits.argmax(dim=-1)  # [1, T]
        out["argmax"] = argmax.squeeze(0).detach().clone()
        return out
