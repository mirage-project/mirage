"""Rank-aware selective weight loader for the DeepSeek V3 reference.

For each rank, loads only the slice of each weight that this rank
holds. Works for any (tp_size, ep_size) topology supported by
`ParallelConfig`.

Sharding rules (must match `modeling.py`'s parallel layers):

    Replicated (full tensor on every rank):
        - embed_tokens.weight, lm_head.weight
        - input_layernorm / post_attention_layernorm / final norm
        - q_a_proj.weight, q_a_layernorm.weight
        - kv_a_proj_with_mqa.weight, kv_a_layernorm.weight
        - MoE gate.weight + e_score_correction_bias
        - MTP enorm/hnorm/eh_proj/shared_head.norm

    ColumnParallel (split output dim by tp_size, take rank's slice):
        - q_b_proj.weight: split out dim by tp_size
        - kv_b_proj.weight: split out dim by tp_size
        - dense MLP gate_up_proj.weight: split out dim by tp_size
        - shared_experts gate_up_proj.weight: split out dim by tp_size

    RowParallel (split input dim by tp_size, take rank's slice):
        - o_proj.weight: split in dim by tp_size
        - dense MLP down_proj.weight: split in dim by tp_size
        - shared_experts down_proj.weight: split in dim by tp_size

    Routed experts (EP × within-EP TP):
        - This rank's local experts: indices
          [ep_rank * num_local : (ep_rank + 1) * num_local]
        - For each local expert:
            gate_up_proj.weight: split out dim by routed_tp_size, take routed_tp_rank's slice
            down_proj.weight: split in dim by routed_tp_size, take routed_tp_rank's slice
        - Other experts (not in this rank's slice) are not loaded.

The loader handles the HF checkpoint's separate `gate_proj.weight` +
`up_proj.weight` storage (combined to `gate_up_proj` at load time via
`cat([gate, up], dim=0)`) and FP8 → BF16 dequantization.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from safetensors import safe_open

from .config import Config
from .modeling import DeepseekV3Model
from .parallel import ParallelConfig


_DEMO_MODELS = (
    Path(__file__).resolve().parents[2] / "demo" / "deepseek_v3" / "models"
)
if str(_DEMO_MODELS) not in sys.path:
    sys.path.insert(0, str(_DEMO_MODELS))


def _index_safetensors(model_dir: Path) -> dict[str, str]:
    import json
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        out: dict[str, str] = {}
        for st in model_dir.glob("*.safetensors"):
            with safe_open(st, framework="pt") as f:
                for k in f.keys():
                    out[k] = str(st)
        return out
    with open(idx_path) as f:
        idx = json.load(f)["weight_map"]
    return {k: str(model_dir / fname) for k, fname in idx.items()}


def _needed_prefixes(
    layer_indices: Iterable[int], num_hidden_layers: int, with_mtp: bool,
    pcfg: ParallelConfig, n_routed_experts: int,
) -> tuple[list[str], set[int]]:
    """Compute which prefixes to load + which routed experts to load.

    Returns (prefixes, local_expert_global_indices_set).
    """
    pfxs = ["model.embed_tokens.", "model.norm.", "lm_head."]
    for li in layer_indices:
        pfxs.append(f"model.layers.{li}.")
    if with_mtp:
        pfxs.append(f"model.layers.{num_hidden_layers}.")

    # Local routed-expert global indices for this rank.
    num_local = pcfg.num_local_routed_experts(n_routed_experts)
    first = pcfg.first_local_routed_expert(n_routed_experts)
    local_experts = set(range(first, first + num_local))
    return pfxs, local_experts


def _key_is_local_expert(key: str, local_experts: set[int]) -> bool:
    """True iff `key` is a routed-expert weight that this rank holds."""
    # Pattern: model.layers.<L>.mlp.experts.<E>.<...>
    parts = key.split(".")
    if len(parts) < 5 or parts[3] != "mlp":
        return False
    if len(parts) < 7 or parts[4] != "experts":
        return False
    try:
        e_idx = int(parts[5])
    except ValueError:
        return False
    return e_idx in local_experts


def _key_has_experts(key: str) -> bool:
    return ".mlp.experts." in key


def _load_state_dict(
    model_dir: Path,
    layer_indices: list[int],
    num_hidden_layers: int,
    with_mtp: bool,
    target_dtype: torch.dtype,
    device: str,
    pcfg: ParallelConfig,
    n_routed_experts: int,
) -> dict[str, torch.Tensor]:
    from convert import dequantize_fp8, is_fp8

    weight_map = _index_safetensors(model_dir)
    needed_pfx, local_experts = _needed_prefixes(
        layer_indices, num_hidden_layers, with_mtp, pcfg, n_routed_experts,
    )

    by_file: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        if not any(k.startswith(p) for p in needed_pfx):
            continue
        # Skip experts not on this rank to save memory + load time.
        if _key_has_experts(k) and not _key_is_local_expert(k, local_experts):
            continue
        by_file.setdefault(fname, []).append(k)

    state_dict: dict[str, torch.Tensor] = {}
    for fname, keys in by_file.items():
        with safe_open(fname, framework="pt", device="cpu") as f:
            for k in keys:
                state_dict[k] = f.get_tensor(k)

    # Pair (weight, weight_scale_inv) → dequantize FP8 → BF16.
    paired: list[str] = []
    for k in list(state_dict.keys()):
        if k.endswith(".weight"):
            scale_k = f"{k}_scale_inv"
            if scale_k in state_dict:
                w = state_dict[k]
                s = state_dict[scale_k]
                if is_fp8(w):
                    w_bf16 = dequantize_fp8(
                        w.to(device), s.to(device)
                    ).to(target_dtype)
                    state_dict[k] = w_bf16
                paired.append(scale_k)
    for k in paired:
        del state_dict[k]
    return state_dict


def _shard_col(t: torch.Tensor, tp_size: int, rank: int) -> torch.Tensor:
    """Split tensor's dim 0 into `tp_size` slices and take `rank`-th."""
    assert t.shape[0] % tp_size == 0, (
        f"col-shard: shape[0]={t.shape[0]} not divisible by tp_size={tp_size}"
    )
    chunk = t.shape[0] // tp_size
    return t[rank * chunk:(rank + 1) * chunk].contiguous()


def _shard_row(t: torch.Tensor, tp_size: int, rank: int) -> torch.Tensor:
    """Split tensor's dim 1 into `tp_size` slices and take `rank`-th."""
    assert t.shape[1] % tp_size == 0, (
        f"row-shard: shape[1]={t.shape[1]} not divisible by tp_size={tp_size}"
    )
    chunk = t.shape[1] // tp_size
    return t[:, rank * chunk:(rank + 1) * chunk].contiguous()


def load_into(
    model: DeepseekV3Model,
    model_dir: str | Path,
    target_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Load weights from a HuggingFace DeepSeek V3 checkpoint into the
    reference model in place. TP/EP-aware: each rank only loads its slice.
    """
    model_dir = Path(model_dir)
    cfg = model.cfg
    pcfg = model.pcfg
    sd = _load_state_dict(
        model_dir,
        layer_indices=model.layer_indices,
        num_hidden_layers=cfg.num_hidden_layers,
        with_mtp=model.enable_mtp,
        target_dtype=target_dtype,
        device=device,
        pcfg=pcfg,
        n_routed_experts=cfg.n_routed_experts,
    )

    def _get(src_key: str) -> torch.Tensor:
        if src_key not in sd:
            raise KeyError(
                f"Missing weight {src_key} in checkpoint (rank {pcfg.rank} "
                f"has {len(sd)} keys)."
            )
        return sd[src_key]

    def _copy_replicated(dst: torch.nn.Parameter, src_key: str) -> None:
        src = _get(src_key).to(dtype=dst.dtype, device=dst.device)
        if src.shape != dst.shape:
            raise ValueError(
                f"Replicated copy shape mismatch for {src_key}: "
                f"src={tuple(src.shape)} dst={tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.copy_(src)

    def _copy_col_tp(
        dst: torch.nn.Parameter, src_key: str,
        tp_size: int = pcfg.tp_size, rank: int = pcfg.rank,
    ) -> None:
        src_full = _get(src_key).to(dtype=dst.dtype, device=dst.device)
        src = _shard_col(src_full, tp_size, rank)
        if src.shape != dst.shape:
            raise ValueError(
                f"Col-TP shape mismatch for {src_key}: "
                f"src={tuple(src.shape)} dst={tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.copy_(src)

    def _copy_row_tp(
        dst: torch.nn.Parameter, src_key: str,
        tp_size: int = pcfg.tp_size, rank: int = pcfg.rank,
    ) -> None:
        src_full = _get(src_key).to(dtype=dst.dtype, device=dst.device)
        src = _shard_row(src_full, tp_size, rank)
        if src.shape != dst.shape:
            raise ValueError(
                f"Row-TP shape mismatch for {src_key}: "
                f"src={tuple(src.shape)} dst={tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.copy_(src)

    def _copy_col_tp_gate_up(
        dst: torch.nn.Parameter, gate_key: str, up_key: str,
        tp_size: int = pcfg.tp_size, rank: int = pcfg.rank,
    ) -> None:
        gate = _get(gate_key).to(dtype=dst.dtype, device=dst.device)
        up = _get(up_key).to(dtype=dst.dtype, device=dst.device)
        merged = torch.cat([gate, up], dim=0)
        sharded = _shard_col(merged, tp_size, rank)
        if sharded.shape != dst.shape:
            raise ValueError(
                f"Col-TP gate_up shape mismatch for [{gate_key}; {up_key}]: "
                f"src={tuple(sharded.shape)} dst={tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.copy_(sharded)

    # =================== Embedding + Final Norm + LM head ===================
    _copy_replicated(model.embed_tokens.weight, "model.embed_tokens.weight")
    _copy_replicated(model.norm.weight, "model.norm.weight")
    if "lm_head.weight" in sd:
        # lm_head is tied to embed_tokens by default; copy if separate
        # tensor exists in the checkpoint (overrides the tied weight).
        _copy_replicated(model.lm_head.weight, "lm_head.weight")

    def _load_decoder_layer(
        layer: torch.nn.Module, pfx: str, *, is_mtp_block: bool = False,
    ) -> None:
        """Load weights for one DeepseekV2DecoderLayer at prefix `pfx`."""
        # Layernorms (replicated).
        _copy_replicated(layer.input_layernorm.weight, f"{pfx}input_layernorm.weight")
        _copy_replicated(
            layer.post_attention_layernorm.weight,
            f"{pfx}post_attention_layernorm.weight",
        )
        # MLA.
        attn = layer.self_attn
        ap = f"{pfx}self_attn."
        # Replicated Q/KV LoRA-down + layernorms.
        _copy_replicated(attn.q_a_proj.weight, f"{ap}q_a_proj.weight")
        _copy_replicated(attn.q_a_layernorm.weight, f"{ap}q_a_layernorm.weight")
        _copy_replicated(
            attn.kv_a_proj_with_mqa.weight, f"{ap}kv_a_proj_with_mqa.weight"
        )
        _copy_replicated(attn.kv_a_layernorm.weight, f"{ap}kv_a_layernorm.weight")
        # ColumnParallel q_b_proj / kv_b_proj.
        _copy_col_tp(attn.q_b_proj.weight, f"{ap}q_b_proj.weight")
        _copy_col_tp(attn.kv_b_proj.weight, f"{ap}kv_b_proj.weight")
        # RowParallel o_proj.
        _copy_row_tp(attn.o_proj.weight, f"{ap}o_proj.weight")

        # MLP.
        mlp_pfx = f"{pfx}mlp."
        is_dense = layer.layer_idx < cfg.first_k_dense_replace and not is_mtp_block
        if is_dense:
            mlp = layer.mlp
            _copy_col_tp_gate_up(
                mlp.gate_up_proj.weight,
                f"{mlp_pfx}gate_proj.weight",
                f"{mlp_pfx}up_proj.weight",
            )
            _copy_row_tp(mlp.down_proj.weight, f"{mlp_pfx}down_proj.weight")
        else:
            moe = layer.mlp
            # Replicated gate + correction bias.
            _copy_replicated(moe.gate.weight, f"{mlp_pfx}gate.weight")
            if f"{mlp_pfx}gate.e_score_correction_bias" in sd:
                _copy_replicated(
                    moe.gate_e_score_correction_bias,
                    f"{mlp_pfx}gate.e_score_correction_bias",
                )
            # Local routed experts.
            for local_idx, global_e in enumerate(
                range(moe.first_local_expert,
                      moe.first_local_expert + moe.num_local_experts)
            ):
                exp = moe.local_experts[local_idx]
                exp_pfx = f"{mlp_pfx}experts.{global_e}."
                _copy_col_tp_gate_up(
                    exp.gate_up_proj.weight,
                    f"{exp_pfx}gate_proj.weight",
                    f"{exp_pfx}up_proj.weight",
                    tp_size=pcfg.routed_tp_size,
                    rank=pcfg.routed_tp_rank,
                )
                _copy_row_tp(
                    exp.down_proj.weight,
                    f"{exp_pfx}down_proj.weight",
                    tp_size=pcfg.routed_tp_size,
                    rank=pcfg.routed_tp_rank,
                )
            # Shared experts (TP across full tp_size).
            shared_pfx = f"{mlp_pfx}shared_experts."
            _copy_col_tp_gate_up(
                moe.shared_gate_up.weight,
                f"{shared_pfx}gate_proj.weight",
                f"{shared_pfx}up_proj.weight",
            )
            _copy_row_tp(
                moe.shared_down.weight,
                f"{shared_pfx}down_proj.weight",
            )

    # =================== Decoder layers ===================
    for li in model.layer_indices:
        _load_decoder_layer(model.layers[str(li)], f"model.layers.{li}.")

    # =================== MTP ===================
    if model.enable_mtp:
        mtp = model.mtp_layer
        mtp_li = cfg.num_hidden_layers
        mtp_pfx = f"model.layers.{mtp_li}."
        # Replicated MTP-specific.
        _copy_replicated(mtp.enorm.weight, f"{mtp_pfx}enorm.weight")
        _copy_replicated(mtp.hnorm.weight, f"{mtp_pfx}hnorm.weight")
        _copy_replicated(mtp.eh_proj.weight, f"{mtp_pfx}eh_proj.weight")
        _copy_replicated(
            mtp.shared_head_norm.weight, f"{mtp_pfx}shared_head.norm.weight"
        )
        # MTP block (full DecoderLayer at the same prefix).
        _load_decoder_layer(mtp.mtp_block, mtp_pfx, is_mtp_block=True)
