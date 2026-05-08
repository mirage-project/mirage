"""Selective weight loader for the DeepSeek V3 reference.

Loads only the layers specified in `layer_indices` (and optionally the
MTP layer at index `num_hidden_layers`) from a HuggingFace DeepSeek V3
checkpoint, dequantizes FP8 → BF16 on the fly, and copies into the
`DeepseekV3Model` parameter tree.

Why selective:
    The published DeepSeek V3 checkpoint is ~671B parameters. We can't
    load the full thing on a single test GPU. MPK already runs partial
    layers (`--layers 0-3`) for correctness comparison; this loader
    mirrors that.

FP8 dequantization:
    DeepSeek V3 weights are FP8 (E4M3) with a separate per-block scale
    tensor (scale_inv) at `<weight_name>.weight_scale_inv`. The dequant
    formula is:
        block_scale = scale_inv[block_i, block_j]                # float32
        weight_bf16[i, j] = float8_to_float32(weight_fp8[i, j])  \
                           * block_scale[i // BS, j // BS]
        with BS = 128 (DeepSeek V3 specific).

    See `demo/deepseek_v3/models/convert.py:dequantize_fp8` for the
    canonical implementation. The loader reuses it via import.

Weight name mapping:
    HF key (in checkpoint)                     → Reference attribute
    -----------------------------------------------------------------
    model.embed_tokens.weight                  → embed_tokens.weight
    model.layers.<L>.input_layernorm.weight    → layers.<L>.input_layernorm.weight
    model.layers.<L>.self_attn.q_a_proj.weight → layers.<L>.self_attn.q_a_proj.weight
    ... (etc; see dictionaries below)
    model.norm.weight                          → norm.weight
    lm_head.weight                             → lm_head.weight  (tied to embed by default)

    MTP (layer index = num_hidden_layers, e.g., 61):
    model.layers.61.embed_tokens.weight        → embed_tokens.weight (shared with main)
    model.layers.61.enorm.weight               → mtp_layer.enorm.weight
    model.layers.61.hnorm.weight               → mtp_layer.hnorm.weight
    model.layers.61.eh_proj.weight             → mtp_layer.eh_proj.weight
    model.layers.61.input_layernorm.weight     → mtp_layer.mtp_block.input_layernorm.weight
    ... (rest of mtp_block follows main-layer key shape)
    model.layers.61.shared_head.norm.weight    → mtp_layer.shared_head_norm.weight
    model.layers.61.shared_head.head.weight    → tied to lm_head.weight
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


# Add demo path so we can reuse `dequantize_fp8` and `is_fp8`.
_DEMO_MODELS = Path(__file__).resolve().parents[2] / "demo" / "deepseek_v3" / "models"
if str(_DEMO_MODELS) not in sys.path:
    sys.path.insert(0, str(_DEMO_MODELS))


def _index_safetensors(model_dir: Path) -> dict[str, str]:
    """Return key → file mapping by reading model.safetensors.index.json."""
    import json
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        # Single-file checkpoint (rare for DeepSeek V3). Fall back to
        # listing all safetensors files and probing.
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
    layer_indices: Iterable[int], num_hidden_layers: int, with_mtp: bool
) -> list[str]:
    pfxs = ["model.embed_tokens.", "model.norm.", "lm_head."]
    for li in layer_indices:
        pfxs.append(f"model.layers.{li}.")
    if with_mtp:
        pfxs.append(f"model.layers.{num_hidden_layers}.")
    return pfxs


def _matches(key: str, prefixes: list[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _load_state_dict(
    model_dir: Path,
    layer_indices: list[int],
    num_hidden_layers: int,
    with_mtp: bool,
    target_dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    """Load + dequantize the needed slice of the HF checkpoint."""
    from convert import dequantize_fp8, is_fp8  # demo/deepseek_v3/models/convert.py

    weight_map = _index_safetensors(model_dir)
    needed_pfx = _needed_prefixes(layer_indices, num_hidden_layers, with_mtp)
    by_file: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        if _matches(k, needed_pfx):
            by_file.setdefault(fname, []).append(k)

    state_dict: dict[str, torch.Tensor] = {}
    for fname, keys in by_file.items():
        with safe_open(fname, framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                if k.endswith(".weight_scale_inv"):
                    # Don't store separately; consumed by dequant pairing.
                    state_dict[k] = t
                else:
                    state_dict[k] = t

    # Pair (weight, weight_scale_inv) and dequantize where both exist.
    paired: list[str] = []
    for k in list(state_dict.keys()):
        if k.endswith(".weight"):
            scale_k = f"{k}_scale_inv"
            if scale_k in state_dict:
                w = state_dict[k]
                s = state_dict[scale_k]
                if is_fp8(w):
                    w_bf16 = dequantize_fp8(w.to(device), s.to(device)).to(
                        target_dtype
                    )
                    state_dict[k] = w_bf16
                paired.append(scale_k)
    for k in paired:
        del state_dict[k]
    return state_dict


def load_into(
    model: DeepseekV3Model,
    model_dir: str | Path,
    target_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Load weights from a HuggingFace DeepSeek V3 checkpoint into the
    reference model in place. Only the layer indices the model was
    constructed with are loaded.

    HF DeepSeek V3 stores MLP weights as separate `gate_proj.weight`
    and `up_proj.weight`. Our reference uses a fused `gate_up_proj`
    matching vLLM's `MergedColumnParallelLinear`, so we combine at
    load time via `cat([gate, up], dim=0)`.

    HF also stores MTP-specific `model.layers.<N>.embed_tokens.weight`
    and `model.layers.<N>.shared_head.head.weight`. vLLM treats these
    as redundant copies of the main model's tied embeddings — we do
    the same and ignore them. (If they ever differ from main's, the
    audit comparator will flag the resulting argmax mismatch.)
    """
    model_dir = Path(model_dir)
    cfg = model.cfg
    sd = _load_state_dict(
        model_dir,
        layer_indices=model.layer_indices,
        num_hidden_layers=cfg.num_hidden_layers,
        with_mtp=model.enable_mtp,
        target_dtype=target_dtype,
        device=device,
    )

    def _get(src_key: str) -> torch.Tensor:
        if src_key not in sd:
            raise KeyError(
                f"Missing weight {src_key} in checkpoint (have "
                f"{len(sd)} keys)."
            )
        return sd[src_key]

    def _copy(dst: torch.nn.Parameter, src_key: str) -> None:
        src = _get(src_key).to(dtype=dst.dtype, device=dst.device)
        if src.shape != dst.shape:
            raise ValueError(
                f"Shape mismatch for {src_key}: src={tuple(src.shape)} "
                f"dst={tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.copy_(src)

    def _copy_gate_up(dst: torch.nn.Parameter, gate_key: str, up_key: str) -> None:
        gate = _get(gate_key).to(dtype=dst.dtype, device=dst.device)
        up = _get(up_key).to(dtype=dst.dtype, device=dst.device)
        # Reference layout: gate_up_proj weight is [2*intermediate, hidden].
        # cat order = [gate; up] (so chunk(2, dim=-1) of the OUTPUT yields
        # (gate_out, up_out) — see DeepseekV2DenseMLP.forward).
        merged = torch.cat([gate, up], dim=0)
        if merged.shape != dst.shape:
            raise ValueError(
                f"Shape mismatch for [gate;up] {gate_key}+{up_key}: "
                f"src={tuple(merged.shape)} dst={tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.copy_(merged)

    # Embedding (also tied to lm_head).
    _copy(model.embed_tokens.weight, "model.embed_tokens.weight")
    # Final norm.
    _copy(model.norm.weight, "model.norm.weight")
    # lm_head — separate copy in case the checkpoint untied it.
    if "lm_head.weight" in sd:
        _copy(model.lm_head.weight, "lm_head.weight")

    # Decoder layers.
    for li in model.layer_indices:
        layer = model.layers[str(li)]
        pfx = f"model.layers.{li}."
        # Layernorms
        _copy(layer.input_layernorm.weight, f"{pfx}input_layernorm.weight")
        _copy(layer.post_attention_layernorm.weight, f"{pfx}post_attention_layernorm.weight")
        # MLA
        attn = layer.self_attn
        ap = f"{pfx}self_attn."
        _copy(attn.q_a_proj.weight, f"{ap}q_a_proj.weight")
        _copy(attn.q_a_layernorm.weight, f"{ap}q_a_layernorm.weight")
        _copy(attn.q_b_proj.weight, f"{ap}q_b_proj.weight")
        _copy(attn.kv_a_proj_with_mqa.weight, f"{ap}kv_a_proj_with_mqa.weight")
        _copy(attn.kv_a_layernorm.weight, f"{ap}kv_a_layernorm.weight")
        _copy(attn.kv_b_proj.weight, f"{ap}kv_b_proj.weight")
        _copy(attn.o_proj.weight, f"{ap}o_proj.weight")
        # MLP
        mlp_pfx = f"{pfx}mlp."
        if li < cfg.first_k_dense_replace:
            mlp = layer.mlp
            _copy_gate_up(
                mlp.gate_up_proj.weight,
                f"{mlp_pfx}gate_proj.weight",
                f"{mlp_pfx}up_proj.weight",
            )
            _copy(mlp.down_proj.weight, f"{mlp_pfx}down_proj.weight")
        else:
            moe = layer.mlp
            _copy(moe.gate.weight, f"{mlp_pfx}gate.weight")
            # The correction-bias ships under `gate.e_score_correction_bias`.
            if f"{mlp_pfx}gate.e_score_correction_bias" in sd:
                _copy(
                    moe.gate_e_score_correction_bias,
                    f"{mlp_pfx}gate.e_score_correction_bias",
                )
            for e in range(cfg.n_routed_experts):
                exp = moe.experts[e]
                exp_pfx = f"{mlp_pfx}experts.{e}."
                _copy_gate_up(
                    exp.gate_up_proj.weight,
                    f"{exp_pfx}gate_proj.weight",
                    f"{exp_pfx}up_proj.weight",
                )
                _copy(exp.down_proj.weight, f"{exp_pfx}down_proj.weight")
            shared_pfx = f"{mlp_pfx}shared_experts."
            _copy_gate_up(
                moe.shared_experts.gate_up_proj.weight,
                f"{shared_pfx}gate_proj.weight",
                f"{shared_pfx}up_proj.weight",
            )
            _copy(moe.shared_experts.down_proj.weight,
                  f"{shared_pfx}down_proj.weight")

    # MTP layer (if enabled).
    if model.enable_mtp:
        mtp = model.mtp_layer
        mtp_li = cfg.num_hidden_layers
        mtp_pfx = f"model.layers.{mtp_li}."
        _copy(mtp.enorm.weight, f"{mtp_pfx}enorm.weight")
        _copy(mtp.hnorm.weight, f"{mtp_pfx}hnorm.weight")
        _copy(mtp.eh_proj.weight, f"{mtp_pfx}eh_proj.weight")
        # mtp_block (a full DecoderLayer with same key layout)
        layer = mtp.mtp_block
        _copy(layer.input_layernorm.weight, f"{mtp_pfx}input_layernorm.weight")
        _copy(layer.post_attention_layernorm.weight,
              f"{mtp_pfx}post_attention_layernorm.weight")
        attn = layer.self_attn
        ap = f"{mtp_pfx}self_attn."
        _copy(attn.q_a_proj.weight, f"{ap}q_a_proj.weight")
        _copy(attn.q_a_layernorm.weight, f"{ap}q_a_layernorm.weight")
        _copy(attn.q_b_proj.weight, f"{ap}q_b_proj.weight")
        _copy(attn.kv_a_proj_with_mqa.weight, f"{ap}kv_a_proj_with_mqa.weight")
        _copy(attn.kv_a_layernorm.weight, f"{ap}kv_a_layernorm.weight")
        _copy(attn.kv_b_proj.weight, f"{ap}kv_b_proj.weight")
        _copy(attn.o_proj.weight, f"{ap}o_proj.weight")
        # MTP block always uses MoE MLP for DeepSeek V3 (per the published
        # checkpoint), so layer_idx >= first_k_dense_replace by construction.
        moe = layer.mlp
        moe_pfx = f"{mtp_pfx}mlp."
        _copy(moe.gate.weight, f"{moe_pfx}gate.weight")
        if f"{moe_pfx}gate.e_score_correction_bias" in sd:
            _copy(moe.gate_e_score_correction_bias,
                  f"{moe_pfx}gate.e_score_correction_bias")
        for e in range(cfg.n_routed_experts):
            exp = moe.experts[e]
            exp_pfx = f"{moe_pfx}experts.{e}."
            _copy_gate_up(
                exp.gate_up_proj.weight,
                f"{exp_pfx}gate_proj.weight",
                f"{exp_pfx}up_proj.weight",
            )
            _copy(exp.down_proj.weight, f"{exp_pfx}down_proj.weight")
        shared_pfx = f"{moe_pfx}shared_experts."
        _copy_gate_up(
            moe.shared_experts.gate_up_proj.weight,
            f"{shared_pfx}gate_proj.weight",
            f"{shared_pfx}up_proj.weight",
        )
        _copy(moe.shared_experts.down_proj.weight,
              f"{shared_pfx}down_proj.weight")
        # Shared head norm.
        _copy(mtp.shared_head_norm.weight,
              f"{mtp_pfx}shared_head.norm.weight")
        # Note: model.layers.<mtp_li>.embed_tokens.weight and
        # model.layers.<mtp_li>.shared_head.head.weight exist in the
        # checkpoint as redundant copies of the main embed / lm_head
        # (vLLM treats them as tied). We don't load them explicitly —
        # the main model's tied embed is reused. If they ever differ
        # in a future checkpoint, the comparator will catch it via an
        # mtp_argmax mismatch.
