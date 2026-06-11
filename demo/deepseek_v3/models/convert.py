# DeepSeek V3 weight conversion script for Mirage MPK.
#
# Converts HuggingFace DeepSeek V3 checkpoints into sharded safetensors files
# with KV weight absorption and FP8-to-BF16 dequantization.
#
# Usage:
#   python convert.py \
#       --hf-ckpt-path /path/to/DeepSeek-V3 \
#       --save-path /path/to/output \
#       --model-parallel 8
#
# Based on the Qwen3 converter pattern in demo/qwen3/models/convert.py and
# the public DeepSeek-V3 inference code.

import json
import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm, trange

# ---------------------------------------------------------------------------
# DeepSeek V3 architecture constants (from config.json defaults)
# ---------------------------------------------------------------------------
DEFAULT_NUM_HEADS = 128           # num_attention_heads
DEFAULT_QK_NOPE_HEAD_DIM = 128    # qk_nope_head_dim
DEFAULT_QK_ROPE_HEAD_DIM = 64     # qk_rope_head_dim
DEFAULT_Q_LORA_RANK = 1536        # q_lora_rank
DEFAULT_KV_LORA_RANK = 512        # kv_lora_rank
DEFAULT_V_HEAD_DIM = 128          # v_head_dim
DEFAULT_NUM_LAYERS = 61           # num_hidden_layers (0..60)
DEFAULT_FIRST_MOE_LAYER = 3       # first layer with MoE MLP (layers 0-2 are dense)
DEFAULT_NUM_EXPERTS = 256         # n_routed_experts
DEFAULT_NUM_SHARED_EXPERTS = 1    # n_shared_experts (DeepSeek V3 uses 1 shared expert)
DEFAULT_MTP_LAYERS = 1            # number of MTP (multi-token prediction) layers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(hf_ckpt_path: str) -> dict:
    """Load and return the HuggingFace config.json for the model."""
    config_path = os.path.join(hf_ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found at {config_path}. "
            "Make sure --hf-ckpt-path points to a valid HuggingFace checkpoint."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_params(config: dict) -> dict:
    """Extract relevant architecture parameters from config.json."""
    return {
        "num_heads": config.get("num_attention_heads", DEFAULT_NUM_HEADS),
        "qk_nope_head_dim": config.get("qk_nope_head_dim", DEFAULT_QK_NOPE_HEAD_DIM),
        "qk_rope_head_dim": config.get("qk_rope_head_dim", DEFAULT_QK_ROPE_HEAD_DIM),
        "q_lora_rank": config.get("q_lora_rank", DEFAULT_Q_LORA_RANK),
        "kv_lora_rank": config.get("kv_lora_rank", DEFAULT_KV_LORA_RANK),
        "v_head_dim": config.get("v_head_dim", DEFAULT_V_HEAD_DIM),
        "num_layers": config.get("num_hidden_layers", DEFAULT_NUM_LAYERS),
        "first_moe_layer": config.get("first_k_dense_replace", DEFAULT_FIRST_MOE_LAYER),
        "num_experts": config.get("n_routed_experts", DEFAULT_NUM_EXPERTS),
        "num_shared_experts": config.get("n_shared_experts", DEFAULT_NUM_SHARED_EXPERTS),
        "mtp_layers": config.get("num_nextn_predict_layers", DEFAULT_MTP_LAYERS),
    }


def dequantize_fp8(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize an FP8 (e4m3fn) weight tensor to target_dtype using per-channel
    or per-tensor scale factors.

    Args:
        weight: FP8 weight tensor.
        scale: Scale tensor. Can be per-tensor (scalar), per-output-channel
               (shape [out_features] or [out_features, 1]), or a block-wise
               scale with shape [ceil(out/block), ceil(in/block)].
        target_dtype: Output dtype (default bfloat16).

    Returns:
        Dequantized weight in target_dtype.
    """
    if weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e4m3fnuz:
        weight_f = weight.to(torch.float32)
    else:
        # Already a normal float type, just cast.
        return weight.to(target_dtype)

    if scale is None:
        return weight_f.to(target_dtype)

    scale = scale.to(torch.float32)

    if scale.numel() == 1:
        # Per-tensor scale
        result = weight_f * scale.item()
    elif scale.dim() == 1 and scale.shape[0] == weight.shape[0]:
        # Per-output-channel scale: scale shape [out_features]
        result = weight_f * scale.unsqueeze(1)
    elif scale.dim() == 2:
        # Block-wise scale: scale shape [ceil(out/block_out), ceil(in/block_in)]
        # Vectorized: expand scale to match weight shape via repeat_interleave
        block_size = 128
        out_features, in_features = weight.shape
        expanded = scale.repeat_interleave(block_size, dim=0)[:out_features]
        expanded = expanded.repeat_interleave(block_size, dim=1)[:, :in_features]
        result = weight_f * expanded
    else:
        # Fallback: try broadcasting
        result = weight_f * scale

    return result.to(target_dtype)


def shard_tensor(
    param: torch.Tensor,
    dim: Optional[int],
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Shard a tensor along the given dimension for the given rank.
    If dim is None, the tensor is replicated (not sharded).
    """
    if dim is None or world_size == 1:
        return param
    size = param.size(dim)
    assert size % world_size == 0, (
        f"Dimension {dim} of size {size} is not divisible by world_size {world_size}"
    )
    shard_size = size // world_size
    return param.narrow(dim, rank * shard_size, shard_size).contiguous()


# ---------------------------------------------------------------------------
# Weight absorption: absorb kv_b_proj into q_b_proj
# ---------------------------------------------------------------------------

def absorb_kv_into_q(
    q_b_proj_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """
    Absorb kv_b_proj into q_b_proj so that attention can be computed directly
    against the compressed KV latent (c_KV) without materialising full K per head.

    Original computation:
        q_nope = (q_b_proj @ q_a_out)[:, :nope_dim]   # per-head nope part
        k_nope = kv_b_proj @ c_KV                       # per-head nope part
        score_nope = sum_d(q_nope_h[d] * k_nope_h[d])

    After absorption the score is computed as:
        q_absorbed_h = q_nope_h @ W_k_nope_h             (dot directly with c_KV)
        score_nope   = q_absorbed_h @ c_KV

    Concretely:
        q_b_proj has shape [num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank]
        kv_b_proj has shape [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]

    We split q_b_proj per head into nope and rope parts, and kv_b_proj per head
    into k_nope and v parts.  Then for each head:
        q_nope_h:  [qk_nope_head_dim, q_lora_rank]
        k_nope_h:  [qk_nope_head_dim, kv_lora_rank]
        q_absorbed_nope_h = k_nope_h^T @ q_nope_h  ->  [kv_lora_rank, q_lora_rank]

    The rope part of q_b_proj stays unchanged.

    Output shape:
        [num_heads * (kv_lora_rank + qk_rope_head_dim), q_lora_rank]

    Args:
        q_b_proj_weight: shape [num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank]
        kv_b_proj_weight: shape [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        params: model architecture parameters dict

    Returns:
        Absorbed q_b_proj weight of shape
        [num_heads * (kv_lora_rank + qk_rope_head_dim), q_lora_rank]
    """
    num_heads = params["num_heads"]
    qk_nope_head_dim = params["qk_nope_head_dim"]
    qk_rope_head_dim = params["qk_rope_head_dim"]
    kv_lora_rank = params["kv_lora_rank"]
    v_head_dim = params["v_head_dim"]
    q_lora_rank = params["q_lora_rank"]

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    kv_head_dim = qk_nope_head_dim + v_head_dim        # 256

    # q_b_proj_weight: [num_heads * q_head_dim, q_lora_rank]
    # Reshape to [num_heads, q_head_dim, q_lora_rank]
    q_b = q_b_proj_weight.float().reshape(num_heads, q_head_dim, q_lora_rank)
    q_nope = q_b[:, :qk_nope_head_dim, :]  # [num_heads, qk_nope_head_dim, q_lora_rank]
    q_rope = q_b[:, qk_nope_head_dim:, :]  # [num_heads, qk_rope_head_dim, q_lora_rank]

    # kv_b_proj_weight: [num_heads * kv_head_dim, kv_lora_rank]
    # Reshape to [num_heads, kv_head_dim, kv_lora_rank]
    kv_b = kv_b_proj_weight.float().reshape(num_heads, kv_head_dim, kv_lora_rank)
    k_nope = kv_b[:, :qk_nope_head_dim, :]  # [num_heads, qk_nope_head_dim, kv_lora_rank]

    # Absorb: q_absorbed_nope_h = k_nope_h^T @ q_nope_h
    # k_nope^T: [num_heads, kv_lora_rank, qk_nope_head_dim]
    # q_nope:   [num_heads, qk_nope_head_dim, q_lora_rank]
    # result:   [num_heads, kv_lora_rank, q_lora_rank]
    q_absorbed_nope = torch.bmm(
        k_nope.transpose(1, 2),  # [num_heads, kv_lora_rank, qk_nope_head_dim]
        q_nope,                  # [num_heads, qk_nope_head_dim, q_lora_rank]
    )  # [num_heads, kv_lora_rank, q_lora_rank]

    # Concatenate absorbed nope with unchanged rope part per head.
    # q_absorbed_nope: [num_heads, kv_lora_rank, q_lora_rank]
    # q_rope:          [num_heads, qk_rope_head_dim, q_lora_rank]
    q_absorbed = torch.cat(
        [q_absorbed_nope, q_rope], dim=1
    )  # [num_heads, kv_lora_rank + qk_rope_head_dim, q_lora_rank]

    # Flatten back: [num_heads * (kv_lora_rank + qk_rope_head_dim), q_lora_rank]
    out_dim = kv_lora_rank + qk_rope_head_dim
    q_absorbed = q_absorbed.reshape(num_heads * out_dim, q_lora_rank)

    return q_absorbed.to(q_b_proj_weight.dtype)


# ---------------------------------------------------------------------------
# Weight collection from safetensors files
# ---------------------------------------------------------------------------

def collect_all_tensors(
    hf_ckpt_path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load every tensor from the checkpoint's safetensors files.

    Returns:
        weights: dict mapping full parameter name -> tensor
        scales:  dict mapping full parameter name -> scale tensor (for FP8 weights)
    """
    safetensor_files = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No *.safetensors files found in {hf_ckpt_path}"
        )

    weights: Dict[str, torch.Tensor] = {}
    scales: Dict[str, torch.Tensor] = {}

    print(f"Loading tensors from {len(safetensor_files)} safetensors file(s) ...")
    for file_path in tqdm(safetensor_files, desc="Reading shards"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.endswith("_scale_inv") or name.endswith("_scale"):
                    scales[name] = tensor
                else:
                    weights[name] = tensor

    return weights, scales


def find_scale_for_weight(
    weight_name: str,
    scales: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Find the FP8 scale tensor corresponding to a weight name.

    DeepSeek V3 stores scales with suffixes like ``weight_scale_inv``.
    """
    # Try common naming patterns
    for suffix in ["_scale_inv", "_scale"]:
        candidate = weight_name + suffix
        if candidate in scales:
            return scales[candidate]
        # Sometimes the scale key drops ".weight" and appends the suffix
        if weight_name.endswith(".weight"):
            candidate = weight_name[: -len(".weight")] + suffix
            if candidate in scales:
                return scales[candidate]
    return None


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def is_fp8(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def main(hf_ckpt_path: str, save_path: str, mp: int) -> None:
    """
    Convert a HuggingFace DeepSeek V3 checkpoint into Mirage MPK sharded format.

    Steps:
        1. Load all weights and FP8 scales from safetensors.
        2. Dequantize FP8 weights to BF16.
        3. Absorb kv_b_proj into q_b_proj for each layer.
        4. Shard weights across ``mp`` ranks.
        5. Save per-rank safetensors files.

    Args:
        hf_ckpt_path: Path to HuggingFace checkpoint directory.
        save_path: Output directory.
        mp: Model-parallel world size.
    """
    torch.set_num_threads(8)

    # ------------------------------------------------------------------
    # 1. Load config & weights
    # ------------------------------------------------------------------
    config = load_config(hf_ckpt_path)
    params = get_model_params(config)

    print("Model parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"  model_parallel: {mp}")

    weights, scales = collect_all_tensors(hf_ckpt_path)
    print(f"Loaded {len(weights)} weight tensors and {len(scales)} scale tensors.")

    # ------------------------------------------------------------------
    # 2. Dequantize FP8 -> BF16
    # ------------------------------------------------------------------
    print("Dequantizing FP8 weights to BF16 ...")
    dequant_count = 0
    for name in tqdm(list(weights.keys()), desc="Dequantize"):
        w = weights[name]
        if is_fp8(w):
            scale = find_scale_for_weight(name, scales)
            weights[name] = dequantize_fp8(w, scale, target_dtype=torch.bfloat16)
            dequant_count += 1
        elif w.dtype == torch.float32 or w.dtype == torch.float16:
            weights[name] = w.to(torch.bfloat16)
    print(f"  Dequantized {dequant_count} FP8 tensors.")

    # ------------------------------------------------------------------
    # 3. Absorb kv_b_proj into q_b_proj
    # ------------------------------------------------------------------
    print("Absorbing kv_b_proj into q_b_proj ...")
    num_layers = params["num_layers"]
    absorbed_count = 0
    for layer_idx in trange(num_layers, desc="Absorb"):
        q_b_key = f"model.layers.{layer_idx}.self_attn.q_b_proj.weight"
        kv_b_key = f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"

        if q_b_key not in weights:
            # Layer 0 of some configs may use full attention instead of MLA;
            # skip gracefully.
            continue
        if kv_b_key not in weights:
            print(f"  WARNING: {kv_b_key} missing, skipping absorption for layer {layer_idx}")
            continue

        q_b_weight = weights[q_b_key]
        kv_b_weight = weights[kv_b_key]

        absorbed = absorb_kv_into_q(q_b_weight, kv_b_weight, params)
        weights[q_b_key] = absorbed.to(torch.bfloat16)

        # Remove kv_b_proj since it's been absorbed.
        del weights[kv_b_key]
        absorbed_count += 1

    print(f"  Absorbed kv_b_proj in {absorbed_count} layers.")

    # ------------------------------------------------------------------
    # 4. Build per-rank sharded state dicts
    # ------------------------------------------------------------------
    print(f"Sharding weights across {mp} rank(s) ...")
    state_dicts: List[Dict[str, torch.Tensor]] = [{} for _ in range(mp)]

    num_heads = params["num_heads"]
    kv_lora_rank = params["kv_lora_rank"]
    qk_rope_head_dim = params["qk_rope_head_dim"]
    absorbed_head_dim = kv_lora_rank + qk_rope_head_dim  # 576
    first_moe_layer = params["first_moe_layer"]

    for name in tqdm(sorted(weights.keys()), desc="Shard"):
        param = weights[name]
        shard_dim = _get_shard_dim(name, first_moe_layer)

        # Validate divisibility when sharding
        if shard_dim is not None and mp > 1:
            size = param.size(shard_dim)
            if size % mp != 0:
                print(
                    f"  WARNING: {name} dim {shard_dim} size {size} not divisible "
                    f"by {mp}, replicating instead."
                )
                shard_dim = None

        for rank in range(mp):
            state_dicts[rank][name] = shard_tensor(param, shard_dim, rank, mp)

    # Free original weights to save memory.
    del weights

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    os.makedirs(save_path, exist_ok=True)

    print(f"Saving {mp} shard file(s) to {save_path} ...")
    for rank in trange(mp, desc="Save"):
        out_path = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
        save_file(state_dicts[rank], out_path)

    # Copy tokenizer and config files
    for pattern in ("*token*", "*config*"):
        for file_path in glob(os.path.join(hf_ckpt_path, pattern)):
            dst = os.path.join(save_path, os.path.basename(file_path))
            shutil.copyfile(file_path, dst)

    print("Conversion complete.")


# ---------------------------------------------------------------------------
# Sharding policy
# ---------------------------------------------------------------------------

# Regex patterns for matching layer weight names to their shard dimension.
# None means replicate (no sharding).
# 0 means shard along dim 0 (output / row-parallel).
# 1 means shard along dim 1 (input / column-parallel).

_SHARD_RULES: List[Tuple[str, Optional[int]]] = [
    # Embedding & final norm / head
    (r"^model\.embed_tokens\.weight$",                           0),
    (r"^model\.norm\.weight$",                                   None),
    (r"^lm_head\.weight$",                                       0),

    # --- Attention (MLA) ---
    # q_a_proj: low-rank down-projection, shard output dim (across heads ultimately)
    (r"self_attn\.q_a_proj\.weight$",                            0),
    # q_a_layernorm: small, replicate
    (r"self_attn\.q_a_layernorm\.weight$",                       None),
    # q_b_proj (absorbed): output dim is num_heads * absorbed_head_dim, shard dim 0
    (r"self_attn\.q_b_proj\.weight$",                            0),
    # kv_a_proj_with_mqa: single KV head, replicate
    (r"self_attn\.kv_a_proj_with_mqa\.weight$",                  None),
    # kv_a_layernorm: replicate
    (r"self_attn\.kv_a_layernorm\.weight$",                      None),
    # kv_b_proj should have been absorbed; if still present, replicate
    (r"self_attn\.kv_b_proj\.weight$",                           None),
    # o_proj: column-parallel, shard dim 1
    (r"self_attn\.o_proj\.weight$",                              1),

    # --- Layer norms ---
    (r"input_layernorm\.weight$",                                None),
    (r"post_attention_layernorm\.weight$",                       None),

    # --- Dense MLP (layers 0 .. first_moe_layer-1) ---
    # gate_proj & up_proj: row-parallel, shard dim 0
    (r"mlp\.gate_proj\.weight$",                                 0),
    (r"mlp\.up_proj\.weight$",                                   0),
    # down_proj: column-parallel, shard dim 1
    (r"mlp\.down_proj\.weight$",                                 1),

    # --- MoE router ---
    (r"mlp\.gate\.weight$",                                      None),
    # In some configs the router bias may be present.
    (r"mlp\.gate\.bias$",                                        None),

    # --- MoE routed experts ---
    # Expert weights are NOT sharded across TP (each rank holds all experts).
    # Expert-parallel strategies are handled elsewhere; for weight conversion
    # we replicate them.
    (r"mlp\.experts\.\d+\.gate_proj\.weight$",                   None),
    (r"mlp\.experts\.\d+\.up_proj\.weight$",                     None),
    (r"mlp\.experts\.\d+\.down_proj\.weight$",                   None),

    # --- Shared experts ---
    (r"mlp\.shared_experts\.gate_proj\.weight$",                 0),
    (r"mlp\.shared_experts\.up_proj\.weight$",                   0),
    (r"mlp\.shared_experts\.down_proj\.weight$",                 1),

    # --- MTP (Multi-Token Prediction) layers ---
    (r"model\.layers\.\d+\.enorm\.weight$",                      None),
    (r"enorm\.weight$",                                          None),
    (r"model\.layers\.\d+\.hnorm\.weight$",                      None),
    (r"hnorm\.weight$",                                          None),
    (r"eh_proj\.weight$",                                        None),
    (r"eh_proj\.bias$",                                          None),
    (r"shared_head\.norm\.weight$",                              None),
    (r"shared_head\.head\.weight$",                              0),
]

# Compiled regex list (built once at import time).
_COMPILED_SHARD_RULES = [(re.compile(pat), dim) for pat, dim in _SHARD_RULES]


def _get_shard_dim(name: str, first_moe_layer: int) -> Optional[int]:
    """
    Determine the sharding dimension for a given weight name.

    Returns None for weights that should be replicated.
    """
    for regex, dim in _COMPILED_SHARD_RULES:
        if regex.search(name):
            return dim

    # Fallback: any unmatched weight is replicated with a warning.
    print(f"  WARNING: No shard rule for '{name}', replicating.")
    return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert HuggingFace DeepSeek V3 weights for Mirage MPK."
    )
    parser.add_argument(
        "--hf-ckpt-path",
        type=str,
        required=True,
        help="Path to the HuggingFace DeepSeek V3 checkpoint directory.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Output directory for converted sharded safetensors.",
    )
    parser.add_argument(
        "--model-parallel",
        type=int,
        required=True,
        help="Tensor-parallel world size (number of GPU shards).",
    )
    args = parser.parse_args()

    main(args.hf_ckpt_path, args.save_path, args.model_parallel)
