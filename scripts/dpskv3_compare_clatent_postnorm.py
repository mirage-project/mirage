#!/usr/bin/env python3
"""Compare MPK's layer0_c_latent_postnorm.pt to a PyTorch reference
that runs kv_a_proj + kv_a_layernorm on the same input.

Inputs:
    - MPK dump dir (must contain layer0_input_norm.pt and
      layer0_c_latent_postnorm.pt)
    - DeepSeek-V3 model path (for kv_a_proj weights + layernorm weight)

Math:
    rmsnorm_out  = MPK's layer0_input_norm.pt (post-input-layernorm
                   residual, fed into kv_a_proj)
    kv_a_proj_W  = state_dict['model.layers.0.self_attn.kv_a_proj_with_mqa.weight']
                   shape [(kv_lora=512) + (k_pe=64), hidden=7168]
    kv_a_W (slice for c_latent) = kv_a_proj_W[:512, :]
    c_latent     = rmsnorm_out @ kv_a_W.T            (M, 512)
    layernorm_W  = state_dict['...self_attn.kv_a_layernorm.weight']
    c_latent_postnorm_ref = rmsnorm(c_latent) * layernorm_W

Compare against MPK's layer0_c_latent_postnorm[:M, :].

Usage:
    python scripts/dpskv3_compare_clatent_postnorm.py <MPK_DUMP_DIR> [--prompt-len 64]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def dequantize_fp8(w_fp8: torch.Tensor, scale_inv: torch.Tensor,
                   block_size: int = 128) -> torch.Tensor:
    """Dequantize DeepSeek-style FP8 (e4m3) with per-128x128-block scale.

    w_fp8: float8_e4m3fn shape [N, K]
    scale_inv: float32 shape [ceil(N/128), ceil(K/128)]
    Returns: float32 shape [N, K]
    """
    N, K = w_fp8.shape
    nb = (N + block_size - 1) // block_size
    nk = (K + block_size - 1) // block_size
    assert scale_inv.shape == (nb, nk), (
        f"scale shape {tuple(scale_inv.shape)} != ({nb}, {nk})")
    out = torch.empty(N, K, dtype=torch.float32, device=w_fp8.device)
    w_f32 = w_fp8.float()
    for bi in range(nb):
        for ki in range(nk):
            br_start = bi * block_size
            br_end = min(br_start + block_size, N)
            bc_start = ki * block_size
            bc_end = min(bc_start + block_size, K)
            out[br_start:br_end, bc_start:bc_end] = (
                w_f32[br_start:br_end, bc_start:bc_end] * scale_inv[bi, ki]
            )
    return out


def find_kv_a_weights(model_path: str, layer_idx: int = 0):
    """Find the safetensors shard that holds layer 0's kv_a_proj weights."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        prefix = f"model.layers.{layer_idx}.self_attn."
        keys = {
            "kv_a_proj_w":     f"{prefix}kv_a_proj_with_mqa.weight",
            "kv_a_proj_scale": f"{prefix}kv_a_proj_with_mqa.weight_scale_inv",
            "kv_a_ln":         f"{prefix}kv_a_layernorm.weight",
        }
        # Each may live in a different shard; load each from its shard
        loaded = {}
        for short, full in keys.items():
            shard = index.get(full)
            if shard is None:
                continue
            shard_path = os.path.join(model_path, shard)
            sd = load_file(shard_path, device="cpu")
            if full in sd:
                loaded[short] = sd[full]
        return loaded

    # Fallback: try the converted single-shard file
    for fname in os.listdir(model_path):
        if fname.startswith("model0-mp") and fname.endswith(".safetensors"):
            full = os.path.join(model_path, fname)
            sd = load_file(full, device="cpu")
            prefix = f"model.layers.{layer_idx}.self_attn."
            return {
                "kv_a_proj_w":     sd.get(f"{prefix}kv_a_proj_with_mqa.weight"),
                "kv_a_proj_scale": sd.get(f"{prefix}kv_a_proj_with_mqa.weight_scale_inv"),
                "kv_a_ln":         sd.get(f"{prefix}kv_a_layernorm.weight"),
            }
    raise FileNotFoundError(f"Cannot find weights in {model_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--model-path", default="/raid/catalyst/models/DeepSeek-V3")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rms-eps", type=float, default=1e-6)
    args = p.parse_args()

    # Load MPK dumps
    rmsnorm_out_p = os.path.join(args.dump_dir, "layer0_input_norm.pt")
    c_latent_p    = os.path.join(args.dump_dir, "layer0_c_latent_postnorm.pt")
    if not os.path.exists(rmsnorm_out_p):
        print(f"missing: {rmsnorm_out_p}", file=sys.stderr)
        return 2
    if not os.path.exists(c_latent_p):
        print(f"missing: {c_latent_p}", file=sys.stderr)
        return 2

    rmsnorm_out = torch.load(rmsnorm_out_p, map_location="cpu",
                             weights_only=True)
    mpk_c_latent_postnorm = torch.load(c_latent_p, map_location="cpu",
                                       weights_only=True)
    print(f"# rmsnorm_out: shape={tuple(rmsnorm_out.shape)} dtype={rmsnorm_out.dtype}")
    print(f"# mpk_c_latent_postnorm: shape={tuple(mpk_c_latent_postnorm.shape)} "
          f"dtype={mpk_c_latent_postnorm.dtype}")

    # Load weights
    print(f"# Loading weights from {args.model_path}")
    weights = find_kv_a_weights(args.model_path, args.layer)
    if any(v is None for v in weights.values()):
        print(f"# missing weights: {[k for k,v in weights.items() if v is None]}",
              file=sys.stderr)
        return 2
    kv_a_proj_w = weights["kv_a_proj_w"]
    kv_a_proj_scale = weights["kv_a_proj_scale"]
    kv_a_ln = weights["kv_a_ln"]
    print(f"# kv_a_proj.weight: shape={tuple(kv_a_proj_w.shape)} dtype={kv_a_proj_w.dtype}")
    print(f"# kv_a_proj.scale_inv: shape={tuple(kv_a_proj_scale.shape)} dtype={kv_a_proj_scale.dtype}")
    print(f"# kv_a_layernorm.weight: shape={tuple(kv_a_ln.shape)} dtype={kv_a_ln.dtype}")

    # Dequantize the FP8 weight to float32
    if kv_a_proj_w.dtype == torch.float8_e4m3fn:
        kv_a_proj_f32 = dequantize_fp8(kv_a_proj_w, kv_a_proj_scale.float())
    else:
        kv_a_proj_f32 = kv_a_proj_w.float()

    # Slice c_latent part: kv_a_proj.weight is [(kv_lora=512)+(k_pe=64), hidden=7168]
    # vLLM stores as kv_a_proj_with_mqa.weight with output dim = 576,
    # split [:512, :] = kv_a (latent), [512:, :] = kv_pe (rope)
    KV_LORA = 512
    HIDDEN = 7168
    assert kv_a_proj_f32.shape == (576, HIDDEN), (
        f"kv_a_proj weight shape {kv_a_proj_f32.shape} != (576, {HIDDEN})")
    kv_a_latent_w = kv_a_proj_f32[:KV_LORA, :]  # [512, 7168]

    # Compute c_latent = rmsnorm_out @ kv_a_latent_w.T
    # rmsnorm_out shape: [mbt, 7168]. Use FP32 precision.
    M = args.prompt_len
    rmsnorm_out_f32 = rmsnorm_out[:M].float()  # only valid prompt rows
    c_latent_pre_norm = rmsnorm_out_f32 @ kv_a_latent_w.t()  # [M, 512]
    print(f"# c_latent (pre-norm) ref: shape={tuple(c_latent_pre_norm.shape)}")
    print(f"#   L2 first 8 rows: {c_latent_pre_norm.norm(dim=-1)[:8].tolist()}")

    # Apply RMSNorm: x_norm = x * weight / sqrt(mean(x**2) + eps)
    var = c_latent_pre_norm.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + args.rms_eps)
    c_latent_postnorm_ref = c_latent_pre_norm * rstd * kv_a_ln.float()
    c_latent_postnorm_ref_bf16 = c_latent_postnorm_ref.to(torch.bfloat16)

    # Compare to MPK
    mpk_postnorm = mpk_c_latent_postnorm[:M].float()

    # Per-row stats
    diff = (mpk_postnorm.float() - c_latent_postnorm_ref).norm(dim=-1)
    cos = F.cosine_similarity(mpk_postnorm, c_latent_postnorm_ref, dim=-1)
    l2_mpk = mpk_postnorm.norm(dim=-1)
    l2_ref = c_latent_postnorm_ref.norm(dim=-1)
    print(f"\n# Per-row comparison (M={M}):")
    print(f"#   mean cos(MPK, ref) = {cos.mean():.6f}")
    print(f"#   min  cos          = {cos.min():.6f}  (row {cos.argmin().item()})")
    print(f"#   max  diff_l2      = {diff.max():.6f}  (row {diff.argmax().item()})")
    print(f"#   mean diff_l2      = {diff.mean():.6f}")
    print(f"#   l2 ratio (MPK/ref) mean = {(l2_mpk / l2_ref.clamp(min=1e-6)).mean():.4f}")
    print(f"#   any NaN MPK: {torch.isnan(mpk_postnorm).any().item()}")
    print(f"#   any Inf MPK: {torch.isinf(mpk_postnorm).any().item()}")

    if cos.mean() > 0.99:
        print("\nVERDICT: MPK c_latent_postnorm matches PyTorch ref (cos>0.99).")
        print("         kv_a_proj + kv_a_layernorm pipeline is CORRECT.")
        return 0
    elif cos.mean() > 0.5:
        print("\nVERDICT: MPK c_latent_postnorm has SOME error (0.5<cos<0.99).")
        print("         Likely FP8 quantization noise; not a structural bug.")
        return 0
    else:
        print("\nVERDICT: MPK c_latent_postnorm is BROKEN (cos<0.5).")
        print("         kv_a_proj or kv_a_layernorm has a bug.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
