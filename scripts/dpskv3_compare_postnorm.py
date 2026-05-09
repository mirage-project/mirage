#!/usr/bin/env python3
"""Compare per-layer MPK vs reference residuals BOTH pre- and
post-RMSNorm.

The pre-norm view is what `dpskv3_compare_perlayer.py` reports today —
it compares `self.x` (MPK) vs `embed + sum(attn + mlp)` (ref) directly
in residual space. The post-norm view applies the next layer's
`input_layernorm` weight to both sides and compares the resulting
post-rmsnorm tensors. RMSNorm is `weight * x / sqrt(mean(x^2) + eps)`;
since cosine is invariant under scalar magnitude scaling, the only way
post-rmsnorm cosine can differ from pre-rmsnorm cosine is via the
per-channel weight (which can amplify or dampen direction differences).

Usage:
    python scripts/dpskv3_compare_postnorm.py \\
        --mpk outputs/dpskv3_mpk_perlayer_<ts> \\
        --ref outputs/dpskv3_ref_dump_20260509_001712_FIXED \\
        --model-path /raid/catalyst/models/DeepSeek-V3 \\
        --layers 0-19 [--prompt-len 84 --cmp-row 83]
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_layernorm_weights(model_path: str, layer_indices: list[int]) -> dict:
    """Load `model.layers.<i>.input_layernorm.weight` for each layer."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    keys = [f"model.layers.{i}.input_layernorm.weight" for i in layer_indices]
    final_norm_key = "model.norm.weight"
    keys.append(final_norm_key)
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        if key not in weight_map:
            print(f"# WARN: {key} not in index — skipping")
            continue
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        shard_path = os.path.join(model_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in shard_keys:
                out[k] = f.get_tensor(k)
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).to(weight.dtype) * weight


def cos_l2(mpk: torch.Tensor, ref: torch.Tensor, n: int) -> tuple[float, float, float]:
    mpk = mpk[:n].float()
    ref = ref[:n].float()
    cos = F.cosine_similarity(mpk, ref, dim=-1)
    l2_m = mpk.norm(dim=-1)
    l2_r = ref.norm(dim=-1)
    ratio = (l2_m / l2_r.clamp(min=1e-6)).mean().item()
    return cos.mean().item(), cos.min().item(), ratio


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mpk", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--layers", default="0-19")
    p.add_argument("--prompt-len", type=int, default=84)
    p.add_argument("--cmp-row", type=int, default=83)
    args = p.parse_args()

    if "-" in args.layers:
        a, b = args.layers.split("-", 1)
        layer_indices = list(range(int(a), int(b) + 1))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    print(f"# Pre- and post-RMSNorm cosine: MPK={args.mpk}")
    print(f"#                                REF={args.ref}")
    print(f"# Loading input_layernorm weights from {args.model_path} ...")
    weights = load_layernorm_weights(args.model_path, layer_indices)
    print(f"# Loaded {len(weights)} weight tensors.")
    print()
    print(f"{'tag':<22} | {'pre_cos':>9} {'pre_min':>9} {'pre_l2':>8}"
          f" | {'post_cos':>9} {'post_min':>9} {'post_l2':>8}")
    print("-" * 88)

    for li in layer_indices:
        tag = f"layer_{li:02d}_residual"
        mpk_p = Path(args.mpk) / f"{tag}.pt"
        ref_p = Path(args.ref) / f"{tag}.pt"
        if not mpk_p.exists() or not ref_p.exists():
            print(f"{tag:<22} (missing)")
            continue
        mpk = torch.load(mpk_p, map_location="cpu", weights_only=True)
        ref = torch.load(ref_p, map_location="cpu", weights_only=True)
        n = min(mpk.shape[0], ref.shape[0], args.prompt_len)

        # Pre-norm
        pre_mean, pre_min, pre_l2 = cos_l2(mpk, ref, n)

        # Post-norm: apply next layer's input_layernorm if available, else
        # apply the final model.norm (this layer's residual feeds into either
        # the next decoder layer's input_layernorm OR — for the LAST layer
        # in the build — the model.norm before lm_head).
        next_key = f"model.layers.{li+1}.input_layernorm.weight"
        if next_key in weights:
            w = weights[next_key]
            note = "input_layernorm"
        else:
            w = weights.get("model.norm.weight")
            note = "model.norm"
        if w is None:
            print(f"{tag:<22} | "
                  f"{pre_mean:>9.4f} {pre_min:>9.4f} {pre_l2:>8.4f}"
                  f" | (no weight)")
            continue
        mpk_pn = rms_norm(mpk[:n], w)
        ref_pn = rms_norm(ref[:n], w)
        post_mean, post_min, post_l2 = cos_l2(mpk_pn, ref_pn, n)
        print(f"{tag:<22} | "
              f"{pre_mean:>9.4f} {pre_min:>9.4f} {pre_l2:>8.4f}"
              f" | {post_mean:>9.4f} {post_min:>9.4f} {post_l2:>8.4f}"
              f"  ({note})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
