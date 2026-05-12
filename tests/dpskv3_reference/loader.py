"""Load HF DeepSeek V3 safetensors checkpoint into our subset model
that wraps the official inference Transformer modules.

Two responsibilities:

1. Translate HF parameter names → official-model parameter names.
2. Dequantize FP8 → BF16 at load time (we run the reference in BF16
   for simplicity; the checkpoint stores FP8 + per-128x128-block
   scale_inv).

The official model lives at `/home/muhengl/DeepSeek-V3/inference/model.py`
and its parameter layout matches the DeepSeek paper's notation
(`wq_a`/`wq_b`, `wkv_a`/`wkv_b`, `w1`/`w2`/`w3`, etc.) — different
from HF (`q_a_proj`/`q_b_proj`, `kv_a_proj_with_mqa`/`kv_b_proj`,
`gate_proj`/`down_proj`/`up_proj`). The map is in `_HF_TO_OFFICIAL`.

Only weights for the layer indices passed in `layer_indices` are
loaded; other layers' Modules don't exist in the subset model.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from safetensors import safe_open


# ===== HF name → official name mapping ========================================
#
# Each entry's pattern is matched against the HF key. The capture groups in
# the pattern are substituted into the corresponding "official" template.
#
# Order matters: more specific patterns first.
_HF_TO_OFFICIAL: list[tuple[re.Pattern, str]] = [
    # Top-level
    (re.compile(r"^model\.embed_tokens\.weight$"), "embed.weight"),
    (re.compile(r"^model\.norm\.weight$"), "norm.weight"),
    (re.compile(r"^lm_head\.weight$"), "head.weight"),
    # Per-layer
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"),
     "layers.{0}.attn_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"),
     "layers.{0}.ffn_norm.weight"),
    # Attention (MLA) — both .weight and .weight_scale_inv
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$"),
     "layers.{0}.attn.wq_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight_scale_inv$"),
     "layers.{0}.attn.wq_a.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_a_layernorm\.weight$"),
     "layers.{0}.attn.q_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$"),
     "layers.{0}.attn.wq_b.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight_scale_inv$"),
     "layers.{0}.attn.wq_b.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight$"),
     "layers.{0}.attn.wkv_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight_scale_inv$"),
     "layers.{0}.attn.wkv_a.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_a_layernorm\.weight$"),
     "layers.{0}.attn.kv_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight$"),
     "layers.{0}.attn.wkv_b.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight_scale_inv$"),
     "layers.{0}.attn.wkv_b.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),
     "layers.{0}.attn.wo.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight_scale_inv$"),
     "layers.{0}.attn.wo.scale"),
    # Dense MLP (HF: mlp.{gate,up,down}_proj → official: ffn.{w1,w3,w2}).
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),
     "layers.{0}.ffn.w1.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.w1.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"),
     "layers.{0}.ffn.w3.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.w3.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),
     "layers.{0}.ffn.w2.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.w2.scale"),
    # MoE router gate (HF: mlp.gate.weight / .e_score_correction_bias)
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"),
     "layers.{0}.ffn.gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$"),
     "layers.{0}.ffn.gate.bias"),
    # MoE routed experts (HF: mlp.experts.{j}.{gate,up,down}_proj)
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$"),
     "layers.{0}.ffn.experts.{1}.w1.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.experts.{1}.w1.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$"),
     "layers.{0}.ffn.experts.{1}.w3.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.experts.{1}.w3.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$"),
     "layers.{0}.ffn.experts.{1}.w2.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.experts.{1}.w2.scale"),
    # MoE shared expert (HF: mlp.shared_experts.{gate,up,down}_proj)
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$"),
     "layers.{0}.ffn.shared_experts.w1.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.shared_experts.w1.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$"),
     "layers.{0}.ffn.shared_experts.w3.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.shared_experts.w3.scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$"),
     "layers.{0}.ffn.shared_experts.w2.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight_scale_inv$"),
     "layers.{0}.ffn.shared_experts.w2.scale"),
]

# Ignore-list: MTP layer (model.layers.61 in DSv3) and tokenizer
# artifacts. We don't model MTP in the reference (yet).
_IGNORE_PATTERNS = [
    re.compile(r"^model\.layers\.61\."),
    re.compile(r"^model\.embed_tokens_v2\."),
]


def _hf_to_official_name(hf_key: str) -> Optional[str]:
    """Return the official-model parameter name for an HF key, or None
    if the key should be ignored."""
    for ignore in _IGNORE_PATTERNS:
        if ignore.match(hf_key):
            return None
    for pat, tmpl in _HF_TO_OFFICIAL:
        m = pat.match(hf_key)
        if m:
            return tmpl.format(*m.groups())
    return None


def _index_safetensors(model_dir: Path) -> dict[str, str]:
    """Map HF parameter key → safetensors shard path."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        return {k: str(model_dir / fname) for k, fname in idx.items()}
    out: dict[str, str] = {}
    for st in model_dir.glob("*.safetensors"):
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                out[k] = str(st)
    return out


def _dequantize_fp8(weight_fp8: torch.Tensor, scale_inv: torch.Tensor,
                   block_size: int = 128) -> torch.Tensor:
    """Convert an FP8 weight + per-(block_size x block_size) BF16-scale
    tensor into a full BF16 weight.

        weight_bf16[m, n] = weight_fp8[m, n].float()
                          * scale_inv[m // block_size, n // block_size]

    Padding: HF stores `scale_inv` with shape (ceil(M/bs), ceil(N/bs))
    so we can fall short on the last row/col. We use repeat_interleave
    then crop to (M, N).
    """
    assert weight_fp8.dtype == torch.float8_e4m3fn, (
        f"expected float8_e4m3fn, got {weight_fp8.dtype}"
    )
    M, N = weight_fp8.shape
    # Upcast scale to float32 for the per-block expand+multiply.
    scale = scale_inv.float()
    # Expand block scale to per-element scale via repeat_interleave.
    expanded = (
        scale.repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
    )[:M, :N]
    w_f32 = weight_fp8.float() * expanded
    return w_f32.to(torch.bfloat16)


def _wanted_keys(
    layer_indices: Iterable[int], num_routed_experts: int = 256,
) -> set[str]:
    """Set of HF parameter keys we want for the requested subset of layers.

    Includes top-level (embed, norm, lm_head) + per-layer attn + per-layer
    MLP (dense or MoE). For MoE we include ALL experts; the loader skips
    routed expert weights that don't exist in the official module for
    layers that aren't MoE.
    """
    wanted = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for li in layer_indices:
        prefix = f"model.layers.{li}."
        wanted.add(prefix + "input_layernorm.weight")
        wanted.add(prefix + "post_attention_layernorm.weight")
        attn = prefix + "self_attn."
        for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            wanted.add(attn + n + ".weight")
            wanted.add(attn + n + ".weight_scale_inv")
        wanted.add(attn + "q_a_layernorm.weight")
        wanted.add(attn + "kv_a_layernorm.weight")
        # MLP — request both dense and MoE keys; the loader silently
        # skips ones that don't exist in the checkpoint.
        for n in ("gate_proj", "up_proj", "down_proj"):
            wanted.add(prefix + "mlp." + n + ".weight")
            wanted.add(prefix + "mlp." + n + ".weight_scale_inv")
        wanted.add(prefix + "mlp.gate.weight")
        wanted.add(prefix + "mlp.gate.e_score_correction_bias")
        for j in range(num_routed_experts):
            ep = prefix + f"mlp.experts.{j}."
            for n in ("gate_proj", "up_proj", "down_proj"):
                wanted.add(ep + n + ".weight")
                wanted.add(ep + n + ".weight_scale_inv")
        sh = prefix + "mlp.shared_experts."
        for n in ("gate_proj", "up_proj", "down_proj"):
            wanted.add(sh + n + ".weight")
            wanted.add(sh + n + ".weight_scale_inv")
    return wanted


def load_official_subset(
    model: torch.nn.Module,
    model_path: str | Path,
    layer_indices: Iterable[int],
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict[str, int]:
    """Load HF safetensors weights into a subset model built from the
    official DeepSeek inference code.

    Args:
        model: Subset model (e.g., `DeepseekV3SubsetModel`) whose
            parameters use the official naming scheme.
        model_path: Path to HF checkpoint directory.
        layer_indices: Layer indices to load (matches the model's
            constructed Blocks).
        device: Target device for loaded weights.
        dtype: Target dtype (BF16 is what the official model uses by
            default when not configured for FP8).

    Returns:
        Stats dict: {'loaded': int, 'dequantized': int, 'missing': int}.

    Raises:
        FileNotFoundError if no checkpoint or shard file found.
        RuntimeError if a required HF key is missing from the
        checkpoint (e.g., an MoE layer is in `layer_indices` but the
        checkpoint is dense-only).
    """
    model_path = Path(model_path)
    idx = _index_safetensors(model_path)

    # Build the wanted-set for the requested layer subset.
    wanted = _wanted_keys(layer_indices)
    # The HF index may not have every "wanted" key — e.g., dense
    # layers don't have `gate.weight`, MoE layers don't have
    # `gate_proj.weight`. We tolerate "key in wanted but not in idx".

    # Group keys by shard for efficient single-pass open.
    by_shard: dict[str, list[str]] = {}
    for k, shard in idx.items():
        if k in wanted:
            by_shard.setdefault(shard, []).append(k)

    state_dict = dict(model.state_dict())  # name → tensor

    stats = {"loaded": 0, "dequantized": 0, "skipped": 0, "missing_official": 0}

    # First pass: load all `*.weight` and `*.weight_scale_inv` into a
    # staging dict so we can dequantize after both halves are loaded.
    staging: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                staging[k] = t

    # Second pass: convert HF names to official names, dequantize FP8,
    # copy into model.
    by_official: dict[str, torch.Tensor] = {}
    for hf_key, hf_tensor in staging.items():
        official = _hf_to_official_name(hf_key)
        if official is None:
            stats["skipped"] += 1
            continue
        # `*.scale` and `*.weight` come separately; pair them.
        by_official[official] = hf_tensor

    # For each FP8 weight that has a matching scale, dequantize.
    handled: set[str] = set()
    for off_name, t in list(by_official.items()):
        if off_name.endswith(".scale"):
            continue
        if t.dtype == torch.float8_e4m3fn:
            scale_name = off_name[: -len(".weight")] + ".scale"
            if scale_name in by_official:
                t = _dequantize_fp8(t, by_official[scale_name])
                stats["dequantized"] += 1
                handled.add(scale_name)
            else:
                if verbose:
                    print(f"  [load] missing scale for {off_name} — using raw FP8")
        # Copy into model.
        if off_name not in state_dict:
            stats["missing_official"] += 1
            if verbose:
                print(f"  [load] {off_name} not in model (probably layer not in subset)")
            continue
        target = state_dict[off_name]
        if target.shape != t.shape:
            raise RuntimeError(
                f"shape mismatch for {off_name}: model={tuple(target.shape)}, "
                f"checkpoint={tuple(t.shape)}"
            )
        target.copy_(t.to(dtype=target.dtype, device=device))
        stats["loaded"] += 1
        handled.add(off_name)

    # gate.bias (the float32 e_score_correction_bias) gets through the
    # same path; the loader stored it as torch.float32 in HF, copy as-is.
    return stats


if __name__ == "__main__":
    # Smoke: just verify the mapping table covers all keys for layer 0
    # + layer 3 (one dense, one MoE).
    idx_path = "/raid/catalyst/models/DeepSeek-V3/model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)["weight_map"]
    unknown = []
    layer0_or_3 = 0
    for k in idx:
        if "model.layers.0." in k or "model.layers.3." in k or k in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
        ):
            layer0_or_3 += 1
            if _hf_to_official_name(k) is None and not any(
                p.match(k) for p in _IGNORE_PATTERNS
            ):
                unknown.append(k)
    print(f"layer0+layer3 keys scanned: {layer0_or_3}")
    print(f"unmapped: {len(unknown)}")
    if unknown:
        for k in unknown[:30]:
            print(f"  {k}")
