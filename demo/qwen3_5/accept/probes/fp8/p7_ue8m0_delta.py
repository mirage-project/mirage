#!/usr/bin/env python3
"""P7 -- UE8M0 dense requant divergence, quantified (M2-I2, v1-architecture.md SS14).

Supporting evidence (optional per spec) for the SS6.2 rejection of MPK's EXISTING dense FP8
path: calls the REAL `DeepSeekV3Builder._requantize_fp8_for_ue8m0` (mirage's actual production
code, `python/mirage/mpk/models/deepseek_v3/builder.py:476-542` -- imported directly, not
reimplemented, for exact fidelity; it is a `@staticmethod` doing only dequant/quantize torch
math, no MPK runtime/kernel/CUDA dependency, verified before writing this script) on real
Qwen3.5-35B-A3B-FP8 checkpoint dense-path tensors, and reports elementwise delta stats between
the checkpoint's own block-dequantized weight ("before") and the UE8M0-requantized-then-
dequantized weight ("after").

Expected: nonzero deltas in BOTH directions (checkpoint scales are not exact powers of two) --
exactly-zero deltas would falsify SS6.2's rejection rationale. Runs on CPU in venv-mpk per
spec (no GPU claim needed -- the exact torch build MPK itself uses performs the math).
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

CKPT_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
SNAPSHOT = os.environ.get(
    "QWEN35_SNAPSHOT",
    os.path.expanduser(
        f"~/mpk-qwen35/hf/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/{CKPT_REVISION}"
    ),
)
DEFAULT_TENSORS = (
    "layers.0.linear_attn.in_proj_qkv,layers.3.self_attn.q_proj,"
    "layers.0.linear_attn.out_proj,layers.0.mlp.shared_expert.gate_proj"
)

_open_shards = {}


def _get(index, key):
    shard = index["weight_map"][key]
    path = os.path.join(SNAPSHOT, shard)
    if path not in _open_shards:
        _open_shards[path] = safe_open(path, framework="pt")
    return _open_shards[path].get_tensor(key)


def load_index():
    with open(os.path.join(SNAPSHOT, "model.safetensors.index.json")) as f:
        return json.load(f)


def resolve_tensor_prefix(requested, index):
    """Resolve a --tensor CLI value (e.g. the spec's literal 'layers.0.self_attn.q_proj') to
    an ACTUAL checkpoint key prefix. Layer 0 is a GDN layer [(0+1)%4 != 0] and structurally
    has no self_attn.* -- if asked for layer-0 self_attn, substitute layer 3 (first
    full-attention layer, i in {3,7,...,39}), logged loudly (never silently), matching the
    same documented substitution used by p10_fp8_dense_bar.py for the same structural reason.
    """
    full = f"model.language_model.{requested}"
    if f"{full}.weight" in index["weight_map"]:
        return full, requested, None
    if "layers.0." in requested and "self_attn" in requested:
        substituted = requested.replace("layers.0.", "layers.3.", 1)
        full2 = f"model.language_model.{substituted}"
        if f"{full2}.weight" in index["weight_map"]:
            note = (f"'{requested}' does not exist (layer 0 is a GDN layer, has no self_attn.*)"
                    f" -- substituted '{substituted}' (first full-attention layer)")
            return full2, substituted, note
    raise KeyError(f"no checkpoint tensor '{full}.weight' and no known substitution applies; "
                   f"check model.safetensors.index.json for the exact key.")


def recompute_ue8m0_scale(weight_fp8, scale_inv):
    """Mirrors _requantize_fp8_for_ue8m0's OWN steps 1-2 EXACTLY (builder.py:498-512,
    verified against the imported source before writing this) to recover the per-block
    power-of-2 scale used to produce `new_fp8`. Deliberately avoids reverse-engineering
    `packed_ue8m0`'s strided-uint32 bit-packing (a separate, more error-prone exercise) --
    recomputing via the same short, already-verified public formula is simpler and equally
    faithful since it is not a reimplementation of the requantization DECISION itself (that
    comes from the real, imported `new_fp8`), only of the scale bookkeeping needed to
    dequantize it for comparison."""
    M, K = weight_fp8.shape
    group_size = 128
    scale_k = K // group_size
    scale_inv_expanded = scale_inv.float().repeat_interleave(
        group_size, dim=0)[:M].repeat_interleave(group_size, dim=1)[:, :K]
    w_before = weight_fp8.float() * scale_inv_expanded
    w_blocks = w_before.reshape(M, scale_k, group_size)
    block_amax = w_blocks.abs().amax(dim=2).clamp(min=1e-12)
    raw_scale = block_amax / 448.0
    ue8m0_exp = torch.ceil(torch.log2(raw_scale.clamp(min=1e-30)))
    new_scale = torch.pow(2.0, ue8m0_exp)
    return new_scale, w_before


def dequant_after(new_fp8, new_scale, group_size=128):
    M, scale_k = new_scale.shape
    K = new_fp8.shape[1]
    s_exp = new_scale.unsqueeze(2).expand(M, scale_k, group_size).reshape(M, scale_k * group_size)[:, :K]
    return new_fp8.float() * s_exp


def analyze_tensor(requested, index):
    from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder

    full_key, resolved, note = resolve_tensor_prefix(requested, index)
    weight_fp8 = _get(index, f"{full_key}.weight")
    scale_inv = _get(index, f"{full_key}.weight_scale_inv")
    assert scale_inv.dtype == torch.bfloat16, f"expected bf16 checkpoint scale, got {scale_inv.dtype}"

    new_fp8, packed_ue8m0 = DeepSeekV3Builder._requantize_fp8_for_ue8m0(weight_fp8, scale_inv)
    new_scale, w_before = recompute_ue8m0_scale(weight_fp8, scale_inv)
    w_after = dequant_after(new_fp8, new_scale)

    diff = w_after - w_before
    n_pos = int((diff > 0).sum().item())
    n_neg = int((diff < 0).sum().item())
    n_zero = int((diff == 0).sum().item())
    frob_rel = diff.norm().item() / max(w_before.norm().item(), 1e-12)

    # Is the checkpoint's OWN scale already an exact power of two? (directly tests the
    # docstring's premise: "Checkpoint float32 scales are NOT powers of 2")
    log2_scale = torch.log2(scale_inv.float().clamp(min=1e-30))
    frac_scale_is_pow2 = (log2_scale.round() - log2_scale).abs().lt(1e-6).float().mean().item()

    return {
        "requested_tensor": requested, "resolved_tensor": resolved, "substitution_note": note,
        "shape": list(weight_fp8.shape),
        "checkpoint_scale_shape": list(scale_inv.shape),
        "checkpoint_scale_dtype": str(scale_inv.dtype),
        "checkpoint_scale_frac_already_pow2": frac_scale_is_pow2,
        "max_abs_delta": diff.abs().max().item(),
        "mean_abs_delta": diff.abs().mean().item(),
        "mean_signed_delta": diff.mean().item(),
        "std_signed_delta": diff.std().item(),
        "frob_rel_delta": frob_rel,
        "n_positive_deltas": n_pos, "n_negative_deltas": n_neg, "n_zero_deltas": n_zero,
        "n_elements": diff.numel(),
        "frac_nonzero": (n_pos + n_neg) / diff.numel(),
        "nonzero_in_both_directions": (n_pos > 0 and n_neg > 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor", default=DEFAULT_TENSORS,
                     help="comma-separated checkpoint tensor prefixes relative to "
                          "model.language_model. (e.g. the spec's 'layers.0.self_attn.q_proj')")
    ap.add_argument("--out", default=os.path.expanduser("~/mpk-qwen35/probes/fp8_out/p7_ue8m0_delta.json"))
    args = ap.parse_args()

    print(f"torch {torch.__version__}", flush=True)
    index = load_index()
    results = []
    for t in args.tensor.split(","):
        t = t.strip()
        r = analyze_tensor(t, index)
        results.append(r)
        print(f"[{t}] resolved={r['resolved_tensor']} shape={r['shape']} "
              f"max_abs_delta={r['max_abs_delta']:.6f} frob_rel={r['frob_rel_delta']:.4e} "
              f"pos={r['n_positive_deltas']} neg={r['n_negative_deltas']} zero={r['n_zero_deltas']} "
              f"ckpt_scale_frac_pow2={r['checkpoint_scale_frac_already_pow2']:.4f} "
              f"both_directions={r['nonzero_in_both_directions']}"
              + (f"  NOTE: {r['substitution_note']}" if r["substitution_note"] else ""), flush=True)

    verdict = {
        "expected_outcome": "nonzero deltas in BOTH directions for every tensor (checkpoint "
                             "scales are not exact powers of two) -- exactly-zero deltas would "
                             "falsify v1-architecture.md SS6.2's rejection of MPK's existing "
                             "UE8M0-requantizing dense fp8 path.",
        "all_tensors_show_bidirectional_nonzero_delta": all(r["nonzero_in_both_directions"] for r in results),
        "tensors": results,
        "method": "Calls the REAL mirage.mpk.models.deepseek_v3.builder.DeepSeekV3Builder."
                   "_requantize_fp8_for_ue8m0 (imported, not reimplemented) on real checkpoint "
                   "tensors; recomputes the per-block UE8M0 scale via the same public formula "
                   "(builder.py steps 1-2, verified against source) to dequantize the 'after' "
                   "side for comparison, rather than reverse-engineering packed_ue8m0's strided "
                   "uint32 bit-packing.",
        "torch_version": torch.__version__,
        "checkpoint": "Qwen/Qwen3.5-35B-A3B-FP8", "checkpoint_revision": CKPT_REVISION,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nWROTE {args.out}")
    print(f"all_tensors_show_bidirectional_nonzero_delta={verdict['all_tensors_show_bidirectional_nonzero_delta']}")


if __name__ == "__main__":
    main()
