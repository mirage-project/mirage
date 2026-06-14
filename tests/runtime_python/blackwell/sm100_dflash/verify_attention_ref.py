"""Verify the pure-torch dflash_attention reference reproduces the HF dump.

This makes dflash_attention() a trustworthy oracle for the MPK K3 attention kernel,
and validates the YaRN cos/sin source (Qwen3RotaryEmbedding from the config).

Run: CUDA_VISIBLE_DEVICES=3 python verify_attention_ref.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import load_weight, dflash_attention, CKPT  # noqa: E402

DUMPS = os.path.join(os.path.dirname(__file__),
                     "../../../../demo/qwen3/dflash_correctness/dumps")


def load_dump(name):
    fn = name.replace("::", "__").replace(".", "_") + ".pt"
    return torch.load(os.path.join(DUMPS, fn))


def build_yarn_cos_sin(ckpt, positions, device, dtype):
    """cos/sin via the reference Qwen3RotaryEmbedding (carries YaRN scaling)."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RotaryEmbedding
    cfg_dict = json.load(open(os.path.join(ckpt, "config.json")))
    cfg = Qwen3Config(**{k: v for k, v in cfg_dict.items() if k != "architectures"})
    cfg.head_dim = cfg_dict["head_dim"]
    rope = Qwen3RotaryEmbedding(cfg).to(device)
    dummy = torch.zeros(1, 1, device=device, dtype=dtype)
    pos = positions.unsqueeze(0).to(device)
    cos, sin = rope(dummy, pos)               # [1,T,d]
    return cos[0].to(dtype), sin[0].to(dtype)


def main():
    device, dtype = "cuda", torch.bfloat16
    meta = json.load(open(os.path.join(DUMPS, "meta.json")))
    n_q, n_kv, d = meta["n_q"], meta["n_kv"], meta["head_dim"]
    sw = meta["sliding_window"] if meta["layer_types"][0] == "sliding_attention" else None

    ctx = load_dump("out::hidden_norm").to(device, dtype).squeeze(0)          # [ctx_len,H]
    h = load_dump("out::layers.0.input_layernorm").to(device, dtype).squeeze(0)  # [B,H]
    ref_attn = load_dump("out::layers.0.self_attn").to(device, torch.float32).squeeze(0)
    ctx_len, B = ctx.shape[0], h.shape[0]
    T = ctx_len + B

    positions = torch.arange(T)
    cos, sin = build_yarn_cos_sin(CKPT, positions, device, dtype)

    q_w = load_weight("layers.0.self_attn.q_proj.weight")
    k_w = load_weight("layers.0.self_attn.k_proj.weight")
    v_w = load_weight("layers.0.self_attn.v_proj.weight")
    o_w = load_weight("layers.0.self_attn.o_proj.weight")
    qn = load_weight("layers.0.self_attn.q_norm.weight")
    kn = load_weight("layers.0.self_attn.k_norm.weight")

    out = dflash_attention(ctx, h, q_w, k_w, v_w, o_w, qn, kn, cos, sin,
                           sw, n_q, n_kv, d).to(torch.float32)
    err = (out - ref_attn).abs().max().item()
    rel = err / ref_attn.abs().max().item()
    print(f"ctx_len={ctx_len} B={B} sliding_window={sw}")
    print(f"attn-ref vs dump: maxerr {err:.4f}  relmax {rel:.4f}")
    ok = rel < 0.02  # relative tol (bf16) — tensor magnitude can be O(100s)
    print("PASSED" if ok else "FAILED")
    assert ok


if __name__ == "__main__":
    main()
