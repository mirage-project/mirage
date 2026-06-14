"""DFlash (Kimi K2.6) PyTorch reference oracle.

Loads the HF reference draft model (`dflash.py: DFlashDraftModel`) with the real
checkpoint weights (optionally sliced to a single layer), runs a controlled
forward, and dumps every op/layer tensor to disk for MPK kernel/layer alignment.

Run in the `mirage00` conda env:
    conda run --no-capture-output -n mirage00 python demo/qwen3/dflash_correctness/ref_dump.py \
        --num-layers 1 --bs 1 --ctx-len 16

Outputs: <out>/meta.json + <out>/*.pt  (one tensor per captured op).
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import torch

CKPT = "/raid/catalyst/models/Kimi-K2.6-DFlash-tmp"


def load_ref_module(ckpt):
    """Import dflash.py from the checkpoint dir as a module."""
    path = os.path.join(ckpt, "dflash.py")
    spec = importlib.util.spec_from_file_location("dflash_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dflash_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_config(ckpt, num_layers):
    """Build a Qwen3Config matching the checkpoint, optionally truncated layers."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    cfg_dict = json.load(open(os.path.join(ckpt, "config.json")))
    if num_layers is not None:
        cfg_dict["num_hidden_layers"] = num_layers
        cfg_dict["layer_types"] = cfg_dict["layer_types"][:num_layers]
    cfg = Qwen3Config(**{k: v for k, v in cfg_dict.items()
                         if k not in ("architectures",)})
    # carry the dflash-specific sub-config + fields HF Qwen3Config drops
    cfg.dflash_config = cfg_dict["dflash_config"]
    cfg.block_size = cfg_dict["block_size"]
    cfg.layer_types = cfg_dict["layer_types"]
    cfg.sliding_window = cfg_dict["sliding_window"]
    cfg.head_dim = cfg_dict["head_dim"]
    return cfg


def load_state_dict(ckpt, num_layers):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    shard_keys = {}
    for k, shard in idx.items():
        shard_keys.setdefault(shard, []).append(k)
    for shard, keys in shard_keys.items():
        with safe_open(os.path.join(ckpt, shard), framework="pt") as f:
            for k in keys:
                if k.startswith("layers."):
                    li = int(k.split(".")[1])
                    if num_layers is not None and li >= num_layers:
                        continue
                sd[k] = f.get_tensor(k)
    return sd


def build_noncausal_sliding_mask(ctx_len, q_len, sliding_window, dtype, device):
    """4D additive mask [1,1,q_len, ctx_len+q_len] for one non-causal block.

    The B block queries attend to [context(ctx_len) + block(q_len)] with NO causal
    constraint inside the visible window. Sliding window (if not None) limits each
    query to keys within `sliding_window` of the query's absolute position.

    Absolute positions: context tokens 0..ctx_len-1, block tokens ctx_len..ctx_len+q_len-1.
    """
    total = ctx_len + q_len
    neg = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, q_len, total, dtype=dtype, device=device)
    if sliding_window is not None:
        q_pos = torch.arange(q_len, device=device) + ctx_len      # [q_len]
        k_pos = torch.arange(total, device=device)                # [total]
        # key visible iff 0 <= q_pos - k_pos < sliding_window  (causal-style window),
        # but DFlash is non-causal: allow future keys within the window too.
        dist = (q_pos[:, None] - k_pos[None, :]).abs()
        blocked = dist >= sliding_window
        mask[0, 0][blocked] = neg
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ctx-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="eager")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dumps"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    dt = torch.bfloat16

    ref = load_ref_module(args.ckpt)
    cfg = build_config(args.ckpt, args.num_layers)
    cfg._attn_implementation = args.attn_impl

    model = ref.DFlashDraftModel(cfg).to(dev, dt).eval()
    sd = load_state_dict(args.ckpt, args.num_layers)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    H = cfg.hidden_size
    K = len(cfg.dflash_config["target_layer_ids"])
    B = cfg.block_size
    ctx_len, bs = args.ctx_len, args.bs

    # ---- controlled inputs (seeded) ----
    noise_embedding = torch.randn(bs, B, H, device=dev, dtype=dt) * 0.1
    target_hidden = torch.randn(bs, ctx_len, K * H, device=dev, dtype=dt) * 0.1
    position_ids = torch.arange(ctx_len + B, device=dev).unsqueeze(0).expand(bs, -1).contiguous()
    sw = cfg.sliding_window if cfg.layer_types[0] == "sliding_attention" else None
    attention_mask = build_noncausal_sliding_mask(ctx_len, B, sw, dt, dev)

    dumps = {}

    def save(name, t):
        dumps[name] = t.detach().to(torch.float32).cpu()

    # ---- hooks on every submodule ----
    handles = []
    for mod_name, module in model.named_modules():
        if mod_name == "":
            continue
        def mk(nm):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(o):
                    save(f"out::{nm}", o)
            return hook
        handles.append(module.register_forward_hook(mk(mod_name)))

    with torch.no_grad():
        out = model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=False,
        )
    save("final_hidden", out)
    save("in::noise_embedding", noise_embedding)
    save("in::target_hidden", target_hidden)
    save("in::position_ids", position_ids.to(torch.float32))
    save("in::attention_mask", attention_mask)

    for h in handles:
        h.remove()

    for name, t in dumps.items():
        fn = name.replace("::", "__").replace(".", "_") + ".pt"
        torch.save(t, os.path.join(args.out, fn))

    meta = dict(num_layers=cfg.num_hidden_layers, bs=bs, ctx_len=ctx_len, B=B,
                H=H, K=K, n_q=cfg.num_attention_heads, n_kv=cfg.num_key_value_heads,
                head_dim=cfg.head_dim, I=cfg.intermediate_size,
                rms_norm_eps=cfg.rms_norm_eps, sliding_window=cfg.sliding_window,
                layer_types=cfg.layer_types, seed=args.seed, attn_impl=args.attn_impl,
                names=sorted(dumps.keys()))
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"\nDumped {len(dumps)} tensors to {args.out}")
    print("final_hidden:", tuple(out.shape), out.dtype,
          "mean", out.float().mean().item(), "std", out.float().std().item())


if __name__ == "__main__":
    main()
