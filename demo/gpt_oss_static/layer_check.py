"""Per-layer check of the GPT-OSS static layers against an exact reference, as the GLM-5.3 / Kimi K3 static layers are
checked (glm53_flash/sglang_moe/moe_static_hook.py: two quantized implementations differ by a few % on their own, so an
exact reference is the yardstick): the model runs --batch requests to position --pos (the prompts teacher-forced:
GSM8K's 5-shot prompts, few_shot_gsm8k's text), then each layer of that step alone; for each layer, its input, its K / V
history (the cache, positions < pos) and its outputs, against Hugging Face's GptOss modules (attention with the sinks and
the sliding window, the norms, the router) and the experts as GptOssExperts computes them (the checkpoint's MXFP4 weights
dequantized as transformers' convert_moe_packed_tensors: exact in bf16 and fp64), run on the same input and history:
  exact   all of it in fp64 (RoPE's angles fp32, as GptOssRotaryEmbedding's)
  hf      all of it in bf16, as Hugging Face's eager model (its softmax in bf16 too: vLLM / SGLang keep it in fp32)
Printed per layer: the attention's share of the residual (x1 - x) and the MoE's (out - x1), static and hf, each as relRMS
against exact (and the worst token's), and how many tokens hf routes to exact's experts.

python layer_check.py CKPT --data-path test.jsonl [--batch 32] [--pos 700] [--plans DIR]
"""
import argparse, json, os


def rel(a, b):
    """relRMS of a against b over all rows, and the worst row's."""
    d = (a - b).double()
    b = b.double()
    rows = d.pow(2).mean(-1).sqrt() / b.pow(2).mean(-1).sqrt().clamp_min(1e-30)
    return (d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item(), rows.max().item()


class History:
    """The K / V cache of one layer for Hugging Face's attention: the given history, then the step's own."""

    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, cache_kwargs=None):
        import torch
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pos", type=int, default=None, help="the step's position (default: the shortest prompt's last)")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--plans", default=None, help="the plan files' directory (default: round robin)")
    ap.add_argument("--out", default="layer_check")
    args = ap.parse_args()
    import torch
    import gpt_oss_static as g
    from gsm8k import one_example
    from transformers import AutoTokenizer
    from transformers.integrations.mxfp4 import FP4_VALUES

    def mxfp4_cpu(blocks, scales):   # transformers' convert_moe_packed_tensors for one expert, on the CPU: [K, N] bf16
        lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16)
        out = torch.stack((lut[(blocks & 15).long()], lut[(blocks >> 4).long()]), -1).flatten(-2)   # [N, G, 32]
        out = torch.ldexp(out, scales.to(torch.int32)[..., None] - 127)
        return out.flatten(-2).t().contiguous()
    from transformers.models.gpt_oss.modeling_gpt_oss import (GptOssAttention, GptOssRMSNorm, GptOssRotaryEmbedding,
                                                              GptOssTopKRouter)
    from mxfp4 import Mxfp4Checkpoint
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    lines = [json.loads(l) for l in open(args.data_path)]
    shots = "".join(one_example(lines, i, True) + "\n\n" for i in range(5))
    R = args.batch
    ids = [tok(shots + one_example(lines, i, False)).input_ids for i in range(R)]
    S = args.pos if args.pos is not None else min(len(i) for i in ids) - 1
    assert S < min(len(i) for i in ids) and S < args.max_seq
    g.T = next(b for b in g.BUCKETS if b >= R)
    m = g.Model(args.ckpt, args.max_seq, R)
    prompt_t = torch.tensor([i[:S + 1] for i in ids], device="cuda")
    gen = torch.zeros(R, 1, dtype=torch.long, device="cuda")
    sliding = [i for i in range(m.L) if m.types[i] == "sliding_attention"]
    full = [i for i in range(m.L) if m.types[i] != "sliding_attention"]
    kinds = {"sliding": (lambda mpk, t: g.build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0])),
             "full": (lambda mpk, t: g.build_layer(m, mpk, t, 0), m.layer_tensors(full[0])),
             "head": (lambda mpk, t: g.build_head(m, mpk, t), m.head_tensors(prompt_t, gen))}
    k_sl, k_fu, k_hd = (g.compile_kernel(k, b, t, args.out, args.plans and g.plan_path(m, args.plans, k))
                        for k, (b, t) in kinds.items())
    kernel_of, inst = {}, {}
    for i in range(m.L):
        kernel_of[i] = k_sl if i in sliding else k_fu
        inst[i] = kernel_of[i].add_instance(m.layer_tensors(i))
    k_hd.add_instance(m.head_tensors(prompt_t, gen))
    m.set_position(0)
    m.x_embed[:R].copy_(m.embed[prompt_t[:, 0]])
    for _ in range(S):          # positions 0 .. S - 1; the head leaves position S's token row and page table
        for i in range(m.L):
            kernel_of[i].launch_instances(inst[i], 1)
        k_hd.launch_instances(0, 1)
    torch.cuda.synchronize()

    cfg = m.cfg
    cfg._attn_implementation = "eager"
    sd = Mxfp4Checkpoint(args.ckpt)
    rot = GptOssRotaryEmbedding(cfg).cuda()
    pos = torch.full((R, 1), S, device="cuda")
    MP = args.max_seq // g.PAGE
    print(f"position {S}, {R} requests; relRMS vs exact fp64 (all rows / worst row)", flush=True)
    print(f"{'layer':>9} | {'attn static':>19} {'attn hf bf16':>19} | {'moe static':>19} {'moe hf bf16':>19} | hf experts = exact")
    for i in range(m.L):
        x_in = (m.x_embed if i == 0 else m.x[i % 2])[:R].clone()
        kernel_of[i].launch_instances(inst[i], 1)
        torch.cuda.synchronize()
        x1_s, out_s = m.shared["x1"][:R].clone(), m.x[(i + 1) % 2][:R].clone()
        p = f"model.layers.{i}."
        mods = {"attn": GptOssAttention(cfg, i), "ln1": GptOssRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps),
                "ln2": GptOssRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), "router": GptOssTopKRouter(cfg)}
        for name, prefix in (("attn", "self_attn."), ("ln1", "input_layernorm."), ("ln2", "post_attention_layernorm."),
                             ("router", "mlp.router.")):
            mods[name].load_state_dict({k: sd[p + prefix + k] for k in mods[name].state_dict()})
        w13b, w13s = sd[p + "mlp.experts.gate_up_proj_blocks"], sd[p + "mlp.experts.gate_up_proj_scales"]
        w2b, w2s = sd[p + "mlp.experts.down_proj_blocks"], sd[p + "mlp.experts.down_proj_scales"]
        b13, b2 = sd[p + "mlp.experts.gate_up_proj_bias"].cuda(), sd[p + "mlp.experts.down_proj_bias"].cuda()
        kc = m.layers[i]["k_cache"].view(R, MP * g.PAGE, m.KVH, m.D)[:, :S].permute(0, 2, 1, 3)
        vc = m.layers[i]["v_cache"].view(R, MP * g.PAGE, m.KVH, m.D)[:, :S].permute(0, 2, 1, 3)
        res = {}
        for dt in (torch.float64, torch.bfloat16):
            md = {k: v.to(device="cuda", dtype=dt) for k, v in mods.items()}
            x = x_in.to(dt)[:, None, :]
            mask = torch.zeros(R, 1, 1, S + 1, dtype=dt, device="cuda")
            if i in sliding:   # Hugging Face's sliding mask: key j visible when j > S - window
                mask[..., :S + 1 - m.window] = torch.finfo(dt).min
            cos, sin = rot(x.float(), pos)
            a, _ = md["attn"](hidden_states=md["ln1"](x), position_embeddings=(cos.to(dt), sin.to(dt)),
                              attention_mask=mask, past_key_values=History(kc.to(dt), vc.to(dt)))
            x1 = x + a
            h = md["ln2"](x1)[:, 0]
            scores, idx = md["router"](h)
            moe = torch.zeros_like(h)
            for e in idx.unique().tolist():
                # [H, 2I] and [I, H], gate / up interleaved (GptOssExperts'); MXFP4 values are exact in bf16.
                # Converted on the CPU, and fp64 (exact anywhere) computed there: 120B's experts in fp64 and the
                # conversion's temporaries do not fit beside its static model at 32 requests
                dev = "cpu" if dt == torch.float64 else "cuda"
                w13, w2e = (mxfp4_cpu(b[e].cpu(), sc[e].cpu()).to(dev, dt) for b, sc in ((w13b, w13s), (w2b, w2s)))
                rows = (idx == e).any(-1).nonzero()[:, 0]
                gate_up = h[rows].to(dev) @ w13 + b13[e].to(dev, dt)
                gate = gate_up[..., ::2].clamp(max=m.limit)
                up = gate_up[..., 1::2].clamp(min=-m.limit, max=m.limit)
                o = (((up + 1) * (gate * torch.sigmoid(gate * 1.702))) @ w2e + b2[e].to(dev, dt)).to("cuda")
                moe.index_add_(0, rows, o * scores[rows, e, None])
            res[dt] = (x1[:, 0], x1[:, 0] + moe, idx.sort(-1).values)
        ex, hf = res[torch.float64], res[torch.bfloat16]
        x = x_in.double()
        a_s, a_h = rel(x1_s.double() - x, ex[0] - x), rel(hf[0].double() - x, ex[0] - x)
        m_s, m_h = rel(out_s.double() - x1_s.double(), ex[1] - ex[0]), rel(hf[1].double() - hf[0].double(), ex[1] - ex[0])
        same = (hf[2] == ex[2]).all(-1).sum().item()
        fmt = lambda r: f"{r[0]:.2e} / {r[1]:.2e}"
        print(f"{i:>3} {'slide' if i in sliding else 'full':>5} | {fmt(a_s):>19} {fmt(a_h):>19} | {fmt(m_s):>19} "
              f"{fmt(m_h):>19} | hf {same}/{R}", flush=True)
    for k in (k_sl, k_fu, k_hd):
        k.finalize()


if __name__ == "__main__":
    main()
