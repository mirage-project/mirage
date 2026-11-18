"""One GPU, one TP4 rank (no all-reduce): the layer with mHC's coefficients in the kernel against the plain layer. Rounds of plain / stats
in one process; y must not change; pre, post, comb against the official formulas in fp32; the span of each, serving build, L2 emptied.
python stats_check.py --kind attn|moe --layer 3 --checkpoint DIR --out DIR [--flag=...] [--rounds 3]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from deepseek_v4_1_csa2 import demo as attn_demo          # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights    # noqa: E402
from deepseek_v4_1_moe import demo as moe_demo            # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights      # noqa: E402
from dsv41_static_layer import mhc_tensors                # noqa: E402
from mirage.mpk.static_harness import Harness             # noqa: E402

T, H, HC, EPS, HC_EPS, ITERS = 8, 5120, 4, 1e-20, 1e-6, 20
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--kind", choices=("attn", "moe"), default="attn")
ap.add_argument("--out", default="/tmp/stats_check")
ap.add_argument("--flag", action="append", default=[])
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
L, attention = a.layer, a.kind == "attn"
dev = torch.device("cuda", 0)
read = attn_weights.checkpoint_reader(a.checkpoint)
cfg = attn_weights.model_config(a.checkpoint)
flags = ["-DSTATIC_RESET_IN_KERNEL"] + a.flag
relrms = lambda x, y: ((x.double() - y.double()).square().mean().sqrt() / y.double().square().mean().sqrt()).item()

if attention:
    variant, ratio = attn_weights.variant_of(cfg, L), cfg["compress_ratios"][L]
    csrc, top = L == cfg["candidate_source_layer"], 4096
    rows = (top + 2 + 255) // 256 * 256
    cap = max(((top + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
    w = {**attn_weights.layer_tensors(read, cfg, L, 4, 0, rows, dev), **attn_demo.state_tensors(variant, ratio, csrc, cap, dev)}

    def build(tag):
        attn_demo.random_inputs(w, variant, ratio, top)
        return attn_demo.build([w], variant, ratio, csrc, 16, os.path.join(a.out, tag), extra_flags=flags)
else:
    w = moe_weights.layer_tensors(read, L, 4, 0, dev)
    w.update(moe_weights.routed_tensors(read, L, 4, [0], [dev])[0])
    w["x"] = torch.randn(T, H, generator=torch.Generator().manual_seed(1)).bfloat16().to(dev).contiguous()
    w["y"] = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)

    def build(tag):
        return moe_demo.build([w], os.path.join(a.out, tag), extra_flags=flags)


def timed(h):
    spans = []
    for _ in range(50):
        h.launch(512 << 20, timeout_s=60)
        spans.append(h.span_us())
    return sorted(spans)[25]


kind = "attn" if attention else "ffn"
R = torch.randn(T, HC * H, generator=torch.Generator().manual_seed(7)).bfloat16()
extra = mhc_tensors(read, L, kind, None, dev, stats=True)
extra["streams"].copy_(R.to(dev))
plain = {}
for r in range(a.rounds):
    for name in ("plain", "stats"):
        for k in extra:
            w.pop(k, None)
        if name == "stats":
            w.update(extra)
        sk, bufs, info = build(f"{name}{r}")
        h = Harness(sk)
        h.launch(timeout_s=60)
        y = w["y"].clone()
        print(f"round {r} {name}: span {timed(h):.1f} us, y bit-identical to plain: "
              f"{torch.equal(y.view(torch.int16), plain[r].view(torch.int16)) if name == 'stats' else '-'}", flush=True)
        if name == "plain":
            plain[r] = y
        sk.finalize()

p = f"layers.{L}."
fn, base, scale = (read(p + f"hc_{kind}_{n}").float() for n in ("fn", "base", "scale"))
x = R.float()
mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
comb = (mixes[:, 2 * HC:].view(T, HC, HC) * scale[2] + base[2 * HC:].view(HC, HC)).softmax(-1) + HC_EPS
comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
for _ in range(ITERS - 1):
    comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
print(f"pre relRMS {relrms(extra['h_pre'].cpu(), pre):.2e}, post {relrms(extra['h_post'].cpu(), post):.2e}, "
      f"comb {relrms(extra['h_comb'].cpu().view(T, HC, HC), comb):.2e}", flush=True)
