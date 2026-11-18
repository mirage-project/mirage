"""DeepSeek-V4.1's mHC in the static layers (StaticMegakernel.dsv41_mhc_nodes), on 4 GPUs, against the official formulas in fp32
on the CPU: first the plain layer on random inputs gives y; then the same layer with the mHC nodes, on the same inputs and random
streams: pre, post, comb, R' and x (the next sublayer's input) relRMS, and the same bits on every GPU.

python mhc_check.py --layer 3 --kind attn|moe [--no-norm] --checkpoint DIR --out DIR
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from deepseek_v4_1_csa2 import demo as attn_demo       # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights  # noqa: E402
from deepseek_v4_1_moe import demo as moe_demo         # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights   # noqa: E402
from mirage.mpk.static_harness import Harness          # noqa: E402

T, H, HC, EPS, HC_EPS, ITERS = 8, 5120, 4, 1e-20, 1e-6, 20
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--kind", choices=("attn", "moe"), default="attn")
ap.add_argument("--no-norm", action="store_true", help="the post only")
ap.add_argument("--out", default="/work/mhc")
ap.add_argument("--flag", action="append", default=[])
ap.add_argument("--timeline", action="store_true", help="the timing build: when each node's tasks ran")
ap.add_argument("--gpus", type=int, default=4, help="the GPUs = the TP size (2: a local run on two GPUs)")
ap.add_argument("--push", default=None, help="instead of mHC: PUSH or ADD (STATIC_WO_B_<it>), y against the plain layer's")
a = ap.parse_args()
L, attention = a.layer, a.kind == "attn"
NG = a.gpus
# MHC_ONE_GPU=1 (code generation only): every rank's tensors on GPU 0; the build writes layer.cu, then the load fails
devs = [torch.device("cuda", 0 if os.environ.get("MHC_ONE_GPU") else g) for g in range(NG)]
read = attn_weights.checkpoint_reader(a.checkpoint)
cfg = attn_weights.model_config(a.checkpoint)


def relrms(x, y):
    return ((x.double() - y.double()).square().mean().sqrt() / y.double().square().mean().sqrt()).item()


if attention:
    variant, ratio = attn_weights.variant_of(cfg, L), cfg["compress_ratios"][L]
    csrc, heads = L == cfg["candidate_source_layer"], 64 // NG
    top = 4096
    rows = (top + 2 + 255) // 256 * 256
    cap = max(((top + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
    shards = [attn_weights.layer_tensors(read, cfg, L, NG, r, rows, d) for r, d in zip(range(NG), devs)]
    ws = [{**s, **attn_demo.state_tensors(variant, ratio, csrc, cap, d)} for s, d in zip(shards, devs)]

    def build(tag, flags):
        for wg in ws:
            attn_demo.random_inputs(wg, variant, ratio, top)
        return attn_demo.build(ws, variant, ratio, csrc, heads, os.path.join(a.out, tag), extra_flags=flags,
                               profile=a.timeline)
else:
    ws = [moe_weights.layer_tensors(read, L, NG, r, d) for r, d in zip(range(NG), devs)]
    for wg, rt in zip(ws, moe_weights.routed_tensors(read, L, NG, list(range(NG)), devs)):
        wg.update(rt)
    x = torch.randn(T, H, generator=torch.Generator().manual_seed(1)).bfloat16()
    for wg, d in zip(ws, devs):
        wg["x"] = x.to(d).contiguous()
        wg["y"] = torch.zeros(T, H, dtype=torch.bfloat16, device=d)

    def build(tag, flags):
        return moe_demo.build(ws, os.path.join(a.out, tag), extra_flags=flags, profile=a.timeline)

def timed(h, info, tag, between=None):
    """30 launches, L2 emptied: the median span; --timeline: per node, on GPU 0, first start and last end after the start barrier,
    the median task, and the two stage stamps' medians."""
    spans, launches = [], []
    for _ in range(int(os.environ.get("MHC_LAUNCHES", "30"))):   # (a profiler run: fewer)
        if between is not None:   # another layer's kernel runs on every SM before each timed launch (as in a server)
            between()
        h.launch(512 << 20, timeout_s=60)
        spans.append(h.span_us())
        if a.timeline:
            launches.append((h.task_times(), h.timing()))
    spans.sort()
    print(f"{tag}: median span {spans[len(spans) // 2]:.1f} us", flush=True)
    if not a.timeline:
        return
    import json
    import statistics
    from mirage.mpk import compiler
    from mirage.mpk.cost_records import stamps_of_gpu
    schedule = json.load(open(info["paths"][0]))
    mpk = info["mpk"]
    nodes = {n.graph_idx: n for n in compiler.graph_nodes(mpk)}
    tasks = schedule["all_tasks"]
    count = {}
    rows = {}
    for times, timing in launches:
        st = stamps_of_gpu(schedule, times[0])
        bar = timing[0][0]
        per = {}
        for t, v in st.items():
            n = tasks[t]["node"]
            per.setdefault(n, []).append((v["start"] - bar, v["end"] - bar))
        for sm, q in enumerate(schedule["worker_task_queues"]):
            for i, t in enumerate(q):
                n = tasks[t]["node"]
                for k in (2, 3):
                    if len(times[0][sm][i]) > k and times[0][sm][i][k]:
                        rows.setdefault(("stamp", n, k), []).append((times[0][sm][i][k] - bar) / 1e3)
        for n, ts in per.items():
            count[n] = len(ts)
            r = rows.setdefault(n, ([], [], []))
            r[0].append(min(x[0] for x in ts) / 1e3)
            r[1].append(max(x[1] for x in ts) / 1e3)
            r[2].append(statistics.median(x[1] - x[0] for x in ts) / 1e3)
    if tag.startswith("mhc"):   # per GPU, the last launch: sum_store's end and the stamps of hc_post_combine, sorted
        tt, tm = launches[-1]
        for r in range(NG):
            b0 = tm[r][0]
            sch = json.load(open(info["paths"][r]))
            tk = sch["all_tasks"]
            ss_end, rows_ = [], []
            for sm, q in enumerate(sch["worker_task_queues"]):
                for i, t in enumerate(q):
                    nm = nodes[tk[t]["node"]].name
                    e = [(x - b0) / 1e3 if x else float("nan") for x in tt[r][sm][i]]
                    if nm == "sum_store":
                        ss_end.append(e[1])
                    if nm == "hc_post_combine":
                        rows_.append((e[2], e[3], e[0], e[1], sm, tk[t].get("index", tk[t].get("task"))))
            rows_.sort()
            print(f"  GPU {r}: sum_store ends {min(ss_end):.1f} .. {max(ss_end):.1f}; hc_post_combine stamp0 first {rows_[0][0]:.1f} median {rows_[len(rows_)//2][0]:.1f} last {rows_[-1][0]:.1f}; "
                  f"late tasks (s0, s1, start, end, sm, idx): " + " ".join(f"({a:.1f},{b:.1f},{c:.1f},{d:.1f},{e},{f})" for a, b, c, d, e, f in rows_[-6:]), flush=True)
    if False:
        tt, tm = launches[-1]
        b0 = tm[0][0]
        for sm, q in enumerate(schedule["worker_task_queues"]):
            names = [nodes[tasks[t]["node"]].name for t in q]
            if "hc_post_combine" in names:
                st_ = [" ".join(f"{(x - b0) / 1e3:.1f}" if x else "-" for x in tt[0][sm][i]) for i in range(len(q))]
                print(f"  SM {sm}: " + " | ".join(f"{compiler.tname(mpk, nodes[tasks[t]['node']].outputs[0])} [{st_[i]}]" for i, t in enumerate(q)), flush=True)
    keys = [k for k in rows if not isinstance(k, tuple)]
    for n in sorted(keys, key=lambda k: statistics.median(rows[k][1])):
        r = rows[n]
        st = "".join(f"  stamp{k - 2} {statistics.median(rows[('stamp', n, k)]):6.2f} (max {max(rows[('stamp', n, k)]):6.2f})"
                     for k in (2, 3) if ("stamp", n, k) in rows)
        print(f"  {nodes[n].name:<18} {compiler.tname(mpk, nodes[n].outputs[0]):<13} tasks {count[n]:>4}  first start "
              f"{statistics.median(r[0]):6.2f}  last end {statistics.median(r[1]):6.2f}  median task {statistics.median(r[2]):5.2f}{st}", flush=True)


GEN_ONLY = bool(os.environ.get("MHC_ONE_GPU"))
# measurement only. MHC_PREALLOC=1: tensors of the mHC nodes' sizes are allocated on every GPU before the plain layer is built
# (does the plain layer's speed depend on where its buffers land?). MHC_TWICE=1: the plain layer is built and timed a second
# time (does a layer built second in the process run differently?).
_keep = []
if os.environ.get("MHC_PREALLOC"):
    for d in devs:
        _keep.append([torch.zeros(128, HC * H, dtype=torch.bfloat16, device=d), torch.zeros(T, HC * H, dtype=torch.bfloat16, device=d),
                      torch.zeros(T, HC * H, dtype=torch.bfloat16, device=d), torch.zeros(24, HC * H, device=d),
                      torch.zeros(27, device=d), torch.zeros(T, HC, device=d), torch.zeros(T, H, dtype=torch.bfloat16, device=d),
                      torch.zeros(H, dtype=torch.bfloat16, device=d)])
if GEN_ONLY:   # code generation only (MHC_ONE_GPU=plain | mhc): that layer's layer.cu; the generated host code then ends the
    # process at the load (one GPU)
    if os.environ["MHC_ONE_GPU"] != "mhc":
        build("plain", a.flag)
    y = torch.zeros(T, H, dtype=torch.bfloat16)
else:
    sk, bufs, info = build("plain", a.flag)
    h = Harness(sk)
    h.launch(timeout_s=60)
    y = ws[0]["y"].cpu().clone()
    timed(h, info, "plain")
    INTER = bool(os.environ.get("MHC_INTERLEAVE"))   # measurement only: the plain kernel stays loaded (below)
    if INTER:
        plain_keep = (sk, bufs, info, h)
    else:
        sk.finalize()
    if os.environ.get("MHC_TWICE"):
        sk, bufs, info = build("plain2", a.flag)
        h = Harness(sk)
        h.launch(timeout_s=60)
        timed(h, info, "plain2")
        sk.finalize()
print(f"L{L} {a.kind}: plain layer done, y rms {y.float().square().mean().sqrt():.4f}", flush=True)

if a.push:   # the same layer and inputs, wo_b pushing
    os.environ[f"STATIC_WO_B_{a.push}"] = "1"
    sk, bufs, info = build("push", a.flag)
    for wg in ws:   # (a stale y would pass the comparison below)
        wg["y"].zero_()
    h = Harness(sk)
    h.launch(timeout_s=60)
    yp = [wg["y"].cpu().clone() for wg in ws]
    print(f"push: y relRMS vs plain {relrms(yp[0].float(), y.float()):.3e}, equal bits "
          f"{(yp[0].view(torch.int16) == y.view(torch.int16)).float().mean().item():.4f}, the same on every GPU "
          f"{all(torch.equal(p.view(torch.int16), yp[0].view(torch.int16)) for p in yp[1:])}", flush=True)
    timed(h, info, "push")
    sk.finalize()
    sys.exit(0)

kind = "attn" if attention else "ffn"
p = f"layers.{L}."
fn, base, scale = (read(p + f"hc_{kind}_{n}").float() for n in ("fn", "base", "scale"))
gamma = read((f"layers.{L}.ffn_norm.weight" if attention else f"layers.{L + 1}.attn_norm.weight")).bfloat16()
R = torch.randn(T, HC * H, generator=torch.Generator().manual_seed(7)).bfloat16()
for wg, d in zip(ws, devs):
    hi = fn.bfloat16()
    fn_w = torch.cat([hi, (fn - hi.float()).bfloat16(), torch.zeros(128 - 2 * fn.shape[0], fn.shape[1], dtype=torch.bfloat16)])
    wg["hc_fn_w"] = fn_w.to(d).contiguous()
    wg.update(streams=R.to(d).contiguous(), streams_out=torch.zeros(T, HC * H, dtype=torch.bfloat16, device=d),
              hc_fn=fn.to(d).contiguous(), hc_small=torch.cat([base, scale]).to(d).contiguous(),
              h_pre=torch.zeros(T, HC, device=d), x_out=torch.zeros(T, H, dtype=torch.bfloat16, device=d))
    if not a.no_norm:
        wg["gamma_next"] = gamma.to(d).contiguous()
if os.environ.get("MHC_STATS_ONLY"):   # measurement only: the plain layer + only the coefficient nodes beside it (hc_mix as a
    # gemm_tile, hc_coef; the layer still ends with sum_store's y): its node timeline, then stop
    for wg, d in zip(ws, devs):
        wg.update(h_post=torch.zeros(T, HC, device=d), h_comb=torch.zeros(T, HC * HC, device=d))
    sk, bufs, info = build("stats", a.flag)
    h = Harness(sk)
    h.launch(timeout_s=60)
    timed(h, info, "stats")
    sk.finalize()
    sys.exit(0)
t0 = time.time()
if GEN_ONLY:
    try:
        build("mhc", a.flag)
    except Exception as e:
        print(f"mhc: generated ({type(e).__name__} at the load, as expected)", flush=True)
    sys.exit(0)
sk, bufs, info = build("mhc", a.flag)
h = Harness(sk)
print(f"mhc build {time.time() - t0:.0f} s, grids {info['grids']}", flush=True)
h.launch(timeout_s=60)
print(f"L{L} {a.kind}: span {h.span_us():.1f} us", flush=True)
timed(h, info, "mhc")
if not GEN_ONLY and os.environ.get("MHC_INTERLEAVE"):   # each kernel again, the other one launched before every timed launch
    ph = plain_keep[3]
    timed(ph, plain_keep[2], "plain after mhc", between=lambda: h.launch(512 << 20, timeout_s=60))
    timed(h, info, "mhc after plain", between=lambda: ph.launch(512 << 20, timeout_s=60))
    plain_keep[0].finalize()

# the official formulas (model.py Block.hc_mixes, hc_post, hc_pre; kernel.py hc_split_sinkhorn), fp32
x = R.float()
mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
comb = (mixes[:, 2 * HC:].view(T, HC, HC) * scale[2] + base[2 * HC:].view(HC, HC)).softmax(-1) + HC_EPS
comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
for _ in range(ITERS - 1):
    comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
Rs = R.view(T, HC, H).float()
Rn = (post[:, :, None] * y.float()[:, None] +
      torch.einsum("tij,tic->tjc", comb, Rs)).bfloat16()
s = (pre[:, :, None] * Rn.float()).sum(1)
xn = (s.bfloat16().float() * torch.rsqrt(s.square().mean(-1, keepdim=True) + EPS) * gamma.float()).bfloat16()

coef = [b["hc_coef"].view(torch.float32).view(T, 32).cpu() for b in bufs]
got_pre = ws[0]["h_pre"].cpu()
print(f"pre relRMS {relrms(got_pre, pre):.2e} (coef buffer {relrms(coef[0][:, :HC], pre):.2e}), post {relrms(coef[0][:, HC:2 * HC], post):.2e}, "
      f"comb {relrms(coef[0][:, 2 * HC:2 * HC + HC * HC], comb.view(T, -1)):.2e}")
print(f"R' relRMS {relrms(ws[0]['streams_out'].cpu().view(T, HC, H).float(), Rn.float()):.2e}, equal bits "
      f"{(ws[0]['streams_out'].cpu().view(torch.int16) == Rn.view(T, -1).view(torch.int16)).float().mean().item():.4f}")
if not a.no_norm:
    print(f"x relRMS {relrms(ws[0]['x_out'].cpu().float(), xn.float()):.2e}, equal bits "
          f"{(ws[0]['x_out'].cpu().view(torch.int16) == xn.view(torch.int16)).float().mean().item():.4f}")
same = [torch.equal(wg["streams_out"].cpu().view(torch.int16), ws[0]["streams_out"].cpu().view(torch.int16)) for wg in ws[1:]]
print(f"R' the same bits on the other GPUs: {same}", flush=True)
sk.finalize()
