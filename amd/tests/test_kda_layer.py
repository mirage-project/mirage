"""The K3 KDA layer (TP8, 8 tokens) on AMD GPUs as one static megakernel, with random weights in the real shapes, checked step by
step against float64 references, then timed like amd/tests/test_layer_one_gpu.py (the MoE harness).

Graph per GPU (12 heads of 128; amd/notes/kda_mi355x.md):
  x -> projection (gemm_tile, Wproj [6400 (6336 + zero rows), 7168] bf16, K parts in slots)
    -> short_conv_silu (the K parts added; q / k / v: 4-tap conv over the pool's window + SiLU, fp32; rings k / v and the
       windows after each token written)
    -> delta_rule_recurrence (per head: f_b, the gate, beta, the gated delta rule from the pool's state, o_norm x sigmoid(g)
       -> y; rings g / beta written)
    -> o_proj (gemm_tile push_all: this GPU's partial of all 7168 rows to every GPU; polls y)
    -> attn_residual (per token: the GPUs' partials added, u = bf16(prefix + a) out, softmax mix over the bank rows and u,
       RMSNorm -> hidden)
Serving contract (SGLang's KDA target-verify): the conv window and the state are read from pools at slot info[5] and not written.

Checks (each step from the kernel's own inputs to it, where readable): q / k / v after the conv vs the float64 projection + conv
(the kernel rounds z to bf16 first, as SGLang: tolerance for rounding flips), the rings and windows, y per GPU from the kernel's
own q / k / v / g / f_a / beta, u and hidden from the kernel's own y of every GPU, the pools unchanged, hidden and u bit-identical
on every GPU, launch 1 == launch 0.

python amd/tests/test_kda_layer.py --gpus 8 [--gemm-shuffled] [--flush-mb 1024] [--e2e-reps 40] [--flag -DX ...] [--profile]
MK_BACKEND=hip python amd/tests/test_kda_layer.py --codegen [--profile] [--flag ...]   (no GPU: graph, plan, layer.cu, hipcc
                                                                                      compile only; registers / scratch)
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "amd", "tools"))
from test_layer_one_gpu import Report, bf16, node_times, relrms, same_bits  # noqa: E402

T, H, HEADS, HD = 8, 7168, 12, 1536
NCONV = 3 * HD
PROJ_N, PROJ_PAD = 6336, 6400
COL_G, COL_FA, COL_BETA = 4608, 6144, 6272
SLOTS, SLOT, ICW_ROWS, ICW_ROW, RING_L = 2, 1, 2, 1, 12   # the pools' sizes and the slot / row used (indexing tested)
NB_ALLOC, NB = 4, 3                                         # bank rows allocated per token, used
ONORM_EPS, RES_EPS = 1e-5, 1e-5
SHARDED = ("wproj", "conv_w", "wfb", "prm", "oproj", "conv_pool", "state_pool")
AR_CHUNKS = 0   # main's --ar-chunks (the u diagnosis)


def random_weights(dev, seed: int, common: dict = None) -> dict:
    """GPU dev's tensors: its own random shard (seed), the tensors every GPU has from `common` (GPU 0's dict) when given."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *shape: torch.randn(shape, generator=g, device=dev)
    bf = lambda t: t.to(torch.bfloat16)
    w = {
        "x": bf(rn(T, H)),
        "wproj": bf(torch.cat([rn(PROJ_N, H) * H ** -0.5, torch.zeros(PROJ_PAD - PROJ_N, H, device=dev)])),
        "conv_w": rn(NCONV, 4) * 0.5,
        "wfb": bf(rn(HD, 128) * 128 ** -0.5),
        "prm": torch.cat([rn(HEADS) * 0.5, rn(HD) * 0.5, 1 + 0.1 * rn(128)]),
        "oproj": bf(rn(H, HD) * HD ** -0.5),
        "conv_pool": bf(rn(SLOTS * 3, NCONV)),
        "state_pool": rn(SLOTS * HEADS * 128, 128) * 0.1,
        "bank": bf(rn(T, NB_ALLOC, H)),
        "prefix": bf(rn(T, H)),
        "q": bf(rn(H) * H ** -0.5),
        "gain": bf(1 + 0.1 * rn(H)),
        "ow": bf(1 + 0.1 * rn(H)),
        # written by the kernel
        "icw": torch.zeros(ICW_ROWS * T * 3, NCONV, dtype=torch.bfloat16, device=dev),
        "ring_k": torch.zeros(SLOTS * HEADS * RING_L, 128, dtype=torch.bfloat16, device=dev),
        "ring_v": torch.zeros(SLOTS * HEADS * RING_L, 128, dtype=torch.bfloat16, device=dev),
        "ring_g": torch.zeros(SLOTS * HEADS * RING_L, 128, dtype=torch.float32, device=dev),
        "ring_beta": torch.zeros(SLOTS * HEADS, RING_L, dtype=torch.float32, device=dev),
        "hidden": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "new_prefix": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "info": torch.tensor([NB, NB_ALLOC * H, H, RING_L, T, -1, -1, 0], dtype=torch.int32, device=dev),
        "slot": torch.tensor([SLOT, -1], dtype=torch.int32, device=dev),        # (the framework's index tensors; [0] used)
        "icw_row": torch.tensor([ICW_ROW, -1], dtype=torch.int32, device=dev),
    }
    if common is not None:
        for k in list(w):
            if k not in SHARDED and k not in ("icw", "ring_k", "ring_v", "ring_g", "ring_beta", "hidden", "new_prefix", "info",
                                              "slot", "icw_row"):
                w[k] = common[k].to(dev)
    return w


# ---- float64 references (mpk-kernels tests/kda_reference.py, our layouts) ----
def ref_conv(w: dict) -> dict:
    """q / k / v after the conv + SiLU [T][NCONV], the windows after each token [T][3][NCONV], the bf16 projection row."""
    proj = bf16(w["x"].double() @ w["wproj"][:PROJ_N].double().T)                       # [T, 6336]
    window = w["conv_pool"][SLOT * 3:SLOT * 3 + 3].double()                             # [3, NCONV] oldest first
    win = torch.cat([window.T, proj[:, :NCONV].T], 1)                                   # [NCONV, 3 + T]
    cw = w["conv_w"].double()
    acc = sum(win[:, j:j + T] * cw[:, j:j + 1] for j in range(4))                       # [NCONV, T]
    qkv = (acc * torch.sigmoid(acc)).T                                                  # [T, NCONV]
    icw = torch.stack([win[:, t + 1:t + 4].T for t in range(T)])                        # [T, 3, NCONV]
    return dict(proj=proj, qkv=qkv, icw=icw)


def ref_recurrence(qkv: torch.Tensor, row: torch.Tensor, w: dict) -> dict:
    """y [T, HD] from q / k / v [T, NCONV] (fp32 values) and the projection row's g, f_a, beta (bf16 values)."""
    q, k, v = (qkv[:, i * HD:(i + 1) * HD].double().view(T, HEADS, 128) for i in range(3))
    prm = w["prm"].double()
    a_log, dt_bias, onw = prm[:HEADS], prm[HEADS:HEADS + HD], prm[HEADS + HD:]
    fb = bf16(row[:, COL_FA:COL_FA + 128].double() @ w["wfb"].double().T).view(T, HEADS, 128)
    sg = torch.sigmoid((fb + dt_bias.view(HEADS, 128)) * torch.exp(a_log.view(HEADS, 1)))
    gk = -5.0 * sg
    beta = torch.sigmoid(row[:, COL_BETA:COL_BETA + HEADS].double())                     # [T, HEADS]
    S = w["state_pool"].double().view(SLOTS, HEADS, 128, 128)[SLOT].clone()
    o = torch.zeros(T, HEADS, 128, dtype=torch.float64, device=qkv.device)
    for m in range(T):
        rk = 1.0 / torch.sqrt((k[m] ** 2).sum(-1, keepdim=True) + 1e-6)
        rq = (128 ** -0.5) / torch.sqrt((q[m] ** 2).sum(-1, keepdim=True) + 1e-6)
        S = S * torch.exp(gk[m]).unsqueeze(1)
        pred = torch.einsum("hvk,hk->hv", S, k[m])
        dlt = rk * beta[m].unsqueeze(1) * (v[m] - rk * pred)
        S = S + torch.einsum("hv,hk->hvk", dlt, k[m])
        o[m] = rq * torch.einsum("hvk,hk->hv", S, q[m])
    o = bf16(o).double()   # (SGLang ROCm: o stored in bf16 before rms_norm_gated)
    rstd = 1.0 / torch.sqrt((o ** 2).mean(-1, keepdim=True) + ONORM_EPS)
    y = bf16(o * rstd * onw.view(1, 1, 128) * torch.sigmoid(row[:, COL_G:COL_G + HD].double().view(T, HEADS, 128)))
    return dict(y=y.view(T, HD), gk=gk, beta=beta)


def ref_residual(a: torch.Tensor, w: dict, prefix: str = "prefix", q: str = "q", gain: str = "gain", ow: str = "ow") -> tuple:
    """u, hidden from the summed partials a [T, H] (float64). (Aggregation 1: a = y, the zero prefix, its own weights.)"""
    u = bf16(w[prefix].double() + bf16(a).double())   # (a: the all-reduce's bf16 output, SGLang ROCm)
    rows = w["bank"][:, :NB].double()
    src = torch.cat([rows, u.unsqueeze(1)], 1)                                          # [T, NB + 1, H]
    c = (w[q].float() * w[gain].float()).double()   # fp32, not rounded (SGLang ROCm cw2)
    sc = (src * c).sum(-1) / torch.sqrt((src * src).mean(-1) + RES_EPS)
    h = (torch.softmax(sc, -1).unsqueeze(-1) * src).sum(1)
    return u, bf16(h / torch.sqrt((h * h).mean(-1, keepdim=True) + RES_EPS) * w[ow].double())


def check_all(ws: list, h, rep: Report, pools0: list) -> None:
    n = len(ws)
    partials, raws = [], []
    for gi, w in enumerate(ws):
        tag = f"GPU {gi}: "
        rec_in = torch.from_numpy(np.frombuffer(h.sk.read(gi, "rec_in"), dtype=np.uint8).copy()).to(w["x"].device)
        row = rec_in[:T * PROJ_PAD * 2].view(torch.bfloat16).view(T, PROJ_PAD)
        qkv = rec_in[T * PROJ_PAD * 2:T * PROJ_PAD * 2 + T * NCONV * 4].view(torch.float32).view(T, NCONV)
        rc = ref_conv(w)
        rep.check(tag + "projection row (g, f_a, beta)", relrms(row[:, NCONV:PROJ_N], rc["proj"][:, NCONV:]) < 2e-3,
                  f"relRMS {relrms(row[:, NCONV:PROJ_N], rc['proj'][:, NCONV:]):.2e} (bf16 of fp32 vs float64 sums)")
        rep.check(tag + "conv q / k / v", relrms(qkv, rc["qkv"]) < 5e-3, f"relRMS {relrms(qkv, rc['qkv']):.2e}")
        icw = w["icw"].view(ICW_ROWS, T, 3, NCONV)[ICW_ROW]
        rep.check(tag + "windows after each token", relrms(icw, rc["icw"]) < 5e-3, f"relRMS {relrms(icw, rc['icw']):.2e}")
        rr = ref_recurrence(qkv, row, w)
        rk = w["ring_k"].view(SLOTS, HEADS, RING_L, 128)[SLOT, :, :T]
        rv = w["ring_v"].view(SLOTS, HEADS, RING_L, 128)[SLOT, :, :T]
        rk_ref = bf16(qkv[:, HD:2 * HD].double().view(T, HEADS, 128).permute(1, 0, 2))
        rv_ref = bf16(qkv[:, 2 * HD:].double().view(T, HEADS, 128).permute(1, 0, 2))
        rep.check(tag + "rings k / v", relrms(rk, rk_ref) < 1e-2 and relrms(rv, rv_ref) < 1e-2,
                  f"relRMS {relrms(rk, rk_ref):.2e} / {relrms(rv, rv_ref):.2e} (bf16 of the kernel's fp32)")
        rg = w["ring_g"].view(SLOTS, HEADS, RING_L, 128)[SLOT, :, :T]
        rb = w["ring_beta"].view(SLOTS, HEADS, RING_L)[SLOT, :, :T]
        rep.check(tag + "rings g / beta", relrms(rg, rr["gk"].permute(1, 0, 2)) < 2e-3 and relrms(rb, rr["beta"].T) < 1e-5,
                  f"relRMS {relrms(rg, rr['gk'].permute(1, 0, 2)):.2e} / {relrms(rb, rr['beta'].T):.2e}")
        y = torch.from_numpy(np.frombuffer(h.sk.read(gi, "y_rec"), dtype=np.uint16).copy()).view(torch.bfloat16)
        y = y[:T * HD].view(T, HD).to(w["x"].device)
        rep.check(tag + "y (recurrence + o_norm)", relrms(y, rr["y"]) < 5e-3, f"relRMS {relrms(y, rr['y']):.2e}")
        raws.append(y.double() @ w["oproj"].double().T)
        partials.append(bf16(raws[-1]).double())   # (bf16 values in float64: added in float64 below)
        unchanged = torch.equal(w["conv_pool"].cpu(), pools0[gi][0]) and torch.equal(w["state_pool"].cpu(), pools0[gi][1])
        rep.check(tag + "pools unchanged", unchanged, "conv window and state pools bit for bit")
    a = sum(p.to(ws[0]["x"].device) for p in partials)
    u_ref, hid_ref = ref_residual(a, ws[0])
    if not relrms(ws[0]["new_prefix"], u_ref) < 2e-3:   # diagnosis: which variant of the reference the kernel's u matches
        dev = ws[0]["x"].device
        u = ws[0]["new_prefix"].double()
        for gi, w in enumerate(ws):   # NaN values: where (attn_residual: chunk = column // (H / chunks), thread = 8 columns)
            bad = (~torch.isfinite(w["new_prefix"].double())).nonzero().cpu()
            if len(bad):
                cw = H // (AR_CHUNKS or H // 1024)
                print(f"    GPU {gi}: {len(bad)} non-finite u values; tokens {sorted(set(bad[:, 0].tolist()))}; chunks "
                      f"{sorted(set((bad[:, 1] // cw).tolist()))}; columns in chunk {sorted(set((bad[:, 1] % cw).tolist()))[:12]}; "
                      f"bits {w['new_prefix'][bad[0, 0], bad[0, 1]].view(torch.int16).item() & 0xFFFF:#06x}", flush=True)
        ul = (u.to(torch.bfloat16).view(torch.int16).int() - u_ref.to(torch.bfloat16).view(torch.int16).int()).abs()
        print(f"    u: {(ul > 0).float().mean().item():.3f} of the values differ, max {ul.max().item()} ulps; per token relRMS "
              f"{[round(relrms(u[t], u_ref[t]), 5) for t in range(T)]}", flush=True)
        ex_set = int(h.layout.get("exchange_set", 0)) if hasattr(h, "layout") else 0
        for gi in range(n):   # the partials as pushed into GPU gi's exchange buffer (both sets), vs y . Wo^T of each GPU
            raw_b = np.frombuffer(h.sk.read(gi, "oproj_sent"), dtype=np.uint8)
            nb_ = n * T * H * 2
            for st in range(2):
                off = st * (len(raw_b) - nb_)
                got = torch.from_numpy(raw_b[off:off + nb_].copy()).view(torch.bfloat16).view(n, T, H).double().to(dev)
                print(f"    GPU {gi} set {st}: pushed partials vs bf16(y Wo^T) relRMS per sender "
                      f"{[round(relrms(got[r], partials[r].to(dev)), 5) for r in range(n)]}", flush=True)
                if st == 0:
                    u_py = bf16(ws[0]["prefix"].double() + got.sum(0))
                    dd = (u - u_py).abs() > 0
                    cols = dd.nonzero()[:, 1]
                    print(f"    u from the pushed bytes: kernel u vs it relRMS {relrms(u, u_py):.2e}, differing {int(dd.sum())}; "
                          f"their column % 16 histogram {torch.bincount(cols % 16, minlength=16).tolist()}; column // 896 "
                          f"{torch.bincount(cols // 896, minlength=8).tolist()}; u_py vs u_ref {relrms(u_py, u_ref):.2e}", flush=True)
                    t0 = int(dd.nonzero()[0, 0]); c0 = int(dd.nonzero()[0, 1])
                    print(f"    first: t {t0} col {c0}: kernel {u[t0, c0].item():.6f} py {u_py[t0, c0].item():.6f} prefix "
                          f"{ws[0]['prefix'][t0, c0].item():.6f} a {got.sum(0)[t0, c0].item():.6f}; kernel - prefix "
                          f"{(u[t0, c0] - ws[0]['prefix'][t0, c0].double()).item():.6f}", flush=True)
            if gi == 0:
                break
        a_raw = sum(r.to(dev) for r in raws)
        print(f"    partials not rounded to bf16: relRMS {relrms(u, bf16(ws[0]['prefix'].double() + a_raw)):.2e}", flush=True)
        for g in range(n):
            a_g = a - partials[g].to(dev)
            print(f"    without GPU {g}'s partial: relRMS {relrms(u, bf16(ws[0]['prefix'].double() + a_g)):.2e}; GPU {g}'s partial "
                  f"twice: {relrms(u, bf16(ws[0]['prefix'].double() + a + partials[g].to(dev))):.2e}", flush=True)
        print(f"    no prefix: {relrms(u, bf16(a)):.2e}; prefix only: {relrms(u, ws[0]['prefix'].double()):.2e}", flush=True)
    rep.check("u (new prefix)", relrms(ws[0]["new_prefix"], u_ref) < 2e-3, f"relRMS {relrms(ws[0]['new_prefix'], u_ref):.2e}")
    rep.check("hidden", relrms(ws[0]["hidden"], hid_ref) < 5e-3, f"relRMS {relrms(ws[0]['hidden'], hid_ref):.2e}")
    if n > 1:
        rep.check("hidden and u the same on every GPU", same_bits([w["hidden"] for w in ws]) and
                  same_bits([w["new_prefix"] for w in ws]), "bits")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--out", default="kda_layer")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wait", type=float, default=60)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--flush-mb", type=int, default=0)
    p.add_argument("--e2e-reps", type=int, default=20)
    p.add_argument("--ar-chunks", type=int, default=0, help="attn_residual tasks per token (0: H / 1024; 4: 1792-column chunks)")
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--grids", default=None, help='JSON {graph_idx: grid}: regrid nodes (plan_one_gpu.simple_plan)')
    p.add_argument("--save-outs", default=None, help="launch 1's outputs of every GPU to this .pt (test_rank_lib.py compares)")
    p.add_argument("--dedicate", default=None, help="comma-separated graph_idx: plan_one_gpu.dedicate_plan")
    p.add_argument("--light", default=None, help="N:H[,N:H]: plan_one_gpu.light_plan (node N off the CUs with the most H tasks)")
    p.add_argument("--apart", default=None, help="N,A1,A2..: plan_one_gpu.apart_plan (node N one per CU, off the CUs of A*)")
    p.add_argument("--first", default=None, help="comma-separated graph_idx: plan_one_gpu.first_plan")
    p.add_argument("--codegen", action="store_true", help="no GPU: build the code and compile it only (amd/tools/codegen_check.py)")
    a = p.parse_args()
    global AR_CHUNKS
    AR_CHUNKS = a.ar_chunks
    if a.codegen and not torch.cuda.is_available():   # as codegen_check.py: the graph's tensors on the CPU, an MI355X's props
        for fn in ("empty", "zeros", "ones", "full", "arange", "tensor", "empty_like", "zeros_like", "randn", "cat"):
            def on_cpu(*args, _orig=getattr(torch, fn), **kw):
                if str(kw.get("device", "")).startswith("cuda"):
                    kw["device"] = "cpu"
                return _orig(*args, **kw)
            setattr(torch, fn, on_cpu)

        class Props:
            major, minor, multi_processor_count, gcnArchName = 9, 5, 256, "gfx950"
            name, total_memory = "no GPU (codegen check)", 288 << 30
        torch.cuda.get_device_properties = lambda *_: Props
        torch.cuda.current_device = lambda: 0
        torch.cuda.device_count = lambda: 8
        torch.cuda.set_device = lambda *_: None
        _gen = torch.Generator
        torch.Generator = lambda device=None: _gen()
    import json
    from mirage.core import bfloat16, float32
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    n = a.gpus
    assert torch.cuda.device_count() >= n
    if a.codegen:   # one GPU's tensors (shapes only), every GPU the same
        dev0 = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda", 0)
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, "CUs x", n,
          flush=True)
    if a.codegen:
        ws = [random_weights(dev0, a.seed)]
    else:
        ws = [random_weights(torch.device("cuda", 0), a.seed)]
        ws += [random_weights(torch.device("cuda", g), a.seed + g, common=ws[0]) for g in range(1, n)]
    pools0 = [(w["conv_pool"].cpu().clone(), w["state_pool"].cpu().clone()) for w in ws]
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        from mirage.mpk.static_megakernel import shuffle_gemm_weight
        mpk.gemm_weight_layout = "shuffled"
        for w in ws:
            for name in ("wproj", "oproj"):
                w[name + "_k"] = shuffle_gemm_weight(w[name])
    else:
        for w in ws:
            w["wproj_k"], w["oproj_k"] = w["wproj"], w["oproj"]
    att = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in (
        "x", "wproj_k", "conv_w", "conv_pool", "icw", "ring_k", "ring_v", "info", "wfb", "prm", "state_pool", "ring_g",
        "ring_beta", "oproj_k", "bank", "prefix", "q", "gain", "ow", "hidden", "new_prefix", "slot", "icw_row")}
    proj = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=float32, name="proj_partial")
    rec_in = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=bfloat16, name="rec_in")
    y_rec = mpk.new_tensor(dims=(T, HD), dtype=bfloat16, name="y_rec")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    xin = att["x"]
    mpk.gemm_tile_layer(input=xin, weight=att["wproj_k"], output=proj, combine="slots",
                        hybrid=os.environ.get("MPK_KDA_PROJ_HYB") == "1")   # (gemm_tile.cuh gemm_wave HYB; grid x 50)
    mpk.short_conv_silu_layer(input=proj, conv_weight=att["conv_w"], conv_pool=att["conv_pool"], output=rec_in,
                              icw=att["icw"], ring_k=att["ring_k"], ring_v=att["ring_v"], info=att["info"], slot=att["slot"],
                              icw_row=att["icw_row"])
    mpk.delta_rule_recurrence_layer(rec_in=rec_in, wfb=att["wfb"], prm=att["prm"], state_pool=att["state_pool"], output=y_rec,
                                    ring_g=att["ring_g"], ring_beta=att["ring_beta"], info=att["info"], slot=att["slot"],
                                    col_g=COL_G, col_fa=COL_FA, col_beta=COL_BETA, onorm_eps=ONORM_EPS)
    mpk.gemm_tile_layer(input=y_rec, weight=att["oproj_k"], output=sent, combine="push_all", input_polled=True)
    mpk.attn_residual_layer(input=sent, bank=att["bank"], prefix=att["prefix"], q=att["q"], gain=att["gain"], ow=att["ow"],
                            output=att["hidden"], u_output=att["new_prefix"], info=att["info"], eps=RES_EPS, chunks=a.ar_chunks)
    plan = simple_plan(mpk, json.loads(a.grids) if a.grids else None)
    if a.first:   # (plan_one_gpu.first_plan: these nodes' tasks first on CUs of their own)
        from plan_one_gpu import first_plan
        plan = first_plan(plan, {int(x) for x in a.first.split(",")})
    if a.light:   # (plan_one_gpu.light_plan: node N's tasks off the CUs with the most tasks of node H)
        from plan_one_gpu import light_plan
        for nh in a.light.split(","):
            plan = light_plan(plan, *(int(x) for x in nh.split(":")))
    if a.apart:   # (plan_one_gpu.apart_plan: node N's tasks one per CU, on CUs without tasks of the nodes A*)
        from plan_one_gpu import apart_plan
        ap = [int(x) for x in a.apart.split(",")]
        plan = apart_plan(plan, ap[0], set(ap[1:]))
    if a.dedicate:   # (plan_one_gpu.dedicate_plan: these nodes' tasks alone on CUs of their own, first)
        from plan_one_gpu import dedicate_plan
        plan = dedicate_plan(plan, {int(x) for x in a.dedicate.split(",")})
    print("plan grids:", {k: v for k, v in plan["grids"].items()}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    compiler.write_plan_file(os.path.join(a.out, f"plan_{n}gpu.json"), plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    if a.codegen:
        import re
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q)) for q in paths], a.profile)
        cu = os.path.join(a.out, "layer.cu")
        with open(cu, "w") as f:
            f.write(code)
        cmd = static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["-Wall", "--save-temps"],
                                              arch="gfx950")
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        print(f"layer.cu {len(code.splitlines())} lines; hipcc exit {r.returncode}, {r.stderr.count('warning:')} warnings",
              flush=True)
        if r.returncode or "warning:" in r.stderr:
            print("\n".join(l for l in r.stderr.splitlines() if "error" in l or "warning:" in l or "note:" in l)[:6000])
        if r.returncode == 0:
            sasm = open(os.path.join(a.out, "layer-hip-amdgcn-amd-amdhsa-gfx950.s")).read()
            print({k: re.search(rf"; {k}: (\d+)", sasm).group(1) for k in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy")})
        return
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(sk)
    outs = ("hidden", "new_prefix", "ring_k", "ring_v", "ring_g", "ring_beta", "icw")
    runs = []
    for launch in range(2):
        try:
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH (launch {launch}): {e}", flush=True)
            os._exit(2)
        for g in range(n):
            torch.cuda.synchronize(g)
        runs.append([{k: w[k].clone() for k in outs} for w in ws])
        print(f"launch {launch}: e2e {h.e2e_us():.1f} us, span {h.span_us():.1f} us", flush=True)
        if a.profile and launch == 1:
            node_times(h, paths, compiler.graph_nodes(mpk))
    rep = Report()
    print("checks of launch 1:")
    check_all(ws, h, rep, pools0)
    for k in outs:
        rep.check(f"launch 1 == launch 0: {k}", all(same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], runs[1])),
                  f"bits, all {n} GPUs")
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)
    if a.save_outs:
        torch.save([{k: v.cpu() for k, v in r.items()} for r in runs[1]], a.save_outs)
        print("saved launch 1's outputs to", a.save_outs, flush=True)
    if a.e2e_reps and (not rep.failed or any("_KO=" in f for f in a.flag)):   # (a knockout build: timed, outputs wrong)
        e2e, spans, per_gpu = [], [], []
        for i in range(a.e2e_reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            per_gpu.append(sk.elapsed_us())
            e2e.append(max(per_gpu[-1]))
            spans.append(h.span_us())
        for g in range(n):
            torch.cuda.synchronize(g)
        last = [{k: w[k].clone() for k in outs} for w in ws]
        ok = all(all(same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], last)) for k in outs)
        print("CHECK OF THE LAST TIMED LAUNCH " + ("PASSED" if ok else "FAILED"), flush=True)
        e2e.sort()
        spans.sort()
        med = [sorted(p[g] for p in per_gpu)[len(per_gpu) // 2] for g in range(n)]
        print(f"LATENCY ({a.e2e_reps} launches, {n} GPUs, {'timing build' if a.profile else 'normal build'}, {a.flush_mb} MB "
              f"flushed before each): e2e median {e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max {e2e[-1]:.1f}); span median "
              f"{spans[len(spans) // 2]:.1f} us; e2e median per GPU {[round(m, 1) for m in med]}", flush=True)


if __name__ == "__main__":
    main()
