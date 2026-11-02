"""The K3 KDA layer in plain decode (TP8, T requests with one new token each) on AMD GPUs as one static megakernel, with random
weights in the real shapes, checked against float64 references, then timed like amd/tests/test_kda_layer.py.

Graph per GPU (12 heads of 128):
  x -> projection (gemm_tile, Wproj [6400, 7168] bf16, K parts in slots)
    -> short_conv_decode (the K parts added; q / k / v per request: 4-tap conv over its window at slot[t] + SiLU, fp32; the
       window moved on in place)
    -> delta_rule_decode (per head and group of requests: f_b, the gate, beta, one gated delta-rule step on the request's
       state at slot[t], written back; o_norm x sigmoid(g) -> y)
    -> o_proj (gemm_tile push_all) -> attn_residual (per request, as in verify)
Serving contract (SGLang's KDA decode, amd/notes/sglang_v0521_decode_contract.md B.3): pools read and written in place at
slot[t]; slot[t] = -1 (a padded row of a captured graph): no state read or written.

Checks: the projection row; q / k / v of the live rows vs float64 (padded row: zeros); the windows and states after the step at
the live slots, every other slot unchanged bit for bit; y of the live rows from the kernel's own q / k / v / row; u and hidden
from every GPU's y; hidden and u the same on every GPU; launch 1 == launch 0 (the pools restored before each).

python amd/tests/test_kda_decode.py --tokens 64 --gpus 8 [--rpt R] [--flush-mb 512] [--e2e-reps 20] [--profile]
MK_BACKEND=hip python amd/tests/test_kda_decode.py --tokens 64 --codegen   (no GPU: graph, plan, layer.cu, hipcc compile only)
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

T, H, HEADS, HD = 8, 7168, 12, 1536   # T: --tokens (main sets it)
NCONV = 3 * HD
PROJ_N, PROJ_PAD = 6336, 6400
COL_G, COL_FA, COL_BETA = 4608, 6144, 6272
NB_ALLOC, NB = 4, 3
ONORM_EPS, RES_EPS = 1e-5, 1e-5
SHARDED = ("wproj", "conv_w", "wfb", "prm", "oproj", "conv_pool", "state_pool")


def n_slots() -> int:
    return T + 3   # slot 0 unused (SGLang reserves it), one more spare


def request_slots(seed: int) -> list:
    """Distinct slots >= 1 in a shuffled order; the last row padded (-1) when T > 1."""
    g = torch.Generator().manual_seed(seed + 1000)
    s = (torch.randperm(n_slots() - 1, generator=g)[:T] + 1).tolist()
    if T > 1:
        s[-1] = -1
    return s


def random_weights(dev, seed: int, common: dict = None) -> dict:
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *shape: torch.randn(shape, generator=g, device=dev)
    bf = lambda t: t.to(torch.bfloat16)
    S = n_slots()
    w = {
        "x": bf(rn(T, H)),
        "wproj": bf(torch.cat([rn(PROJ_N, H) * H ** -0.5, torch.zeros(PROJ_PAD - PROJ_N, H, device=dev)])),
        "conv_w": rn(NCONV, 4) * 0.5,
        "wfb": bf(rn(HD, 128) * 128 ** -0.5),
        "prm": torch.cat([rn(HEADS) * 0.5, rn(HD) * 0.5, 1 + 0.1 * rn(128)]),
        "oproj": bf(rn(H, HD) * HD ** -0.5),
        "conv_pool": bf(rn(S * 3, NCONV)),
        "state_pool": rn(S * HEADS * 128, 128) * 0.1,
        "bank": bf(rn(T, NB_ALLOC, H)),
        "prefix": bf(rn(T, H)),
        "q": bf(rn(H) * H ** -0.5),
        "gain": bf(1 + 0.1 * rn(H)),
        "ow": bf(1 + 0.1 * rn(H)),
        "hidden": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "new_prefix": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "info": torch.tensor([NB, NB_ALLOC * H, H, 0, 0, -1, -1, 0], dtype=torch.int32, device=dev),
        "slots": torch.tensor(request_slots(seed), dtype=torch.int32, device=dev),
    }
    if common is not None:
        for k in list(w):
            if k not in SHARDED and k not in ("hidden", "new_prefix", "info"):
                w[k] = common[k].to(dev)
    return w


def ref_conv(w: dict, pool0: torch.Tensor, slots: list) -> dict:
    """The bf16 projection row, q / k / v [T][NCONV] (zeros on padded rows), the pool after the step."""
    proj = bf16(w["x"].double() @ w["wproj"][:PROJ_N].double().T)                       # [T, 6336]
    cw = w["conv_w"].double()
    pool = pool0.double().view(n_slots(), 3, NCONV).clone()
    qkv = torch.zeros(T, NCONV, dtype=torch.float64, device=proj.device)
    for t, s in enumerate(slots):
        if s < 0:
            continue
        win = torch.cat([pool[s], proj[t:t + 1, :NCONV]])                              # [4, NCONV] oldest first
        acc = (win * cw.T).sum(0)
        qkv[t] = acc * torch.sigmoid(acc)
        pool[s] = win[1:]
    return dict(proj=proj, qkv=qkv, pool=pool)


def ref_recurrence(qkv: torch.Tensor, row: torch.Tensor, w: dict, state0: torch.Tensor, slots: list) -> dict:
    """y [T, HD] of the live rows and the states after the step, from the kernel's q / k / v and row."""
    q, k, v = (qkv[:, i * HD:(i + 1) * HD].double().view(T, HEADS, 128) for i in range(3))
    prm = w["prm"].double()
    a_log, dt_bias, onw = prm[:HEADS], prm[HEADS:HEADS + HD], prm[HEADS + HD:]
    fb = bf16(row[:, COL_FA:COL_FA + 128].double() @ w["wfb"].double().T).view(T, HEADS, 128)
    gk = -5.0 * torch.sigmoid((fb + dt_bias.view(HEADS, 128)) * torch.exp(a_log.view(HEADS, 1)))
    beta = torch.sigmoid(row[:, COL_BETA:COL_BETA + HEADS].double())                     # [T, HEADS]
    states = state0.double().view(n_slots(), HEADS, 128, 128).clone()
    o = torch.zeros(T, HEADS, 128, dtype=torch.float64, device=qkv.device)
    for m, s in enumerate(slots):
        if s < 0:
            continue
        S = states[s]
        rk = 1.0 / torch.sqrt((k[m] ** 2).sum(-1, keepdim=True) + 1e-6)
        rq = (128 ** -0.5) / torch.sqrt((q[m] ** 2).sum(-1, keepdim=True) + 1e-6)
        S = S * torch.exp(gk[m]).unsqueeze(1)
        pred = torch.einsum("hvk,hk->hv", S, k[m])
        dlt = rk * beta[m].unsqueeze(1) * (v[m] - rk * pred)
        S = S + torch.einsum("hv,hk->hvk", dlt, k[m])
        o[m] = rq * torch.einsum("hvk,hk->hv", S, q[m])
        states[s] = S
    o = bf16(o).double()
    rstd = 1.0 / torch.sqrt((o ** 2).mean(-1, keepdim=True) + ONORM_EPS)
    y = bf16(o * rstd * onw.view(1, 1, 128) * torch.sigmoid(row[:, COL_G:COL_G + HD].double().view(T, HEADS, 128)))
    return dict(y=y.view(T, HD), states=states)


def ref_residual(a: torch.Tensor, w: dict) -> tuple:
    u = bf16(w["prefix"].double() + bf16(a).double())
    rows = w["bank"][:, :NB].double()
    src = torch.cat([rows, u.unsqueeze(1)], 1)
    c = (w["q"].float() * w["gain"].float()).double()
    sc = (src * c).sum(-1) / torch.sqrt((src * src).mean(-1) + RES_EPS)
    h = (torch.softmax(sc, -1).unsqueeze(-1) * src).sum(1)
    return u, bf16(h / torch.sqrt((h * h).mean(-1, keepdim=True) + RES_EPS) * w["ow"].double())


def build_graph(mpk, w: dict, rpt: int, x_in=None, prefix_in=None, prefix_wait=None):
    """the KDA decode nodes; returns (hidden, new_prefix). x_in / prefix_in: graph tensors of the layer before in the same
    kernel (test_kda_moe_layer.py --order moe), else the inputs x / prefix; prefix_wait: attn_residual's (counter, need)"""
    from mirage.core import bfloat16, float32
    given = {"x": x_in, "prefix": prefix_in}
    att = {k: given[k] if given.get(k) is not None else mpk.attach_input(torch_tensor=w[k], name=k) for k in (
        "x", "wproj_k", "conv_w", "conv_pool", "info", "wfb", "prm", "state_pool", "oproj_k", "bank", "prefix", "q", "gain",
        "ow", "hidden", "new_prefix", "slots")}
    proj = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=float32, name="proj_partial")
    rec_in = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=bfloat16, name="rec_in")
    y_rec = mpk.new_tensor(dims=(T, HD), dtype=bfloat16, name="y_rec")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    mpk.gemm_tile_layer(input=att["x"], weight=att["wproj_k"], output=proj, combine="slots",
                        hybrid={"1": True, "4": 4}.get(os.environ.get("MPK_KDA_PROJ_HYB"), False))   # (gemm_wave HYB, grid x 50; 4: HYB4, grid x 25)
    # MPK_KDA_SPLIT_PROJ=1: the conv only for q / k / v; the recurrence reads g, f_a, beta from the projection's K parts itself
    # (with plan_one_gpu.late_plan: those rows' projection tasks the second wave, the conv and the recurrence not waiting for them)
    split = os.environ.get("MPK_KDA_SPLIT_PROJ") in ("1", "2")   # (2: the same graph, test_kda_moe_layer.py orders the rows)
    if os.environ.get("MPK_KDA_CONV_IN_REC") == "1":   # no conv node: each recurrence task makes its q / k / v (delta_rule_decode.cuh CIR)
        mpk.delta_rule_decode_layer(rec_in=None, wfb=att["wfb"], prm=att["prm"], state_pool=att["state_pool"], output=y_rec,
                                    slots=att["slots"], requests_per_task=rpt, col_g=COL_G, col_fa=COL_FA, col_beta=COL_BETA,
                                    onorm_eps=ONORM_EPS, proj=proj, conv=(att["conv_w"], att["conv_pool"]),
                                    pairs_per_task=int(os.environ.get("MPK_KDA_FLAT", "0")))   # (FLAT + CIR: P pairs per task)
    else:
        mpk.short_conv_decode_layer(input=proj, conv_weight=att["conv_w"], conv_pool=att["conv_pool"], output=rec_in,
                                    slots=att["slots"], conv_only=split)
        mpk.delta_rule_decode_layer(rec_in=rec_in, wfb=att["wfb"], prm=att["prm"], state_pool=att["state_pool"], output=y_rec,
                                    slots=att["slots"], requests_per_task=rpt, col_g=COL_G, col_fa=COL_FA, col_beta=COL_BETA,
                                    onorm_eps=ONORM_EPS, proj=proj if split else None,
                                    pairs_per_task=int(os.environ.get("MPK_KDA_FLAT", "0")))   # (P (head, request) pairs per task)
    # o_proj: polls y (T <= 16), or waits for the recurrence's count and fills its activation by DMA (T > 16: 64 x 1536 bf16
    # polled in register rounds took ~20 us per task)
    polled = T <= 16 if os.environ.get("MPK_OPROJ_POLLED") is None else os.environ["MPK_OPROJ_POLLED"] == "1"
    # MPK_AR_RS=1: o_proj's rows to their GPU only (push_rs) and attn_residual over this GPU's columns, its dots and u / out
    # columns to every GPU (attn_residual.cuh RS; T 64: the push_all push ~16 us at ~917 KB per peer)
    rs = os.environ.get("MPK_AR_RS") == "1"
    # (MPK_OPROJ_FRS=1, with RS at T > 16: K parts, a row block's last part adds them and sends it (combine finish_rs))
    frs = rs and not polled and os.environ.get("MPK_OPROJ_FRS") == "1"
    mpk.gemm_tile_layer(input=y_rec, weight=att["oproj_k"], output=sent,
                        combine="finish_rs" if frs else "push_rs" if rs else "push_all", input_polled=polled)
    mpk.attn_residual_layer(input=sent, bank=att["bank"], prefix=att["prefix"], q=att["q"], gain=att["gain"], ow=att["ow"],
                            output=att["hidden"], u_output=att["new_prefix"], info=att["info"], eps=RES_EPS,
                            prefix_wait=prefix_wait, rs=rs,
                            chunks=int(os.environ.get("MPK_AR_CHUNKS", "0")))   # (MPK_AR_CHUNKS 4: 1792-column chunks, A/B)
    return att["hidden"], att["new_prefix"]


def check_all(ws: list, h, rep: Report, pools0: list) -> None:
    partials = []
    for gi, w in enumerate(ws):
        tag = f"GPU {gi}: "
        dev = w["x"].device
        slots = w["slots"].tolist()
        live = [t for t, s in enumerate(slots) if s >= 0]
        dead = [t for t, s in enumerate(slots) if s < 0]
        used = {s for s in slots if s >= 0}
        other = [s for s in range(n_slots()) if s not in used]
        pool0, state0 = (p.to(dev) for p in pools0[gi])
        rc = ref_conv(w, pool0, slots)
        cir = os.environ.get("MPK_KDA_CONV_IN_REC") == "1"
        if cir:   # (no conv node, no rec_in: the recurrence's reference from the reference row and q / k / v, bf16 as the kernel's)
            row = torch.zeros(T, PROJ_PAD, dtype=torch.bfloat16, device=dev)
            row[:, :PROJ_N] = rc["proj"].to(torch.bfloat16)
            qkv = rc["qkv"].to(torch.bfloat16).float()
        else:
            rec_in = torch.from_numpy(np.frombuffer(h.sk.read(gi, "rec_in"), dtype=np.uint8).copy()).to(dev)
            row = rec_in[:T * PROJ_PAD * 2].view(torch.bfloat16).view(T, PROJ_PAD)
            qkv = rec_in[T * PROJ_PAD * 2:T * PROJ_PAD * 2 + T * NCONV * 4].view(torch.float32).view(T, NCONV)
        if cir:
            pass
        elif os.environ.get("MPK_KDA_SPLIT_PROJ") in ("1", "2"):
            # (the conv writes only q / k / v; the recurrence read g, f_a, beta from the projection's K parts: its reference takes
            # them from the reference projection row, bf16 as the kernel rounds the parts' sum)
            row = row.clone()
            row[:, NCONV:PROJ_N] = rc["proj"][:, NCONV:].to(row.dtype)
        else:
            rep.check(tag + "projection row (g, f_a, beta)", relrms(row[:, NCONV:PROJ_N], rc["proj"][:, NCONV:]) < 2e-3,
                      f"relRMS {relrms(row[:, NCONV:PROJ_N], rc['proj'][:, NCONV:]):.2e}")
        if not cir:
            rep.check(tag + "conv q / k / v (live rows)", relrms(qkv[live], rc["qkv"][live]) < 5e-3,
                      f"relRMS {relrms(qkv[live], rc['qkv'][live]):.2e}")
            if dead:
                rep.check(tag + "conv q / k / v (padded rows) zero", bool((qkv[dead] == 0).all()), "exact")
        pool = w["conv_pool"].view(n_slots(), 3, NCONV)
        p0 = pool0.view(n_slots(), 3, NCONV)
        lu = sorted(used)
        shifted = torch.equal(pool[lu, :2], p0[lu, 1:])
        rep.check(tag + "windows after the step (live slots)", shifted and relrms(pool[lu, 2], rc["pool"][lu, 2]) < 5e-3,
                  f"taps 0, 1 = old 1, 2 bit for bit: {shifted}; new input relRMS {relrms(pool[lu, 2], rc['pool'][lu, 2]):.2e}")
        rep.check(tag + "windows of the other slots unchanged", torch.equal(pool[other], p0[other]), "bits")
        rr = ref_recurrence(qkv, row, w, state0, slots)
        y = torch.from_numpy(np.frombuffer(h.sk.read(gi, "y_rec"), dtype=np.uint16).copy()).view(torch.bfloat16)
        y = y[:T * HD].view(T, HD).to(dev)
        rep.check(tag + "y (live rows)", relrms(y[live], rr["y"][live]) < 5e-3, f"relRMS {relrms(y[live], rr['y'][live]):.2e}")
        rep.check(tag + "y finite", bool(torch.isfinite(y.double()).all()), "all rows")
        st = w["state_pool"].view(n_slots(), HEADS, 128, 128)
        s0 = state0.view(n_slots(), HEADS, 128, 128)
        rep.check(tag + "states after the step (live slots)", relrms(st[lu], rr["states"][lu]) < 1e-4,
                  f"relRMS {relrms(st[lu], rr['states'][lu]):.2e}")
        rep.check(tag + "states of the other slots unchanged", torch.equal(st[other], s0[other]), "bits")
        partials.append(bf16(y.double() @ w["oproj"].double().T).double())
    a = sum(p.to(ws[0]["x"].device) for p in partials)
    u_ref, hid_ref = ref_residual(a, ws[0])
    rep.check("u (new prefix)", relrms(ws[0]["new_prefix"], u_ref) < 2e-3, f"relRMS {relrms(ws[0]['new_prefix'], u_ref):.2e}")
    rep.check("hidden", relrms(ws[0]["hidden"], hid_ref) < 5e-3, f"relRMS {relrms(ws[0]['hidden'], hid_ref):.2e}")
    if len(ws) > 1:
        rep.check("hidden and u the same on every GPU", same_bits([w["hidden"] for w in ws]) and
                  same_bits([w["new_prefix"] for w in ws]), "bits")


def main():
    global T
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=int(os.environ.get("MPK_TOKENS", "8")), help="requests (batch size)")
    p.add_argument("--rpt", type=int, default=0, help="requests per delta_rule_decode task (0: the plan's grid)")
    p.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--out", default="kda_decode")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wait", type=float, default=60)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--flush-mb", type=int, default=512)
    p.add_argument("--e2e-reps", type=int, default=20)
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--grids", default=None, help='JSON {graph_idx: grid}: regrid nodes (plan_one_gpu.simple_plan)')
    p.add_argument("--codegen", action="store_true", help="no GPU: build the code and compile it only")
    a = p.parse_args()
    T = a.tokens
    os.environ["MPK_TOKENS"] = str(T)
    if a.codegen and not torch.cuda.is_available():   # as test_kda_layer.py: the graph's tensors on the CPU, an MI355X's props
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
    print("device:", torch.cuda.get_device_name(0) if not a.codegen else "codegen", "x", n, f"T = {T}", flush=True)
    if a.codegen:
        ws = [random_weights(torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda", 0), a.seed)]
    else:
        ws = [random_weights(torch.device("cuda", 0), a.seed)]
        ws += [random_weights(torch.device("cuda", g), a.seed + g, common=ws[0]) for g in range(1, n)]
    print("slots:", ws[0]["slots"].tolist(), flush=True)
    pools0 = [(w["conv_pool"].cpu().clone(), w["state_pool"].cpu().clone()) for w in ws]
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        from mirage.mpk.static_megakernel import shuffle_gemm_weight
        mpk.gemm_weight_layout = "shuffled"
    for w in ws:
        for name in ("wproj", "oproj"):
            w[name + "_k"] = shuffle_gemm_weight(w[name]) if a.gemm_shuffled else w[name]
    build_graph(mpk, ws[0], a.rpt or 1)
    plan = simple_plan(mpk, json.loads(a.grids) if a.grids else None)
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
    outs = ("hidden", "new_prefix", "conv_pool", "state_pool")

    def restore():
        for w, (cp, sp) in zip(ws, pools0):
            w["conv_pool"].copy_(cp)
            w["state_pool"].copy_(sp)
        for g in range(n):
            torch.cuda.synchronize(g)
    runs = []
    inter = {}   # MPK_KDA_INTER=1: the named intermediates after launches 1 and 3 (the same arena copy), bits compared
    nl = 4 if os.environ.get("MPK_KDA_INTER") == "1" else 2
    for launch in range(nl):
        restore()
        try:
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH (launch {launch}): {e}", flush=True)
            os._exit(2)
        for g in range(n):
            torch.cuda.synchronize(g)
        if launch < 2:
            runs.append([{k: w[k].clone() for k in outs} for w in ws])
        if launch in (1, 3) and nl == 4:
            for nm in ("rec_in", "y_rec", "oproj_sent"):
                try:
                    inter.setdefault(nm, []).append(np.frombuffer(h.sk.read(0, nm), dtype=np.uint16).copy())
                except Exception as e:  # noqa
                    print(f"  ({nm} not readable: {e})", flush=True)
            inter.setdefault("hidden", []).append(ws[0]["hidden"].view(torch.int16).cpu().numpy().copy())
        if nl == 4:
            inter.setdefault("_all", []).append(ws[0]["hidden"].float().cpu().clone())
        if launch == 3:
            hs = inter.pop("_all")
            for i in range(4):
                for j in range(i + 1, 4):
                    d = (hs[i] - hs[j]).abs()
                    print(f"  hidden, launch {i} vs {j}: {'same bits' if torch.equal(hs[i], hs[j]) else f'differ: max |d| {d.max().item():.3e}, relRMS {relrms(hs[i], hs[j]):.2e}, rows {sorted(set(d.amax(1).nonzero().flatten().tolist()))[:16]}'}", flush=True)
            for nm, (a1, a3) in inter.items():
                d = np.nonzero(a1 != a3)[0]
                print(f"  launch 1 vs 3, GPU 0, {nm}: {len(d)} of {a1.size} bf16 words differ" +
                      (f"; rows (of {T}) {sorted(set((d // (a1.size // T)).tolist()))[:20]}" if len(d) else ""), flush=True)
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
    if a.e2e_reps and not rep.failed:
        e2e, spans, per_gpu = [], [], []
        for i in range(a.e2e_reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            per_gpu.append(sk.elapsed_us())
            e2e.append(max(per_gpu[-1]))
            spans.append(h.span_us())
        for g in range(n):
            torch.cuda.synchronize(g)
        e2e.sort()
        spans.sort()
        print(f"LATENCY (T = {T}, {a.e2e_reps} launches, {n} GPUs, {'timing build' if a.profile else 'normal build'}, "
              f"{a.flush_mb} MB flushed before each): e2e median {e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max "
              f"{e2e[-1]:.1f}); span median {spans[len(spans) // 2]:.1f} us", flush=True)


if __name__ == "__main__":
    main()
