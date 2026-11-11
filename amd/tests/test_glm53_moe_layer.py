"""GLM-5.3 MoE sublayer on AMD GPUs (TP 8, plain decode): x_ffn -> the router (gemm_tile, K parts in slots) -> glm_topk_route
(sigmoid + bias, top 8, renormalised x 2.5, + the shared expert) -> glm_moe_experts (FP8 experts, this GPU's 256 of each expert's
2048 features) -> allreduce_send (a token's 9 rows added, bf16; the senders land the sums and do SGLang's fused residual add +
RMSNorm with layer L + 1's input_layernorm: the next attention input).
Real weights of MoE layer --layer (router bf16, bias fp32, 256 experts + the shared one, e4m3 + fp32 block scales) and layer
L + 1's input_layernorm; random x_ffn and residual. Checked per launch: the pairs (experts = the reference's top 8, weights), R per
pair on GPU 0 (vs float64 of the dequantized weights and the kernel's own pairs), residual_out and x_attn vs the reference chain;
GPUs equal; relaunches bit-identical. amd/notes/sglang_v0522_glm53_layer_contract.md.

    python amd/tests/test_glm53_moe_layer.py --tokens 8 [--gpus 8] [--layer 4] [--codegen [--serving]] [--profile]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_glm_mhc import bf, cpu_standins, relrms  # noqa: E402

H, E, TOPK, I_FULL, SCALE = 6144, 256, 8, 2048, 2.5
NL = NE_PAD = 256   # router rows (no padding: 256 experts, a multiple of 64 and of 128)


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.files[f].get_tensor(name)


def load_moe(ck: Ckpt, layer: int):
    p = f"model.layers.{layer}.mlp."
    w = {"router": ck.get(p + "gate.weight"), "bias": ck.get(p + "gate.e_score_correction_bias").float(), "experts": []}
    for e in range(E + 1):
        q = p + (f"experts.{e}." if e < E else "shared_experts.")
        w["experts"].append({k: (ck.get(q + f"{k}_proj.weight"), ck.get(q + f"{k}_proj.weight_scale_inv").float())
                             for k in ("gate", "up", "down")})
    return w


def pack_experts(w, g: int, n: int):
    """GPU g's bank in tasks/glm_moe_experts.cuh's layout (as amd/tests/test_glm_moe_layer.py pack_experts) and the raw shards"""
    IR = I_FULL // n
    fs, bs = slice(g * IR, (g + 1) * IR), slice(g * IR // 128, (g + 1) * IR // 128)
    w13, s13, w2, s2, raw = [], [], [], [], []
    for ex in w["experts"]:
        (gq, gs), (uq, us), (dq, ds) = ex["gate"], ex["up"], ex["down"]
        gq, uq, dq = gq.view(torch.uint8)[fs], uq.view(torch.uint8)[fs], dq.view(torch.uint8)[:, fs]
        gs, us, ds = gs[bs], us[bs], ds[:, bs]
        u13 = torch.stack([gq.reshape(IR // 16, 16, H), uq.reshape(IR // 16, 16, H)], 1)
        w13.append(u13.reshape(IR // 16, 2, 16, H // 128, 4, 2, 16).permute(0, 3, 1, 5, 4, 2, 6).reshape(2 * IR, H))
        s13.append(torch.stack([gs, us], -1))
        w2.append(dq.reshape(H // 16, 16, IR // 128, 4, 2, 16).permute(0, 2, 4, 3, 1, 5).reshape(H, IR))
        s2.append(ds.contiguous())
        raw.append(((gq, gs), (uq, us), (dq, ds)))
    return (torch.cat(w13, 0).contiguous(), torch.stack(s13).reshape(-1).contiguous(), torch.cat(w2, 0).contiguous(),
            torch.stack(s2).reshape(-1).contiguous(), raw)


_DEQ = {}


def dequant(raw, g, e):
    """expert e's shard on GPU g, dequantized to fp64 on that GPU (the reference runs there), cached"""
    if (g, e) not in _DEQ:
        dev = torch.device("cuda", g) if torch.cuda.is_available() else torch.device("cpu")
        f8 = lambda q: q.to(dev).view(torch.float8_e4m3fn).double()
        blk = lambda q, s: f8(q) * s.to(dev).double().repeat_interleave(128, 0).repeat_interleave(128, 1)
        _DEQ[(g, e)] = tuple(blk(q, s) for q, s in raw[g][e])
    return _DEQ[(g, e)]


def route_ref(x, router, bias):
    """noaux_tc, n_group 1: sigmoid scores, + bias for the choice, top 8, the scores renormalised x 2.5"""
    sc = torch.sigmoid(x.float() @ router.float().t())
    idx = torch.topk(sc + bias, TOPK, dim=-1).indices
    wt = sc.gather(1, idx)
    return idx, wt / wt.sum(-1, keepdim=True) * SCALE


def add_rmsnorm_ref(parts, res, gamma, eps):
    """aiter's fused all-reduce + add + RMSNorm (contract section 7): parts [GPUS][T][H] bf16"""
    acc = torch.zeros(parts[0].shape, dtype=torch.float32)
    for p_ in parts:
        acc = acc + p_.float()
    v = bf(acc).float() + res.float()
    rstd = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return bf(v), bf(v * rstd * gamma.float())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--model-dir", default="/dev/shm/glm53")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--router-grid", default="", help="the router gemm's grid (default: the wide grid)")
    p.add_argument("--gemm-cap", type=int, default=256)
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm53_moe")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--serving", action="store_true", help="with --codegen: the server library's graph (scratch tensors inside)")
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    KP = TOPK + 1
    os.environ["MPK_TOKENS"] = str(T)
    sys.path.insert(0, os.path.join(HERE, "..", "serving", "glm53"))
    import glm53_layers as gl
    a.flag = gl.apply_settings(a.flag)   # (GLM-5.3's build settings: glm53_layers.G53_FLAGS / G53_ENV)
    if a.codegen:
        cpu_standins()
    from mirage.core import bfloat16, float32, int32, int64  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    rms_eps = cfg["rms_norm_eps"]
    assert (cfg["hidden_size"], cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]) == (H, E, TOPK, I_FULL)
    ck = Ckpt(a.model_dir)
    if a.codegen:   # (the shapes only)
        z = lambda *sh: torch.zeros(*sh, dtype=torch.uint8).view(torch.float8_e4m3fn)
        ex = {"gate": (z(I_FULL, H), torch.ones(16, 48)), "up": (z(I_FULL, H), torch.ones(16, 48)),
              "down": (z(H, I_FULL), torch.ones(48, 16))}
        W = {"router": torch.zeros(E, H, dtype=torch.bfloat16), "bias": torch.zeros(E), "experts": [ex] * (E + 1)}
        gamma = torch.ones(H, dtype=torch.bfloat16)
    else:
        W = load_moe(ck, a.layer)
        gamma = ck.get(f"model.layers.{a.layer + 1}.input_layernorm.weight")
    gen = torch.Generator().manual_seed(17)
    x = bf(torch.randn(T, H, generator=gen))
    res = bf(torch.randn(T, H, generator=gen) * 4)
    router = torch.zeros(NL, H, dtype=torch.bfloat16)
    router[:E] = W["router"]
    bias = torch.full((NE_PAD,), float("-inf"))
    bias[:E] = W["bias"]
    ws, deqs = [], []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        w13, s13, w2, s2, raw = pack_experts(W, g, n)
        deqs.append(raw)
        ws.append({"x_ffn": x.to(dev), "router": router.to(dev), "bias": bias.to(dev), "w13": w13.to(dev),
                   "w13_sf": s13.to(dev), "w2": w2.to(dev), "w2_sf": s2.to(dev),
                   "pairs": torch.zeros(T, KP, dtype=torch.int64, device=dev),
                   "R_moe": torch.zeros(T * KP, H, dtype=torch.float32, device=dev),
                   "res_in": res.to(dev), "attn_norm": gamma.to(dev),
                   "res_out": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                   "x_attn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev)})
        if any("STATIC_GME_TIMES" in fl for fl in a.flag):   # (per wave 8 times, long long: test_glm_moe_layer.py's print)
            ws[-1]["gme_dbg"] = torch.zeros(1024 * 8 * 2, dtype=torch.int32, device=dev)
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    SCRATCH = ['pairs', 'R_moe'] if a.serving else []
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    logits = mpk.new_tensor(dims=(T, NL), dtype=float32, name="router_logits")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="moe_sent")
    mpk.gemm_tile_layer(input=at["x_ffn"], weight=at["router"], output=logits, combine="slots")
    mpk.glm_topk_route_layer(input=logits, bias=at["bias"], output=at["pairs"], k=TOPK, shared_experts=1, shared_first=E,
                             routed_scaling_factor=SCALE)
    mpk.glm_moe_experts_layer(x=at["x_ffn"], pairs=at["pairs"], w13=at["w13"], w13_sf=at["w13_sf"], w2=at["w2"],
                              w2_sf=at["w2_sf"], output=at["R_moe"], swiglu_limit=None, dbg=at.get("gme_dbg"))
    # (the add + RMSNorm folded into the senders, as glm53_layers)
    mpk.allreduce_send_layer(input=at["R_moe"], output=sent, norm=dict(residual=at["res_in"], residual_out=at["res_out"],
                                                                        gamma=at["attn_norm"], output=at["x_attn"], eps=rms_eps,
                                                                        rs=gl.rs_on(T, n)))
    nodes = compiler.graph_nodes(mpk)
    from glm_plan import wide_gemm_grids
    grids = wide_gemm_grids(mpk, a.gemm_cap) if a.gemm_cap else {}
    if a.router_grid:
        grids[nodes[0].graph_idx] = tuple(int(v) for v in a.router_grid.split(","))
    plan = simple_plan(mpk, grids)
    print("nodes:", [(q.graph_idx, q.name, q.grid) for q in nodes], flush=True)
    os.makedirs(a.out, exist_ok=True)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    if a.codegen:
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q)) for q in paths], False)
        cu = os.path.join(a.out, "layer.cu")
        open(cu, "w").write(code)
        r = subprocess.run(static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["--save-temps"],
                                                           arch="gfx950"), cwd=a.out, capture_output=True, text=True)
        print(f"hipcc exit {r.returncode}", r.stderr[-3000:] if r.returncode else "", flush=True)
        return
    from mirage.mpk.static_harness import Harness
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(sk)
    idx_ref, wt_ref = route_ref(x, W["router"], W["bias"])
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)

    def expert_ref(xt, g, e):   # (on GPU g, fp64; the result on the CPU)
        gt, up, dn = dequant(deqs, g, e)
        xd = xt.to(gt.device).double()
        gv, uv = xd @ gt.t(), xd @ up.t()
        return ((gv * torch.sigmoid(gv) * uv) @ dn.t()).cpu()
    first = None
    for launch in range(a.launches):
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        pk = ws[0]["pairs"].cpu()
        ek, wk = (pk >> 32).int(), (pk & 0xFFFFFFFF).int().view(torch.float32)
        if launch == 0:
            same_set = all(set(ek[t, :TOPK].tolist()) == set(idx_ref[t].tolist()) for t in range(T))
            check("pairs: the routed experts = the reference's top 8", same_set,
                  f"tokens differing: {[t for t in range(T) if set(ek[t, :TOPK].tolist()) != set(idx_ref[t].tolist())]}")
            wr = torch.stack([wt_ref[t][[idx_ref[t].tolist().index(e) for e in ek[t, :TOPK].tolist()]] if same_set else
                              wt_ref[t] for t in range(T)])
            check("pairs: weights", relrms(wk[:, :TOPK], wr) < 1e-5 and bool((ek[:, TOPK] == E).all()) and
                  bool((wk[:, TOPK] == 1.0).all()), f"relRMS {relrms(wk[:, :TOPK], wr):.2e}")
            Rk = ws[0]["R_moe"].cpu().view(T, KP, H)
            ref = torch.zeros(T, KP, H, dtype=torch.float64)
            for t in range(T):
                for j in range(KP):
                    ref[t, j] = expert_ref(x[t:t + 1], 0, int(ek[t, j]))[0] * float(wk[t, j])
            check("GPU 0: R per pair vs float64 (dequantized weights)", relrms(Rk, ref) < 1e-2, f"relRMS {relrms(Rk, ref):.2e}")
            parts = []
            for g in range(n):   # (each GPU's send: a token's KP rows added in order, rounded to bf16)
                part = torch.zeros(T, H, dtype=torch.float64)
                for t in range(T):
                    for j in range(KP):
                        part[t] += expert_ref(x[t:t + 1], g, int(ek[t, j]))[0] * float(wk[t, j])
                parts.append(bf(part.float()))
            res_ref, x_ref = add_rmsnorm_ref(parts, res, gamma, rms_eps)
        G0 = {k: ws[0][k].cpu().clone() for k in ("res_out", "x_attn", "pairs")}
        rs = gl.rs_on(T, n)   # (RSN: residual_out only on each token's owner GPU)
        if rs:
            G0["res_out"] = gl.owner_rows([ws[g]["res_out"].cpu() for g in range(n)], T, n)
        check(f"launch {launch}: residual_out vs the reference chain", relrms(G0["res_out"], res_ref) < 1e-2,
              f"relRMS {relrms(G0['res_out'], res_ref):.2e}")
        check(f"launch {launch}: x_attn vs the reference chain", relrms(G0["x_attn"], x_ref) < 2e-2,
              f"relRMS {relrms(G0['x_attn'], x_ref):.2e}")
        same = all(torch.equal(ws[g][k].cpu(), G0[k]) for g in range(1, n) for k in G0 if not (rs and k == "res_out"))
        check(f"launch {launch}: every GPU's outputs equal", same)
        if first is None:
            first = G0
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G0[k], first[k]) for k in G0))
    if not failed and a.reps:
        us = []
        for _ in range(a.reps):
            h.launch(512 << 20, timeout_s=60)
            us.append(max(h.sk.elapsed_us()))
        us.sort()
        print(f"LATENCY e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, L2 "
              f"emptied, slowest GPU)", flush=True)
        if a.profile:
            from test_glm_moe_layer import node_times
            node_times(h, paths, compiler.graph_nodes(mpk))
        if any("STATIC_GME_TIMES" in fl for fl in a.flag):   # GPU 0's waves, us after its start barrier (the last launch)
            import numpy as np
            st = int([b for b, _ in h.timing()][0])
            tm = ws[0]["gme_dbg"].cpu().view(torch.int64).view(1024, 8).numpy().astype(np.float64)
            np.save(os.path.join(a.out, "gme_times.npy"), tm - st)   # (ns after GPU 0's start barrier, per wave gw = wave NCU + cu)
            ok = tm[:, 1] > 0
            rel = (tm - st) / 1e3
            q = lambda v: f"{np.percentile(v, 5):.1f} / {np.median(v):.1f} / {v.max():.1f}" if len(v) else "-"
            print(f"GME_TIMES GPU 0 ({ok.sum()} waves with a W13 item; 5 % / median / max us): table built {q(rel[tm[:, 0] > 0, 0])}; "
                  f"first item issued {q(rel[ok, 1])}; its sums done {q(rel[ok, 2])}; W13 end counted {q(rel[tm[:, 3] > 0, 3])}", flush=True)
            w2 = tm[:, 6] > 0
            if w2.any():
                print(f"GME_TIMES GPU 0 W2 ({w2.sum()} waves): first W2 item past its wait {q(rel[w2, 4])}; last item done "
                      f"{q(rel[w2, 5])}", flush=True)
    print("CHECKS " + ("PASS" if not failed else f"FAILED {failed}"), flush=True)


if __name__ == "__main__":
    main()
