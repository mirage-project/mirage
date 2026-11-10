"""GLM-5.3-Flash dense MLP sublayer (layers 0 .. 2) on AMD GPUs (TP4, plain decode): x_ffn -> gemm [gate | up] (this GPU's
F = 12288 / GPUs features, K parts in slots) -> glm_swiglu_mul (SiLU(min(g, 10)) clamp(u, -10, 10), SGLang swiglu_clamped)
-> gemm down (push_all) -> hc_post_mix (the next attention's mixes) -> hc_pre (x_attn of the next layer). Real weights of
layer --layer (FP8 128 x 128 blocks dequantized to bf16 for gemm_tile), random x_ffn and streams. Checked against a torch
reference with the kernel's rounding points: h (the kernel's gate / up rounded to bf16 vs a bf16 matmul), R', x_attn;
launch k bit-identical to launch 0; every GPU's mHC outputs equal.

    python amd/tests/test_glm_dense_layer.py --tokens 8 [--gpus 4] [--layer 0] [--codegen]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_glm_mhc import H, HC, bf, cpu_standins, hc_post_ref, hc_pre_ref, load_mhc, relrms  # noqa: E402

I_FULL, LIMIT = 12288, 10.0


def load_dense(model_dir: str, layer: int):
    """gate, up [12288, 4096], down [4096, 12288]: e4m3 with fp32 128 x 128 block scales, dequantized to fp32"""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    p = f"model.language_model.layers.{layer}.mlp."
    out = {}
    for k in ("gate_proj", "up_proj", "down_proj"):
        with safe_open(os.path.join(model_dir, idx[p + k + ".weight"]), "pt") as f:
            w = f.get_tensor(p + k + ".weight")
        with safe_open(os.path.join(model_dir, idx[p + k + ".weight_scale_inv"]), "pt") as f:
            s = f.get_tensor(p + k + ".weight_scale_inv").float()
        out[k] = w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:w.shape[0], :w.shape[1]]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--gemm-cap", type=int, default=256, help="gemm_tile grids: the most tasks up to this (glm_plan.py; 0: the layer defaults)")
    p.add_argument("--oproj-ar", action="store_true", help="o_proj as a store gemm + allreduce_send (instead of push_all)")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_dense")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--serving", action="store_true", help="with --codegen: the server library's graph (scratch tensors inside)")
    p.add_argument("--codegen", action="store_true")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    F = I_FULL // n
    os.environ["MPK_TOKENS"] = str(T)
    if a.codegen:
        cpu_standins()
    from mirage.core import bfloat16, float32, int32, int64  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    rms_eps, hc_eps, iters = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"]
    m = load_mhc(a.model_dir, a.layer + 1, "attn")   # (the next layer's attention pre)
    gen = torch.Generator().manual_seed(17)
    x = bf(torch.randn(T, H, generator=gen))
    R = bf(torch.randn(T, HC, H, generator=gen))
    hp = (torch.rand(T, HC, generator=gen) * 2).float()
    hr = torch.rand(T, HC * HC, generator=gen).float()
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = m["base"], m["scale"]
    Wd = {k: torch.zeros(I_FULL if k != "down_proj" else H, H if k != "down_proj" else I_FULL) for k in
          ("gate_proj", "up_proj", "down_proj")} if a.codegen else load_dense(a.model_dir, a.layer)
    ws, gus, dns = [], [], []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        fs = slice(g * F, (g + 1) * F)
        gu = bf(torch.cat([Wd["gate_proj"][fs], Wd["up_proj"][fs]]))
        dn = bf(Wd["down_proj"][:, fs].contiguous())
        gus.append(gu)
        dns.append(dn)
        ws.append({"x_ffn": x.to(dev), "w_gu": gu.to(dev), "w_dn": dn.to(dev),
                   "h_mlp": torch.zeros(T, F, dtype=torch.bfloat16, device=dev),
                   "R_in": R.reshape(T, HC * H).to(dev), "h_post_in": hp.to(dev), "h_res_in": hr.to(dev),
                   "hc_attn_fn": m["fn"].to(dev), "hc_attn_small": hc_small.to(dev), "attn_norm": m["gamma"].to(dev),
                   "R_out": torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
                   "x_attn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                   "h_post_out": torch.zeros(T, HC, dtype=torch.float32, device=dev),
                   "h_res_out": torch.zeros(T, HC * HC, dtype=torch.float32, device=dev)})
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    # --serving (the server libraries): the scratch tensors inside the kernel (node buffers the kernel resets every launch;
    # bound as run tensors they would keep the last launch's values: a polled reader could read them before the new ones)
    SCRATCH = ['h_mlp'] if a.serving else []
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    gup = mpk.new_tensor(dims=(T, 2 * F), dtype=float32, name="gate_up")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="mlp_sent")
    mix = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="attn_hc_mix")
    mpk.gemm_tile_layer(input=at["x_ffn"], weight=at["w_gu"], output=gup, combine="slots")
    mpk.glm_swiglu_mul_layer(gate_up=gup, output=at["h_mlp"], swiglu_limit=LIMIT)
    if a.oproj_ar:   # (o_proj as a store gemm - an fp32 local partial - then allreduce_send, which lands and adds the sums
        # (STATIC_ARS_LAND, T >= 32): the MoE kernel's structure; default push_all)
        opart = mpk.new_tensor(dims=(T, H), dtype=float32, name="o_part")
        mpk.gemm_tile_layer(input=at["h_mlp"], weight=at["w_dn"], output=opart, combine="store")
        mpk.allreduce_send_layer(input=opart, output=sent)
    elif os.environ.get("MPK_DENSE_FIN", "1" if T >= 16 else "0") == "1":   # (probe: the down gemm as finish_all in MPK_DENSE_KP K parts, as the pair harness)
        kp_old = os.environ.get("MPK_STORE_KP")
        os.environ["MPK_STORE_KP"] = os.environ.get("MPK_DENSE_KP", "4")
        mpk.gemm_tile_layer(input=at["h_mlp"], weight=at["w_dn"], output=sent, combine="finish_all", input_polled=True)
        if kp_old is None:
            os.environ.pop("MPK_STORE_KP")
        else:
            os.environ["MPK_STORE_KP"] = kp_old
    else:
        mpk.gemm_tile_layer(input=at["h_mlp"], weight=at["w_dn"], output=sent, combine="push_all", input_polled=os.environ.get("MPK_OPROJ_COUNTED", "0") != "1")
    mpk.hc_post_mix_layer(sent=sent, residual=at["R_in"], coef=(at["h_post_in"], at["h_res_in"]), fn=at["hc_attn_fn"],
                          output=at["R_out"], mix=mix)
    mpk.hc_pre_layer(mix=mix, residual=at["R_out"], hc_small=at["hc_attn_small"], gamma=at["attn_norm"], output=at["x_attn"],
                     rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                     export=(at["h_post_out"], at["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    from glm_plan import wide_gemm_grids
    plan = simple_plan(mpk, wide_gemm_grids(mpk, a.gemm_cap) if a.gemm_cap else {})
    print("nodes:", [(q.graph_idx, q.name, q.grid) for q in nodes], flush=True)
    os.makedirs(a.out, exist_ok=True)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    if a.codegen:
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q)) for q in paths], a.profile)
        cu = os.path.join(a.out, "layer.cu")
        open(cu, "w").write(code)
        r = subprocess.run(static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["--save-temps"],
                                                           arch="gfx950"), cwd=a.out, capture_output=True, text=True)
        print(f"hipcc exit {r.returncode}", r.stderr[-3000:] if r.returncode else "", flush=True)
        return
    from mirage.mpk.static_harness import Harness
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag,
                            profile=a.profile)
    h = Harness(sk)
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)

    def swiglu(gv, uv):
        gv, uv = gv.clamp(max=LIMIT), uv.clamp(-LIMIT, LIMIT)
        return gv * torch.sigmoid(gv) * uv
    first = None
    for launch in range(a.launches):
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        G = {k: ws[0][k].cpu().clone() for k in ("h_mlp", "R_out", "x_attn", "h_post_out", "h_res_out")}
        if launch == 0:
            hs = []
            for g in range(n):
                guv = bf(x.float() @ gus[g].float().t()).float()
                hr_ = bf(swiglu(guv[:, :F], guv[:, F:]))
                hk = ws[g]["h_mlp"].cpu()
                if g == 0:
                    check("h (GPU 0)", relrms(hk, hr_) < 1e-2, f"relRMS {relrms(hk, hr_):.2e}")
                hs.append(hk)
            acc = torch.zeros(T, H)
            for g in range(n):   # the chain from the kernel's h
                acc = acc + bf(hs[g].float() @ dns[g].float().t()).float()
            R2 = hc_post_ref(bf(acc), R, hp, hr)
            x_ref, _, _ = hc_pre_ref(R2, m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
        check(f"launch {launch}: R' vs the chain (the kernel's h)", relrms(G["R_out"].view(T, HC, H), R2) < 2e-2,
              f"relRMS {relrms(G['R_out'].view(T, HC, H), R2):.2e}")
        check(f"launch {launch}: x_attn vs the chain", relrms(G["x_attn"], x_ref) < 3e-2, f"relRMS {relrms(G['x_attn'], x_ref):.2e}")
        mh = ("R_out", "x_attn", "h_post_out", "h_res_out")
        check(f"launch {launch}: every GPU's mHC outputs equal", all(torch.equal(ws[g][k].cpu(), G[k]) for g in range(1, n)
                                                                     for k in mh))
        if first is None:
            first = G
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G[k], first[k]) for k in G))
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
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}", flush=True)


if __name__ == "__main__":
    main()
