"""Timing probe: one GLM-5.3-Flash layer as ONE kernel - the KDA sublayer (projection, glm_kda_decode, o_proj push_all,
hc_post_mix, hc_pre -> x_ffn) followed by the MoE sublayer (router, topk, glm_moe_experts, allreduce_send, hc_post_mix,
hc_pre -> the next layer's x_attn) - vs each sublayer as its own kernel (--mode kda / moe), with the server's own weight
loaders (amd/serving/glm/sglang_glm.py). Checks only that the outputs are finite and equal on every GPU (a timing probe; the
sublayers' numerics are checked by test_glm_kda_layer.py / test_glm_moe_layer.py).

    python amd/tests/test_glm_pair_layer.py --tokens 8 --mode pair|kda|moe [--layer 4]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "amd", "serving", "glm"))
os.environ.setdefault("MPK_GLM", "")
import sglang_glm as sg  # noqa: E402

H, HC, D = 4096, 4, 128
TOPK, E, SCALE = 8, 288, 2.5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--layer", type=int, default=4, help="the KDA layer; the MoE sublayer is that layer's MLP")
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--mode", default="pair", choices=("pair", "kda", "moe"))
    p.add_argument("--mlp", default="moe", choices=("moe", "dense"), help="the MLP sublayer: MoE, or the dense MLP of layers 0..2 "
                   "(--layer 0; mode pair only; rank kind gkdadense)")
    p.add_argument("--last", action="store_true", help="the model's last layer (mode pair, MoE): no final hc_pre - nothing reads the "
                   "next layer's x / coefficients, only R_out (rank kind gkdamoelast)")
    p.add_argument("--rpt", type=int, default=0, help="requests per glm_kda_decode task (0: 4 at T 64, 2 at T 32, else 1)")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_pair")
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--codegen", action="store_true", help="generate + compile layer.cu only (CPU; the rank library's input)")
    p.add_argument("--serving", action="store_true", help="with --codegen: the server library's graph (scratch tensors inside: "
                   "kda_y, x_ffn, pairs, R_moe; run tensors x_attn, state_pool, conv_pool, slots, x_attn_next, R_mid, h_post_mid, "
                   "h_res_mid, mHC io)")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    os.environ["MPK_TOKENS"] = str(T)
    if a.codegen:
        from test_glm_mhc import cpu_standins
        cpu_standins()
    rpt = a.rpt or (4 if T >= 64 else 2 if T >= 32 else 1)
    from mirage.core import bfloat16, float32, int32, int64  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel
    from plan_one_gpu import simple_plan
    from glm_plan import wide_gemm_grids
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    rms_eps, hc_eps, iters, limit = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"], cfg["swiglu_limit"]
    sk = sg._Slices(a.model_dir)
    gen = torch.Generator().manual_seed(29)
    rnd = lambda *sh: torch.randn(*sh, generator=gen).to(torch.bfloat16)
    x_attn, R_in = rnd(T, H), rnd(T, HC * H)
    hp, hr = (torch.rand(T, HC, generator=gen) * 2).float(), torch.rand(T, HC * HC, generator=gen).float()
    x_ffn0 = rnd(T, H)
    if os.environ.get("MPK_LOAD_XFFN"):   # (moe mode on the pair's own x_ffn, saved by MPK_SAVE_XFFN: the same routing)
        x_ffn0 = torch.load(os.environ["MPK_LOAD_XFFN"]).to(torch.bfloat16).cpu()
    ws = []
    HD = None
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        w = {}
        if a.mode in ("pair", "kda"):
            kw, HD = sg._kda_weights(a.model_dir, a.layer, g, n, dev)
            w.update(kw)
            w.update(sg._mhc(sk, a.layer, "ffn", dev)[0])
            HL = HD // D
            slots_n = T + 3
            w.update({"state_pool": torch.randn(slots_n * HL * 128, 128, generator=gen).mul(0.05).to(dev),
                      "conv_pool": rnd(slots_n * 3, 3 * HD).to(dev),
                      "slots": torch.randperm(slots_n, generator=gen)[:T].to(torch.int32).to(dev),
                      "kda_y": torch.zeros(T, HD, dtype=torch.bfloat16, device=dev),
                      "x_attn": x_attn.to(dev)})
        if a.mode == "pair" and a.mlp == "dense":
            w.update(sg._dense_weights(sk, a.layer, g, n, dev)[0])
            w.update(sg._mhc(sk, a.layer + 1, "attn", dev)[0])
            w.update({"h_mlp": torch.zeros(T, w["w_dn"].shape[1], dtype=torch.bfloat16, device=dev),
                      "x_attn_next": torch.zeros(T, H, dtype=torch.bfloat16, device=dev)})
        elif a.mode in ("pair", "moe"):
            w.update(sg._moe_weights(sk, a.layer, g, n, dev))
            w.update(sg._mhc(sk, a.layer + 1, "attn", dev)[0])
            w.update({"pairs": torch.zeros(T, TOPK + 1, dtype=torch.int64, device=dev),
                      "R_moe": torch.zeros(T * (TOPK + 1), H, dtype=torch.float32, device=dev),
                      "x_attn_next": torch.zeros(T, H, dtype=torch.bfloat16, device=dev)})
        # the residual streams and coefficients: in -> (KDA) mid -> (MoE) out
        z = lambda *sh, dt=torch.float32: torch.zeros(*sh, dtype=dt, device=dev)
        w.update({"R_in": R_in.to(dev), "h_post_in": hp.to(dev), "h_res_in": hr.to(dev),
                  "R_mid": z(T, HC * H, dt=torch.bfloat16), "h_post_mid": z(T, HC), "h_res_mid": z(T, HC * HC),
                  "R_out": z(T, HC * H, dt=torch.bfloat16), "h_post_out": z(T, HC), "h_res_out": z(T, HC * HC),
                  "x_ffn": (x_ffn0.to(dev) if a.mode == "moe" else z(T, H, dt=torch.bfloat16))})
        if a.last:   # (the last layer: the mean of R' over the streams, SGLang's hc_contract)
            w["hfin"] = z(T, H, dt=torch.bfloat16)
        ws.append(w)
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    # --serving: the tensors only this kernel uses stay inside it (node buffers reset every launch; bound as run tensors they
    # would keep the last launch's values for a polled reader)
    SCRATCH = ([k for k in ("kda_y", "x_ffn", "pairs", "R_moe", "h_mlp") if k in ws[0]]   # (R_mid, h_*_mid: graph tensors, run tensors)
               if a.serving and a.mode == "pair" else [])
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    if a.mode in ("pair", "kda"):
        PROJ_N = ws[0]["wproj"].shape[0]
        proj = mpk.new_tensor(dims=(T, PROJ_N), dtype=float32, name="proj")
        sent_a = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="o_sent")
        mix_a = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="ffn_hc_mix")
        mpk.gemm_tile_layer(input=at["x_attn"], weight=at["wproj"], output=proj, combine="slots")
        mpk.glm_kda_decode_layer(proj=proj, wfb=at["wfb"], wgb=at["wgb"], prm=at["prm"], state_pool=at["state_pool"],
                                 output=at["kda_y"], slots=at["slots"], conv_weight=at["conv_w"], conv_pool=at["conv_pool"],
                                 col_ga=3 * HD, col_fa=3 * HD + 128, col_beta=3 * HD + 256, requests_per_task=rpt,
                                 onorm_eps=rms_eps)
        # MPK_GLM_OPROJ_AR=1 (probe): o_proj as a polled store gemm (fp32 local partial) + allreduce_send in H / 32 slices (lands
        # and adds the sums)
        oproj_ar = os.environ.get("MPK_GLM_OPROJ_AR", "0") == "1" and T >= 16
        # MPK_OPROJ_FIN (default 1 at T >= 32): o_proj as gemm_tile finish_all in MPK_STORE_KP (default 4) K parts - each task 1 / 4 of
        # the activation, the row block's last part pushes the sums (MI355X 11-09 glmkda26, KDA layer: T 64 106.4 / 107.1 vs push_all
        # 113.0 / 114.1 us, T 32 80.8 / 80.6 vs 82.9 / 82.7, T 16 slower)
        if os.environ.get("MPK_OPROJ_FIN", "1" if T >= 32 and not oproj_ar else "0") == "1":
            os.environ.setdefault("MPK_STORE_KP", "4")
            mpk.gemm_tile_layer(input=at["kda_y"], weight=at["w_o"], output=sent_a, combine="finish_all", input_polled=True)
        elif oproj_ar:
            opart = mpk.new_tensor(dims=(T, H), dtype=float32, name="o_part")
            mpk.gemm_tile_layer(input=at["kda_y"], weight=at["w_o"], output=opart, combine="store", input_polled=True)
            mpk.allreduce_send_layer(input=opart, output=sent_a, num_tasks=H // 32)
        else:
            mpk.gemm_tile_layer(input=at["kda_y"], weight=at["w_o"], output=sent_a, combine="push_all", input_polled=True)
        rout, hpo, hro = (("R_mid", "h_post_mid", "h_res_mid") if a.mode == "pair" else ("R_out", "h_post_out", "h_res_out"))
        mpk.hc_post_mix_layer(sent=sent_a, residual=at["R_in"], coef=(at["h_post_in"], at["h_res_in"]), fn=at["hc_ffn_fn"],
                              output=at[rout], mix=mix_a)
        mpk.hc_pre_layer(mix=mix_a, residual=at[rout], hc_small=at["hc_ffn_small"], gamma=at["ffn_norm"],
                         output=at["x_ffn"], rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                         export=(at[hpo], at[hro]))
    hand = False
    if a.mode == "pair" and a.mlp == "dense":   # (test_glm_dense_layer.py's graph: gate/up slots gemm, SwiGLU, down push_all, mHC)
        F = ws[0]["w_dn"].shape[1]
        gup = mpk.new_tensor(dims=(T, 2 * F), dtype=float32, name="gate_up")
        sent_b = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="mlp_sent")
        mix_b = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="attn_hc_mix")
        mpk.gemm_tile_layer(input=at["x_ffn"], weight=at["w_gu"], output=gup, combine="slots")
        mpk.glm_swiglu_mul_layer(gate_up=gup, output=at["h_mlp"], swiglu_limit=10.0)
        # MPK_DENSE_FIN (default 1 at T >= 16; MI355X 11-09 glmpair19 / 20, checks pass: dense layer T 64 79.4 vs 90.4 us, T 32 63.4 vs 68.4;
        # pair T 16 116.1 / 116.2 vs 119.4 / 118.0, T 8 level;
        # KDA + dense pair T 64 168.9 / 172.4 vs 180.0 / 179.9, T 32 130.8 / 128.4 vs 135.8 / 134.1): the down gemm as finish_all in
        # MPK_DENSE_KP (default 4) K parts, as the KDA o_proj at T >= 32
        # (push_all: each of its 256 tasks loads all of h_mlp, [T][3072] bf16 = 384 KB at T 64; MI355X 11-09 glmprof35 gkdadense T 64: the
        # down gemm ends 22 us after the SwiGLU, its all-reduce lands 28.8 us later)
        if os.environ.get("MPK_DENSE_FIN", "1" if T >= 16 else "0") == "1":
            kp_old = os.environ.get("MPK_STORE_KP")
            os.environ["MPK_STORE_KP"] = os.environ.get("MPK_DENSE_KP", "4")
            mpk.gemm_tile_layer(input=at["h_mlp"], weight=at["w_dn"], output=sent_b, combine="finish_all", input_polled=True)
            if kp_old is None:
                os.environ.pop("MPK_STORE_KP")
            else:
                os.environ["MPK_STORE_KP"] = kp_old
        else:
            mpk.gemm_tile_layer(input=at["h_mlp"], weight=at["w_dn"], output=sent_b, combine="push_all", input_polled=True)
        mpk.hc_post_mix_layer(sent=sent_b, residual=at["R_mid"], coef=(at["h_post_mid"], at["h_res_mid"]), fn=at["hc_attn_fn"],
                              output=at["R_out"], mix=mix_b)
        mpk.hc_pre_layer(mix=mix_b, residual=at["R_out"], hc_small=at["hc_attn_small"], gamma=at["attn_norm"],
                         output=at["x_attn_next"], rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                         export=(at["h_post_out"], at["h_res_out"]))
    elif a.mode in ("pair", "moe"):
        NL = ws[0]["router"].shape[0]
        logits = mpk.new_tensor(dims=(T, NL), dtype=float32, name="router_logits")
        sent_b = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="moe_sent")
        mix_b = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="attn_hc_mix")
        rin, hpi, hri = (("R_mid", "h_post_mid", "h_res_mid") if a.mode == "pair" else ("R_in", "h_post_in", "h_res_in"))
        mpk.gemm_tile_layer(input=at["x_ffn"], weight=at["router"], output=logits, combine="slots")
        mpk.glm_topk_route_layer(input=logits, bias=at["bias"], output=at["pairs"], k=TOPK, shared_experts=1, shared_first=E,
                                 routed_scaling_factor=SCALE)
        mpk.glm_moe_experts_layer(x=at["x_ffn"], pairs=at["pairs"], w13=at["w13"], w13_sf=at["w13_sf"], w2=at["w2"],
                                  w2_sf=at["w2_sf"], output=at["R_moe"], swiglu_limit=limit)
        # (the MoE harness's default at T >= 16: H / 32 send slices + hc_post_mix on the send's CU, STATIC_ARS_LDSHAND)
        hand = os.environ.get("MPK_ARS_HAND", "1") == "1" and T >= int(os.environ.get("MPK_ARS_HAND_MIN_T", "4"))
        mpk.allreduce_send_layer(input=at["R_moe"], output=sent_b, num_tasks=H // 32 if hand else None,
                                 prefetch=(at[rin], at["hc_attn_fn"]) if hand and os.environ.get("MPK_ARS_PREFETCH") == "1" else None)
        mpk.hc_post_mix_layer(sent=sent_b, residual=at[rin], coef=(at[hpi], at[hri]), fn=at["hc_attn_fn"],
                              output=at["R_out"], mix=mix_b, contract=at["hfin"] if a.last else None)
        if not a.last:
            mpk.hc_pre_layer(mix=mix_b, residual=at["R_out"], hc_small=at["hc_attn_small"], gamma=at["attn_norm"],
                             output=at["x_attn_next"], rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                             export=(at["h_post_out"], at["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    grids = wide_gemm_grids(mpk, 256)
    for q in nodes:   # (the router: the MoE harness's default grid)
        if q.name == "gemm_tile" and q.params[2] == ws[0].get("router", torch.zeros(1, 1)).shape[0]:
            grids[q.graph_idx] = tuple(int(v) for v in os.environ.get("MPK_ROUTER_GRID", "12,8,1").split(","))   # (probe override)
    plan = simple_plan(mpk, grids)
    if a.mode in ("pair", "moe") and a.mlp == "moe" and hand:
        from test_glm_moe_layer import pair_send_postmix
        snd = max(q.graph_idx for q in nodes if q.name == "allreduce_send")   # (the MoE part's send: the last one)
        pm = next(q.graph_idx for q in nodes if q.name == "hc_post_mix" and q.graph_idx > snd)
        plan = pair_send_postmix(plan, snd, pm)
    print("nodes:", [(q.graph_idx, q.name, q.grid) for q in compiler.graph_nodes(mpk)], flush=True)
    if a.mode == "pair" and "MPK_SPLIT_LOOP" not in os.environ:   # (one task loop per sublayer: glmpair2 -6..-17 us vs one loop)
        first_moe = min(q.graph_idx for q in nodes if q.graph_idx > next(
            r.graph_idx for r in nodes if r.name == "hc_pre"))
        os.environ["MPK_SPLIT_LOOP"] = str(first_moe)
        print(f"MPK_SPLIT_LOOP={first_moe} (the MoE part's first node)", flush=True)
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
    skc = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(skc)
    for _ in range(2):
        h.launch(512 << 20, timeout_s=60)
    outs = ("R_out", "hfin") if a.last else ("R_out", "h_post_out", "h_res_out", "x_attn_next" if a.mode != "kda" else "x_ffn")
    fin = all(bool(torch.isfinite(ws[g][k].float()).all()) for g in range(n) for k in outs)
    same = all(torch.equal(ws[g][k].cpu(), ws[0][k].cpu()) for g in range(1, n) for k in outs)
    print(f"{'ok  ' if fin else 'FAIL'} outputs finite; {'ok  ' if same else 'FAIL'} every GPU's outputs equal", flush=True)
    if a.mode == "pair" and os.environ.get("MPK_SAVE_XFFN"):
        torch.save(ws[0]["x_ffn"].cpu(), os.environ["MPK_SAVE_XFFN"])
    if a.mode != "kda" and "pairs" in ws[0] and a.mlp == "moe":   # (the experts used: routing must match for a like-for-like comparison)
        pr = ws[0]["pairs"].cpu().contiguous().view(torch.int64).flatten()
        print(f"experts used (incl. shared): {len(set(((pr >> 32) & 0xFFFF).tolist()))}", flush=True)
    us = []
    for _ in range(a.reps):
        h.launch(512 << 20, timeout_s=60)
        us.append(max(h.sk.elapsed_us()))
    us.sort()
    print(f"LATENCY {a.mode} T {T} e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, "
          f"L2 emptied, slowest GPU)", flush=True)
    if a.profile:
        from test_glm_moe_layer import node_times
        node_times(h, paths, compiler.graph_nodes(mpk))


if __name__ == "__main__":
    main()
