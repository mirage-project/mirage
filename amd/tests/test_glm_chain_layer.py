"""N consecutive GLM-5.3-Flash KDA + MoE layers as ONE kernel (a chain of test_glm_pair_layer.py's pairs): per layer j the KDA
sublayer (projection, glm_kda_decode, o_proj push_all, hc_post_mix, hc_pre -> x_ffn_j) then the MoE sublayer (router, topk,
glm_moe_experts, allreduce_send, hc_post_mix, hc_pre -> layer j + 1's x_attn), the server's own weight loaders
(amd/serving/glm/sglang_glm.py). A timing probe (outputs finite and equal on every GPU), or --codegen [--serving]: the rank
library's layer.cu (kind gkdamoe<N>: per-layer tensors suffixed _j; the streams between the sublayers are run tensors R_a_j,
h_post_a_j, h_res_a_j (after layer j's KDA part) and R_b_j, .. (after its MoE part, j < N - 1)).

    python amd/tests/test_glm_chain_layer.py --tokens 2 --layers 3 [--layer 4]
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
    p.add_argument("--tokens", type=int, default=2)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--layer", type=int, default=4, help="the first layer (KDA attention, MoE MLP: 4..6, 8..10, ..)")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--rpt", type=int, default=0, help="requests per glm_kda_decode task (0: 4 at T 64, 2 at T 32, else 1)")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_chain")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--serving", action="store_true")
    a = p.parse_args()
    T, n, NL_ = a.tokens, a.gpus, a.layers
    os.environ["MPK_TOKENS"] = str(T)
    # (a chain's graph has more than 64 buffer slots / maps: the kernel and the hosts take up to 128 slots - G::par_mask_hi)
    os.environ.setdefault("MPK_HIP_MAX_BUFS", "128")
    os.environ.setdefault("MPK_HIP_MAX_MAPS", "128")
    if not any("STATIC_MAX_RESETS" in f for f in a.flag):   # (reset buffers: ~12 per layer, core.cuh MAX_RESETS 24 by default)
        a.flag.append("-DSTATIC_MAX_RESETS=48")
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
    ws = []
    HD = None
    slots_n = T + 3
    slots = torch.randperm(slots_n, generator=gen)[:T].to(torch.int32)
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        z = lambda *sh, dt=torch.float32: torch.zeros(*sh, dtype=dt, device=dev)
        w = {"x_attn": x_attn.to(dev), "slots": slots.to(dev), "R_in": R_in.to(dev), "h_post_in": hp.to(dev),
             "h_res_in": hr.to(dev), "R_out": z(T, HC * H, dt=torch.bfloat16), "h_post_out": z(T, HC), "h_res_out": z(T, HC * HC),
             "x_attn_next": z(T, H, dt=torch.bfloat16)}
        for j in range(NL_):
            L = a.layer + j
            kw, HD = sg._kda_weights(a.model_dir, L, g, n, dev)
            kw.update(sg._mhc(sk, L, "ffn", dev)[0])
            kw.update(sg._moe_weights(sk, L, g, n, dev))
            kw.update(sg._mhc(sk, L + 1, "attn", dev)[0])
            w.update({f"{k}_{j}": v for k, v in kw.items()})
            HL = HD // D
            w.update({f"state_pool_{j}": torch.randn(slots_n * HL * 128, 128, generator=gen).mul(0.05).to(dev),
                      f"conv_pool_{j}": rnd(slots_n * 3, 3 * HD).to(dev),
                      f"kda_y_{j}": z(T, HD, dt=torch.bfloat16), f"x_ffn_{j}": z(T, H, dt=torch.bfloat16),
                      f"pairs_{j}": torch.zeros(T, TOPK + 1, dtype=torch.int64, device=dev),
                      f"R_moe_{j}": z(T * (TOPK + 1), H),
                      f"R_a_{j}": z(T, HC * H, dt=torch.bfloat16), f"h_post_a_{j}": z(T, HC), f"h_res_a_{j}": z(T, HC * HC)})
            if j < NL_ - 1:
                w.update({f"R_b_{j}": z(T, HC * H, dt=torch.bfloat16), f"h_post_b_{j}": z(T, HC), f"h_res_b_{j}": z(T, HC * HC),
                          f"x_attn_{j + 1}": z(T, H, dt=torch.bfloat16)})
            if any("STATIC_GME_TIMES" in f for f in a.flag):   # (per layer: per wave 8 times, test_glm_moe_layer.py's print)
                w[f"gme_dbg_{j}"] = torch.zeros(1024 * 8 * 2, dtype=torch.int32, device=dev)
        if os.environ.get("MPK_CHAIN_HYB", "0") == "1":   # (hybrid gemms need gemm_tile's shuffled weight layout: layers 1 ..)
            from mirage.mpk.static_megakernel import shuffle_gemm_weight
            for j in range(1, NL_):
                w[f"wproj_{j}"] = shuffle_gemm_weight(w[f"wproj_{j}"])
        ws.append(w)
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    # (the tensors only this kernel uses: node buffers inside it; the streams between sublayers are graph tensors: run tensors)
    SCRATCH = [k for k in ws[0] if k.split("_")[0] in ("kda", "pairs") or k.startswith(("x_ffn_", "R_moe_"))
               or (k.startswith("x_attn_") and k != "x_attn_next")] if a.serving else []
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    hand = os.environ.get("MPK_ARS_HAND", "1") == "1" and T >= int(os.environ.get("MPK_ARS_HAND_MIN_T", "4"))
    sends, hyb_nodes = [], []
    for j in range(NL_):
        s = lambda k: at[f"{k}_{j}"]
        xin = at["x_attn"] if j == 0 else at[f"x_attn_{j}"]
        rin, hpi, hri = (at["R_in"], at["h_post_in"], at["h_res_in"]) if j == 0 else \
            (at[f"R_b_{j - 1}"], at[f"h_post_b_{j - 1}"], at[f"h_res_b_{j - 1}"])
        proj = mpk.new_tensor(dims=(T, ws[0][f"wproj_{j}"].shape[0]), dtype=float32, name=f"proj_{j}")
        sent_a = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name=f"o_sent_{j}")
        mix_a = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name=f"ffn_hc_mix_{j}")
        # MPK_CHAIN_HYB=1 (probe): layers 1 .. of the chain project with a hybrid gemm (128-row tasks, (N / 128, 4) grid): its tasks
        # wait on the previous layer's hc_pre with their weights already in LDS / registers (issued during that layer's tail)
        hyb = os.environ.get("MPK_CHAIN_HYB", "0") == "1" and j > 0
        if hyb:
            mpk.gemm_weight_layout = "shuffled"
        mpk.gemm_tile_layer(input=xin, weight=s("wproj"), output=proj, combine="slots", **({"hybrid": True} if hyb else {}))
        mpk.gemm_weight_layout = "rowmajor"
        if hyb:
            hyb_nodes.append(compiler.graph_nodes(mpk)[-1].graph_idx)
        mpk.glm_kda_decode_layer(proj=proj, wfb=s("wfb"), wgb=s("wgb"), prm=s("prm"), state_pool=s("state_pool"),
                                 output=s("kda_y"), slots=at["slots"], conv_weight=s("conv_w"), conv_pool=s("conv_pool"),
                                 col_ga=3 * HD, col_fa=3 * HD + 128, col_beta=3 * HD + 256, requests_per_task=rpt,
                                 onorm_eps=rms_eps)
        # (MPK_OPROJ_FIN, default 1 at T >= 32 as test_glm_pair_layer.py: o_proj as finish_all in MPK_STORE_KP (4) K parts)
        if os.environ.get("MPK_OPROJ_FIN", "1" if T >= 32 else "0") == "1":
            os.environ.setdefault("MPK_STORE_KP", "4")
            mpk.gemm_tile_layer(input=s("kda_y"), weight=s("w_o"), output=sent_a, combine="finish_all", input_polled=True)
        else:
            mpk.gemm_tile_layer(input=s("kda_y"), weight=s("w_o"), output=sent_a, combine="push_all", input_polled=True)
        mpk.hc_post_mix_layer(sent=sent_a, residual=rin, coef=(hpi, hri), fn=s("hc_ffn_fn"), output=s("R_a"), mix=mix_a)
        mpk.hc_pre_layer(mix=mix_a, residual=s("R_a"), hc_small=s("hc_ffn_small"), gamma=s("ffn_norm"), output=s("x_ffn"),
                         rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                         export=(s("h_post_a"), s("h_res_a")))
        last = j == NL_ - 1
        rout, hpo, hro, xo = (at["R_out"], at["h_post_out"], at["h_res_out"], at["x_attn_next"]) if last else \
            (s("R_b"), s("h_post_b"), s("h_res_b"), at[f"x_attn_{j + 1}"])
        logits = mpk.new_tensor(dims=(T, ws[0][f"router_{j}"].shape[0]), dtype=float32, name=f"router_logits_{j}")
        sent_b = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name=f"moe_sent_{j}")
        mix_b = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name=f"attn_hc_mix_{j}")
        mpk.gemm_tile_layer(input=s("x_ffn"), weight=s("router"), output=logits, combine="slots")
        mpk.glm_topk_route_layer(input=logits, bias=s("bias"), output=s("pairs"), k=TOPK, shared_experts=1, shared_first=E,
                                 routed_scaling_factor=SCALE)
        mpk.glm_moe_experts_layer(x=s("x_ffn"), pairs=s("pairs"), w13=s("w13"), w13_sf=s("w13_sf"), w2=s("w2"),
                                  w2_sf=s("w2_sf"), output=s("R_moe"), swiglu_limit=limit, dbg=at.get(f"gme_dbg_{j}"))
        mpk.allreduce_send_layer(input=s("R_moe"), output=sent_b, num_tasks=H // 32 if hand else None,
                                 prefetch=(s("R_a"), s("hc_attn_fn")) if hand and os.environ.get("MPK_ARS_PREFETCH") == "1" else None)
        mpk.hc_post_mix_layer(sent=sent_b, residual=s("R_a"), coef=(s("h_post_a"), s("h_res_a")), fn=s("hc_attn_fn"),
                              output=rout, mix=mix_b)
        mpk.hc_pre_layer(mix=mix_b, residual=rout, hc_small=s("hc_attn_small"), gamma=s("attn_norm"), output=xo,
                         rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters, export=(hpo, hro))
    nodes = compiler.graph_nodes(mpk)
    grids = wide_gemm_grids(mpk, 256)
    for q in nodes:   # (MPK_CHAIN_HYB: hybrid = 128-row tasks, 4 K parts)
        if q.graph_idx in hyb_nodes:
            grids[q.graph_idx] = (q.params[2] // 128, 4, 1)
    for q in nodes:   # (the routers: the MoE harness's default grid)
        if q.name == "gemm_tile" and q.params[2] == ws[0]["router_0"].shape[0]:
            grids[q.graph_idx] = (12, 8, 1)
    plan = simple_plan(mpk, grids)
    if hand:
        from test_glm_moe_layer import pair_send_postmix
        for snd in [q.graph_idx for q in nodes if q.name == "allreduce_send"]:
            pm = min(q.graph_idx for q in nodes if q.name == "hc_post_mix" and q.graph_idx > snd)
            plan = pair_send_postmix(plan, snd, pm)
    if "MPK_SPLIT_LOOP" not in os.environ:   # (one task loop per sublayer: the first node after every hc_pre but the last)
        pres = sorted(q.graph_idx for q in nodes if q.name == "hc_pre")[:-1]
        splits = [min(q.graph_idx for q in nodes if q.graph_idx > hp_) for hp_ in pres]
        os.environ["MPK_SPLIT_LOOP"] = ",".join(str(x) for x in splits)
        print(f"MPK_SPLIT_LOOP={os.environ['MPK_SPLIT_LOOP']}", flush=True)
    print("nodes:", len(nodes), flush=True)
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
    outs = ("R_out", "h_post_out", "h_res_out", "x_attn_next")
    fin = all(bool(torch.isfinite(ws[g][k].float()).all()) for g in range(n) for k in outs)
    same = all(torch.equal(ws[g][k].cpu(), ws[0][k].cpu()) for g in range(1, n) for k in outs)
    print(f"{'ok  ' if fin else 'FAIL'} outputs finite; {'ok  ' if same else 'FAIL'} every GPU's outputs equal", flush=True)
    for j in range(NL_):   # (each layer's experts used and the most tokens one expert has: the chain's later layers route their own x)
        if f"pairs_{j}" in ws[0]:
            pr = ws[0][f"pairs_{j}"].cpu().contiguous().view(torch.int64).flatten()
            ex = ((pr >> 32) & 0xFFFF).tolist()
            cnt = {}
            for e in ex:
                cnt[e] = cnt.get(e, 0) + 1
            print(f"layer {j}: experts used (incl. shared) {len(cnt)}, most tokens per routed expert "
                  f"{max(v for e, v in cnt.items() if e != max(cnt))}", flush=True)
    # (a cross-variant check: the same seeded inputs and launch count give close sums - e.g. MPK_CHAIN_HYB vs default)
    print("CHECKSUM " + " ".join(f"{k} {ws[0][k].float().abs().sum().item():.6e}" for k in ("x_attn_next", "R_out")), flush=True)
    us = []
    for _ in range(a.reps):
        h.launch(512 << 20, timeout_s=60)
        us.append(max(h.sk.elapsed_us()))
    us.sort()
    print(f"LATENCY chain {NL_} T {T} e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, "
          f"L2 emptied, slowest GPU)", flush=True)
    if any("STATIC_GME_TIMES" in f for f in a.flag):   # per layer, GPU 0's waves, us after its start barrier (the last launch)
        import numpy as np
        st = int([b for b, _ in h.timing()][0])
        q = lambda v: f"{np.percentile(v, 5):.1f} / {np.median(v):.1f} / {v.max():.1f}"
        for j in range(NL_):
            tm = ws[0][f"gme_dbg_{j}"].cpu().view(torch.int64).view(1024, 8).numpy().astype(np.float64)
            ok, rel, w2 = tm[:, 1] > 0, (tm - st) / 1e3, tm[:, 6] > 0
            print(f"GME_TIMES layer {j} (5 % / median / max us): table {q(rel[:, 0])}; first item issued {q(rel[ok, 1])}; W13 end "
                  f"{q(rel[:, 3])}; first W2 past its wait {q(rel[w2, 4])}; last W2 done {q(rel[w2, 5])}; W2 per item "
                  f"{q((rel[w2, 5] - rel[w2, 4]) / tm[w2, 6])}", flush=True)
    if a.profile:
        from test_glm_moe_layer import node_times
        node_times(h, paths, compiler.graph_nodes(mpk))


if __name__ == "__main__":
    main()
