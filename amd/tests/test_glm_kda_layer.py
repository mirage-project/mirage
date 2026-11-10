"""GLM-5.3-Flash KDA sublayer on AMD GPUs (TP4, plain decode): x_attn -> the merged projection (gemm_tile, K parts in slots)
-> glm_kda_decode (conv, gates, delta rule, gated o_norm per (head, request)) -> o_proj (push_all) -> hc_post_mix -> hc_pre
(x_ffn, the FFN's h_post / h_res). Real weights of KDA layer --layer (upstream demo/glm53_flash/kda_reference.py loads
them), random x_attn, streams, conv windows and states. Checked per launch against kda_reference.ref_kda (vLLM's glm5next
step with its bf16 rounding points) per GPU and the mHC references (test_glm_mhc.py): y (the gated o_norm output), the new
states and windows, R', x_ffn; launch k (states restored) bit-identical to launch 0; every GPU's mHC outputs equal.

    python amd/tests/test_glm_kda_layer.py --tokens 8 [--gpus 4] [--layer 4] [--proj-grid 51,4,1] [--rpt 1] [--codegen]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_glm_mhc import H, HC, bf, cpu_standins, hc_post_ref, hc_pre_ref, load_mhc, relrms  # noqa: E402

D = 128


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--proj-grid", default="", help="the projection's grid (default: glm_plan.wide_gemm_grids)")
    p.add_argument("--o-grid", default="", help="o_proj's grid (default: glm_plan.wide_gemm_grids)")
    p.add_argument("--gemm-cap", type=int, default=256, help="gemm_tile grids: the most tasks up to this (0: the layer defaults)")
    p.add_argument("--rpt", type=int, default=1, help="requests per glm_kda_decode task")
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--oproj-ar", action="store_true", help="o_proj as a store gemm + allreduce_send (instead of push_all)")
    p.add_argument("--oproj-slots", type=int, default=0, help="o_proj as a slots gemm in this many K parts (each task a 1 / P "
                   "share of the activation) + allreduce_send adding the parts locally before the one all-reduce")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_kda")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--profile", action="store_true", help="the timing build: per-task times per node (median over CUs)")
    p.add_argument("--serving", action="store_true", help="with --codegen: the server library's graph (scratch tensors inside)")
    p.add_argument("--codegen", action="store_true")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    os.environ["MPK_TOKENS"] = str(T)
    if a.codegen:
        cpu_standins()
    sys.path.insert(0, os.path.join(HERE, "..", "..", "demo", "glm53_flash"))
    import kda_reference as kref
    from mirage.core import bfloat16, float32, int32, int64  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel, shuffle_gemm_weight
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    rms_eps, hc_eps, iters = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"]
    W = kref.load_kda_layer(kref.Checkpoint(a.model_dir), a.layer)
    NH = W["num_heads"]
    HL, HD = NH // n, NH // n * D
    COL_GA, COL_FA, COL_BETA = 3 * HD, 3 * HD + 128, 3 * HD + 256
    PROJ_N = -(-(COL_BETA + HL) // 128) * 128
    m = load_mhc(a.model_dir, a.layer, "ffn")
    gen = torch.Generator().manual_seed(11)
    SLOTS = T + 3
    slots = torch.randperm(SLOTS, generator=gen)[:T].to(torch.int32)
    x = bf(torch.randn(T, H, generator=gen))
    R = bf(torch.randn(T, HC, H, generator=gen))
    hp = (torch.rand(T, HC, generator=gen) * 2).float()
    hr = torch.rand(T, HC * HC, generator=gen).float()
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = m["base"], m["scale"]
    ws, sh = [], []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        cs, hs = slice(g * HD, (g + 1) * HD), slice(g * HL, (g + 1) * HL)
        wproj = torch.zeros(PROJ_N, H, dtype=torch.bfloat16)
        wproj[:HD], wproj[HD:2 * HD], wproj[2 * HD:3 * HD] = W["q"][cs], W["k"][cs], W["v"][cs]
        wproj[COL_GA:COL_GA + 128], wproj[COL_FA:COL_FA + 128], wproj[COL_BETA:COL_BETA + HL] = W["g_a"], W["f_a"], W["b"][hs]
        w_o = W["o_proj"][:, cs].contiguous()
        rank = {"q": W["q"][cs], "k": W["k"][cs], "v": W["v"][cs], "b": W["b"][hs], "f_a": W["f_a"], "g_a": W["g_a"],
                "f_b": W["f_b"][cs], "g_b": W["g_b"][cs], "q_conv": W["q_conv"][cs], "k_conv": W["k_conv"][cs],
                "v_conv": W["v_conv"][cs], "A_log": W["A_log"][hs], "dt_bias": W["dt_bias"][cs], "o_norm": W["o_norm"],
                "o_proj": w_o, "num_heads": HL}
        sh.append(rank)
        pool = bf(torch.randn(SLOTS * 3, 3 * HD, generator=gen))
        state = torch.randn(SLOTS * HL * 128, 128, generator=gen) * 0.05
        ws.append({"x_attn": x.to(dev),
                   "wproj": (shuffle_gemm_weight(wproj) if a.gemm_shuffled else wproj).to(dev),
                   "wfb": rank["f_b"].contiguous().to(dev), "wgb": rank["g_b"].contiguous().to(dev),
                   "prm": torch.cat([rank["A_log"], rank["dt_bias"], rank["o_norm"]]).float().to(dev),
                   "state_pool": state.to(dev), "conv_pool": pool.to(dev), "slots": slots.to(dev),
                   "conv_w": torch.cat([rank["q_conv"], rank["k_conv"], rank["v_conv"]], 0).float().contiguous().to(dev),
                   "w_o": (shuffle_gemm_weight(w_o) if a.gemm_shuffled else w_o).to(dev),
                   "R_in": R.reshape(T, HC * H).to(dev), "h_post_in": hp.to(dev), "h_res_in": hr.to(dev),
                   "hc_ffn_fn": m["fn"].to(dev), "hc_ffn_small": hc_small.to(dev), "ffn_norm": m["gamma"].to(dev),
                   "kda_y": torch.zeros(T, HD, dtype=torch.bfloat16, device=dev),
                   "R_out": torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
                   "x_ffn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                   "h_post_out": torch.zeros(T, HC, dtype=torch.float32, device=dev),
                   "h_res_out": torch.zeros(T, HC * HC, dtype=torch.float32, device=dev)})
    init = [{k: ws[g][k].clone() for k in ("state_pool", "conv_pool")} for g in range(n)]
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        mpk.gemm_weight_layout = "shuffled"
    # --serving (the server libraries): the scratch tensors inside the kernel (node buffers the kernel resets every launch;
    # bound as run tensors they would keep the last launch's values: a polled reader could read them before the new ones)
    SCRATCH = ['kda_y'] if a.serving else []
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    proj = mpk.new_tensor(dims=(T, PROJ_N), dtype=float32, name="proj")
    y = at["kda_y"]   # (a graph tensor: the launch's copy of the node's buffer lands in it)
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="o_sent")
    mix = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="ffn_hc_mix")
    mpk.gemm_tile_layer(input=at["x_attn"], weight=at["wproj"], output=proj, combine="slots")
    mpk.glm_kda_decode_layer(proj=proj, wfb=at["wfb"], wgb=at["wgb"], prm=at["prm"], state_pool=at["state_pool"], output=y,
                             slots=at["slots"], conv_weight=at["conv_w"], conv_pool=at["conv_pool"], col_ga=COL_GA,
                             col_fa=COL_FA, col_beta=COL_BETA, requests_per_task=a.rpt, onorm_eps=rms_eps)
    # MPK_OPROJ_FIN (default 1 at T >= 32): o_proj as gemm_tile finish_all in MPK_STORE_KP (default 4) K parts - each task 1 / 4 of
    # the activation, the row block's last part pushes the sums (MI355X 11-09 glmkda26, KDA layer: T 64 106.4 / 107.1 vs push_all
    # 113.0 / 114.1 us, T 32 80.8 / 80.6 vs 82.9 / 82.7, T 16 slower)
    if os.environ.get("MPK_OPROJ_FIN", "1" if T >= 32 and not (a.oproj_ar or a.oproj_slots) else "0") == "1":
        os.environ.setdefault("MPK_STORE_KP", "4")
        mpk.gemm_tile_layer(input=y, weight=at["w_o"], output=sent, combine="finish_all", input_polled=True)
    elif a.oproj_slots:   # (a slots gemm, K parts P: allreduce_send polls and adds the parts, then lands the all-reduce)
        opart = mpk.new_tensor(dims=(T, H), dtype=float32, name="o_part")
        mpk.gemm_tile_layer(input=y, weight=at["w_o"], output=opart, combine="slots")   # (counted: polled slots need rows split over the GPUs)
        mpk.allreduce_send_layer(input=opart, output=sent)
    elif a.oproj_ar:   # (o_proj as a store gemm - an fp32 local partial - then allreduce_send, which lands and adds the sums
        # (from T STATIC_ARS_LAND_MIN_T): the MoE kernel's structure; default push_all)
        opart = mpk.new_tensor(dims=(T, H), dtype=float32, name="o_part")
        mpk.gemm_tile_layer(input=y, weight=at["w_o"], output=opart, combine="store",   # (MPK_OPROJ_AR_POLLED=0: counted input)
                            input_polled=os.environ.get("MPK_OPROJ_AR_POLLED", "1") == "1")
        # (H / 32 send slices at T >= 16: the sender holds its own partial, one vector per thread)
        mpk.allreduce_send_layer(input=opart, output=sent, num_tasks=H // 32 if T >= 16 else None)
    else:
        mpk.gemm_tile_layer(input=y, weight=at["w_o"], output=sent, combine="push_all", input_polled=os.environ.get("MPK_OPROJ_COUNTED", "0") != "1")
    mpk.hc_post_mix_layer(sent=sent, residual=at["R_in"], coef=(at["h_post_in"], at["h_res_in"]), fn=at["hc_ffn_fn"],
                          output=at["R_out"], mix=mix)
    mpk.hc_pre_layer(mix=mix, residual=at["R_out"], hc_small=at["hc_ffn_small"], gamma=at["ffn_norm"], output=at["x_ffn"],
                     rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                     export=(at["h_post_out"], at["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    from glm_plan import wide_gemm_grids
    grids = wide_gemm_grids(mpk, a.gemm_cap) if a.gemm_cap else {}
    if a.proj_grid:
        grids[nodes[0].graph_idx] = tuple(int(v) for v in a.proj_grid.split(","))
    if a.o_grid:
        grids[nodes[2].graph_idx] = tuple(int(v) for v in a.o_grid.split(","))
    plan = simple_plan(mpk, grids)
    print("nodes:", [(q.graph_idx, q.name, q.grid) for q in compiler.graph_nodes(mpk)], flush=True)
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
    # references per GPU (CPU, the initial states)
    refs = []
    for g in range(n):
        pool0 = init[g]["conv_pool"].cpu().view(SLOTS, 3, 3 * HD)
        st0 = init[g]["state_pool"].cpu().view(SLOTS, HL, 128, 128)
        conv_state = torch.stack([pool0[int(s)].t() for s in slots])          # [T, 3 HD, 3] (oldest first)
        rec = torch.stack([st0[int(s)] for s in slots])                       # [T, HL, V, K]
        _, yr, new_conv, new_rec = kref.ref_kda(x, sh[g], conv_state, rec, lower_bound=-5.0, norm_eps=rms_eps, tp=1)
        refs.append((yr, new_conv, new_rec))
    acc = torch.zeros(T, H)
    for g in range(n):
        acc = acc + bf(refs[g][0].float() @ sh[g]["o_proj"].float().t()).float()
    out = bf(acc)
    R2 = hc_post_ref(out, R, hp, hr)
    x_ref, _, _ = hc_pre_ref(R2, m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)

    def restore():
        for g in range(n):
            for k in ("state_pool", "conv_pool"):
                ws[g][k].copy_(init[g][k])
            torch.cuda.synchronize(g)
    first = None
    for launch in range(a.launches):
        restore()
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        for g in range(n):
            yk = ws[g]["kda_y"].cpu()
            yr, new_conv, new_rec = refs[g]
            pool1 = ws[g]["conv_pool"].cpu().view(SLOTS, 3, 3 * HD)
            st1 = ws[g]["state_pool"].cpu().view(SLOTS, HL, 128, 128)
            kc = torch.stack([pool1[int(s)].t() for s in slots])
            kr = torch.stack([st1[int(s)] for s in slots])
            if launch == 0 or g == 0:
                check(f"launch {launch} GPU {g}: y (gated o_norm)", relrms(yk, yr) < 2e-2,
                      f"relRMS {relrms(yk, yr):.2e}, bit-equal {(yk == bf(yr)).float().mean().item():.4f}")
                nc = new_conv.to(kc.dtype)
                check(f"launch {launch} GPU {g}: new conv windows", torch.equal(kc[:, :, :2], nc[:, :, :2]) and
                      relrms(kc[:, :, 2], nc[:, :, 2]) < 1e-2,
                      f"shifted entries bit-equal {torch.equal(kc[:, :, :2], nc[:, :, :2])}, the new input relRMS "
                      f"{relrms(kc[:, :, 2], nc[:, :, 2]):.2e} ({(kc[:, :, 2] == nc[:, :, 2]).float().mean().item():.4f} "
                      f"bit-equal; the projection's K parts summed in fp32 vs a bf16 matmul)")
                check(f"launch {launch} GPU {g}: new states", relrms(kr, new_rec) < 1e-3, f"relRMS {relrms(kr, new_rec):.2e}")
        G0 = {k: ws[0][k].cpu().clone() for k in ("R_out", "x_ffn", "h_post_out", "h_res_out")}
        R2k = G0["R_out"].view(T, HC, H)
        check(f"launch {launch}: R' vs the reference chain", relrms(R2k, R2) < 2e-2, f"relRMS {relrms(R2k, R2):.2e}")
        check(f"launch {launch}: x_ffn vs the reference chain", relrms(G0["x_ffn"], x_ref) < 3e-2,
              f"relRMS {relrms(G0['x_ffn'], x_ref):.2e}")
        same = all(torch.equal(ws[g][k].cpu(), G0[k]) for g in range(1, n) for k in G0)
        check(f"launch {launch}: every GPU's mHC outputs equal", same)
        if first is None:
            first = G0
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G0[k], first[k]) for k in G0))
    if (not failed or os.environ.get("MPK_TIME_FAILED") == "1") and a.reps:   # (MPK_TIME_FAILED=1: timing probes with wrong results)
        us = []
        for _ in range(a.reps):
            restore()
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
