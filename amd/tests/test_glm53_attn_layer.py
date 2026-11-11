"""GLM-5.3 attention sublayer on AMD GPUs (TP 8, plain decode), amd/notes/sglang_v0522_glm53_layer_contract.md sections 2, D:
x_attn -> q_a | kv_a (gemm_tile, K parts in slots) -> glm53_mla_norm (q_lora; [k_nope | rope(k_pe)] e4m3 into the MLA cache row
out_loc) -> q_b (local gemm_tile) -> glm53_q_absorb (q_nope . w_kc, rope(q_pe)) -> [a "full" layer: wk (gemm_tile slots) ->
glm53_index_k (index-K cache row, the dense row list)] -> glm53_sparse_attn (the listed rows; a "shared" layer: the list from a
graph tensor) -> dsa_o_absorb (w_vc) -> o_proj (store gemm in K parts) -> allreduce_send (its senders do the
residual add + post_attention_layernorm: the FFN input).
Real weights of layer --layer (dequantized to bf16; this GPU's 8 heads), random x / residual / earlier cache rows (--ctx per
request), sequence lengths <= 2048 (the dense list). Checked: the cache rows written (MLA and index-K), the list, q_lora, the
query, attn, the residual out and the FFN input vs a float reference of the same math; GPUs equal; relaunches bit-identical.

    python amd/tests/test_glm53_attn_layer.py --tokens 8 [--layer 6 (full) | 7 (shared)] [--ctx 200] [--codegen [--serving]]
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
from test_glm53_moe_layer import Ckpt  # noqa: E402

H, NHT, QL, KVL, NOPE, RP, DV, ID = 6144, 64, 2048, 512, 192, 64, 256, 128
K_TOP, THETA = 2048, 8000000.0


def deq(w, s):   # e4m3 [N, K] with fp32 128x128 block scales -> fp32
    return w.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:w.shape[0], :w.shape[1]]


def rope_tables(npos):
    """aiter RotaryEmbedding: inv_freq = 1 / theta^(2i / 64), cos / sin [npos][32] fp32 -> bf16"""
    inv = 1.0 / (THETA ** (torch.arange(0, RP, 2, dtype=torch.float32) / RP))
    f = torch.einsum("i,j->ij", torch.arange(npos, dtype=torch.float32), inv)
    return bf(f.cos()), bf(f.sin())


def rope_il(x, cos, sin):
    """interleaved rope of x [..., 64] (fp32) with cos / sin [..., 32] (fp32): pairs (2i, 2i + 1)"""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    o = torch.empty_like(x)
    o[..., 0::2] = x0 * cos - x1 * sin
    o[..., 1::2] = x1 * cos + x0 * sin
    return o


def f8(x):   # fp32 -> e4m3 -> fp32 (round to nearest even)
    return x.to(torch.float8_e4m3fn).float()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--model-dir", default="/dev/shm/glm53")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--ctx", type=int, default=200, help="earlier tokens per request (+ T r: requests differ)")
    p.add_argument("--splits", type=int, default=0, help="sparse-attention splits (0: 8 at T <= 8, else 4)")
    p.add_argument("--gemm-cap", type=int, default=256, help="wide gemm grids up to this many tasks (0: the defaults)")
    p.add_argument("--store-kp", type=int, default=4, help="o_proj's K parts")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--fp8", type=int, default=-1, help="the gemms on e4m3 weights + scales (gemm_tile GEMM_FP8; -1: glm53_layers.FP8)")
    p.add_argument("--out", default="glm53_attn")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--serving", action="store_true")
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    NH = NHT // n
    os.environ["MPK_TOKENS"] = str(T)
    sys.path.insert(0, os.path.join(HERE, "..", "serving", "glm53"))
    import glm53_layers as gl
    STORE_GROUPS = gl.STORE_GROUPS   # (the store gemms counted in 8 row groups, glm53_layers)
    a.flag = gl.apply_settings(a.flag)   # (GLM-5.3's build settings: glm53_layers.G53_FLAGS / G53_ENV)
    fp8 = gl.FP8 if a.fp8 < 0 else bool(a.fp8)
    os.environ["MPK_STORE_KP"] = str(a.store_kp)
    if a.codegen:
        cpu_standins()
    from mirage.core import bfloat16, float32, int32, int64, uint8  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    eps = cfg["rms_norm_eps"]
    full = cfg["indexer_types"][a.layer] == "full"
    print(f"layer {a.layer}: indexer {'full' if full else 'shared'}", flush=True)
    ck = gl.Ckpt(a.model_dir)   # (get + sl: glm53_layers.attn_weights reads slices)
    pfx = f"model.layers.{a.layer}."
    if a.codegen:
        W = {"q_a": torch.zeros(QL, H), "kv_a": torch.zeros(KVL + RP, H), "q_b": torch.zeros(NHT * (NOPE + RP), QL),
             "kv_b": torch.zeros(NHT * (NOPE + DV), KVL), "o": torch.zeros(H, NHT * DV), "q_norm": torch.ones(QL),
             "kv_norm": torch.ones(KVL), "wk": torch.zeros(ID, H), "k_nw": torch.ones(ID), "k_nb": torch.zeros(ID),
             "post_norm": torch.ones(H)}
    else:
        g8 = lambda nm: deq(ck.get(pfx + nm + ".weight"), ck.get(pfx + nm + ".weight_scale_inv"))
        W = {"q_a": g8("self_attn.q_a_proj"), "kv_a": g8("self_attn.kv_a_proj_with_mqa"), "q_b": g8("self_attn.q_b_proj"),
             "kv_b": g8("self_attn.kv_b_proj"), "o": g8("self_attn.o_proj"),
             "q_norm": ck.get(pfx + "self_attn.q_a_layernorm.weight").float(),
             "kv_norm": ck.get(pfx + "self_attn.kv_a_layernorm.weight").float(),
             "post_norm": ck.get(pfx + "post_attention_layernorm.weight").float()}
        if full:
            W.update({"wk": g8("self_attn.indexer.wk"), "k_nw": ck.get(pfx + "self_attn.indexer.k_norm.weight").float(),
                      "k_nb": ck.get(pfx + "self_attn.indexer.k_norm.bias").float()})
    gen = torch.Generator().manual_seed(29)
    x = bf(torch.randn(T, H, generator=gen))
    res = bf(torch.randn(T, H, generator=gen) * 4)
    # requests: request t has ctx + 3 t earlier tokens; rows: a random permutation (row 0: SGLang's pad slot)
    seq = torch.tensor([a.ctx + 3 * t + 1 for t in range(T)], dtype=torch.int32)
    assert int(seq.max()) <= K_TOP, "the dense list: seq_len <= 2048"
    W2T = ((int(seq.max()) + 63) // 64) * 64 + 64
    NROW = 64 * (T * W2T // 64 + 2)
    perm = torch.randperm(NROW - 64, generator=gen)[:T * W2T] + 64
    r2t = perm.view(T, W2T).to(torch.int32)
    req = torch.arange(T, dtype=torch.int64)
    pos = (seq - 1).long()
    out_loc = r2t[torch.arange(T), pos].long()
    cache0 = (torch.randn(NROW, KVL + RP, generator=gen) * 2).to(torch.float8_e4m3fn).view(torch.uint8)
    icache0 = torch.zeros(NROW // 64, 64 * (ID + 4), dtype=torch.uint8)
    cos, sin = rope_tables(int(seq.max()) + 8)
    rows_in = torch.full((T, K_TOP), -1, dtype=torch.int32)   # (a shared layer's list: the dense list, as a full layer wrote it)
    for t in range(T):
        rows_in[t, :int(seq[t])] = r2t[t, :int(seq[t])]
    NA = (QL + KVL + RP + 127) // 128 * 128   # (2624 rows padded to 2688: gemm_tile takes whole 128-row tiles)
    # (fp8: the reference on the dequantized fp32 weights, the kernel on the e4m3 bytes + scales; bf16: both on the bf16 weights)
    rw = (lambda v: v.float()) if fp8 else (lambda v: bf(v).float())
    w_qkva = torch.cat([W["q_a"], W["kv_a"], torch.zeros(NA - QL - KVL - RP, H)], 0)
    w_qkva_ref = rw(w_qkva)
    w_qkva = bf(w_qkva)
    ws, shard = [], []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        hs = slice(g * NH, (g + 1) * NH)
        qb = W["q_b"].view(NHT, NOPE + RP, QL)[hs].reshape(NH * (NOPE + RP), QL)
        kvb = W["kv_b"].view(NHT, NOPE + DV, KVL)[hs]
        wkc = kvb[:, :NOPE, :]   # [NH][192][512]
        wvc = kvb[:, NOPE:, :]   # [NH][256][512]
        wo = W["o"][:, g * NH * DV:(g + 1) * NH * DV]
        shard.append({"qb": rw(qb), "wkc": bf(wkc).float(), "wvc": bf(wvc).float(), "wo": rw(wo)})
        d = {"x_attn": x.to(dev), "w_qkva": w_qkva.to(dev), "q_norm": bf(W["q_norm"]).to(dev), "kv_norm": bf(W["kv_norm"]).to(dev),
             "cache": cache0.clone().to(dev), "out_loc": out_loc.to(dev), "positions": pos.to(dev), "cos": cos.to(dev),
             "sin": sin.to(dev), "w_qb": bf(qb).to(dev),
             "w_kc": bf(wkc.transpose(1, 2)).contiguous().view(NH * KVL, NOPE).to(dev),   # [h][l][n]
             "w_vc": bf(wvc).contiguous().view(NH * DV, KVL).to(dev),                     # [h][d][l]
             "seq_lens": seq.to(dev), "w_o": bf(wo).contiguous().to(dev),
             "res_in": res.to(dev), "post_norm": bf(W["post_norm"]).to(dev),
             "q_lora": torch.zeros(T, QL, dtype=torch.bfloat16, device=dev),
             "q_mla": torch.zeros(T, NH * (KVL + RP), dtype=torch.bfloat16, device=dev),
             "attn": torch.zeros(T, NH * KVL, dtype=torch.bfloat16, device=dev),
             "v_out": torch.zeros(T, NH * DV, dtype=torch.bfloat16, device=dev),
             "res_out": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
             "x_ffn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev)}
        if fp8 and not a.codegen:   # (the server's loader: e4m3 + scales)
            f8w = gl.attn_weights(ck, cfg, a.layer, g, n, dev, fp8=True)
            d.update({k: f8w[k] for k in ("w_qkva", "w_qkva_s", "w_qb", "w_qb_s", "w_o", "w_o_s") + (("w_k", "w_k_s") if full else ())})
        elif fp8:   # (codegen: the shapes)
            for k in ("w_qkva", "w_qb", "w_o") + (("w_k",) if full else ()):
                sh = (d[k].shape[0], d[k].shape[1]) if k in d else (ID, H)
                d[k], d[k + "_s"] = torch.zeros(*sh, dtype=torch.uint8), torch.ones(sh[0] // 128, sh[1] // 128)
        if full:
            d.update({"w_k": d["w_k"] if fp8 else bf(W["wk"]).to(dev), "k_nw": W["k_nw"].float().to(dev), "k_nb": W["k_nb"].float().to(dev),
                      "icache": icache0.clone().to(dev), "r2t": r2t.to(dev), "req_idx": req.to(dev),
                      "rows_g": torch.zeros(T, K_TOP, dtype=torch.int32, device=dev),
                      "rows": torch.zeros(T, K_TOP, dtype=torch.int32, device=dev)})
        else:
            d["rows"] = rows_in.to(dev)
        ws.append(d)
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    SCRATCH = (["q_lora", "q_mla", "attn", "v_out"] + (["rows"] if full else [])) if a.serving else []
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32, torch.uint8: uint8}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    latent = mpk.new_tensor(dims=(T, NA), dtype=float32, name="latent")
    qb_t = mpk.new_tensor(dims=(T, NH * (NOPE + RP)), dtype=bfloat16, name="q_b")
    opart = mpk.new_tensor(dims=(T, H), dtype=float32, name="o_part")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="o_sent")
    mpk.gemm_tile_layer(input=at["x_attn"], weight=at["w_qkva"], output=latent, combine="slots", weight_scales=at.get("w_qkva_s"))
    if full:
        kp = mpk.new_tensor(dims=(T, ID), dtype=float32, name="idx_k")
        mpk.gemm_tile_layer(input=at["x_attn"], weight=at["w_k"], output=kp, combine="slots", weight_scales=at.get("w_k_s"))
    mpk.glm53_mla_norm_layer(latent=latent, w_q=at["q_norm"], w_kv=at["kv_norm"], kv_cache=at["cache"], out_loc=at["out_loc"],
                             positions=at["positions"], cos=at["cos"], sin=at["sin"], q_lora=at["q_lora"], eps_q=eps, eps_kv=eps)
    mpk.gemm_tile_layer(input=at["q_lora"], weight=at["w_qb"], output=qb_t, combine="local", input_polled=True, weight_scales=at.get("w_qb_s"))
    mpk.glm53_q_absorb_layer(q=qb_t, w_kc=at["w_kc"], positions=at["positions"], cos=at["cos"], sin=at["sin"], output=at["q_mla"],
                             heads=NH)
    if full:
        mpk.glm53_index_k_layer(k_parts=kp, norm_w=at["k_nw"], norm_b=at["k_nb"], positions=at["positions"], cos=at["cos"],
                                sin=at["sin"], out_loc=at["out_loc"], index_cache=at["icache"], req_to_token=at["r2t"],
                                req_idx=at["req_idx"], seq_lens=at["seq_lens"], rows_copy=at["rows_g"], rows=at["rows"])
    mpk.glm53_sparse_attn_layer(q=at["q_mla"], cache=at["cache"], seq_lens=at["seq_lens"], out_loc=at["out_loc"],
                                norm_out=at["q_lora"], rows=at["rows"], output=at["attn"], heads=NH, scale=(NOPE + RP) ** -0.5,
                                splits=a.splits or (8 if T <= 8 else 4),
                                **({"req_to_token": at["r2t"], "req_idx": at["req_idx"]} if full else {}))
    mpk.dsa_o_absorb_layer(input=at["attn"], w_uv=at["w_vc"], output=at["v_out"])
    mpk.gemm_tile_layer(input=at["v_out"], weight=at["w_o"], output=opart, combine="store", input_polled=True, store_groups=STORE_GROUPS, weight_scales=at.get("w_o_s"))
    # (the add + RMSNorm in the senders, x_ffn the node's output buffer, as glm53_layers)
    mpk.allreduce_send_layer(input=opart, output=at["x_ffn"], norm=dict(residual=at["res_in"], residual_out=at["res_out"],
                                                                         gamma=at["post_norm"], output=at["x_ffn"], eps=eps,
                                                                         in_kernel=True, rs=gl.rs_on(T, n)))
    nodes = compiler.graph_nodes(mpk)
    from glm_plan import wide_gemm_grids
    plan = simple_plan(mpk, wide_gemm_grids(mpk, a.gemm_cap) if a.gemm_cap else {})
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
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)

    # ---- the reference (fp32 / fp64 of the same math), GPU-independent parts first
    lat = bf(x.float() @ w_qkva_ref.t()).float()[:, :QL + KVL + RP]
    rms = lambda v, w: v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w
    q_lora = bf(rms(lat[:, :QL], bf(W["q_norm"]).float()))
    k_nope = bf(rms(lat[:, QL:QL + KVL], bf(W["kv_norm"]).float())).float()
    cp, sp = cos.float()[pos], sin.float()[pos]
    k_pe = bf(rope_il(lat[:, QL + KVL:], cp, sp)).float()
    row_new = f8(torch.cat([k_nope, k_pe], 1))   # [T][576] the cache rows written
    cache_ref = cache0.view(torch.float8_e4m3fn).float().clone()
    cache_ref[out_loc] = row_new
    rows_ref = rows_in
    if full:
        kk = bf(x.float() @ rw(W["wk"]).t()).float()
        mu = kk.mean(-1, keepdim=True)
        var = (kk - mu).pow(2).mean(-1, keepdim=True)
        kn = bf((kk - mu) * torch.rsqrt(var + 1e-6) * W["k_nw"].float() + W["k_nb"].float()).float()
        kn[:, :RP] = bf(rope_il(kn[:, :RP], cp, sp)).float()
        Hd = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(ID)] for i in range(ID)], dtype=torch.float32)
        kh = bf((kn @ Hd) * ID ** -0.5).float()
        ksc = torch.exp2(torch.ceil(torch.log2(kh.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)))
        k_f8 = (kh / ksc).to(torch.float8_e4m3fn)
    attn_ref, part = [], []
    for g in range(n):
        S = shard[g]
        qb = bf(q_lora.float() @ S["qb"].t()).float().view(T, NH, NOPE + RP)
        q_abs = bf(torch.einsum("thn,hnl->thl", qb[..., :NOPE], S["wkc"])).float()
        q_pe = bf(rope_il(qb[..., NOPE:], cp[:, None, :], sp[:, None, :])).float()
        q_cat = torch.cat([q_abs, q_pe], -1)   # [T][NH][576]
        o = torch.zeros(T, NH, KVL)
        for t in range(T):
            rr = cache_ref[rows_ref[t, :int(seq[t])].long()]   # [L][576]
            s = (q_cat[t] @ rr.t()) * (NOPE + RP) ** -0.5
            pr = torch.softmax(s.double(), -1).float()
            o[t] = pr @ rr[:, :KVL]
        o = bf(o).float()
        v = bf(torch.einsum("thl,hdl->thd", o, S["wvc"])).float().reshape(T, NH * DV)
        part.append(bf(v @ S["wo"].t()))
        attn_ref.append((q_cat, o))
    acc = torch.zeros(T, H)
    for p_ in part:
        acc = acc + p_.float()
    vsum = bf(acc).float() + res.float()
    res_ref, xffn_ref = bf(vsum), bf(rms(vsum, bf(W["post_norm"]).float()))

    first = None
    for launch in range(a.launches):
        for g in range(n):   # (the step's state reset: the same step every launch)
            ws[g]["cache"].copy_(cache0.to(ws[g]["cache"].device))
            if full:
                ws[g]["icache"].copy_(icache0.to(ws[g]["icache"].device))
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        G0 = {k: ws[0][k].cpu().clone() for k in ("q_lora", "q_mla", "attn", "res_out", "x_ffn")}
        rs = gl.rs_on(T, n)   # (RSN: residual_out only on each token's owner GPU)
        if rs:
            G0["res_out"] = gl.owner_rows([ws[g]["res_out"].cpu() for g in range(n)], T, n)
        if launch == 0:
            ck_rows = ws[0]["cache"].cpu().view(torch.float8_e4m3fn).float()[out_loc]
            check("MLA cache rows written (e4m3 [k_nope | rope(k_pe)])", relrms(ck_rows, row_new) < 2e-2,
                  f"relRMS {relrms(ck_rows, row_new):.2e}, bytes equal {float((ws[0]['cache'].cpu()[out_loc] == row_new.to(torch.float8_e4m3fn).view(torch.uint8)).float().mean()):.4f}")
            check("q_lora", relrms(G0["q_lora"], q_lora) < 1e-2, f"relRMS {relrms(G0['q_lora'], q_lora):.2e}")
            qk = G0["q_mla"].view(T, NH, KVL + RP)
            check("GPU 0: the query (q_abs | rope(q_pe))", relrms(qk, attn_ref[0][0]) < 2e-2, f"relRMS {relrms(qk, attn_ref[0][0]):.2e}")
            ak = G0["attn"].view(T, NH, KVL)
            check("GPU 0: attn", relrms(ak, attn_ref[0][1]) < 3e-2, f"relRMS {relrms(ak, attn_ref[0][1]):.2e}")
            if full:
                rk = ws[0]["rows_g"].cpu()
                check("the dense row list (and its graph copy)", torch.equal(rk, rows_in) and torch.equal(ws[0]["rows"].cpu(), rows_in))
                ic = ws[0]["icache"].cpu()
                kf, ks = [], []
                for t in range(T):
                    sl = int(out_loc[t]); pg, off = sl // 64, sl % 64
                    pgb = ic[pg]
                    idx = torch.tensor([(off // 16) * 2048 + (d // 16) * 256 + (off % 16) * 16 + d % 16 for d in range(ID)])
                    kf.append(pgb[idx].view(torch.float8_e4m3fn).float())
                    ks.append(pgb[8192 + 4 * off: 8192 + 4 * off + 4].view(torch.float32))
                kf, ks = torch.stack(kf), torch.stack(ks)
                check("index-K rows written (values x scale)", relrms(kf * ks, k_f8.float() * ksc) < 3e-2,
                      f"relRMS {relrms(kf * ks, k_f8.float() * ksc):.2e}, scales equal {torch.equal(ks, ksc)}")
        check(f"launch {launch}: residual_out", relrms(G0["res_out"], res_ref) < 1e-2, f"relRMS {relrms(G0['res_out'], res_ref):.2e}")
        check(f"launch {launch}: x_ffn", relrms(G0["x_ffn"], xffn_ref) < 3e-2, f"relRMS {relrms(G0['x_ffn'], xffn_ref):.2e}")
        same = all(torch.equal(ws[g][k].cpu(), G0[k]) for g in range(1, n) for k in (("x_ffn",) if rs else ("res_out", "x_ffn")))
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
    print("CHECKS " + ("PASS" if not failed else f"FAILED {failed}"), flush=True)


if __name__ == "__main__":
    main()
