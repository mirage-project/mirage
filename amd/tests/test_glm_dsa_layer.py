"""GLM-5.3-Flash DSA attention sublayer on AMD GPUs (TP4, plain decode), SGLang ROCm's cache layouts
(amd/notes/sglang_v0522_glm_dsa_contract.md):
  x_attn -> gemm [q_a | kv_a] (K parts in slots) -> dsa_norm (q_c bf16; the kv row -> the fp8 latent cache at out_loc)
         -> gemm [wk | gate | weights_proj] (slots) -> dsa_kpool_update (LayerNorm k, the tail ring, the pool it closes ->
            the preshuffled pooled page)
  q_c    -> gemm wq_b (slots) -> dsa_index_score (Hadamard q, e4m3 per head, scores over the pooled keys) -> dsa_index_topk
         -> gemm q_b (local, bf16) -> mla_absorb (W_UK) -> dsa_sparse_attn (the top pools' positions + the tail, fp8 rows)
         -> dsa_o_absorb (W_UV) -> o_proj (push_all) -> hc_post_mix -> hc_pre (x_ffn)
Real weights of DSA layer --layer (upstream demo/glm53_flash dsa_mla / dsa_layer load them; the FP8 projections
dequantized to bf16 for gemm_tile), random x, streams, latent cache rows, pooled keys, tails. Checked per stage against
torch references with the kernel's rounding points: q_c, the new cache rows (bytes), the tails and the closed pools (bytes,
scales), the scores, the top-K pools (as sets), the absorbed query, the attention (from the kernel's own query and pools),
v, R', x_ffn; launch k (state restored) bit-identical to launch 0; every GPU's mHC outputs equal.

    python amd/tests/test_glm_dsa_layer.py --tokens 8 [--gpus 4] [--layer 3] [--splits 1] [--codegen]
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

QL, KVL, NOPE, DV, IH, ID, KP, TOPK = 1536, 512, 256, 256, 32, 128, 4, 512
MAXP = 1024                 # pools in a score row (4096 positions; --maxctx)
MAXPP = MAXP // 64          # pooled block table columns
W_TOK = MAXP * 4            # req_to_token row stride (positions)
RING = 4
# (the server library: --maxctx 8192; the hook passes page_table = req_to_token[req_pool_indices, :8192] as r2t with
# req_idx = arange(T), and pooled_bt = page_table[:, ::256] // 256)


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3 (torch: round to nearest even) -> fp32"""
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def ue8m0_quant(v: torch.Tensor):
    """[..., 128] fp32 -> (e4m3 values as fp32, the pow2 scale): s = 2^ceil(log2(max(amax, 1e-4) / 448))"""
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return e4m3(v / s), s


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().float()
    for st in (1, 2, 4, 8, 16, 32, 64):
        y = x.view(*x.shape[:-1], 128 // (2 * st), 2, st)
        a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
    return x * torch.tensor(0.08838834764831845, dtype=torch.float32)


def key_off(o: int, c: int) -> int:
    return (o // 16) * 2048 + (c // 16) * 256 + (o % 16) * 16 + c % 16


KIDX = torch.tensor([[key_off(o, c) for c in range(128)] for o in range(64)])


def page_keys(page: torch.Tensor):
    """(the 64 pooled keys' e4m3 values as fp32 [64][128], their scales [64]) of a preshuffled page (uint8 [8448])"""
    return page[KIDX].view(torch.float8_e4m3fn).float(), page[8192:8448].view(torch.float32).clone()


def page_key(page: torch.Tensor, o: int):
    k, s = page_keys(page)
    return k[o], s[o].item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--splits", type=int, default=1)
    p.add_argument("--ctx", type=int, default=0, help="every request's context about this many positions (+-25 %%; 0: the check "
                   "mix, 64 .. 2000 and one of 3001)")
    p.add_argument("--score-chunks", type=int, default=8,
                   help="dsa_index_score tasks per token (11-08 glmdsa5: 8 vs 2 T 8 201.3 vs 214, T 64 297.1 vs 307, T 1 level)")
    p.add_argument("--gemm-cap", type=int, default=256, help="gemm_tile grids: the most tasks up to this (glm_plan.py; 0: the layer defaults)")
    p.add_argument("--oproj-ar", action="store_true", help="o_proj as a store gemm + allreduce_send (instead of push_all)")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_dsa")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--profile", action="store_true", help="the timing build: per node its task times after the timed launches")
    p.add_argument("--serving", action="store_true", help="with --codegen: the server library's graph (scratch tensors inside)")
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--maxctx", type=int, default=4096, help="positions per request row (req_to_token width; pools = / 4)")
    p.add_argument("--pair-moe", action="store_true", help="one kernel per layer: this DSA sublayer then the layer's MoE "
                   "sublayer (server loaders; timing + finite / equal outputs only, or --codegen --serving: rank kind gdsamoe)")
    a = p.parse_args()
    global MAXP, MAXPP, W_TOK
    MAXP, W_TOK = a.maxctx // 4, a.maxctx
    MAXPP = MAXP // 64
    T, n = a.tokens, a.gpus
    if a.pair_moe and T < 32 and int(os.environ.get("MPK_DSA_OPROJ_KP", "4" if T >= 16 else "0")) > 0:
        os.environ.setdefault("MPK_HIP_MAX_BUFS", "128")   # (the store o_proj's parts + send buffers: more than 64 slots at T 16)
    os.environ["MPK_TOKENS"] = str(T)
    os.environ["STATIC_TOKENS"] = str(T)
    if a.codegen:
        cpu_standins()
    sys.path.insert(0, os.path.join(HERE, "..", "..", "demo", "glm53_flash"))
    from mirage.core import bfloat16, float32, int32, int64, uint8  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    rms_eps, hc_eps, iters = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"]
    NH = cfg["num_attention_heads"] // n
    m = load_mhc(a.model_dir, a.layer, "ffn")
    gen = torch.Generator().manual_seed(13)
    # ---- the state: requests, positions, cache rows, pooled keys, tails
    REQS = T + 2
    req = torch.randperm(REQS, generator=gen)[:T]
    seq = torch.randint(64, 2000, (T,), generator=gen)
    if a.ctx:
        seq = torch.randint(int(a.ctx * 0.75), int(a.ctx * 1.25) + 1, (T,), generator=gen)
    seq[0] = (seq[0] // 4) * 4            # its token closes a pool
    if T > 1 and not a.ctx:
        seq[1] = 3001                       # 750 pools > 512: the radix top-K
    if T > 2 and not a.ctx:
        seq[2] = 37
    ROWS = 1 + REQS * W_TOK
    r2t = torch.zeros(REQS, W_TOK, dtype=torch.int32)
    perm = torch.randperm(ROWS - 1, generator=gen) + 1
    r2t.view(-1)[:] = perm[:REQS * W_TOK].to(torch.int32)
    out_loc = torch.stack([r2t[req[t], seq[t] - 1] for t in range(T)]).long()
    cache0 = (torch.randn(ROWS, KVL, generator=gen) * 0.7).to(torch.float8_e4m3fn).view(torch.uint8)
    PAGES = T * MAXPP + 1
    pbt = (torch.randperm(PAGES - 1, generator=gen)[:T * MAXPP] + 1).view(T, MAXPP).to(torch.int32)
    keys = (torch.randn(PAGES, 64 * 128, generator=gen) * 64).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.exp2(-torch.randint(4, 9, (PAGES, 64), generator=gen).float())
    pcache0 = torch.cat([keys, scales.view(torch.uint8).view(PAGES, 256)], 1).contiguous()
    tail_k0 = bf(torch.randn(REQS, RING, ID, generator=gen))
    tail_s0 = bf(torch.randn(REQS, RING, ID, generator=gen))
    x = bf(torch.randn(T, H, generator=gen))
    R = bf(torch.randn(T, HC, H, generator=gen))
    hp = (torch.rand(T, HC, generator=gen) * 2).float()
    hr = torch.rand(T, HC * HC, generator=gen).float()
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = m["base"], m["scale"]

    def deq(w, s):   # e4m3 [N, K] with fp32 128x128 block scales -> bf16
        return bf(w.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:w.shape[0], :w.shape[1]])
    if a.codegen:
        import dsa_mla as mla
        import dsa_layer as dl
        Ws = [{"w_qkva": torch.zeros(QL + KVL, H, dtype=torch.float8_e4m3fn), "w_qkva_s": torch.ones(16, 32),
               "w_qb": torch.zeros(NH * NOPE, QL, dtype=torch.float8_e4m3fn), "w_qb_s": torch.ones(NH * NOPE // 128, 12),
               "w_o": torch.zeros(H, NH * DV, dtype=torch.float8_e4m3fn), "w_o_s": torch.ones(32, NH * DV // 128),
               "w_uk": torch.zeros(NH, NOPE, KVL, dtype=torch.bfloat16), "w_uv": torch.zeros(NH, KVL, DV, dtype=torch.bfloat16),
               "q_a_norm": torch.ones(QL), "kv_a_norm": torch.ones(KVL)} for _ in range(n)]
        IX = {"w_kgw": torch.zeros(384, H, dtype=torch.bfloat16), "w_iqb": torch.zeros(IH * ID, QL, dtype=torch.bfloat16),
              "k_norm_w": torch.ones(ID), "k_norm_b": torch.zeros(ID), "ape": torch.zeros(4, ID)}
    else:
        import dsa_mla as mla
        import dsa_layer as dl
        Ws = [mla.load_layer(a.model_dir, a.layer, n, g, "cpu") for g in range(n)]
        IX = dl.load_indexer(a.model_dir, a.layer, "cpu")
    w_qkva = deq(Ws[0]["w_qkva"], Ws[0]["w_qkva_s"])
    ws = []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        Wg = Ws[g]
        ws.append({"x_attn": x.to(dev), "w_qkva": w_qkva.to(dev),
                   "q_norm": bf(Wg["q_a_norm"]).to(dev), "kv_norm": bf(Wg["kv_a_norm"]).to(dev),
                   "cache": cache0.clone().to(dev), "out_loc": out_loc.to(dev),
                   "w_kgw": IX["w_kgw"].to(dev), "k_norm_w": IX["k_norm_w"].float().to(dev),
                   "k_norm_b": IX["k_norm_b"].float().to(dev), "ape": IX["ape"].float().contiguous().to(dev),
                   "seq_lens": seq.to(torch.int32).to(dev), "req_slots": req.long().to(dev), "pooled_bt": pbt.to(dev),
                   "tail_k": tail_k0.clone().to(dev), "tail_score": tail_s0.clone().to(dev), "pcache": pcache0.clone().to(dev),
                   "w_iqb": IX["w_iqb"].to(dev), "w_qb": deq(Wg["w_qb"], Wg["w_qb_s"]).to(dev),
                   "w_kc": Wg["w_uk"].transpose(1, 2).contiguous().view(NH * KVL, NOPE).to(dev),   # [h][l][n]
                   "w_vc": Wg["w_uv"].transpose(1, 2).contiguous().view(NH * DV, KVL).to(dev),     # [h][d][l]
                   "r2t": r2t.to(dev), "req_idx": req.long().to(dev),
                   "w_o": deq(Wg["w_o"], Wg["w_o_s"]).to(dev),
                   "R_in": R.reshape(T, HC * H).to(dev), "h_post_in": hp.to(dev), "h_res_in": hr.to(dev),
                   "hc_ffn_fn": m["fn"].to(dev), "hc_ffn_small": hc_small.to(dev), "ffn_norm": m["gamma"].to(dev),
                   "q_c": torch.zeros(T, QL, dtype=torch.bfloat16, device=dev),
                   "scores": torch.zeros(T, MAXP, dtype=torch.float32, device=dev),
                   "idx_q": torch.zeros(T, IH * ID, dtype=torch.bfloat16, device=dev),
                   "pool_idx": torch.full((T, TOPK), -1, dtype=torch.int32, device=dev),
                   "q_mla": torch.zeros(T, NH * KVL, dtype=torch.bfloat16, device=dev),
                   "attn": torch.zeros(T, NH * KVL, dtype=torch.bfloat16, device=dev),
                   "v_out": torch.zeros(T, NH * DV, dtype=torch.bfloat16, device=dev),
                   "R_out": torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
                   "x_ffn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                   "h_post_out": torch.zeros(T, HC, dtype=torch.float32, device=dev),
                   "h_res_out": torch.zeros(T, HC * HC, dtype=torch.float32, device=dev)})
    if a.pair_moe:   # (the MoE sublayer's weights and tensors; the DSA part writes the mid streams, the MoE part R_out ..)
        sys.path.insert(0, os.path.join(HERE, "..", "serving", "glm"))
        os.environ.setdefault("MPK_GLM", "")
        import sglang_glm as sg
        skm = sg._Slices(a.model_dir)
        for g in range(n):
            dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
            ws[g].update(sg._moe_weights(skm, a.layer, g, n, dev))
            ws[g].update(sg._mhc(skm, a.layer + 1, "attn", dev)[0])
            zf = lambda *sh, dt=torch.float32: torch.zeros(*sh, dtype=dt, device=dev)
            ws[g].update({"pairs": torch.zeros(T, 9, dtype=torch.int64, device=dev), "R_moe": zf(T * 9, H),
                          "x_attn_next": zf(T, H, dt=torch.bfloat16), "R_mid": zf(T, HC * H, dt=torch.bfloat16),
                          "h_post_mid": zf(T, HC), "h_res_mid": zf(T, HC * HC)})
    state_keys = ("cache", "pcache", "tail_k", "tail_score")
    init = [{k: ws[g][k].clone() for k in state_keys} for g in range(n)]
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    # --serving (the server libraries): the scratch tensors inside the kernel (node buffers the kernel resets every launch;
    # bound as run tensors they would keep the last launch's values: a polled reader could read them before the new ones)
    SCRATCH = (['q_c', 'idx_q', 'scores', 'pool_idx', 'q_mla', 'attn', 'v_out'] + (['x_ffn', 'pairs', 'R_moe'] if a.pair_moe else [])
               if a.serving else [])
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    latent = mpk.new_tensor(dims=(T, QL + KVL), dtype=float32, name="latent")
    kw = mpk.new_tensor(dims=(T, 384), dtype=float32, name="idx_kw")
    iq = at["idx_q"]   # (a graph tensor: the score reference reads the kernel's own bf16 indexer query)
    qb = mpk.new_tensor(dims=(T, NH * NOPE), dtype=bfloat16, name="q_b")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="o_sent")
    mix = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="ffn_hc_mix")
    mpk.gemm_tile_layer(input=at["x_attn"], weight=at["w_qkva"], output=latent, combine="slots")
    mpk.gemm_tile_layer(input=at["x_attn"], weight=at["w_kgw"], output=kw, combine="slots")
    mpk.dsa_norm_layer(latent=latent, w_q=at["q_norm"], w_kv=at["kv_norm"], kv_cache=at["cache"], out_loc=at["out_loc"],
                       q_c=at["q_c"], eps_q=rms_eps, eps_kv=rms_eps)
    mpk.dsa_kpool_update_layer(kw=kw, k_norm_weight=at["k_norm_w"], k_norm_bias=at["k_norm_b"], ape=at["ape"],
                               seq_lens=at["seq_lens"], req_slots=at["req_slots"], pooled_block_table=at["pooled_bt"],
                               tail_k=at["tail_k"], tail_score=at["tail_score"], pooled_cache=at["pcache"], eps=1e-6)
    # (MPK_DSA_IQ_SKIP, default 1 at T <= 16: the indexer query gemm returns at once when no token has more than TOPK pools. MI355X
    # 11-09 glmdsa19, DSA + MoE pair --ctx 200: T 16 251.3 / 250.2 vs 257.9 / 257.4 us; T 32 / 64 +1..3 - there the q_b gemms run side
    # by side on CU halves)
    iq_skip = os.environ.get("MPK_DSA_IQ_SKIP", "1" if T <= 16 else "0") == "1"
    mpk.gemm_tile_layer(input=at["q_c"], weight=at["w_iqb"], output=iq, combine="local", input_polled=True,
                        skip_seq=(at["seq_lens"], TOPK) if iq_skip else None)
    mpk.gemm_tile_layer(input=at["q_c"], weight=at["w_qb"], output=qb, combine="local", input_polled=True)
    # MPK_DSA_SCORE_SKIP (default 1): a token with <= TOPK pools computes no indexer scores (dsa_index_score params[4] = TOPK; topk
    # takes every pool in order, the sparse attention reads no list; the scores check skips those tokens). MI355X 11-09 glmdsa18, DSA
    # + MoE pair --ctx 200: T 16 254.8 / 252.8 vs 268.0 / 260.9 us; server glmab59 bs 16 gdsamoe -9 us per layer, glmab60 bs 32 -7, bs
    # 64 level
    score_skip = os.environ.get("MPK_DSA_SCORE_SKIP", "1") == "1"
    mpk.dsa_index_score_layer(iq=iq, kw=kw, index_cache=at["pcache"], cos_sin=None, seq_lens=at["seq_lens"],
                              block_table=at["pooled_bt"], output=at["scores"], chunks=a.score_chunks, flash=True,
                              skip_k=TOPK if score_skip else 0)
    mpk.dsa_index_topk_layer(scores=at["scores"], seq_lens=at["seq_lens"], output=at["pool_idx"], unit=4)
    mpk.mla_absorb_layer(q=qb, w_kc=at["w_kc"], output=at["q_mla"], heads=NH, nope=NOPE, rp=0, kv=KVL)
    mpk.dsa_sparse_attn_layer(q=at["q_mla"], cache=at["cache"], req_to_token=at["r2t"], seq_lens=at["seq_lens"],
                              req_idx=at["req_idx"], out_loc=at["out_loc"], norm_out=at["q_c"], indices=at["pool_idx"],
                              output=at["attn"], heads=NH, scale=NOPE ** -0.5, splits=a.splits)
    mpk.dsa_o_absorb_layer(input=at["attn"], w_uv=at["w_vc"], output=at["v_out"])
    # MPK_DSA_OPROJ_KP P (default 4 at T >= 16, else 0 = push_all; DSA + MoE pair glmdsa13 T 16 280.0 / 277.8 vs 285.0 / 281.9, T 8 slower): o_proj as a polled store gemm in P K parts ([part][T][N] fp32,
    # each task 1 / P of the 512 KB activation at T 64) + allreduce_send adding the parts and landing the all-reduce (MI355X 11-09
    # glmkda25, --ctx 512: T 64 170.5 / 169.0 vs push_all 191.0 / 190.2 us, T 32 127.9 / 126.3 vs 135.0 / 133.5)
    dsa_kp = int(os.environ.get("MPK_DSA_OPROJ_KP", "4" if T >= 16 else "0"))
    if dsa_kp and os.environ.get("MPK_OPROJ_FIN", "0") == "1":   # (probe: finish_all - the last K part pushes the sums, no send)
        os.environ.setdefault("MPK_STORE_KP", str(dsa_kp))
        mpk.gemm_tile_layer(input=at["v_out"], weight=at["w_o"], output=sent, combine="finish_all", input_polled=True)
    elif a.oproj_ar or dsa_kp:
        os.environ.setdefault("MPK_STORE_KP", str(max(dsa_kp, 1)))
        opart = mpk.new_tensor(dims=(T, H), dtype=float32, name="o_part")
        mpk.gemm_tile_layer(input=at["v_out"], weight=at["w_o"], output=opart, combine="store",   # (MPK_OPROJ_AR_POLLED=0: counted)
                            input_polled=os.environ.get("MPK_OPROJ_AR_POLLED", "1") == "1")
        mpk.allreduce_send_layer(input=opart, output=sent, num_tasks=H // 32 if T >= 16 else None)   # (OWNPOLL: one vector a thread)
    else:
        mpk.gemm_tile_layer(input=at["v_out"], weight=at["w_o"], output=sent, combine="push_all", input_polled=os.environ.get("MPK_OPROJ_COUNTED", "0") != "1")
    ro, hpo, hro = ("R_mid", "h_post_mid", "h_res_mid") if a.pair_moe else ("R_out", "h_post_out", "h_res_out")
    mpk.hc_post_mix_layer(sent=sent, residual=at["R_in"], coef=(at["h_post_in"], at["h_res_in"]), fn=at["hc_ffn_fn"],
                          output=at[ro], mix=mix)
    mpk.hc_pre_layer(mix=mix, residual=at[ro], hc_small=at["hc_ffn_small"], gamma=at["ffn_norm"], output=at["x_ffn"],
                     rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                     export=(at[hpo], at[hro]))
    hand = False
    if a.pair_moe:   # (the MoE sublayer as in test_glm_pair_layer.py: router (12, 8, 1), the LDS hand-off at T >= 16)
        limit = cfg["swiglu_limit"]
        logits = mpk.new_tensor(dims=(T, ws[0]["router"].shape[0]), dtype=float32, name="router_logits")
        sent_b = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="moe_sent")
        mix_b = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="attn_hc_mix")
        mpk.gemm_tile_layer(input=at["x_ffn"], weight=at["router"], output=logits, combine="slots")
        mpk.glm_topk_route_layer(input=logits, bias=at["bias"], output=at["pairs"], k=8, shared_experts=1, shared_first=288,
                                 routed_scaling_factor=2.5)
        mpk.glm_moe_experts_layer(x=at["x_ffn"], pairs=at["pairs"], w13=at["w13"], w13_sf=at["w13_sf"], w2=at["w2"],
                                  w2_sf=at["w2_sf"], output=at["R_moe"], swiglu_limit=limit)
        hand = os.environ.get("MPK_ARS_HAND", "1") == "1" and T >= int(os.environ.get("MPK_ARS_HAND_MIN_T", "4"))
        mpk.allreduce_send_layer(input=at["R_moe"], output=sent_b, num_tasks=H // 32 if hand else None,
                                 prefetch=(at["R_mid"], at["hc_attn_fn"]) if hand and os.environ.get("MPK_ARS_PREFETCH") == "1" else None)
        mpk.hc_post_mix_layer(sent=sent_b, residual=at["R_mid"], coef=(at["h_post_mid"], at["h_res_mid"]), fn=at["hc_attn_fn"],
                              output=at["R_out"], mix=mix_b)
        mpk.hc_pre_layer(mix=mix_b, residual=at["R_out"], hc_small=at["hc_attn_small"], gamma=at["attn_norm"],
                         output=at["x_attn_next"], rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                         export=(at["h_post_out"], at["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    from glm_plan import wide_gemm_grids
    grids = wide_gemm_grids(mpk, a.gemm_cap) if a.gemm_cap else {}
    # MPK_DSA_QB_CAP N (probe): the two q_b gemms (index wq_b and q_b, both on q_c) at most N tasks each - with N = 128 they take
    # disjoint CU halves (simple_plan's round robin goes on across nodes) and run side by side instead of one after the other on
    # every CU (each its own q_c fill: DSA T 64 stage 0 at +10 us each, glmprof6). Default 128 at T >= 32 (MI355X 11-09 glmdsa16, DSA
    # + MoE pair --ctx 200: T 64 439.9 / 435.4 vs 447.0 / 444.3 us, T 32 312.6 / 307.7 vs 316.8 / 315.1, T 16 level; checks pass)
    qcap = int(os.environ.get("MPK_DSA_QB_CAP", "128" if T >= 32 else "0"))
    if qcap:
        qb = {q.graph_idx for q in nodes if q.name == "gemm_tile" and q.inputs[0].guid == at["q_c"].guid}
        assert len(qb) == 2, qb
        grids.update(wide_gemm_grids(mpk, qcap, only=qb))
        print("MPK_DSA_QB_CAP grids:", {k: grids[k] for k in qb}, flush=True)
    # MPK_DSA_XA_CAP N (probe): the latent gemm [q_a | kv_a] (on x_attn) at most N tasks, so the indexer's w_kgw gemm (also on x_attn, 96
    # tasks) takes the CUs past it and the two run side by side (DSA T 64: w_kgw ran 12.1..18.1 after the latent gemm on CUs 0..95)
    xcap = int(os.environ.get("MPK_DSA_XA_CAP", "0"))
    if xcap:
        xa = [q for q in nodes if q.name == "gemm_tile" and q.inputs[0].guid == at["x_attn"].guid]
        assert len(xa) == 2, [q.graph_idx for q in xa]
        grids.update(wide_gemm_grids(mpk, xcap, only={xa[0].graph_idx}))
        print("MPK_DSA_XA_CAP grids:", {q.graph_idx: grids.get(q.graph_idx) for q in xa}, flush=True)
    if a.pair_moe:
        for q in nodes:
            if q.name == "gemm_tile" and q.params[2] == ws[0]["router"].shape[0]:
                grids[q.graph_idx] = (12, 8, 1)
    plan = simple_plan(mpk, grids)
    if a.pair_moe:
        first_moe = min(q.graph_idx for q in nodes if q.graph_idx > next(r.graph_idx for r in nodes if r.name == "hc_pre"))
        if hand:
            from test_glm_moe_layer import pair_send_postmix
            snd = max(q.graph_idx for q in nodes if q.name == "allreduce_send")   # (the MoE part's send: the last one)
            pm = next(q.graph_idx for q in nodes if q.name == "hc_post_mix" and q.graph_idx > snd)
            plan = pair_send_postmix(plan, snd, pm)
        os.environ.setdefault("MPK_SPLIT_LOOP", str(first_moe))
        print(f"MPK_SPLIT_LOOP={os.environ['MPK_SPLIT_LOOP']} (the MoE part's first node)", flush=True)
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
        print(f"hipcc exit {r.returncode}", r.stderr[-4000:] if r.returncode else "", flush=True)
        return
    from mirage.mpk.static_harness import Harness
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag,
                            profile=a.profile)
    h = Harness(sk)
    if a.pair_moe:   # (timing probe: finite / equal outputs, the e2e latency)
        for _ in range(2):
            h.launch(512 << 20, timeout_s=60)
        outs = ("R_out", "h_post_out", "h_res_out", "x_attn_next")
        fin = all(bool(torch.isfinite(ws[g][k].float()).all()) for g in range(n) for k in outs)
        same = all(torch.equal(ws[g][k].cpu(), ws[0][k].cpu()) for g in range(1, n) for k in outs)
        print(f"{'ok  ' if fin else 'FAIL'} outputs finite; {'ok  ' if same else 'FAIL'} every GPU's outputs equal", flush=True)
        us = []
        for _ in range(a.reps):
            h.launch(512 << 20, timeout_s=60)
            us.append(max(h.sk.elapsed_us()))
        us.sort()
        print(f"LATENCY dsa+moe T {T} e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, "
              f"L2 emptied, slowest GPU)", flush=True)
        if a.profile:
            from test_glm_moe_layer import node_times
            node_times(h, paths, compiler.graph_nodes(mpk))
        return
    # ---- references (CPU)
    lat = bf(x.float() @ w_qkva.float().t()).float()
    rms = lambda v, w, eps: v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w
    qc_ref = bf(rms(lat[:, :QL], bf(Ws[0]["q_a_norm"]).float(), rms_eps))
    kvrow_ref = bf(rms(lat[:, QL:], bf(Ws[0]["kv_a_norm"]).float(), rms_eps)).float().to(torch.float8_e4m3fn).view(torch.uint8)
    kwr = x.float() @ IX["w_kgw"].float().t()
    k_bf, g_bf, wp = bf(kwr[:, :ID]).float(), bf(kwr[:, ID:2 * ID]).float(), kwr[:, 2 * ID:2 * ID + IH]
    kn = bf(torch.nn.functional.layer_norm(k_bf, (ID,), IX["k_norm_w"].float(), IX["k_norm_b"].float(), 1e-6)).float()
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)

    def restore():
        for g in range(n):
            for k in state_keys:
                ws[g][k].copy_(init[g][k])
            torch.cuda.synchronize(g)
    first = None
    for launch in range(a.launches):
        restore()
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        G = {k: ws[0][k].cpu().clone() for k in ("q_c", "idx_q", "cache", "pcache", "tail_k", "tail_score", "scores", "pool_idx",
                                                 "q_mla", "attn", "v_out", "R_out", "x_ffn", "h_post_out", "h_res_out")}
        if launch == 0:
            check("q_c", relrms(G["q_c"], qc_ref) < 1e-2, f"relRMS {relrms(G['q_c'], qc_ref):.2e}")
            rows = torch.stack([G["cache"][out_loc[t]] for t in range(T)])
            check("the new fp8 cache rows", (rows == kvrow_ref).float().mean().item() > 0.98,
                  f"bytes equal {(rows == kvrow_ref).float().mean().item():.4f} (bf16 sums of the K parts vs a bf16 matmul)")
            other = G["cache"].clone()
            other[out_loc] = cache0[out_loc]
            check("no other cache row written", torch.equal(other, cache0))
            # tails and closed pools
            tk, ts = G["tail_k"], G["tail_score"]
            ok_t, ok_p, worst = True, True, 0.0
            pc = G["pcache"]
            for t in range(T):
                pos, r = int(seq[t]) - 1, int(req[t])
                # (the K parts' fp32 sums vs a torch matmul: a bf16 value may round the other way)
                ok_t &= relrms(tk[r, pos % RING], kn[t]) < 1e-2 and relrms(ts[r, pos % RING], g_bf[t]) < 1e-2
                if pos % 4 == 3:
                    kv = torch.stack([tail_k0[r, (pos - 3 + s) % RING].float() if s < 3 else kn[t] for s in range(4)])
                    sc = torch.stack([tail_s0[r, (pos - 3 + s) % RING].float() if s < 3 else g_bf[t] for s in range(4)])
                    sc = sc + IX["ape"].float()
                    pr = torch.softmax(sc, 0)
                    pooled = bf(bf((kv * pr).sum(0)).float().unsqueeze(0))
                    hq = bf(hadamard128(pooled.float())).float()
                    q8, s8 = ue8m0_quant(hq)
                    pool = pos // 4
                    k8k, s8k = page_key(pc[int(pbt[t, pool // 64])], pool % 64)
                    worst = max(worst, relrms(k8k * s8k, q8[0] * s8.item()))
                    ok_p &= relrms(k8k * s8k, q8[0] * s8.item()) < 2e-2
            check("tails (LayerNorm k, gate)", ok_t)
            check("the closed pools (e4m3 x scale vs the reference)", ok_p, f"worst relRMS {worst:.2e}")
            # scores from the kernel's own pooled cache and the reference q of the kernel's q_c
            iqr = G["idx_q"].float().view(T, IH, ID)   # (the kernel's bf16 query: wq_b(q_c))
            q8, qs = ue8m0_quant(bf(hadamard128(iqr)).float())
            wh = (wp * IH ** -0.5) * qs.squeeze(-1) * ID ** -0.5
            sbad, sworst = 0, 0.0
            for t in range(T):
                npl = int(seq[t]) // 4
                pk = [page_keys(pc[int(pbt[t, pg])]) for pg in range((npl + 63) // 64)]
                ks = torch.cat([k for k, _ in pk])[:npl]
                ss = torch.cat([s_ for _, s_ in pk])[:npl]
                ref = ss * (torch.relu(ks.double() @ q8[t].double().t()) * wh[t].double()).sum(-1)
                if npl > TOPK or not score_skip:   # (skipped tokens: no scores)
                    e = relrms(G["scores"][t, :npl], ref)
                    sworst = max(sworst, e)
                    sbad += e > 1e-3
                kidx = G["pool_idx"][t]
                if npl <= TOPK:
                    want = torch.cat([torch.arange(npl), torch.full((TOPK - npl,), -1)]).int()
                    sbad += not torch.equal(kidx, want)
                else:
                    top = set(torch.topk(G["scores"][t, :npl], TOPK).indices.tolist())
                    got = set(kidx.tolist())
                    sbad += len(top ^ got) > 4 or bool((kidx.sort().values != kidx).any())
            check("scores and the top-K pools", sbad == 0, f"worst score relRMS {sworst:.2e}")
            # the absorbed query from the kernel's q_c
            qbr = bf(G["q_c"].float() @ ws[0]["w_qb"].cpu().float().t()).float().view(T, NH, NOPE)
            qm_ref = bf(torch.einsum("thn,hnl->thl", qbr, Ws[0]["w_uk"].float())).reshape(T, NH * KVL)
            check("q_mla (W_UK)", relrms(G["q_mla"], qm_ref) < 1e-2, f"relRMS {relrms(G['q_mla'], qm_ref):.2e}")
            # the attention from the kernel's q_mla, pools and cache
            cf = G["cache"].view(torch.float8_e4m3fn).float()
            att_ref = torch.zeros(T, NH, KVL)
            for t in range(T):
                s_t, npl = int(seq[t]), int(seq[t]) // 4
                npk = min(npl, TOPK)
                pools = list(range(npk)) if npl <= TOPK else G["pool_idx"][t, :npk].tolist()
                posl = [4 * q + i for q in pools for i in range(4)] + list(range(4 * npl, s_t))
                rows_t = cf[r2t[req[t]][torch.tensor(posl)].long()]
                qv = G["q_mla"][t].float().view(NH, KVL)
                sc = (qv @ rows_t.t()) * NOPE ** -0.5
                pr = torch.exp(sc - sc.amax(-1, keepdim=True))
                att_ref[t] = bf((bf(pr).float() @ rows_t) / pr.sum(-1, keepdim=True)).float()
            att_k = G["attn"].float().view(T, NH, KVL)
            check("attention (the kernel's query, pools, cache)", relrms(att_k, att_ref) < 2e-2,
                  f"relRMS {relrms(att_k, att_ref):.2e}")
            v_ref = bf(torch.einsum("thl,hld->thd", att_k, Ws[0]["w_uv"].float())).reshape(T, NH * DV)
            check("v (W_UV)", relrms(G["v_out"], v_ref) < 1e-2, f"relRMS {relrms(G['v_out'], v_ref):.2e}")
            # the chain: every GPU's v from its own kernel outputs -> o_proj -> mHC
            acc = torch.zeros(T, H)
            for g in range(n):
                acc = acc + bf(ws[g]["v_out"].cpu().float() @ ws[g]["w_o"].cpu().float().t()).float()
            R2 = hc_post_ref(bf(acc), R, hp, hr)
            x_ref, _, _ = hc_pre_ref(R2, m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
        R2k = G["R_out"].view(T, HC, H)
        check(f"launch {launch}: R' vs the chain (the kernel's v)", relrms(R2k, R2) < 2e-2, f"relRMS {relrms(R2k, R2):.2e}")
        check(f"launch {launch}: x_ffn vs the chain", relrms(G["x_ffn"], x_ref) < 3e-2, f"relRMS {relrms(G['x_ffn'], x_ref):.2e}")
        mh = ("R_out", "x_ffn", "h_post_out", "h_res_out")
        same = all(torch.equal(ws[g][k].cpu(), G[k]) for g in range(1, n) for k in mh)
        check(f"launch {launch}: every GPU's mHC outputs equal", same)
        if first is None:
            first = G
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G[k], first[k]) for k in G))
    if not failed and a.reps:
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
