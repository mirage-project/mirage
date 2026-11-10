"""Our GLM-5.3-Flash sublayer kernels (per-rank libraries: amd/serving/glm/build_glm_libs.sh) inside SGLang ROCm v0.5.22's
glm5_next.py, plain decode, one TP rank per process (TP4, no EP). The contract: amd/notes/sglang_v0522_glm_layer_contract.md.

  prepare(model)                 end of the model's load_weights: this rank's weights from the checkpoint (the harnesses'
                                 loaders and packing: amd/tests/test_glm_*_layer.py), the libraries (each a collective: the
                                 exchange regions over HIP IPC), per layer and batch size a library instance, the run buffers
  h = forward(model, ids, fb)    Glm5NextModel.forward on a decode step of a listed batch size (None: SGLang's path): embed,
                                 R = x.repeat(1, 4), layer 0's attention mHC pre (torch, SGLang's _mhc_pre_torch formulas);
                                 per layer i kernel A_i (KDA / DSA sublayer: attention, o_proj, all-reduce, attention mHC post,
                                 FFN mHC pre) then B_i (MoE / dense sublayer ... the next layer's attention mHC pre); then
                                 hc_contract (the mean over the streams) and the final RMSNorm
Our kernels read / write SGLang's caches in place: KDA conv / recurrent states at mamba_cache_indices, the DSA fp8 latent rows
at out_cache_loc, the pooled index-K pages and the tail rings; per step one gather page_table = req_to_token[req_pool_indices,
:MAXCTX] for the DSA layers (positions -> cache rows), pooled_bt = page_table[:, ::256] // 256.
Environment:
  MPK_GLM            "run": ours (unset: SGLang's path everywhere)
  MPK_GLM_TOKENS     the batch sizes with libraries (comma-separated; SGLang's captured decode sizes)
  MPK_GLM_DLIB       the libraries' directory (<kind>_t<T>/lib.so, kinds gkda gdsa gmoe gdense)
  MPK_GLM_MODEL      the checkpoint directory (default /dev/shm/glm53f)
  MPK_GLM_PAIR_DSA   "1" (default): DSA + MoE layers as one kernel too (kind gdsamoe: test_glm_dsa_layer.py --pair-moe), up to
                     MPK_GLM_PAIR_DSA_MAX_T (default 64)
  MPK_GLM_PAIR       "1" (default; "0" off): a KDA layer with a MoE MLP runs as ONE kernel (kind gkdamoe: amd/tests/test_glm_pair_layer.py
                     --codegen --serving); its next-layer input goes to the other of two x_attn buffers;
                     MPK_GLM_PAIR_MAX_T: only at batch sizes up to it (default 64; DLIB needs gkdamoe_t<T> for those)
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for d in (os.path.join(ROOT, "amd", "serving"), os.path.join(ROOT, "amd", "tests"), os.path.join(ROOT, "amd", "tools"),
          os.path.join(ROOT, "demo", "glm53_flash")):
    if d not in sys.path:
        sys.path.insert(0, d)
from rank_lib import RankKernel  # noqa: E402

H, HC, D = 4096, 4, 128
MAXCTX = 8192
ON = os.environ.get("MPK_GLM", "") == "run"
TOKENS = sorted({int(x) for x in os.environ.get("MPK_GLM_TOKENS", "").split(",") if x})
DLIB = os.environ.get("MPK_GLM_DLIB", "/mpk-amd/glm/dlib")
MODEL = os.environ.get("MPK_GLM_MODEL", "/dev/shm/glm53f")
# MPK_GLM_DEBUG_SYNC=1 (with --disable-cuda-graph): after every kernel a device sync and a line (layer, kind, finite outputs)
DEBUG_SYNC = os.environ.get("MPK_GLM_DEBUG_SYNC", "") == "1"
# MPK_GLM_DUMP=dir (with DEBUG_SYNC): each B kernel's inputs saved before its launch (dir/r<rank>_t<T>_l<layer>_B.pt, the
# first decode step of each T), to replay a fault outside SGLang (amd/tests/test_glm_rank_lib.py --inputs)
DUMP = os.environ.get("MPK_GLM_DUMP", "")
# (MI355X 11-08 glmab22, same session: pair 2.05 / 1.89 / 1.64 / 1.50 / 1.21 / 1.22 / 1.10 vs 1.98 / 1.84 / 1.63 / 1.48 / 1.18 /
# 1.12 / 1.11 at bs 1..64; glmgsm3 GSM8K 0.907 with pair at every T, SGLang 0.908)
PAIR = os.environ.get("MPK_GLM_PAIR", "1") == "1"
# (glmab44 A B A B at bs 64, pairs to T 64 for KDA and DSA (dlib30: gkdamoe / gdsamoe / gkdamoelast T 64) vs to 32: 15725.1 / 16211.3 vs
# 16544.3 / 16397.3 us)
PAIR_MAX_T = int(os.environ.get("MPK_GLM_PAIR_MAX_T", "64"))   # (the pair kernels only up to this batch size)
# (DSA + MoE layers as one kernel too, kind gdsamoe, up to MPK_GLM_PAIR_DSA_MAX_T; MI355X 11-08 glmab23 same session: bs 1 / 2 / 4 / 8
# 4111.0 / 4717.1 / 5975.4 / 7588.3 vs 4213.5 / 4794.3 / 6000.4 / 7434.8 us)
PAIR_DSA = os.environ.get("MPK_GLM_PAIR_DSA", "1") == "1"
# (glmab35 A B A B with gdsamoe T 8 / 16 / 32 rebuilt: DSA pairs to T 32 vs 4 bs 8 7484.0 / 7426.4 vs 7499.0 / 7401.7, bs 16 9608.0 /
# 9741.3 vs 9706.4 / 9935.5, bs 32 11869.0 / 11864.7 vs 12011.0 / 11824.9 us)
PAIR_DSA_MAX_T = int(os.environ.get("MPK_GLM_PAIR_DSA_MAX_T", "64"))
# (KDA + dense MLP layers 0..2 as one kernel, kind gkdadense; MI355X 11-08 glmab27 A B A B: bs 1 / 2 / 4 / 8 3951.3 / 4590.4 / 5905.1 /
# 7534.4 and 3950.9 / 4588.9 / 5852.2 / 7474.1 vs 3964.5 / 4620.0 / 5906.2 / 7486.3 and 3962.6 / 4628.1 / 5903.7 / 7483.8 us)
# (consecutive KDA + MoE pair layers as ONE kernel, up to MPK_GLM_CHAIN layers (2 or 3; 1 = off), kinds gkdamoe2 / gkdamoe3
# (amd/tests/test_glm_chain_layer.py), at the batch sizes in MPK_GLM_CHAIN_TS (comma list))
# MI355X 11-08 glmab30 (dlib26, A B A B): chain 3 vs off bs 1 3950.4 / 3938.5 vs 3919.2 / 3934.1, bs 2 4620.3 / 4598.6 vs 4577.1 /
# 4585.3, bs 4 5836.8 / 5814.4 vs 5885.6 / 5863.6, bs 8 7633.8 / 7535.7 vs 7476.1 / 7515.6 us -> on at T 4 only (MPK_GLM_CHAIN_TS)
CHAIN = int(os.environ.get("MPK_GLM_CHAIN", "3"))
# (glmab31 A B A B, chains also at T 16 / 32: bs 16 9844.9 / 9871.5 vs 9912.8 / 9892.8, bs 32 12121.7 / 11847.6 vs 11954.6 / 12028.0)
# (11-09 glmab64, chains with hybrid projections (chain_hyb libraries, dlib42) at T 1 / 2 / 8: bs 1 3723.8 / 3735.2 vs 3885.4 / 3884.8, bs 2
# 4427.0 / 4464.0 vs 4534.6 / 4538.2, bs 8 7484.6 / 7443.7 vs 7406.9 / 7385.5 us -> on at T 1 / 2 / 4 / 16)
CHAIN_TS = {int(x) for x in os.environ.get("MPK_GLM_CHAIN_TS", "1,2,4,16").split(",") if x}
# (MPK_GLM_LAST=1: the last layer, a KDA + MoE pair at that T, without its final hc_pre - kind gkdamoelast)
# (MI355X 11-09 glmab41 A B A B: bs 1 / 2 / 4 3914.3 / 4569.7 / 5812.5 and 3926.3 / 4570.1 / 5814.1 vs 3925.1 / 4576.0 / 5820.7 and
# 3927.0 / 4583.3 / 5833.7 us; bs 8..32 within the spread; GSM8K 0.921. Needs gkdamoelast_t<T> in DLIB, e.g. dlib30)
LAST = os.environ.get("MPK_GLM_LAST", "1") == "1"
PAIR_DENSE = os.environ.get("MPK_GLM_PAIR_DENSE", "1") == "1"
# (the dense layers 0..2 as one kernel up to T 64: MI355X 11-09 glmab55 dense-layer kernel sums bs 16 355.6 vs 373.1, bs 32 413 vs 426, bs 64
# 555 vs 567 us; needs gkdadense_t8 .. t64 in DLIB, e.g. dlib36)
PAIR_DENSE_MAX_T = int(os.environ.get("MPK_GLM_PAIR_DENSE_MAX_T", "64"))
_S = {}   # the process's state (prepare)


def _log(msg):
    print(f"[mpk-glm rank {_S.get('rank', '?')}] {msg}", flush=True)


def _bf(x):
    return x.to(torch.bfloat16)


# ---- this rank's weights (the harnesses' layouts) ----
def _mhc(ck, layer, which, dev):
    p = f"model.language_model.layers.{layer}."
    norm = "post_attention_layernorm.weight" if which == "ffn" else "input_layernorm.weight"
    fn, base, scale = ck.get(p + f"hc_{which}_fn").float(), ck.get(p + f"hc_{which}_base").float(), ck.get(p + f"hc_{which}_scale").float()
    small = torch.zeros(32, dtype=torch.float32)
    small[:HC * (HC + 2)], small[HC * (HC + 2):HC * (HC + 2) + 3] = base, scale
    pre = "hc_ffn" if which == "ffn" else "hc_attn"
    return {f"{pre}_fn": fn.to(dev), f"{pre}_small": small.to(dev),
            ("ffn_norm" if which == "ffn" else "attn_norm"): ck.get(p + norm).to(torch.bfloat16).to(dev)}, \
        {"fn": fn, "base": base, "scale": scale, "gamma": ck.get(p + norm)}


def _kda_weights(model_dir, layer, rank, n, dev):
    import kda_reference as kref
    W = kref.load_kda_layer(kref.Checkpoint(model_dir), layer)
    NH = W["num_heads"]
    HL, HD = NH // n, NH // n * D
    col_ga, col_fa, col_beta = 3 * HD, 3 * HD + 128, 3 * HD + 256
    proj_n = -(-(col_beta + HL) // 128) * 128
    cs, hs = slice(rank * HD, (rank + 1) * HD), slice(rank * HL, (rank + 1) * HL)
    wproj = torch.zeros(proj_n, H, dtype=torch.bfloat16)
    wproj[:HD], wproj[HD:2 * HD], wproj[2 * HD:3 * HD] = W["q"][cs], W["k"][cs], W["v"][cs]
    wproj[col_ga:col_ga + 128], wproj[col_fa:col_fa + 128], wproj[col_beta:col_beta + HL] = W["g_a"], W["f_a"], W["b"][hs]
    return {"wproj": wproj.to(dev), "wfb": W["f_b"][cs].contiguous().to(dev), "wgb": W["g_b"][cs].contiguous().to(dev),
            "prm": torch.cat([W["A_log"][hs], W["dt_bias"][cs], W["o_norm"]]).float().to(dev),
            "conv_w": torch.cat([W["q_conv"][cs], W["k_conv"][cs], W["v_conv"][cs]], 0).float().contiguous().to(dev),
            "w_o": W["o_proj"][:, cs].contiguous().to(dev)}, HD


def _deq(w, s):
    return _bf(w.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:w.shape[0], :w.shape[1]])


def _dsa_weights(sk, layer, rank, n, dev, HEADS=64, QL=1536, KVL=512, NOPE=256, DV=256, H_IDX=32, ID=128, KGW=384):
    """this rank's DSA tensors (upstream demo/glm53_flash dsa_mla.load_layer / dsa_layer.load_indexer's selection, without
    their mirage imports): [q_a | kv_a] FP8 dequantized to bf16 (replicated), q_b / o_proj FP8 dequantized (this rank's
    heads), W_UK / W_UV from kv_b_proj (bf16), the norms; the indexer (replicated, bf16)"""
    p = f"model.language_model.layers.{layer}.self_attn."
    NH = HEADS // n
    hs, hb = slice(rank * NH * DV, (rank + 1) * NH * DV), slice(rank * NH * 2, (rank + 1) * NH * 2)
    g = lambda nm: sk.get(p + nm)
    qkva = torch.cat([g("q_a_proj.weight").view(torch.uint8), g("kv_a_proj_with_mqa.weight").view(torch.uint8)]) \
        .view(torch.float8_e4m3fn)
    qkva_s = torch.cat([g("q_a_proj.weight_scale_inv"), g("kv_a_proj_with_mqa.weight_scale_inv")]).float()
    kvb = g("kv_b_proj.weight").to(torch.bfloat16).view(HEADS, NOPE + DV, KVL)[rank * NH:(rank + 1) * NH]
    w_uk, w_uv = kvb[:, :NOPE].contiguous(), kvb[:, NOPE:].transpose(1, 2).contiguous()   # [NH, 256, 512], [NH, 512, 256]
    ip = p + "indexer."
    gi = lambda nm: sk.get(ip + nm)
    wk, gate, wp = gi("wk.weight"), gi("index_kpool_compress_gate"), gi("weights_proj.weight")
    kgw = torch.cat([wk, gate, wp, torch.zeros(KGW - 2 * ID - H_IDX, H, dtype=torch.bfloat16)]).to(torch.bfloat16)
    return {"w_qkva": _deq(qkva, qkva_s).to(dev), "q_norm": _bf(g("q_a_layernorm.weight").float()).to(dev),
            "kv_norm": _bf(g("kv_a_layernorm.weight").float()).to(dev), "w_kgw": kgw.contiguous().to(dev),
            "k_norm_w": gi("k_norm.weight").float().to(dev), "k_norm_b": gi("k_norm.bias").float().to(dev),
            "ape": gi("index_kpool_compress_ape").float().contiguous().to(dev),
            "w_iqb": gi("wq_b.weight").to(torch.bfloat16).contiguous().to(dev),
            "w_qb": _deq(g("q_b_proj.weight")[hs], g("q_b_proj.weight_scale_inv")[hb]).to(dev),
            "w_kc": w_uk.transpose(1, 2).contiguous().view(NH * KVL, NOPE).to(dev),
            "w_vc": w_uv.transpose(1, 2).contiguous().view(NH * DV, KVL).to(dev),
            "w_o": _deq(g("o_proj.weight")[:, hs].contiguous(), g("o_proj.weight_scale_inv")[:, hb]).to(dev)}, NH


class _Slices:
    """this rank's slices of checkpoint tensors (safetensors get_slice: only the rows / columns read)"""
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def sl(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.files[f].get_slice(name)

    def get(self, name):
        return self.sl(name)[:]


def _moe_weights(sk, layer, rank, n, dev, NB=289, I_FULL=2048, NL=384, NE_PAD=320):
    """the bank of this rank (amd/tests/test_glm_moe_layer.py pack_experts' layout, packed on the GPU): W13 [NB 2 IR, H]
    units of 16 gate + 16 up rows in the load order, scales [NB][IR / 128][H / 128][2], W2 [NB H, IR] 16-row tiles in the
    load order, scales [NB][H / 128][IR / 128]; router [384, H] bf16, bias [320] (-inf past the 288)"""
    p = f"model.language_model.layers.{layer}.mlp."
    IR = I_FULL // n
    r0, r1, b0, b1 = rank * IR, (rank + 1) * IR, rank * IR // 128, (rank + 1) * IR // 128
    w13 = torch.empty(NB * 2 * IR, H, dtype=torch.uint8, device=dev)
    s13 = torch.empty(NB, IR // 128, H // 128, 2, dtype=torch.float32, device=dev)
    w2 = torch.empty(NB * H, IR, dtype=torch.uint8, device=dev)
    s2 = torch.empty(NB, H // 128, IR // 128, dtype=torch.float32, device=dev)
    for e in range(NB):
        q = p + (f"experts.{e}." if e < NB - 1 else "shared_experts.")
        gq = sk.sl(q + "gate_proj.weight")[r0:r1].to(dev).view(torch.uint8)
        uq = sk.sl(q + "up_proj.weight")[r0:r1].to(dev).view(torch.uint8)
        dq = sk.sl(q + "down_proj.weight")[:, r0:r1].to(dev).contiguous().view(torch.uint8)
        gs = sk.sl(q + "gate_proj.weight_scale_inv")[b0:b1].float().to(dev)
        us = sk.sl(q + "up_proj.weight_scale_inv")[b0:b1].float().to(dev)
        ds = sk.sl(q + "down_proj.weight_scale_inv")[:, b0:b1].float().to(dev)
        u13 = torch.stack([gq.reshape(IR // 16, 16, H), uq.reshape(IR // 16, 16, H)], 1)
        w13[e * 2 * IR:(e + 1) * 2 * IR] = u13.reshape(IR // 16, 2, 16, H // 128, 4, 2, 16).permute(0, 3, 1, 5, 4, 2, 6) \
            .reshape(2 * IR, H)
        s13[e] = torch.stack([gs, us], -1)
        w2[e * H:(e + 1) * H] = dq.reshape(H // 16, 16, IR // 128, 4, 2, 16).permute(0, 2, 4, 3, 1, 5).reshape(H, IR)
        s2[e] = ds
    router = torch.zeros(NL, H, dtype=torch.bfloat16, device=dev)
    router[:NB - 1] = sk.get(p + "gate.weight").to(torch.bfloat16).to(dev)
    # (dtype explicit: SGLang loads the model with torch's default dtype set to bf16 - a bf16 bias here (glmsrv9: 640 B) was
    # read as 320 fp32 by glm_topk_route: garbage biases and a read past its end, faulting where the next page was unmapped)
    bias = torch.full((NE_PAD,), float("-inf"), dtype=torch.float32, device=dev)
    bias[:NB - 1] = sk.get(p + "gate.e_score_correction_bias").float().to(dev)
    return {"router": router, "bias": bias, "w13": w13, "w13_sf": s13.reshape(-1), "w2": w2, "w2_sf": s2.reshape(-1)}


def _dense_weights(sk, layer, rank, n, dev, I_FULL=12288):
    p = f"model.language_model.layers.{layer}.mlp."
    F = I_FULL // n
    r0, r1, b0, b1 = rank * F, (rank + 1) * F, rank * F // 128, (rank + 1) * F // 128

    def dq(w, s):
        return _bf(w.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:w.shape[0], :w.shape[1]])
    g = dq(sk.sl(p + "gate_proj.weight")[r0:r1].to(dev), sk.sl(p + "gate_proj.weight_scale_inv")[b0:b1].to(dev))
    u = dq(sk.sl(p + "up_proj.weight")[r0:r1].to(dev), sk.sl(p + "up_proj.weight_scale_inv")[b0:b1].to(dev))
    d = dq(sk.sl(p + "down_proj.weight")[:, r0:r1].to(dev), sk.sl(p + "down_proj.weight_scale_inv")[:, b0:b1].to(dev))
    return {"w_gu": torch.cat([g, u]).contiguous(), "w_dn": d.contiguous()}, F


# the weights' dtypes the kernels were built for (amd/tests/test_glm_*_layer.py): checked in prepare (SGLang loads with torch's
# default dtype bf16: a tensor made without an explicit dtype silently becomes bf16)
_F32 = {"bias", "w13_sf", "w2_sf", "prm", "conv_w", "k_norm_w", "k_norm_b", "ape", "hc_ffn_fn", "hc_ffn_small", "hc_attn_fn",
        "hc_attn_small"}
_U8 = {"w13", "w2"}


def _check_dtypes(layer, w):
    for k, v in w.items():
        want = torch.float32 if k in _F32 else torch.uint8 if k in _U8 else torch.bfloat16
        if v.dtype != want or not v.is_contiguous():
            raise RuntimeError(f"[mpk-glm] layer {layer}: weight {k} is {v.dtype} {tuple(v.shape)} contiguous "
                               f"{v.is_contiguous()}, the kernel reads {want}")


def prepare(model):
    """the libraries, this rank's layer instances, the run buffers (once, after the weights are loaded)"""
    if not ON or not TOKENS:
        return
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group
    import test_glm_moe_layer as tm
    t0 = time.time()
    group = get_tp_group().device_group
    rank, n = dist.get_rank(group), dist.get_world_size(group)
    dev = torch.device("cuda", torch.cuda.current_device())
    _S.update(rank=rank, n=n, dev=dev)
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    cfg = cfg.get("text_config", cfg)
    L = cfg["num_hidden_layers"]
    attn_types = ["dsa" if (i % 4 == 3) else "kda" for i in range(L)]   # (GLM-5.3-Flash: layers 3, 7, .., 43 DSA)
    if "layer_types" in cfg:
        attn_types = ["kda" if "linear" in t else "dsa" for t in cfg["layer_types"]]
    mlp_types = cfg.get("mlp_layer_types", ["dense"] * cfg["first_k_dense_replace"] + ["sparse"] * (L - cfg["first_k_dense_replace"]))
    _S.update(L=L, attn=attn_types, mlp=mlp_types, rms_eps=cfg["rms_norm_eps"], hc_eps=cfg["hc_eps"],
              iters=cfg["hc_sinkhorn_iters"])
    paired = [PAIR and ((mlp_types[i] == "sparse" and (attn_types[i] == "kda" or PAIR_DSA)) or
                        (mlp_types[i] != "sparse" and attn_types[i] == "kda" and PAIR_DENSE)) for i in range(L)]
    # (the batch sizes up to which layer i runs as one kernel)
    pmax = [(PAIR_DENSE_MAX_T if mlp_types[i] != "sparse" else PAIR_MAX_T if attn_types[i] == "kda"
             else min(PAIR_MAX_T, PAIR_DSA_MAX_T)) if paired[i] else 0 for i in range(L)]
    pkind = [f"g{attn_types[i]}{'moe' if mlp_types[i] == 'sparse' else 'dense'}" for i in range(L)]
    if LAST and pkind[L - 1] == "gkdamoe" and paired[L - 1]:
        pkind[L - 1] = "gkdamoelast"
    _S["pmax"] = pmax
    _S["pkind"] = pkind
    kinds = ({("g" + attn_types[i]) for i in range(L) if any(T > pmax[i] for T in TOKENS)} |
             {("gmoe" if mlp_types[i] == "sparse" else "gdense") for i in range(L) if any(T > pmax[i] for T in TOKENS)})
    _S["paired"] = paired
    # (chains: per T, start layer -> length; runs of KDA + MoE pair layers cut into pieces of at most CHAIN)
    kchain = [paired[i] and attn_types[i] == "kda" and mlp_types[i] == "sparse" for i in range(L)]
    groups = {}
    for T in TOKENS:
        groups[T] = {}
        if CHAIN < 2 or T not in CHAIN_TS:
            continue
        i = 0
        while i < L:
            if kchain[i] and T <= pmax[i]:
                j = i
                while j < L and kchain[j] and T <= pmax[j] and j - i < CHAIN:
                    j += 1
                if j - i >= 2:
                    groups[T][i] = j - i
                i = j
            else:
                i += 1
    _S["groups"] = groups
    libs = {}
    for T in TOKENS:
        pk = {pkind[i] for i in range(L) if T <= pmax[i]} | {f"gkdamoe{nn}" for nn in groups[T].values()}
        for k in sorted(kinds | pk):
            path = os.path.join(DLIB, f"{k}_t{T}", "lib.so")
            libs[(k, T)] = RankKernel(path, group, dev)
    _log(f"libraries {sorted(libs)} ({time.time() - t0:.0f} s)")
    ck = _Slices(MODEL)
    layers, keep = {}, []
    hd = None
    nh_dsa = None
    for i in range(L):
        a, m = attn_types[i], mlp_types[i]
        if a == "kda":
            wa, hd = _kda_weights(MODEL, i, rank, n, dev)
        else:
            wa, nh_dsa = _dsa_weights(ck, i, rank, n, dev)
        mf, _ = _mhc(ck, i, "ffn", dev)
        wa.update(mf)
        wb = _moe_weights(ck, i, rank, n, dev) if m == "sparse" else _dense_weights(ck, i, rank, n, dev)[0]
        ma, _ = _mhc(ck, min(i + 1, L - 1), "attn", dev)   # (the last layer: its own attention pre, results unused)
        wb.update(ma)
        _check_dtypes(i, wa)
        _check_dtypes(i, wb)
        keep += [wa, wb]
        for T in TOKENS:
            if T <= pmax[i]:
                layers[("P", i, T)] = libs[(pkind[i], T)].create_layer({**wa, **wb})
                continue
            layers[("A", i, T)] = libs[("g" + a, T)].create_layer(wa)
            layers[("B", i, T)] = libs[("gmoe" if m == "sparse" else "gdense", T)].create_layer(wb)
        if i % 8 == 0:
            _log(f"layer {i} weights ({time.time() - t0:.0f} s, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)")
    # (a chain library built with MPK_CHAIN_HYB - the marker file chain_hyb next to lib.so - projects layers 1 .. with hybrid gemms,
    # which read gemm_tile's shuffled layout: those layers' wproj shuffled once, shared by every T; MI355X 11-09 glmchain6, chain 3
    # T 16 672.6 / 673.7 vs 687.4 / 684.7 us, T 4 368.6 / 369.0 vs 380.8 / 383.8)
    def shuffle_gemm_weight(w):   # (static_megakernel.shuffle_gemm_weight: mirage is not importable in the server process)
        N, K = w.shape
        return w.reshape(N // 16, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().reshape(N, K)
    shuf = {}
    for T in TOKENS:   # (the chains' instances: every layer's weights suffixed _j)
        for i0, nn in groups[T].items():
            hyb = os.path.exists(os.path.join(DLIB, f"gkdamoe{nn}_t{T}", "chain_hyb"))
            wc = {}
            for j in range(nn):
                wc.update({f"{k}_{j}": v for k, v in {**keep[2 * (i0 + j)], **keep[2 * (i0 + j) + 1]}.items()})
                if hyb and j > 0:
                    if i0 + j not in shuf:
                        shuf[i0 + j] = shuffle_gemm_weight(keep[2 * (i0 + j)]["wproj"])
                    wc[f"wproj_{j}"] = shuf[i0 + j]
            layers[("C", i0, T)] = libs[(f"gkdamoe{nn}", T)].create_layer(wc)
    _, m0 = _mhc(ck, 0, "attn", dev)
    _S.update(libs=libs, layers=layers, keep=keep, m0={k: v.to(dev) for k, v in m0.items()}, hd=hd, nh_dsa=nh_dsa)
    # run buffers per batch size (persistent: the captured graphs hold their pointers)
    bufs = {}
    KP = tm.TOPK + 1
    for T in TOKENS:
        z = lambda *sh, dt=torch.bfloat16: torch.zeros(*sh, dtype=dt, device=dev)
        bufs[T] = {"R": [z(T, HC * H), z(T, HC * H)], "hp": [z(T, HC, dt=torch.float32), z(T, HC, dt=torch.float32)],
                   "hr": [z(T, HC * HC, dt=torch.float32), z(T, HC * HC, dt=torch.float32)],
                   "x_attn": z(T, H), "x_attn2": z(T, H), "x_ffn": z(T, H), "R_mid": z(T, HC * H), "hfin": z(T, H),
                   "hp_mid": z(T, HC, dt=torch.float32), "hr_mid": z(T, HC * HC, dt=torch.float32), "kda_y": z(T, hd), "pairs": z(T, KP, dt=torch.int64),
                   "R_moe": z(T * KP, H, dt=torch.float32), "h_mlp": z(T, 12288 // n),
                   "q_c": z(T, 1536), "scores": z(T, MAXCTX // 4, dt=torch.float32), "idx_q": z(T, 32 * 128),
                   "pool_idx": z(T, 512, dt=torch.int32), "q_mla": z(T, nh_dsa * 512), "attn": z(T, nh_dsa * 512),
                   "v_out": z(T, nh_dsa * 256), "req_idx": torch.arange(T, dtype=torch.int64, device=dev),
                   # (chains: the streams between the sublayers, run tensors)
                   "R_a": [z(T, HC * H) for _ in range(3)], "hp_a": [z(T, HC, dt=torch.float32) for _ in range(3)],
                   "hr_a": [z(T, HC * HC, dt=torch.float32) for _ in range(3)], "R_b": [z(T, HC * H) for _ in range(2)],
                   "hp_b": [z(T, HC, dt=torch.float32) for _ in range(2)], "hr_b": [z(T, HC * HC, dt=torch.float32) for _ in range(2)]}
    _S["bufs"] = bufs
    if DEBUG_SYNC:   # (diagnosis) the address ranges, to place a fault report's address
        for i in range(L):
            for nm, w in (("A", keep[2 * i]), ("B", keep[2 * i + 1])):
                rng = " ".join(f"{k}=[{v.data_ptr():#x},{v.data_ptr() + v.numel() * v.element_size():#x})" for k, v in w.items())
                _log(f"ranges layer {i} {nm}: {rng}")
        for T in TOKENS:
            rng = " ".join(f"{k}=[{v.data_ptr():#x},{v.data_ptr() + v.numel() * v.element_size():#x})"
                           for k, v in bufs[T].items() if torch.is_tensor(v))
            _log(f"ranges bufs T {T}: {rng}")
    _log(f"ready: {L} layers x {len(TOKENS)} sizes ({time.time() - t0:.0f} s, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)")


def _hc_pre(R, m, rms_eps, hc_eps, iters):
    """layer 0's attention mHC pre: SGLang's own hc_pre (aiter's fused mhc_pre + RMSNorm on gfx950; MI355X 11-07 glmab1: the
    torch formulas were ~120 small kernels, ~475 us of a 5.9-ms bs-1 step), else the torch reference"""
    T = R.shape[0]
    try:
        from sglang.kernels.ops.layernorm.mhc import hc_pre
        x, hr, hp, fused = hc_pre(R, m["fn"], m["scale"], m["base"], HC, rms_eps, hc_eps, iters, 2.0, None, m["gamma"], rms_eps)
        if not fused:
            xf = x.float()
            x = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + rms_eps) * m["gamma"].float()).to(torch.bfloat16)
        return x, hp.reshape(T, HC).float(), hr.reshape(T, HC * HC).float()
    except ImportError:
        from test_glm_mhc import hc_pre_ref
        x, hp, hr = hc_pre_ref(R.view(T, HC, H), m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
        return x, hp.reshape(T, HC).float(), hr.reshape(T, HC * HC).float()


def _check_once(T, i, io):
    """the first call per (T, layer kind): the SGLang tensors' dtype / shape / stride logged; a strided one is an error (the
    kernels read them by pointer as dense rows)"""
    key = ("chk", T, i)
    if key in _S:
        return
    _S[key] = True
    lines = []
    for k, v in io.items():
        lines.append(f"{k} {str(v.dtype).replace('torch.', '')} {tuple(v.shape)} st {tuple(v.stride())}")
        if not v.is_contiguous():
            raise RuntimeError(f"[mpk-glm] T {T} layer {i}: {k} is not contiguous ({tuple(v.shape)} strides {tuple(v.stride())})")
    _log(f"T {T} layer {i} run tensors: " + "; ".join(lines))


def active(forward_batch) -> bool:
    if not ON or "layers" not in _S:
        return False
    fm = forward_batch.forward_mode
    return fm.is_decode() and forward_batch.input_ids.shape[0] in _S["bufs"]


def forward(model, input_ids, forward_batch, input_embeds=None):
    """one decode step with our kernels (Glm5NextModel.forward's result: the final-normed hidden [T, H]); input_embeds: the
    multimodal wrapper's (general_mm_embed_routine embeds the tokens and passes input_ids None)"""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    h = input_embeds if input_embeds is not None else model.embed_tokens(input_ids)
    T = h.shape[0]
    b, layers = _S["bufs"][T], _S["layers"]
    b["R"][0].view(T, HC, H).copy_(h.view(T, 1, H).expand(T, HC, H))   # (one kernel: R = x repeated over the streams)
    # (layer 0 reads the mHC pre's outputs in place - inside a captured graph their addresses are fixed - no copies into the
    # persistent buffers: MI355X 11-08 glmab28 bs 2 trace, 4 copyBuffer + 2 copy kernels ~27 us per step)
    x0, hp0, hr0 = _hc_pre(b["R"][0], _S["m0"], _S["rms_eps"], _S["hc_eps"], _S["iters"])
    x0, hp0, hr0 = x0.contiguous(), hp0.contiguous(), hr0.contiguous()
    be = get_attn_backend()
    lin = be.linear_attn_backend.forward_metadata
    slots = lin.mamba_cache_indices[:T]
    full = be.full_attn_backend.forward_metadata
    seq32 = full.cache_seqlens_int32[:T]
    rtt = be.req_to_token_pool.req_to_token
    page_table = rtt[forward_batch.req_pool_indices[:T], :MAXCTX].contiguous()
    pooled_bt = (page_table[:, ::256] // 256).contiguous()
    pool = be.token_to_kv_pool
    cur = 0
    xa, xi = [x0, b["x_attn2"]], 0   # (a paired layer reads xa[xi], writes the next layer's input into xa[1 - xi])
    skip_to = 0
    for i in range(_S["L"]):
        if i == 1:
            xa[0] = b["x_attn"]   # (from layer 1 on the two persistent buffers; layer 0's x was the mHC pre's output)
        if i < skip_to:
            continue
        nn = _S["groups"][T].get(i, 0)
        if nn:   # (a chain of nn KDA + MoE layers: one kernel)
            io = {"R_in": b["R"][cur], "h_post_in": b["hp"][cur], "h_res_in": b["hr"][cur], "R_out": b["R"][1 - cur],
                  "h_post_out": b["hp"][1 - cur], "h_res_out": b["hr"][1 - cur], "x_attn": xa[xi], "x_attn_next": xa[1 - xi],
                  "slots": slots}
            for j in range(nn):
                c = be.req_to_token_pool.mamba2_layer_cache(i + j)
                io.update({f"state_pool_{j}": c.temporal, f"conv_pool_{j}": c.conv[0], f"R_a_{j}": b["R_a"][j],
                           f"h_post_a_{j}": b["hp_a"][j], f"h_res_a_{j}": b["hr_a"][j]})
                if j < nn - 1:
                    io.update({f"R_b_{j}": b["R_b"][j], f"h_post_b_{j}": b["hp_b"][j], f"h_res_b_{j}": b["hr_b"][j]})
            layers[("C", i, T)].run(io)
            if DEBUG_SYNC:
                torch.cuda.synchronize()
                _log(f"T {T} layers {i}..{i + nn - 1} C done: R_out finite {bool(torch.isfinite(io['R_out'].float()).all())}")
            cur, xi = 1 - cur, 1 - xi
            skip_to = i + nn
            continue
        hp_in = hp0 if i == 0 else b["hp"][cur]
        hr_in = hr0 if i == 0 else b["hr"][cur]
        if T <= _S["pmax"][i]:
            io = {"R_in": b["R"][cur], "h_post_in": hp_in, "h_res_in": hr_in, "R_out": b["R"][1 - cur],
                  "h_post_out": b["hp"][1 - cur], "h_res_out": b["hr"][1 - cur], "x_attn": xa[xi], "x_attn_next": xa[1 - xi],
                  "R_mid": b["R_mid"], "h_post_mid": b["hp_mid"], "h_res_mid": b["hr_mid"],
                  "hfin": b["hfin"]}   # (gkdamoelast: R_out and the streams' mean hfin; the other outputs unused)
            if _S["attn"][i] == "kda":
                c = be.req_to_token_pool.mamba2_layer_cache(i)
                io.update(state_pool=c.temporal, conv_pool=c.conv[0], slots=slots)
            else:
                tk, ts = pool.get_compress_tail_buffers(i)
                io.update(cache=pool.get_key_buffer(i).view(torch.uint8), out_loc=forward_batch.out_cache_loc[:T], seq_lens=seq32,
                          req_slots=forward_batch.req_pool_indices[:T], pooled_bt=pooled_bt, tail_k=tk, tail_score=ts,
                          pcache=pool.get_index_k_with_scale_buffer(i), r2t=page_table, req_idx=b["req_idx"])
            layers[("P", i, T)].run(io)
            if DEBUG_SYNC:
                torch.cuda.synchronize()
                _log(f"T {T} layer {i} P ({_S['attn'][i]} + {_S['mlp'][i]}) done: R_out finite {bool(torch.isfinite(io['R_out'].float()).all())}, "
                     f"x_attn_next finite {bool(torch.isfinite(io['x_attn_next'].float()).all())}")
            cur, xi = 1 - cur, 1 - xi
            continue
        io = {"R_in": b["R"][cur], "h_post_in": hp_in, "h_res_in": hr_in, "R_out": b["R"][1 - cur],
              "h_post_out": b["hp"][1 - cur], "h_res_out": b["hr"][1 - cur], "x_attn": xa[xi], "x_ffn": b["x_ffn"]}
        if _S["attn"][i] == "kda":
            c = be.req_to_token_pool.mamba2_layer_cache(i)
            io.update(state_pool=c.temporal, conv_pool=c.conv[0], slots=slots, kda_y=b["kda_y"])
        else:
            tk, ts = pool.get_compress_tail_buffers(i)
            io.update(cache=pool.get_key_buffer(i).view(torch.uint8), out_loc=forward_batch.out_cache_loc[:T], seq_lens=seq32,
                      req_slots=forward_batch.req_pool_indices[:T], pooled_bt=pooled_bt, tail_k=tk, tail_score=ts,
                      pcache=pool.get_index_k_with_scale_buffer(i), r2t=page_table, req_idx=b["req_idx"], q_c=b["q_c"],
                      scores=b["scores"], idx_q=b["idx_q"], pool_idx=b["pool_idx"], q_mla=b["q_mla"], attn=b["attn"],
                      v_out=b["v_out"])
        if i < 4:
            _check_once(T, i, io)
        layers[("A", i, T)].run(io)
        if DEBUG_SYNC:
            torch.cuda.synchronize()
            _log(f"T {T} layer {i} A ({_S['attn'][i]}) done: R_out finite {bool(torch.isfinite(io['R_out'].float()).all())}, "
                 f"x_ffn finite {bool(torch.isfinite(io['x_ffn'].float()).all())}")
        cur = 1 - cur
        io = {"R_in": b["R"][cur], "h_post_in": b["hp"][cur], "h_res_in": b["hr"][cur], "R_out": b["R"][1 - cur],
              "h_post_out": b["hp"][1 - cur], "h_res_out": b["hr"][1 - cur], "x_attn": b["x_attn"] if i == 0 else xa[xi],
              "x_ffn": b["x_ffn"]}
        if _S["mlp"][i] == "sparse":
            io.update(pairs=b["pairs"], R_moe=b["R_moe"])
        else:
            io.update(h_mlp=b["h_mlp"])
        if DUMP and ("dumped", T, i) not in _S:
            _S[("dumped", T, i)] = True
            os.makedirs(DUMP, exist_ok=True)
            torch.save({k: v.detach().cpu().clone() for k, v in io.items() if k in ("x_ffn", "R_in", "h_post_in", "h_res_in")},
                       os.path.join(DUMP, f"r{_S['rank']}_t{T}_l{i}_B.pt"))
        layers[("B", i, T)].run(io)
        if DEBUG_SYNC:
            torch.cuda.synchronize()
            _log(f"T {T} layer {i} B ({_S['mlp'][i]}) done: R_out finite {bool(torch.isfinite(io['R_out'].float()).all())}, "
                 f"x_attn finite {bool(torch.isfinite(io['x_attn'].float()).all())}")
        cur = 1 - cur
    if _S["pkind"][_S["L"] - 1] == "gkdamoelast" and T <= _S["pmax"][_S["L"] - 1]:
        return model.norm(b["hfin"])   # (the last kernel wrote the streams' mean)
    hfin = b["R"][cur].unflatten(-1, (HC, -1)).mean(dim=-2)   # (SGLang hc_contract)
    return model.norm(hfin)
