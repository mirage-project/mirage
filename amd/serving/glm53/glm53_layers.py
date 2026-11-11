"""GLM-5.3 (zai-org/GLM-5.3, GlmMoeDsaForCausalLM) on MI355X, TP 8: one GPU's weights of a layer and the graph of one layer as
ONE kernel - the attention sublayer (MLA + the DSA row list) then the MLP sublayer (MoE, or the dense MLP of layers 0 .. 2) - used by
the pair harness (amd/tests/test_glm53_pair_layer.py, --codegen --serving: the rank libraries) and by the server hook
(sglang_glm53.py). amd/notes/sglang_v0522_glm53_layer_contract.md. The math of each sublayer is checked by
amd/tests/test_glm53_attn_layer.py / test_glm53_moe_layer.py (the same task graphs).

Run tensors of a layer kernel (per step; the hook binds SGLang's own buffers by these names):
  x_attn [T, H] bf16 (the attention input: input_layernorm of the residual), res_in [T, H] bf16 (the residual stream),
  cache [rows, 576] uint8 (this layer's MLA cache), out_loc / positions [T] int64, seq_lens [T] int32, cos / sin [P, 32] bf16,
  full layers: icache [pages, 8448] uint8 (index-K), r2t [reqs, W] int32 (req_to_token), req_idx [T] int64, rows_g [T, 2048] int32
  (the list written for the shared layers after it); shared layers: rows [T, 2048] int32 (the list of the last full layer);
  res_mid [T, H] bf16 (the residual between the sublayers: a buffer of the caller's); outputs res_out [T, H] bf16 (the residual
  after the layer), x_next [T, H] bf16 (the next layer's input: its input_layernorm of res_out; the model's last layer: model.norm).
"""
import json
import os

import torch

STORE_GROUPS = 8   # (the store gemms counted in 8 row groups; g53prof5)
# both all-reduces fold the residual add + RMSNorm into their landing senders (allreduce_send_layer norm=, g53arn1: pair T 1 -6.4,
# T 2 -4.4, T 8 -7.3, T 64 -4 us; the attention's in kernel: x_ffn the senders' own output), and from T ARRS_MIN_T they run by token
# owner (allreduce_send RSN: residual_out only on each token's owner GPU; g53rs1: pair T 64 303.4 299.2 -> 271.7 271.3, T 32 221.4
# 226.1 -> 212.7 211.7 us)
ARRS_MIN_T = 16


def rs_on(T, n):
    return T >= ARRS_MIN_T and T % n == 0


def owner_rows(per_gpu, T, n):
    """RSN: a [T][H] tensor whose rows are valid only on their owner GPU (t // (T / n)): the owners' rows put together"""
    import torch as _t
    tpg = T // n
    return _t.cat([per_gpu[g][g * tpg:(g + 1) * tpg] for g in range(n)])   # (g53arn3: pair T 1 88.4 88.4 -> 87.4 85.5, T 8 134.0 132.8 -> 128.0 128.2, T 64 306.7 305.5 -> 303.9 304.9 us)

H, NHT, QL, KVL, NOPE, RP, DV, ID = 6144, 64, 2048, 512, 192, 64, 256, 128
NA = (QL + KVL + RP + 127) // 128 * 128   # (q_a | kv_a 2624 rows, padded to whole 128-row tiles)
E, TOPK, I_MOE, I_DENSE, SCALE = 256, 8, 2048, 12288, 2.5
K_TOP = 2048
# GLM-5.3's build settings (every harness and library build applies them; core.cuh's defaults stay GLM-5.3-Flash's):
# the all-reduce senders land the sums from T 1 (MI355X 11-10 g53l6, pair full L6 A B A B: T 1 127.4 / 125.9 -> 103.4 / 108.0 us,
# T 2 162.2 / 159.8 -> 111.1 / 109.7)
G53_FLAGS = ["-DSTATIC_ARS_LAND_MIN_T=1", "-DSTATIC_RBLK_PAR=1", "-DSTATIC_GME_W13DB=1", "-DSTATIC_GME_W2R=128", "-DSTATIC_GME_W2XQ=2", "-DSTATIC_GME_CUPOST=1"]   # (CUPOST: g53cup1 MoE T 8 -3.2, T 4 -3.0; pair T 4 -0.8, T 1 -0.6, T 8 level; W2XQ 2: g53xq1 pair T 16 -10.5, T 64 -12.3; g53xq2 T 32 -10.5, T 8 -3, T 1 -2, T 4 +1 us; W2R 128: g53w2r2 pair L7 T 4 100.2 102.6 -> 98.8 97.5, T 16 161.6 163.2 -> 159.8 158.4, T 1 2 8 level; W13DB: g53db1 MoE T 16 131.8 131.6 -> 125.0 124.7, T 32 167.1 168.5 -> 164.1 164.2, T 64 200.0 200.2 -> 190.1 190.2 us; RBLK_PAR: g53rbp2 pair L7 T 64 277.7 275.0 -> 266.6 266.2, T 32 215.0 214.1 -> 208.9 208.9 us)
# the attention / dense MLP gemms on the checkpoint's e4m3 weights + fp32 block scales (gemm_tile GEMM_FP8)
FP8 = True
G53_ENV = {"MPK_ARS_LAND_MIN_T": "1"}


def apply_settings(flags):
    """the harness's --flag list with G53_FLAGS (unless the same macro is given), G53_ENV into the environment (unless set)"""
    for k, v in G53_ENV.items():
        os.environ.setdefault(k, v)
    given = {f.split("=")[0] for f in flags}
    return flags + [f for f in G53_FLAGS if f.split("=")[0] not in given]
RUN_NAMES = {"x_attn", "res_in", "res_mid", "cache", "out_loc", "positions", "seq_lens", "cos", "sin", "icache", "r2t", "req_idx", "rows_g",
             "rows", "res_out", "x_next"}


def bf(x):
    return x.to(torch.bfloat16)


class Ckpt:
    """the checkpoint's tensors by name (safetensors, slices read on demand)"""

    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def _f(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.files[f]

    def get(self, name):
        return self._f(name).get_tensor(name)

    def sl(self, name):
        return self._f(name).get_slice(name)


def deq(w, s):   # e4m3 [N, K] with fp32 128 x 128 block scales -> fp32
    return w.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:w.shape[0], :w.shape[1]]


def _deq_rows(ck, name, r0, r1):   # rows [r0, r1) (whole 128-row blocks) of a block-fp8 weight, dequantized (fp32)
    assert r0 % 128 == 0 and r1 % 128 == 0
    w = ck.sl(name + ".weight")[r0:r1]
    s = ck.sl(name + ".weight_scale_inv")[r0 // 128:r1 // 128]
    return w.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:, :w.shape[1]]


def _deq_cols(ck, name, c0, c1):   # columns [c0, c1) (whole 128 blocks)
    w = ck.sl(name + ".weight")[:, c0:c1]
    s = ck.sl(name + ".weight_scale_inv")[:, c0 // 128:c1 // 128]
    return w.float() * s.float().repeat_interleave(128, 0)[:w.shape[0]].repeat_interleave(128, 1)


def _f8_rows(ck, name, r0, r1):   # rows [r0, r1) (whole 128-row blocks) of a block-fp8 weight: (e4m3 bytes, fp32 scales)
    assert r0 % 128 == 0 and r1 % 128 == 0
    return ck.sl(name + ".weight")[r0:r1].view(torch.uint8), ck.sl(name + ".weight_scale_inv")[r0 // 128:r1 // 128].float()


def _f8_cols(ck, name, c0, c1):   # columns [c0, c1) (whole 128 blocks)
    return ck.sl(name + ".weight")[:, c0:c1].view(torch.uint8), ck.sl(name + ".weight_scale_inv")[:, c0 // 128:c1 // 128].float()


def _f8_all(ck, name):
    return ck.get(name + ".weight").view(torch.uint8), ck.get(name + ".weight_scale_inv").float()


def attn_weights(ck, cfg, layer, g, n, dev, fp8=None):
    """GPU g's attention weights of a layer (this GPU's NHT / n heads): the gemms' weights as e4m3 + fp32 scales (*_s) with fp8
    (default FP8), else bf16; w_kc / w_vc bf16 (kv_b_proj dequantized); full layers: the indexer's k side"""
    fp8 = FP8 if fp8 is None else fp8
    p, NH = f"model.layers.{layer}.self_attn.", NHT // n
    full = cfg["indexer_types"][layer] == "full"
    if fp8:
        (qa, qas), (kva, kvas) = _f8_all(ck, p + "q_a_proj"), _f8_all(ck, p + "kv_a_proj_with_mqa")
        pad = NA - QL - KVL - RP
        w = {"w_qkva": torch.cat([qa, kva, torch.zeros(pad, H, dtype=torch.uint8)], 0),
             "w_qkva_s": torch.cat([qas, kvas], 0)[:NA // 128]}
        w["w_qb"], w["w_qb_s"] = _f8_rows(ck, p + "q_b_proj", g * NH * (NOPE + RP), (g + 1) * NH * (NOPE + RP))
        w["w_o"], w["w_o_s"] = _f8_cols(ck, p + "o_proj", g * NH * DV, (g + 1) * NH * DV)
    else:
        qa, kva = deq(ck.get(p + "q_a_proj.weight"), ck.get(p + "q_a_proj.weight_scale_inv")), \
            deq(ck.get(p + "kv_a_proj_with_mqa.weight"), ck.get(p + "kv_a_proj_with_mqa.weight_scale_inv"))
        w = {"w_qkva": bf(torch.cat([qa, kva, torch.zeros(NA - QL - KVL - RP, H)], 0)),
             "w_qb": bf(_deq_rows(ck, p + "q_b_proj", g * NH * (NOPE + RP), (g + 1) * NH * (NOPE + RP))),
             "w_o": bf(_deq_cols(ck, p + "o_proj", g * NH * DV, (g + 1) * NH * DV))}
    w.update({"q_norm": ck.get(p + "q_a_layernorm.weight").to(torch.bfloat16), "kv_norm": ck.get(p + "kv_a_layernorm.weight").to(torch.bfloat16),
              "post_norm": ck.get(f"model.layers.{layer}.post_attention_layernorm.weight").to(torch.bfloat16)})
    kvb = bf(_deq_rows(ck, p + "kv_b_proj", g * NH * (NOPE + DV), (g + 1) * NH * (NOPE + DV))).view(NH, NOPE + DV, KVL)
    w["w_kc"] = kvb[:, :NOPE, :].transpose(1, 2).contiguous().view(NH * KVL, NOPE)   # [h][l][n]
    w["w_vc"] = kvb[:, NOPE:, :].contiguous().view(NH * DV, KVL)                    # [h][d][l]
    if full:
        if fp8:
            w["w_k"], w["w_k_s"] = _f8_all(ck, p + "indexer.wk")
        else:
            w["w_k"] = bf(deq(ck.get(p + "indexer.wk.weight"), ck.get(p + "indexer.wk.weight_scale_inv")))
        w.update({"k_nw": ck.get(p + "indexer.k_norm.weight").float(), "k_nb": ck.get(p + "indexer.k_norm.bias").float()})
    return {k: v.contiguous().to(dev) for k, v in w.items()}


def pack_shard(ex_list):
    """one GPU's bank in tasks/glm_moe_experts.cuh's layout (amd/tests/test_glm53_moe_layer.py pack_experts) from its shards:
    per expert ((gate e4m3 [IR, H], scales [IR / 128, H / 128]), (up ..), (down e4m3 [H, IR], scales [H / 128, IR / 128]))"""
    w13, s13, w2, s2 = [], [], [], []
    for (gq, gs), (uq, us), (dq, ds) in ex_list:
        IR = gq.shape[0]
        gq, uq, dq = gq.view(torch.uint8), uq.view(torch.uint8), dq.view(torch.uint8)
        u13 = torch.stack([gq.reshape(IR // 16, 16, H), uq.reshape(IR // 16, 16, H)], 1)
        w13.append(u13.reshape(IR // 16, 2, 16, H // 128, 4, 2, 16).permute(0, 3, 1, 5, 4, 2, 6).reshape(2 * IR, H))
        s13.append(torch.stack([gs, us], -1))
        w2.append(dq.reshape(H // 16, 16, IR // 128, 4, 2, 16).permute(0, 2, 4, 3, 1, 5).reshape(H, IR))
        s2.append(ds.contiguous())
    return (torch.cat(w13, 0).contiguous(), torch.stack(s13).reshape(-1).contiguous(), torch.cat(w2, 0).contiguous(),
            torch.stack(s2).reshape(-1).contiguous())


def next_norm(ck, cfg, layer):
    """the gamma of the RMSNorm after a layer: the next layer's input_layernorm, the last layer's model.norm"""
    L = cfg["num_hidden_layers"]
    return ck.get(f"model.layers.{layer + 1}.input_layernorm.weight" if layer + 1 < L else "model.norm.weight").to(torch.bfloat16)


def moe_weights(ck, cfg, layer, g, n, dev):
    """GPU g's MoE weights (router bf16, bias fp32 (-inf past 256), the bank of 256 routed + the shared expert: e4m3 + fp32 scales)"""
    p = f"model.layers.{layer}.mlp."
    IR = I_MOE // n
    ex = []
    for e in range(E + 1):   # (GPU g's IR rows of gate / up, IR columns of down, and their blocks' scales)
        q = p + (f"experts.{e}." if e < E else "shared_experts.")
        gu = [(ck.sl(q + f"{k}_proj.weight")[g * IR:(g + 1) * IR],
               ck.sl(q + f"{k}_proj.weight_scale_inv")[g * IR // 128:(g + 1) * IR // 128].float()) for k in ("gate", "up")]
        dn = (ck.sl(q + "down_proj.weight")[:, g * IR:(g + 1) * IR],
              ck.sl(q + "down_proj.weight_scale_inv")[:, g * IR // 128:(g + 1) * IR // 128].float())
        ex.append((gu[0], gu[1], dn))
    w13, s13, w2, s2 = pack_shard(ex)
    bias = ck.get(p + "gate.e_score_correction_bias").float()
    w = {"router": ck.get(p + "gate.weight").to(torch.bfloat16), "bias": bias, "w13": w13, "w13_sf": s13, "w2": w2, "w2_sf": s2,
         "next_norm": next_norm(ck, cfg, layer)}
    return {k: v.contiguous().to(dev) for k, v in w.items()}


def dense_weights(ck, cfg, layer, g, n, dev, fp8=None):
    """GPU g's dense MLP (layers 0 .. 2): gate | up rows of its F = 12288 / n features, down's columns (e4m3 + scales, or bf16)"""
    fp8 = FP8 if fp8 is None else fp8
    p, F = f"model.layers.{layer}.mlp.", I_DENSE // n
    if fp8:
        (gt, gs), (up, us) = _f8_rows(ck, p + "gate_proj", g * F, (g + 1) * F), _f8_rows(ck, p + "up_proj", g * F, (g + 1) * F)
        dn, ds = _f8_cols(ck, p + "down_proj", g * F, (g + 1) * F)
        w = {"w_gu": torch.cat([gt, up]), "w_gu_s": torch.cat([gs, us]), "w_dn": dn, "w_dn_s": ds}
    else:
        gt, up = _deq_rows(ck, p + "gate_proj", g * F, (g + 1) * F), _deq_rows(ck, p + "up_proj", g * F, (g + 1) * F)
        w = {"w_gu": bf(torch.cat([gt, up])), "w_dn": bf(_deq_cols(ck, p + "down_proj", g * F, (g + 1) * F))}
    w["next_norm"] = next_norm(ck, cfg, layer)
    return {k: v.contiguous().to(dev) for k, v in w.items()}


def run_shapes(T, full, n, rows_cache, pages, reqs, w2t, npos):
    """the run tensors' shapes and dtypes (stand-ins for the graph: the hook binds SGLang's own tensors)"""
    s = {"x_attn": ((T, H), torch.bfloat16), "res_in": ((T, H), torch.bfloat16), "res_mid": ((T, H), torch.bfloat16), "cache": ((rows_cache, KVL + RP), torch.uint8),
         "out_loc": ((T,), torch.int64), "positions": ((T,), torch.int64), "seq_lens": ((T,), torch.int32),
         "cos": ((npos, RP // 2), torch.bfloat16), "sin": ((npos, RP // 2), torch.bfloat16),
         "res_out": ((T, H), torch.bfloat16), "x_next": ((T, H), torch.bfloat16)}
    if full:
        s.update({"icache": ((pages, 64 * (ID + 4)), torch.uint8), "r2t": ((reqs, w2t), torch.int32), "req_idx": ((T,), torch.int64),
                  "rows_g": ((T, K_TOP), torch.int32)})
    else:
        s["rows"] = ((T, K_TOP), torch.int32)
    return s


def build_layer(mpk, at, T, n, full, mlp, eps, splits=1, store_kp=4):
    """the layer's graph: attention (MLA + the DSA list) -> res_mid / x_ffn -> MLP ("moe" | "dense") -> res_out / x_next.
    at: the graph tensors by name (weights + run tensors); scratch tensors made here."""
    from mirage.core import bfloat16, float32, int32
    NH = NHT // n
    os.environ["MPK_STORE_KP"] = str(store_kp)
    nt = lambda nm, dims, dt: mpk.new_tensor(dims=dims, dtype=dt, name=nm)
    latent, qb = nt("latent", (T, NA), float32), nt("q_b", (T, NH * (NOPE + RP)), bfloat16)
    q_lora, q_mla = nt("q_lora", (T, QL), bfloat16), nt("q_mla", (T, NH * (KVL + RP)), bfloat16)
    attn, v_out = nt("attn", (T, NH * KVL), bfloat16), nt("v_out", (T, NH * DV), bfloat16)
    opart = nt("o_part", (T, H), float32)
    res_mid, x_ffn = at["res_mid"], nt("x_ffn", (T, H), bfloat16)   # (res_mid: a run tensor - read by pointer)
    mpk.gemm_tile_layer(input=at["x_attn"], weight=at["w_qkva"], output=latent, combine="slots", weight_scales=at.get("w_qkva_s"))
    if full:
        kp = nt("idx_k", (T, ID), float32)
        mpk.gemm_tile_layer(input=at["x_attn"], weight=at["w_k"], output=kp, combine="slots", weight_scales=at.get("w_k_s"))
    mpk.glm53_mla_norm_layer(latent=latent, w_q=at["q_norm"], w_kv=at["kv_norm"], kv_cache=at["cache"], out_loc=at["out_loc"],
                             positions=at["positions"], cos=at["cos"], sin=at["sin"], q_lora=q_lora, eps_q=eps, eps_kv=eps)
    mpk.gemm_tile_layer(input=q_lora, weight=at["w_qb"], output=qb, combine="local", input_polled=True, weight_scales=at.get("w_qb_s"))
    mpk.glm53_q_absorb_layer(q=qb, w_kc=at["w_kc"], positions=at["positions"], cos=at["cos"], sin=at["sin"], output=q_mla, heads=NH)
    if full:
        rows = nt("rows_n", (T, K_TOP), int32)
        mpk.glm53_index_k_layer(k_parts=kp, norm_w=at["k_nw"], norm_b=at["k_nb"], positions=at["positions"], cos=at["cos"],
                                sin=at["sin"], out_loc=at["out_loc"], index_cache=at["icache"], req_to_token=at["r2t"],
                                req_idx=at["req_idx"], seq_lens=at["seq_lens"], rows_copy=at["rows_g"], rows=rows)
    else:
        rows = at["rows"]
    # (a full layer's sparse attention reads its list from req_to_token itself - glm53_index_k's seq_len <= K list, the same
    # rows, no wait for the indexer's k side; g53sad1 pair L6 T 1 -3.4, T 2 -2.2, T 4 -4.0, T 8 -5.0, T 16 -0.8 us)
    mpk.glm53_sparse_attn_layer(q=q_mla, cache=at["cache"], seq_lens=at["seq_lens"], out_loc=at["out_loc"], norm_out=q_lora,
                                rows=rows, output=attn, heads=NH, scale=(NOPE + RP) ** -0.5, splits=splits,
                                **({"req_to_token": at["r2t"], "req_idx": at["req_idx"]} if full else {}))
    mpk.dsa_o_absorb_layer(input=attn, w_uv=at["w_vc"], output=v_out)
    mpk.gemm_tile_layer(input=v_out, weight=at["w_o"], output=opart, combine="store", input_polled=True, store_groups=STORE_GROUPS, weight_scales=at.get("w_o_s"))
    # (the add + RMSNorm in the senders, x_ffn their own output buffer: allreduce_send_layer norm in_kernel)
    mpk.allreduce_send_layer(input=opart, output=x_ffn, norm=dict(residual=at["res_in"], residual_out=res_mid,
                                                                   gamma=at["post_norm"], output=x_ffn, eps=eps, in_kernel=True,
                                                                   rs=rs_on(T, n)))
    sent2 = nt("mlp_sent", (T, H), bfloat16)
    tail = dict(residual=res_mid, residual_out=at["res_out"], gamma=at["next_norm"], output=at["x_next"], eps=eps, rs=rs_on(T, n))
    if mlp == "moe":
        logits = nt("router_logits", (T, E), float32)
        pairs = nt("pairs", (T, TOPK + 1), mpk_int64())
        R = nt("R_moe", (T * (TOPK + 1), H), float32)
        mpk.gemm_tile_layer(input=x_ffn, weight=at["router"], output=logits, combine="slots")
        mpk.glm_topk_route_layer(input=logits, bias=at["bias"], output=pairs, k=TOPK, shared_experts=1, shared_first=E,
                                 routed_scaling_factor=SCALE)
        mpk.glm_moe_experts_layer(x=x_ffn, pairs=pairs, w13=at["w13"], w13_sf=at["w13_sf"], w2=at["w2"], w2_sf=at["w2_sf"],
                                  output=R, swiglu_limit=None)
        mpk.allreduce_send_layer(input=R, output=sent2, norm=tail)
    else:
        F = I_DENSE // n
        gup, hm, dpart = nt("gate_up", (T, 2 * F), float32), nt("h_mlp", (T, F), bfloat16), nt("dn_part", (T, H), float32)
        mpk.gemm_tile_layer(input=x_ffn, weight=at["w_gu"], output=gup, combine="slots", weight_scales=at.get("w_gu_s"))
        mpk.glm_swiglu_mul_layer(gate_up=gup, output=hm, swiglu_limit=None)
        mpk.gemm_tile_layer(input=hm, weight=at["w_dn"], output=dpart, combine="store", input_polled=True, store_groups=STORE_GROUPS, weight_scales=at.get("w_dn_s"))
        mpk.allreduce_send_layer(input=dpart, output=sent2, norm=tail)


def mpk_int64():
    from mirage.core import int64
    return int64
