"""Torch float64 reference of one Kimi K3 MLA layer's attention part on one rank (TP8: 12 of the 96 heads), as SGLang 0.5.19
computes it on a target-verify step of 8 tokens (models/kimi_k3.py KimiK3MLAAttention over deepseek_v2.py
DeepseekV2AttentionMLA's absorb path, deepseek_common/attention_forward_methods/forward_mla.py; the attention core
layers/attention/trtllm_mla_backend.py forward_extend -> flashinfer trtllm_batch_decode_with_kv_cache_mla), rounded to bf16
where SGLang's kernels write bf16:

  latent = x . W_a^T                          W_a = fused_qkv_a_proj_with_mqa [1536 + 512 + 64, H]   (fused_a_gemm, bf16 out)
  q_a    = rmsnorm(latent[:, :1536]) q_a_layernorm;  k_nope = rmsnorm(latent[:, 1536:2048]) kv_a_layernorm;  k_pe = latent[:, 2048:]
  q      = q_a . W_qb^T  -> [T, 12, 192]: q_nope = [:, :, :128], q_pe = [:, :, 128:]                       (no RoPE: skip_rope)
  q_abs  = q_nope . w_kc                      w_kc [12, 128, 512] (kv_b_proj's k part, per head)          (bmm, bf16 out)
  kv row = [k_nope | k_pe] (576 bf16) of each of the 8 tokens, written into the paged KV cache
  attn   = softmax(scaling (q_abs . K[:, :512] + q_pe . K[:, 512:])) . K[:, :512]   over the KV rows (exp rounded to bf16 for
           the product, fp32 sum of the unrounded exps), token t of the 8 seeing
           rows 0 .. seq_len + t (causal within the verify window); scaling = 1 / sqrt(128 + 64)              (bf16 out)
  v      = attn . w_vc                        w_vc [12, 512, 128]                                          (bmm, bf16 out)
  gate   = x . W_g^T [T, 1536]                                                                              (bf16 out)
  o_in   = bf16(v * bf16(sigmoid(gate)))                                     (mla_output_gate: the unfused pair's rounding)
  o      = o_in . W_o^T [T, H]: this rank's partial                                                        (bf16 out)
  u      = bf16(sum over the 8 ranks of o + prefix)   (k3_ar_fusion all_reduce_push_res: fp32 over ranks 0..7, then the prefix)
  hidden = aggregation 2 over the bank rows and u (attn_residual)

Also the kernel's weight layout: to_kernel_weights (SGLang's loaded weights -> the kernel's) and layer_info.
"""
import torch


T, HEADS, Q_LORA, KV_LORA, NOPE, ROPE, VD = 8, 12, 1536, 512, 128, 64, 128


def rmsnorm(x, w, eps):
    x = x.double()
    return bf(x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w.double())


def split_kv_b(kv_b):
    """kv_b_proj [12 * (128 + 128), 512] (this rank's heads) -> w_kc [12, 128, 512], w_vc [12, 512, 128] (deepseek_v2
    post_load_weights: per head the first 128 rows are k_nope's, the next 128 v's)."""
    w = kv_b.double().view(HEADS, NOPE + VD, KV_LORA)
    return w[:, :NOPE], w[:, NOPE:].transpose(1, 2)


def core(x, w, kv_prefix, causal=True):
    """x [T, H] bf16 (aggregation 1's output); w: the dump's weights (sglang_k3_mla._weights); kv_prefix [L, 576] (the KV rows
    before the step). Returns dict of every stage (float64 tensors holding bf16 values where SGLang rounds)."""
    xd = x.double()
    wa = w["fused_qkv_a_proj_with_mqa"].double()
    latent = bf(xd @ wa.T)                                                            # [T, 2112]
    q_a = rmsnorm(latent[:, :Q_LORA], w["q_a_layernorm"], w["q_a_layernorm.eps"])
    k_nope = rmsnorm(latent[:, Q_LORA:Q_LORA + KV_LORA], w["kv_a_layernorm"], w["kv_a_layernorm.eps"])
    k_pe = latent[:, Q_LORA + KV_LORA:]
    q = bf(q_a @ w["q_b_proj"].double().T).view(T, HEADS, NOPE + ROPE)
    q_nope, q_pe = q[..., :NOPE], q[..., NOPE:]
    if "w_kc" in w:
        w_kc, w_vc = w["w_kc"].double(), w["w_vc"].double()
    else:
        w_kc, w_vc = split_kv_b(w["kv_b_proj"])
    q_abs = bf(torch.einsum("thn,hnl->thl", q_nope, w_kc))                           # [T, 12, 512]
    kv_new = torch.cat([k_nope, k_pe], -1)                                          # [T, 576]
    K = torch.cat([kv_prefix.double(), kv_new], 0)                                  # [L + T, 576]
    L = kv_prefix.shape[0]
    s = (torch.einsum("thl,jl->thj", q_abs, K[:, :KV_LORA]) + torch.einsum("thr,jr->thj", q_pe, K[:, KV_LORA:])) * w["scaling"]
    if causal:
        j = torch.arange(L + T).view(1, 1, -1)
        t = torch.arange(T).view(-1, 1, 1)
        s = s.masked_fill(j > L + t, float("-inf"))
    # SGLang's attention kernel (trtllm-gen; measured on server dumps 10-12): e = exp(s - max) rounded
    # to bf16 for the P.V product, divided at the end by the fp32 sum of the UNROUNDED e
    e = torch.exp(s - s.amax(-1, keepdim=True))
    attn = bf(torch.einsum("thj,jl->thl", bf(e), K[:, :KV_LORA]) / e.sum(-1, keepdim=True))   # [T, 12, 512]
    v = bf(torch.einsum("thl,hlv->thv", attn, w_vc)).reshape(T, HEADS * VD)        # [T, 1536]
    gate = bf(xd @ w["g_proj"].double().T)                                          # [T, 1536]
    o_in = bf(v * bf(torch.sigmoid(gate)))
    o = bf(o_in @ w["o_proj"].double().T)                                           # [T, H]
    return dict(latent=latent, q_a=q_a, k_nope=k_nope, k_pe=k_pe, q_nope=q_nope, q_pe=q_pe, q_abs=q_abs, kv_new=kv_new,
                attn=attn, v=v, gate=gate, o_in=o_in, o=o)


def all_reduce_prefix(parts, prefix):
    """u = bf16(sum of the ranks' partials, in rank order, + prefix) (prefix None: zeros)."""
    acc = sum(p.double() for p in parts)
    if prefix is not None:
        acc = acc + prefix.double()
    return bf(acc)


def aggregation2(u, bank, w, eps=None):
    """SGLang's attn_res over the bank rows and u with mlp_res_proj (q), mlp_res_norm (gain), post_attention_layernorm (ow)."""
    e = eps if eps is not None else w.get("post_attention_layernorm.eps", 1e-5)
    return attn_residual(u, bank, w["mlp_res_proj"], w["mlp_res_norm"], w["post_attention_layernorm"], e)


def bf(t):
    return t.to(torch.bfloat16).double()


def attn_residual(u, rows, q, gain, ow, eps=1e-5):
    """u [T, H] (bf16 values: prefix + the summed o_proj partials, rounded); rows [T, n, H] the bank rows; q, gain, ow [H].
    Sources = the n rows then u; score = (c . src) / sqrt(mean(src^2) + eps), c = bf16(q * gain); h = softmax-weighted sum
    (not rounded); out = bf16(h / sqrt(mean(h^2) + eps) * ow): SGLang's attn_res_fused_tma (server dumps 10-12: 7.5e-6)."""
    src = torch.cat([rows.double(), u.double().unsqueeze(1)], 1)           # [T, n + 1, H]
    c = bf(q.double().reshape(-1) * gain.double().reshape(-1))            # SGLang's cw, rounded once
    sc = (src * c).sum(-1) / torch.sqrt((src * src).mean(-1) + eps)
    h = (torch.softmax(sc, -1).unsqueeze(-1) * src).sum(1)                  # fp32 into the RMSNorm (SGLang: not rounded)
    return bf(h / torch.sqrt((h * h).mean(-1, keepdim=True) + eps) * ow.double().reshape(-1))


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


# ---- the kernel's weight layout (the tensors the serving package passes) ----
NB_MAX = 16


def score_weights(res_q: torch.Tensor, res_gain: torch.Tensor) -> dict:
    """Aggregation 2's score weight as SGLang forms it (kernels/ops/kimi_k3/attn_res.py: cw = bf16(norm.w * proj.w), rounded
    once): the kernel's q := cw, gain := 1 (it scores with q * gain)."""
    cw = (res_gain.reshape(-1).float() * res_q.reshape(-1).float()).to(torch.bfloat16).contiguous()
    return {"q": cw, "gain": torch.ones_like(cw)}


def to_kernel_weights(fused_qkv_a: torch.Tensor, q_a_norm: torch.Tensor, kv_a_norm: torch.Tensor, q_b: torch.Tensor,
                      w_kc: torch.Tensor, w_vc: torch.Tensor, g_proj: torch.Tensor, o_proj: torch.Tensor,
                      res_q: torch.Tensor, res_gain: torch.Tensor, res_ow: torch.Tensor) -> dict:
    """This rank's MLA weights as SGLang holds them after loading (KimiK3MLAAttention: fused_qkv_a_proj_with_mqa [1536 + 512 +
    64, H], q_a_layernorm [1536], kv_a_layernorm [512], q_b_proj [12 x 192, 1536], w_kc [12, 128, 512], w_vc [12, 512, 128],
    g_proj [1536, H], o_proj [H, 1536]; aggregation 2's mlp_res_proj [H], mlp_res_norm [H], post_attention_layernorm [H]) ->
    the kernel's (w_kc / w_vc transposed per head: the MMAs' row operands)."""
    bf = torch.bfloat16
    c = lambda t: t.to(bf).contiguous()
    return {"wa": c(fused_qkv_a), "wq_norm": c(q_a_norm.reshape(-1)), "wkv_norm": c(kv_a_norm.reshape(-1)), "wqb": c(q_b),
            "wkcT": c(w_kc.reshape(HEADS, NOPE, KV_LORA).transpose(1, 2).reshape(HEADS * KV_LORA, NOPE)),
            "wvcT": c(w_vc.reshape(HEADS, KV_LORA, VD).transpose(1, 2).reshape(HEADS * VD, KV_LORA)), "wg": c(g_proj), "wo": c(o_proj),
            **score_weights(res_q, res_gain), "ow": c(res_ow.reshape(-1))}


def layer_info(nvb: int, bank_stride_token: int, bank_stride_row: int, device) -> torch.Tensor:
    """aggregation 2's run-time geometry (int32 [8]): the bank rows it mixes and the bank's strides in elements."""
    if not 1 <= nvb <= NB_MAX:
        raise ValueError(f"MLA kernel: {nvb} bank rows, the kernel mixes 1..{NB_MAX}")
    return torch.tensor([nvb, bank_stride_token, bank_stride_row, 0, 0, 0, 0, 0], dtype=torch.int32, device=device)
