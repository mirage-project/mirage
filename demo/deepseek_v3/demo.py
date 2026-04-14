from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import sys
import json

from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder
from mirage.mpk.models.graph_builder import MirageModelConfig


DEFAULT_SAVE_DIR = os.path.join("outputs", "deepseek_v3")
MAX_SAVE_TOKENS = 100

# DeepSeek V3 architecture constants
# MLA: 128 heads, compressed KV dim 512, rope dim 64, total head dim 576
DEEPSEEK_V3_NUM_HEADS = 128
DEEPSEEK_V3_KV_LORA_RANK = 512
DEEPSEEK_V3_QK_ROPE_HEAD_DIM = 64
DEEPSEEK_V3_HEAD_DIM_TOTAL = DEEPSEEK_V3_KV_LORA_RANK + DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 576


def grid_for_rmsnorm_linear_layer(size: int):
    if size / 96 > 400:
        assert size % 256 == 0, f"FATAL: Linear layer size not supported, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64


def max_factor_leq_n(m: int, n: int) -> int:
    """Return the largest factor of m that is less than or equal to n."""
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def run_correctness_test(args, state_dict, layer_indices, rank, world_size,
                         first_token_id=1):
    """Run PyTorch reference on selected layers and compare against MPK.

    Both use the SAME real weights from the checkpoint.
    Layer indices specify which layers to include (e.g., [0, 3] = 1 dense + 1 MoE).
    The MTP layer (index 61) is included if --mtp is set.
    """
    import torch.nn.functional as F
    import math

    device = f"cuda:{rank}"
    layer_indices = sorted(layer_indices)
    num_layers = len(layer_indices)
    include_mtp = args.mtp

    # DeepSeek V3 constants (after weight absorption)
    KV_LORA_RANK = DEEPSEEK_V3_KV_LORA_RANK  # 512
    QK_ROPE_HEAD_DIM = DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 64
    QK_HEAD_DIM = DEEPSEEK_V3_HEAD_DIM_TOTAL  # 576
    V_HEAD_DIM = KV_LORA_RANK  # 512
    NUM_Q_HEADS = DEEPSEEK_V3_NUM_HEADS // world_size
    HIDDEN = 7168
    FIRST_MOE = 3
    NUM_EXPERTS = 256
    TOPK = 8

    def rms_norm(x, weight, eps=1e-6):
        orig = x.dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (weight.float() * x.float() * torch.rsqrt(v + eps)).to(orig)

    def sigmoid_topk(logits, bias, k):
        """DeepSeek V3 MoE routing: noaux_tc (grouped topk) + norm + scaling.
        Matches `MoEGate.forward` in modeling_deepseek.py:425-471."""
        # Read MoE routing config from real model
        cfg = config.to_dict() if hasattr(config, "to_dict") else config
        n_group = cfg.get("n_group", 8)
        topk_group = cfg.get("topk_group", 4)
        n_routed_experts = cfg.get("n_routed_experts", 256)
        norm_topk = cfg.get("norm_topk_prob", True)
        routed_scaling = cfg.get("routed_scaling_factor", 2.5)

        scores = torch.sigmoid(logits.float())  # [bs, n_experts]
        bs = scores.shape[0]
        # Group selection: top-2 scores per group, sum them, pick top topk_group groups
        scores_for_choice = scores + bias.float().unsqueeze(0)
        group_scores = scores_for_choice.view(bs, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)  # [bs, n_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [bs, topk_group]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bs, n_group, n_routed_experts // n_group)
            .reshape(bs, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if k > 1 and norm_topk:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * routed_scaling
        return topk_weight, topk_idx

    QK_NOPE = 128   # per-head nope dim (before absorption)
    V_ORIG = 128     # per-head V dim (before absorption)
    Q_HEAD_DIM = QK_NOPE + QK_ROPE_HEAD_DIM  # 192 — used for softmax scale (NOT 576)

    def dequant_fp8(weight, scale, block_k=128):
        """Dequantize FP8 weight using block-wise scale_inv (checkpoint format).

        Weight: [M, K], Scale: [M/block_k, K/block_k].
        Each scale element covers a block_k × block_k tile.
        """
        w = weight.float()
        s = scale.float()
        if s.dim() == 2 and w.dim() == 2:
            # Expand scale to match weight shape
            s = s.repeat_interleave(block_k, dim=0)[:w.shape[0]]
            s = s.repeat_interleave(block_k, dim=1)[:, :w.shape[1]]
        return (w * s).to(torch.bfloat16)

    # ---- MPK-matched precision path -------------------------------------
    # MPK uses re-quantized FP8 weights with per-row UE8M0 (power-of-2) scale
    # in 128-wide K blocks, and quantizes inputs the same way. To produce a
    # PyTorch reference whose precision matches MPK exactly, we replicate the
    # quantize → dequant round-trip on both sides before doing the matmul.
    _w_cache = {}

    def _requant_weight_to_ue8m0(weight_fp8, scale_inv, block_k=128):
        """Match builder._requantize_fp8_for_ue8m0:
        dequant 2D-block scale_inv → re-quantize as per-row 1D UE8M0
        (block_k along K) → return (new_fp8, power_of_2_scale[M, K/block_k])."""
        M, K = weight_fp8.shape
        scale_k = K // block_k
        scale_inv_expanded = scale_inv.float().repeat_interleave(
            block_k, dim=0)[:M].repeat_interleave(block_k, dim=1)[:, :K]
        w_float = weight_fp8.float() * scale_inv_expanded
        w_blocks = w_float.reshape(M, scale_k, block_k)
        block_amax = w_blocks.abs().amax(dim=2).clamp(min=1e-12)
        raw_scale = block_amax / 448.0
        ue8m0_exp = torch.ceil(torch.log2(raw_scale.clamp(min=1e-30)))
        new_scale = torch.pow(2.0, ue8m0_exp)  # [M, scale_k]
        new_scale_expanded = new_scale.unsqueeze(2).expand_as(w_blocks)
        w_rescaled = (w_blocks / new_scale_expanded).clamp(-448, 448)
        new_fp8 = w_rescaled.reshape(M, K).to(torch.float8_e4m3fn)
        return new_fp8, new_scale

    def _dequant_ue8m0_weight(weight_fp8, ue8m0_scale, block_k=128):
        """Per-row UE8M0 dequant: weight[M, K] * scale[M, K/block_k]."""
        s = ue8m0_scale.repeat_interleave(block_k, dim=1)[:, :weight_fp8.shape[1]]
        return weight_fp8.float() * s

    def _quant_dequant_input_ue8m0(x_bf16, block_k=128):
        """Per-token UE8M0 quant + dequant round-trip on the input.

        x: [batch, K] bf16 → quantize per (token, K-block) to FP8 + UE8M0
        scale → dequant back. Returns float32 tensor that mirrors what MPK
        sees inside the GEMM (i.e. fp8 × fp8 with block-scaled accumulation)."""
        bs, K = x_bf16.shape
        sk = K // block_k
        x_blocks = x_bf16.float().reshape(bs, sk, block_k)
        amax = x_blocks.abs().amax(dim=2).clamp(min=1e-12)
        raw_scale = amax / 448.0
        ue8m0_exp = torch.ceil(torch.log2(raw_scale.clamp(min=1e-30)))
        scale = torch.pow(2.0, ue8m0_exp)  # [bs, sk]
        scale_exp = scale.unsqueeze(2).expand_as(x_blocks)
        x_q = (x_blocks / scale_exp).clamp(-448, 448).to(torch.float8_e4m3fn)
        x_deq = x_q.float() * scale_exp
        return x_deq.reshape(bs, K)

    def _fp8_block_scaled_mm(x_fp8, x_scale, w_fp8, w_scale, block_k=128):
        """Block-scaled FP8 matmul using torch._scaled_mm per K-block.
        Matches MPK's UMMA: per-block FP8×FP8 with scale applied per block.
        x_fp8: (M, K) fp8, x_scale: (M, K//block_k) float32
        w_fp8: (N, K) fp8, w_scale: (N, K//block_k) float32
        Returns: (M, N) bfloat16."""
        M, K = x_fp8.shape
        N = w_fp8.shape[0]
        num_blocks = K // block_k
        acc = torch.zeros(M, N, dtype=torch.float32, device=x_fp8.device)
        for bi in range(num_blocks):
            ks, ke = bi * block_k, (bi + 1) * block_k
            # _scaled_mm: (M, block_k) @ (block_k, N) with per-row scales
            # _scaled_mm requires scale_a stride(0)==1; force contiguous layout
            sa = x_scale[:, bi:bi+1].clone().view(-1, 1).contiguous()   # (M, 1)
            sb = w_scale[:, bi:bi+1].clone().view(1, -1).contiguous()   # (1, N)
            block_out = torch._scaled_mm(
                x_fp8[:, ks:ke].contiguous(),
                w_fp8[:, ks:ke].contiguous().t(),
                scale_a=sa, scale_b=sb,
                out_dtype=torch.bfloat16,
            )
            acc += block_out.float()
        return acc.to(torch.bfloat16)

    def fp8_linear(x, weight_key, sd, block_k=128):
        """FP8 matmul. MPK_REF_NO_QUANT=1 → use raw checkpoint dequant.
        MPK_REF_TRUE_FP8=1 → use torch._scaled_mm per K-block (matches
        MPK's actual FP8×FP8 tensor core precision, not float32 sim)."""
        if os.environ.get("MPK_REF_NO_QUANT", "0") == "1":
            w = dequant_fp8(sd[weight_key], sd[f"{weight_key}_scale_inv"], block_k)
            return F.linear(x.float(), w.float()).to(x.dtype)
        # Prepare UE8M0 requantized weight (cached)
        if weight_key not in _w_cache:
            new_fp8, new_scale = _requant_weight_to_ue8m0(
                sd[weight_key], sd[f"{weight_key}_scale_inv"], block_k)
            _w_cache[weight_key] = (new_fp8, new_scale)
        cached = _w_cache[weight_key]
        if isinstance(cached, tuple):
            w_fp8, w_scale = cached
        else:
            # Legacy: was storing dequanted weight. Re-cache as (fp8, scale).
            new_fp8, new_scale = _requant_weight_to_ue8m0(
                sd[weight_key], sd[f"{weight_key}_scale_inv"], block_k)
            _w_cache[weight_key] = (new_fp8, new_scale)
            w_fp8, w_scale = new_fp8, new_scale

        if os.environ.get("MPK_REF_TRUE_FP8", "0") == "1":
            # True FP8 matmul: quantize input to FP8+UE8M0 scale, then
            # use _scaled_mm per K-block (matches MPK UMMA precision).
            x_flat = x.reshape(-1, x.shape[-1])
            bs, K = x_flat.shape
            sk = K // block_k
            x_blocks = x_flat.float().reshape(bs, sk, block_k)
            amax = x_blocks.abs().amax(dim=2).clamp(min=1e-12)
            raw_s = amax / 448.0
            ue8m0_exp = torch.ceil(torch.log2(raw_s.clamp(min=1e-30)))
            x_scale = torch.pow(2.0, ue8m0_exp)  # (bs, sk) UE8M0 scale
            x_fp8 = (x_blocks / x_scale.unsqueeze(2)).clamp(-448, 448).to(torch.float8_e4m3fn)
            x_fp8 = x_fp8.reshape(bs, K)
            out = _fp8_block_scaled_mm(x_fp8, x_scale, w_fp8, w_scale, block_k)
            return out.reshape(*x.shape[:-1], w_fp8.shape[0])
        else:
            # Default: dequant both to float32, matmul in float32 (higher precision).
            w_deq = _dequant_ue8m0_weight(w_fp8, w_scale, block_k)
            x_deq = _quant_dequant_input_ue8m0(x.reshape(-1, x.shape[-1]), block_k)
            out = F.linear(x_deq, w_deq).to(x.dtype)
            return out.reshape(*x.shape[:-1], w_deq.shape[0])

    # Precompute RoPE cos/sin with the same formula as MPK builder
    # (use args.max_seq_length so we cover all positions).
    _rope_theta = 10000.0
    _rope_half = QK_ROPE_HEAD_DIM // 2
    _rope_max_seq = args.max_seq_length
    _rope_freqs = 1.0 / (_rope_theta ** (torch.arange(0, _rope_half, dtype=torch.float32, device=device) / _rope_half))
    _rope_positions = torch.arange(_rope_max_seq, dtype=torch.float32, device=device)
    _rope_angles = torch.outer(_rope_positions, _rope_freqs)  # [max_seq, half]
    _rope_cos = torch.cat([_rope_angles.cos(), _rope_angles.cos()], dim=-1).to(torch.bfloat16)
    _rope_sin = torch.cat([-_rope_angles.sin(), _rope_angles.sin()], dim=-1).to(torch.bfloat16)

    # DeepSeek V3 softmax_scale (modeling_deepseek.py:689-695):
    #   softmax_scale = q_head_dim^-0.5 * mscale^2  (where q_head_dim = 192, NOT 576)
    #   mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    #         = 0.1 * mscale_all_dim * log(scaling_factor) + 1.0
    cfg_d = config.to_dict() if hasattr(config, "to_dict") else config
    rope_scale = cfg_d.get("rope_scaling", None)
    if rope_scale and rope_scale.get("type") == "yarn":
        mscale_all_dim = rope_scale.get("mscale_all_dim", 0)
        scaling_factor = rope_scale.get("factor", 1.0)
        if mscale_all_dim:
            _ms = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
            _softmax_scale = (Q_HEAD_DIM ** -0.5) * _ms * _ms
        else:
            _softmax_scale = Q_HEAD_DIM ** -0.5
    else:
        _softmax_scale = Q_HEAD_DIM ** -0.5
    print(f"  [REF] Q_HEAD_DIM={Q_HEAD_DIM}, softmax_scale={_softmax_scale:.6f}")

    def apply_rope(x, position):
        """Apply RoPE to x [..., rope_dim] at the given position(s).
        Matches MPK's half-spliced formulation."""
        cos = _rope_cos[position]  # [..., rope_dim]
        sin = _rope_sin[position]
        # rotate_half: split into halves, swap
        half = x.shape[-1] // 2
        rotated = torch.cat([x[..., half:], x[..., :half]], dim=-1)
        return (x.float() * cos.float() + rotated.float() * sin.float()).to(x.dtype)

    def mla_attention(hidden, prefix, sd, kv_cache, seq_pos, num_heads):
        """MLA attention following raw checkpoint flow (no weight absorption)."""
        bs = hidden.shape[0]
        p = prefix + "self_attn."

        # Q path: q_a_proj → norm → q_b_proj → split(q_nope, q_pe)
        q_a = fp8_linear(hidden, f"{p}q_a_proj.weight", sd)
        q_a = rms_norm(q_a, sd[f"{p}q_a_layernorm.weight"])
        q_full = fp8_linear(q_a, f"{p}q_b_proj.weight", sd)
        q_full = q_full.view(bs, num_heads, QK_NOPE + QK_ROPE_HEAD_DIM)  # [bs, H, 192]
        q_nope = q_full[:, :, :QK_NOPE]   # [bs, H, 128]
        q_pe = q_full[:, :, QK_NOPE:]      # [bs, H, 64]
        # Apply RoPE to q_pe at position seq_pos (single token at a time)
        q_pe = apply_rope(q_pe, seq_pos)

        # KV path: kv_a_proj → split → norm(c_latent)
        kv_full = fp8_linear(hidden, f"{p}kv_a_proj_with_mqa.weight", sd)
        c_lat = kv_full[:, :KV_LORA_RANK]   # [bs, 512]
        k_pe_raw = kv_full[:, KV_LORA_RANK:]  # [bs, 64]
        c_lat = rms_norm(c_lat, sd[f"{p}kv_a_layernorm.weight"])
        # Apply RoPE to k_pe at position seq_pos BEFORE writing to cache
        k_pe_rotated = apply_rope(k_pe_raw, seq_pos)

        # Cache write (store rotated k_pe so reads get already-rotated values)
        kv_new = torch.cat([c_lat, k_pe_rotated], dim=-1)  # [bs, 576]
        for b in range(bs):
            kv_cache[seq_pos + b] = kv_new[b]
        kv_all = kv_cache[:seq_pos + bs]  # [kv_len, 576]

        # Weight absorption via kv_b_proj
        kv_b = dequant_fp8(sd[f"{p}kv_b_proj.weight"],
                           sd[f"{p}kv_b_proj.weight_scale_inv"])
        kv_b = kv_b.view(num_heads, V_ORIG + QK_NOPE, KV_LORA_RANK)  # [H, 256, 512]
        W_UK = kv_b[:, V_ORIG:, :]   # [H, 128, 512] K nope absorption
        W_UV = kv_b[:, :V_ORIG, :]   # [H, 128, 512] V absorption

        # Absorbed Q: q_nope_abs = q_nope @ W_UK → [bs, H, 512]
        q_nope_abs = torch.einsum('bhd,hdk->bhk', q_nope.float(), W_UK.float()).to(hidden.dtype)

        # Attention: Q_abs × c_kv^T + q_pe × k_pe^T (RoPE already applied to both)
        k_nope = kv_all[:, :KV_LORA_RANK]   # [kv_len, 512]
        k_pe_all = kv_all[:, KV_LORA_RANK:]  # [kv_len, 64]
        s = (torch.einsum('bhd,sd->bhs', q_nope_abs.float(), k_nope.float()) +
             torch.einsum('bhd,sd->bhs', q_pe.float(), k_pe_all.float()))
        # NOTE: Use precomputed softmax_scale matching DeepSeek V3 (q_head_dim=192 + YARN mscale).
        s = s * _softmax_scale
        attn_probs = F.softmax(s, dim=-1)

        # V absorption: attn @ c_kv → [bs, H, 512], then × W_UV^T → [bs, H, 128]
        attn_v = torch.einsum('bhs,sd->bhd', attn_probs, kv_all[:, :KV_LORA_RANK].float())
        attn_out = torch.einsum('bhd,hkd->bhk', attn_v, W_UV.float()).to(hidden.dtype)

        # o_proj
        flat = attn_out.reshape(bs, num_heads * V_ORIG)
        return fp8_linear(flat, f"{p}o_proj.weight", sd)

    def dense_mlp(hidden, prefix, sd):
        p = prefix + "mlp."
        gate = F.silu(fp8_linear(hidden, f"{p}gate_proj.weight", sd))
        up = fp8_linear(hidden, f"{p}up_proj.weight", sd)
        return fp8_linear(gate * up, f"{p}down_proj.weight", sd)

    def moe_mlp(hidden, prefix, sd):
        bs = hidden.shape[0]
        p = prefix + "mlp."
        # Router (BF16)
        logits = F.linear(hidden.float(), sd[f"{p}gate.weight"].float()).to(hidden.dtype)
        weights, topk_idx = sigmoid_topk(logits, sd[f"{p}gate.e_score_correction_bias"], TOPK)
        # MPK_DUMP_MOE=<layer_idx>: print routing for last token to verify MPK alignment
        _dmp = os.environ.get("MPK_DUMP_MOE", "")
        if _dmp and prefix == f"model.layers.{int(_dmp)}.":
            for b in range(bs):
                w_sorted, idx_sort = weights[b].sort(descending=True)
                topk_sorted = topk_idx[b][idx_sort]
                print(f"  [REF L{_dmp} b={b}] experts={topk_sorted.tolist()} weights={[f'{x:.4f}' for x in w_sorted.tolist()]}")
        out = torch.zeros(bs, HIDDEN, device=device, dtype=hidden.dtype)
        # Routed experts (per-expert individual weights, FP8)
        # MPK_MOE_RAW_DEQUANT=1: use raw checkpoint dequant for expert GEMMs
        # (float32 per-block scale, matching MPK's group GEMM scale format)
        # instead of UE8M0 round-trip. If this gives same cosine as default,
        # the gap is NOT from quantization scheme difference.
        _moe_raw = os.environ.get("MPK_MOE_RAW_DEQUANT", "0") == "1"
        def _expert_linear(x, weight_key, sd_dict, block_k=128):
            """Expert FP8 linear: raw dequant when _moe_raw, else UE8M0 sim."""
            if _moe_raw:
                w = dequant_fp8(sd_dict[weight_key],
                                sd_dict[f"{weight_key}_scale_inv"], block_k)
                # Also quant-dequant input with float32 scale (not UE8M0)
                bs_, K = x.shape
                sk = K // block_k
                x_blocks = x.float().reshape(bs_, sk, block_k)
                amax = x_blocks.abs().amax(dim=2).clamp(min=1e-12)
                scale = amax / 448.0  # float32 scale (NOT power-of-2)
                scale_exp = scale.unsqueeze(2).expand_as(x_blocks)
                x_q = (x_blocks / scale_exp).clamp(-448, 448).to(torch.float8_e4m3fn)
                x_deq = x_q.float() * scale_exp
                return F.linear(x_deq.reshape(bs_, K), w.float()).to(x.dtype)
            else:
                return fp8_linear(x, weight_key, sd_dict, block_k)
        for b in range(bs):
            for ki in range(TOPK):
                eid = topk_idx[b, ki].item()
                w = weights[b, ki].item()
                ep = f"{p}experts.{eid}."
                gate = F.silu(_expert_linear(hidden[b:b+1], f"{ep}gate_proj.weight", sd))
                up = _expert_linear(hidden[b:b+1], f"{ep}up_proj.weight", sd)
                down = _expert_linear(gate * up, f"{ep}down_proj.weight", sd)
                out[b] += w * down.squeeze(0)
        # Shared expert (FP8)
        sp = p + "shared_experts."
        sg = F.silu(fp8_linear(hidden, f"{sp}gate_proj.weight", sd))
        su = fp8_linear(hidden, f"{sp}up_proj.weight", sd)
        sd_out = fp8_linear(sg * su, f"{sp}down_proj.weight", sd)
        return out + sd_out

    print(f"\n{'='*60}")
    print(f"Correctness Test: layers={layer_indices}, mtp={include_mtp}")
    print(f"{'='*60}")

    # Run PyTorch reference — process full prompt token-by-token (matching MPK offline mode)
    from transformers import AutoTokenizer
    tokenizer_ref = AutoTokenizer.from_pretrained(args.model_path)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer_ref.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    all_token_ids = tokenizer_ref([text], return_tensors="pt").input_ids[0].to(device)
    prompt_len = len(all_token_ids)
    print(f"  Full prompt: {prompt_len} tokens, ids[:5]={all_token_ids[:5].tolist()}")

    max_seq = prompt_len + 16
    kv_caches = [torch.zeros(max_seq, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
                 for _ in range(num_layers + (1 if include_mtp else 0))]

    # Ablation env vars (must match the same vars in builder.py for fair comparison)
    skip_layer = os.environ.get("MPK_SKIP_LAYER", "0") == "1"
    skip_attn = os.environ.get("MPK_SKIP_ATTN", "0") == "1"
    skip_mlp = os.environ.get("MPK_SKIP_MLP", "0") == "1"
    # MPK_NO_RESIDUAL=1: drop the residual `+` so the reference matches the
    # current MPK builder (which OVERWRITES self.x instead of adding).
    no_residual = os.environ.get("MPK_NO_RESIDUAL", "0") == "1"

    # Token-by-token prefill (matching persistent kernel offline mode)
    for step in range(prompt_len):
        tid = all_token_ids[step]
        hidden = F.embedding(tid.unsqueeze(0), state_dict["model.embed_tokens.weight"])
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        for cache_idx, layer_idx in enumerate(layer_indices):
            prefix = f"model.layers.{layer_idx}."
            if skip_layer:
                continue
            normed = rms_norm(hidden, state_dict[f"{prefix}input_layernorm.weight"])
            if skip_attn:
                # bypass attention: hidden unchanged
                pass
            else:
                attn_out = mla_attention(normed, prefix, state_dict, kv_caches[cache_idx], step, NUM_Q_HEADS)
                hidden = attn_out if no_residual else (hidden + attn_out)
            normed = rms_norm(hidden, state_dict[f"{prefix}post_attention_layernorm.weight"])
            if skip_mlp:
                pass
            else:
                if layer_idx < FIRST_MOE:
                    mlp_out = dense_mlp(normed, prefix, state_dict)
                else:
                    mlp_out = moe_mlp(normed, prefix, state_dict)
                hidden = mlp_out if no_residual else (hidden + mlp_out)

    hidden = rms_norm(hidden, state_dict["model.norm.weight"])
    logits = F.linear(hidden.float(), state_dict["lm_head.weight"].float())
    ref_token = logits.argmax(dim=-1).item()
    print(f"PyTorch reference output token: {ref_token}")
    print(f"PyTorch logits[0,:5]: {logits[0,:5].tolist()}")
    print(f"PyTorch reference completed successfully.")
    # Return both: the logits (last position, full vocab) and the argmax token.
    # The logits tensor lets the caller compare distributions, not just argmax.
    return ref_token, logits[0].detach().clone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 demo with Mirage megakernel")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to converted DeepSeek V3 weights")
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage megakernel")
    parser.add_argument("--profiling", action="store_true",
                        help="Enable profiling to generate trace")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int,
                        help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int,
                        help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=128, type=int,
                        help="Page size for KV cache")
    parser.add_argument("--max-num-pages", default=64, type=int,
                        help="Max number of pages")
    parser.add_argument("--max-seq-length", default=4096, type=int,
                        help="Max sequence length")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.",
                        help="Input prompt text")
    parser.add_argument("--mtp", action="store_true",
                        help="Enable MTP speculative decoding")
    parser.add_argument("--num-speculative-tokens", default=1, type=int,
                        choices=range(1, 8),
                        help="Number of speculative tokens for MTP (1-7)")
    parser.add_argument("--rejection-sample-method", default="strict", type=str,
                        choices=["strict", "probabilistic", "synthetic"],
                        help="Rejection sampling method for speculative decoding")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore eos token during generation")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true",
                        help="Enable sampling (default off)")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help=(
                            "Optionally dump first N generated token_ids, text, and latency to JSON. "
                            "If path omitted, saves to outputs/deepseek_v3/{torch_output.json|mpk_output.json}."
                        ))
    # Developer correctness testing
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness test: compare MPK output against PyTorch reference")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layer indices to load (e.g. '0,3,60'). "
                             "Used with --correctness to test a reduced model.")

    args = parser.parse_args()

    # Multi-GPU setup via MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    # Load model config and tokenizer from converted weights
    print(f"Loading model config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Extract DeepSeek V3 architecture parameters
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    # MLA parameters
    kv_lora_rank = getattr(config, "kv_lora_rank", DEEPSEEK_V3_KV_LORA_RANK)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", DEEPSEEK_V3_QK_ROPE_HEAD_DIM)
    ckv_kpe_dim = kv_lora_rank + qk_rope_head_dim  # 576 for DeepSeek V3
    num_attention_heads = config.num_attention_heads

    print(f"Model config: hidden_size={hidden_size}, num_layers={num_layers}, "
          f"vocab_size={vocab_size}, num_heads={num_attention_heads}, "
          f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
          f"ckv_kpe_dim={ckv_kpe_dim}")

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests

    # Allocate token buffers
    tokens = torch.full(
        (total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda"
    )
    input_tokens = torch.full(
        (args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
    )
    output_tokens = torch.full(
        (args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
    )

    # Tokenize prompt
    messages = [
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full(
        (total_num_requests,), model_inputs.input_ids.shape[-1],
        dtype=torch.int, device="cuda"
    )

    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    if args.use_mirage:
        import mirage as mi

        # Pad vocab_size for task graph creation
        padded_vocab_size = ((vocab_size + 255) // 256) * 256

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None

        # MTP speculative decoding config
        if args.mtp:
            spec_decode_config = mi.spec_decode_class(
                "lookahead",
                ngram_size=3,
                spec_length=args.num_speculative_tokens,
            )
        else:
            spec_decode_config = None

        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)

        # Meta tensor buffers for paged attention
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda"
        )
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda"
        )

        # MLA uses a single combined ckv_kpe cache per layer
        # Shape: (num_layers, max_num_pages, page_size, ckv_kpe_dim)
        # where ckv_kpe_dim = kv_lora_rank + qk_rope_head_dim = 576
        ckv_kpe_cache = torch.zeros(
            (num_layers, args.max_num_pages, args.page_size, ckv_kpe_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        eos_token_id = config.eos_token_id if not args.ignore_eos else -1
        # Handle eos_token_id being a list (common in DeepSeek V3)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=eos_token_id,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=True,
        )

        # Load state dict from converted weights
        print(f"Loading model weights from: {args.model_path}")
        from safetensors.torch import load_file
        from safetensors import safe_open

        # Determine which layers we need
        layer_indices_for_load = None
        if args.correctness and args.layers:
            layer_indices_for_load = [int(x) for x in args.layers.split(',')]
            # Also need MTP layer if --mtp
            if args.mtp:
                layer_indices_for_load.append(num_layers)  # layer 61

        weight_file = os.path.join(
            args.model_path, f"model{rank}-mp{world_size}.safetensors"
        )
        if os.path.exists(weight_file):
            state_dict = load_file(weight_file, device="cuda")
        else:
            # Selective loading: only load needed layers from sharded files
            index_file = os.path.join(args.model_path, "model.safetensors.index.json")
            if os.path.exists(index_file) and layer_indices_for_load is not None:
                # Smart loading: use index to only load relevant shards/keys
                print(f"  Selective loading for layers: {layer_indices_for_load}")
                state_dict = {}
                with open(index_file) as f:
                    index = json.load(f)
                # Build key filter: global keys + selected layer keys
                needed_prefixes = ["model.embed_tokens.", "model.norm.", "lm_head."]
                for li in layer_indices_for_load:
                    needed_prefixes.append(f"model.layers.{li}.")
                # Group keys by shard file
                shard_to_keys = {}
                for key, shard in index["weight_map"].items():
                    if any(key.startswith(p) for p in needed_prefixes):
                        shard_to_keys.setdefault(shard, []).append(key)
                # Load only needed shards and keys
                for shard, keys in sorted(shard_to_keys.items()):
                    shard_path = os.path.join(args.model_path, shard)
                    print(f"  Loading {len(keys)} keys from {shard}")
                    with safe_open(shard_path, framework="pt", device="cuda") as f:
                        for key in keys:
                            state_dict[key] = f.get_tensor(key)
                print(f"  Loaded {len(state_dict)} keys total")
            else:
                # Full loading (no index or no layer filter)
                import glob
                shard_files = sorted(glob.glob(
                    os.path.join(args.model_path, "model-*.safetensors")
                ))
                if shard_files:
                    state_dict = {}
                    for shard_file in shard_files:
                        state_dict.update(load_file(shard_file, device="cuda"))
                else:
                    candidates = [
                        os.path.join(args.model_path, "model.safetensors"),
                    ]
                    state_dict = None
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            state_dict = load_file(candidate, device="cuda")
                            break
                    if state_dict is None:
                        raise FileNotFoundError(
                            f"Could not find model weights at {args.model_path}. "
                            f"Expected {weight_file} or model.safetensors or model-*.safetensors"
                        )

        # Parse layer indices for correctness mode
        layer_indices_arg = None
        if args.correctness and args.layers:
            layer_indices_arg = [int(x) for x in args.layers.split(',')]

        # Correctness test: run PyTorch reference first (on raw weights)
        ref_token = None
        ref_logits = None
        if args.correctness:
            test_layers = layer_indices_arg if layer_indices_arg else list(range(num_layers))
            first_tok = model_inputs.input_ids[0, 0].item()
            ref_token, ref_logits = run_correctness_test(
                args, state_dict, test_layers, rank, world_size,
                first_token_id=first_tok)

            # Convert weights for MPK builder:
            # - Keep FP8 weights + scale_inv as-is (builder uses FP8 GEMM pipeline)
            # - Only dequant kv_b_proj for absorption into q_b_proj
            # - Fuse gate+up for dense layers and per-expert weights
            print("\nConverting weights for MPK builder (in-memory, FP8 preserved)...")
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
            from convert import (
                dequantize_fp8, absorb_kv_into_q, get_model_params, is_fp8,
                find_scale_for_weight,
            )
            config_dict = AutoConfig.from_pretrained(args.model_path).to_dict()
            mp = get_model_params(config_dict)

            # Absorb kv_b_proj into q_b_proj (requires dequant for matmul)
            # After absorption, q_b_proj becomes BF16 (absorbed), kv_b_proj is deleted
            # Include MTP layer (61) if --mtp is set
            absorb_layers = list(test_layers)
            if args.mtp:
                absorb_layers.append(num_layers)  # layer 61
            for li in absorb_layers:
                attn = f"model.layers.{li}.self_attn."
                q_key = f"{attn}q_b_proj.weight"
                kv_key = f"{attn}kv_b_proj.weight"
                if q_key in state_dict and kv_key in state_dict:
                    # Dequant both for absorption (GPU)
                    q_w = state_dict[q_key]
                    kv_w = state_dict[kv_key]
                    q_s_key = f"{q_key}_scale_inv"
                    kv_s_key = f"{kv_key}_scale_inv"
                    if is_fp8(q_w) and q_s_key in state_dict:
                        q_bf16 = dequantize_fp8(q_w.cuda(), state_dict[q_s_key].cuda()).to(torch.bfloat16)
                    else:
                        q_bf16 = q_w.cuda().to(torch.bfloat16)
                    if is_fp8(kv_w) and kv_s_key in state_dict:
                        kv_bf16 = dequantize_fp8(kv_w.cuda(), state_dict[kv_s_key].cuda()).to(torch.bfloat16)
                    else:
                        kv_bf16 = kv_w.cuda().to(torch.bfloat16)
                    absorbed = absorb_kv_into_q(q_bf16, kv_bf16, mp).to(torch.bfloat16)
                    # Fuse V un-absorption into o_proj:
                    # o_proj_fused[h] = o_proj[h] @ W_UV[h]
                    # where W_UV[h] = kv_b_proj V part: [v_orig, kv_lora_rank]
                    num_heads_loc = mp["num_heads"]  # use full heads, shard later
                    qk_nope = mp["qk_nope_head_dim"]
                    v_dim = mp["v_head_dim"]
                    kv_lora_rank = mp["kv_lora_rank"]
                    kv_head_dim = qk_nope + v_dim
                    kv_b_reshaped = kv_bf16.reshape(num_heads_loc, kv_head_dim, kv_lora_rank)
                    W_UV = kv_b_reshaped[:, :v_dim, :]  # [H, v_dim, kv_lora_rank]

                    # Fuse into o_proj: o_proj [hidden, H*v_dim] → o_proj_fused [hidden, H*kv_lora]
                    o_key = f"{attn}o_proj.weight"
                    if o_key in state_dict:
                        o_w = state_dict[o_key]
                        if is_fp8(o_w):
                            o_s_key = f"{o_key}_scale_inv"
                            if o_s_key in state_dict:
                                o_bf16 = dequantize_fp8(o_w.cuda(), state_dict[o_s_key].cuda()).to(torch.bfloat16)
                                del state_dict[o_s_key]
                            else:
                                o_bf16 = o_w.cuda().to(torch.bfloat16)
                        else:
                            o_bf16 = o_w.cuda().to(torch.bfloat16)
                        # o_bf16: [hidden, H*v_dim] → reshape [hidden, H, v_dim]
                        hidden = o_bf16.shape[0]
                        o_reshaped = o_bf16.reshape(hidden, num_heads_loc, v_dim)
                        # o_fused[h] = o_reshaped[:, h, :] @ W_UV[h] → [hidden, kv_lora_rank]
                        # Batched: o_fused = einsum('dhn,hnk->dhk', o_reshaped, W_UV) → [hidden, H, kv_lora]
                        o_fused = torch.einsum('dhn,hnk->dhk', o_reshaped.float(), W_UV.float())
                        state_dict[o_key] = o_fused.reshape(hidden, num_heads_loc * kv_lora_rank).to(torch.bfloat16)
                        print(f"  Fused o_proj: [{hidden}, {num_heads_loc*v_dim}] → [{hidden}, {num_heads_loc*kv_lora_rank}]")

                    # Replace q_b_proj with absorbed BF16 version, remove scale
                    state_dict[q_key] = absorbed
                    if q_s_key in state_dict:
                        del state_dict[q_s_key]
                    # Remove kv_b_proj (absorbed into q)
                    del state_dict[kv_key]
                    if kv_s_key in state_dict:
                        del state_dict[kv_s_key]

            # Fuse gate_proj + up_proj for dense MLP layers (keep FP8).
            # IMPORTANT: silu_mul kernel reads `gate` from the first half of
            # each block and `up` from the second half (per-block layout). The
            # FP8 linear partitions the output dim by `grid_for_rmsnorm_linear_layer`
            # into N tasks, so silu_mul uses N/2 tasks. We must INTERLEAVE
            # gate/up at granularity = N/2 chunks per tensor (NOT torch.cat).
            from mirage.mpk.models.utils import shuffle_tensors as _shuffle_tensors
            from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer as _grid_fn
            for li in absorb_layers:
                prefix = f"model.layers.{li}.mlp."
                gate_key = f"{prefix}gate_proj.weight"
                up_key = f"{prefix}up_proj.weight"
                if gate_key in state_dict and up_key in state_dict:
                    g = state_dict.pop(gate_key)
                    u = state_dict.pop(up_key)
                    out_dim_total = g.shape[0] + u.shape[0]
                    split = _grid_fn(out_dim_total) // 2
                    state_dict[f"{prefix}gate_up_proj.weight"] = _shuffle_tensors(
                        [g, u], split=split, dim=0)
                    # Fuse scales with the SAME interleave (each scale row covers
                    # 128 weight rows, so the chunk-row count = chunk_weight_rows/128).
                    gs_key = f"{gate_key}_scale_inv"
                    us_key = f"{up_key}_scale_inv"
                    if gs_key in state_dict and us_key in state_dict:
                        gs = state_dict.pop(gs_key)
                        us = state_dict.pop(us_key)
                        state_dict[f"{prefix}gate_up_proj.weight_scale_inv"] = (
                            _shuffle_tensors([gs, us], split=split, dim=0))

            # Fuse per-expert weights into experts.w13/w2 tensors (keep FP8)
            for li in absorb_layers:
                ep = f"model.layers.{li}.mlp.experts."
                expert_keys = [k for k in list(state_dict.keys())
                               if k.startswith(ep) and ".gate_proj.weight" in k
                               and not k.endswith("_scale_inv")]
                if expert_keys:
                    n_exp = len(expert_keys)
                    print(f"  Fusing {n_exp} experts for layer {li}")
                    w13_list, w2_list = [], []
                    s13_list, s2_list = [], []
                    has_scale = f"{ep}0.gate_proj.weight_scale_inv" in state_dict
                    for e in range(n_exp):
                        g = state_dict.pop(f"{ep}{e}.gate_proj.weight")
                        u = state_dict.pop(f"{ep}{e}.up_proj.weight")
                        d = state_dict.pop(f"{ep}{e}.down_proj.weight")
                        w13_list.append(torch.cat([g, u], dim=0))
                        w2_list.append(d)
                        if has_scale:
                            gs = state_dict.pop(f"{ep}{e}.gate_proj.weight_scale_inv")
                            us = state_dict.pop(f"{ep}{e}.up_proj.weight_scale_inv")
                            ds = state_dict.pop(f"{ep}{e}.down_proj.weight_scale_inv")
                            s13_list.append(torch.cat([gs, us], dim=0))
                            s2_list.append(ds)
                    state_dict[f"{ep}w13.weight"] = torch.stack(w13_list)
                    state_dict[f"{ep}w2.weight"] = torch.stack(w2_list)
                    if has_scale:
                        state_dict[f"{ep}w13.weight_scale_inv"] = torch.stack(s13_list)
                        state_dict[f"{ep}w2.weight_scale_inv"] = torch.stack(s2_list)

            # Ensure all tensors are on GPU and 16B-aligned
            for k in list(state_dict.keys()):
                t = state_dict[k]
                if not t.is_cuda:
                    t = t.cuda()
                if not t.is_contiguous():
                    t = t.contiguous()
                if t.data_ptr() % 16 != 0:
                    aligned = torch.empty_like(t)
                    aligned.copy_(t)
                    t = aligned
                state_dict[k] = t

            print(f"  Converted: {len(state_dict)} keys (FP8 weights preserved)")

        # Build MLA model config for the builder
        model_config = MirageModelConfig(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "intermediate_size", None) or getattr(config, "moe_intermediate_size", 18432),
            vocab_size=vocab_size,
            local_num_q_heads=num_attention_heads // world_size,
            local_num_kv_heads=1,  # MLA uses single KV head (shared latent)
            head_dim=ckv_kpe_dim,  # 576 for MLA
            num_layers=num_layers,
            k_cache=[ckv_kpe_cache[i] for i in range(num_layers)],
            v_cache=[ckv_kpe_cache[i] for i in range(num_layers)],
            position_embeddings=None,
            state_dict=state_dict,
            with_lm_head=True,
        )

        # Build the computation graph using the DeepSeek V3 builder
        builder = DeepSeekV3Builder(mpk)
        if os.environ.get("MPK_SKIP_LM_HEAD", "0") == "1":
            model_config.with_lm_head = False
        # In correctness mode, automatically expose lm_head_out so we can
        # compare logits distributions (not just argmax tokens).
        if args.correctness:
            os.environ["MPK_DUMP_LOGITS"] = "1"
        builder.build_from_config(model_config, layer_indices=layer_indices_arg)

        results = mpk.kn_graph.generate_task_graph(
            num_gpus=world_size, my_gpu_id=rank
        )
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        # ABLATION: MPK_DRY_RUN=1 → stop after task_graph gen (for inspecting offsets without kernel launch)
        if os.environ.get("MPK_DRY_RUN", "0") == "1":
            print(f"[DRY RUN] task_graph_{rank}.json written. Exiting before kernel launch.")
            sys.exit(0)

        mpk.compile(output_dir=args.output_dir)

        # Run inference
        print("Starting inference with Mirage megakernel...")
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        # Debug: check token buffers and step counter
        prompt_len_val = prompt_lengths[0].item()
        # Correctness comparison: first generated token is at tokens[0, prompt_len]
        if args.correctness and ref_token is not None:
            mpk_token = tokens[0, prompt_len_val].item()
            print(f"  tokens around prompt_len={prompt_len_val}: {tokens[0, max(0,prompt_len_val-2):prompt_len_val+3].tolist()}")
            print(f"\n{'='*60}")
            print(f"Correctness comparison:")
            print(f"  PyTorch reference token: {ref_token}")
            print(f"  MPK output token:        {mpk_token}")
            if ref_token == mpk_token:
                print(f"  PASS: tokens match!")
            else:
                print(f"  FAIL: tokens differ!")
            print(f"{'='*60}\n")

            # Distribution comparison (logits, before argmax)
            # NOTE: MPK reuses lm_head_out across iterations (mbt rows). After
            # 4000+ iterations only the last few steps' logits remain. To find
            # the row matching ref_logits, scan all mbt rows by cosine sim.
            mlp_out_buf = getattr(builder, "mlp_out_buf", None)
            if mlp_out_buf is not None:
                print("[MLP_OUT] each row stats:")
                for r in range(mlp_out_buf.shape[0]):
                    row = mlp_out_buf[r].float()
                    print(f"  row {r}: amax={row.abs().max().item():.4f} mean_abs={row.abs().mean().item():.6f} nonzero={(row != 0).sum().item()}/{row.numel()}")
            attn_out_buf = getattr(builder, "attn_out_buf", None)
            if attn_out_buf is not None:
                print("[ATTN_OUT] each row stats:")
                for r in range(attn_out_buf.shape[0]):
                    row = attn_out_buf[r].float()
                    print(f"  row {r}: amax={row.abs().max().item():.4f} mean_abs={row.abs().mean().item():.6f} nonzero={(row != 0).sum().item()}/{row.numel()}")
            qnope_buf = getattr(builder, "q_nope_pe_buf", None)
            if qnope_buf is not None:
                print("[Q_NOPE_PE] each row stats:")
                for r in range(qnope_buf.shape[0]):
                    row = qnope_buf[r].float()
                    print(f"  row {r}: amax={row.abs().max().item():.4f} mean_abs={row.abs().mean().item():.6f} nonzero={(row != 0).sum().item()}/{row.numel()}")
            attn_proj_buf = getattr(builder, "attn_proj_out_buf", None)
            if attn_proj_buf is not None:
                print("[ATTN_PROJ_OUT] each row stats:")
                for r in range(attn_proj_buf.shape[0]):
                    row = attn_proj_buf[r].float()
                    print(f"  row {r}: amax={row.abs().max().item():.4f} mean_abs={row.abs().mean().item():.6f} nonzero={(row != 0).sum().item()}/{row.numel()}")
            # MPK_DUMP_MOE: print MPK's selected experts and weights to compare with ref
            moe_w = getattr(builder, "moe_topk_weights_buf", None)
            moe_idx_buf = getattr(builder, "moe_routing_indices_buf", None)
            if moe_w is not None and moe_idx_buf is not None:
                _dmp_layer = os.environ.get("MPK_DUMP_MOE", "?")
                # routing_indices is [NUM_EXPERTS, batch], values 0=not selected, k+1 = rank k
                idx = moe_idx_buf.cpu()  # [NUM_EXPERTS, batch]
                w = moe_w.cpu()  # [batch, TOPK]
                for b in range(idx.shape[1]):
                    selected = []  # list of (k_rank, expert_id)
                    for e in range(idx.shape[0]):
                        rank = idx[e, b].item()
                        if rank > 0:
                            selected.append((rank - 1, e))
                    selected.sort()  # by k_rank (matches output order)
                    expert_order = [e for _, e in selected]
                    weight_order = [w[b, k].item() for k, _ in selected]
                    # Sort by weight descending for easier comparison with ref
                    pairs = sorted(zip(weight_order, expert_order), reverse=True)
                    if pairs:
                        s_experts = [p[1] for p in pairs]
                        s_weights = [f'{p[0]:.4f}' for p in pairs]
                        print(f"  [MPK L{_dmp_layer} b={b}] experts={s_experts} weights={s_weights}")
            mpk_logits_buf = getattr(builder, "lm_head_out_buf", None)
            if mpk_logits_buf is not None and ref_logits is not None:
                ref_logits_cpu = ref_logits.detach().float().cpu()
                vocab = ref_logits.shape[-1]
                mpk_all = mpk_logits_buf[:, :vocab].detach().float().cpu()
                print(f"Scanning {mpk_all.shape[0]} mbt rows of lm_head_out for best match:")
                for r in range(mpk_all.shape[0]):
                    row = mpk_all[r]
                    cs = torch.nn.functional.cosine_similarity(
                        row.unsqueeze(0), ref_logits_cpu.unsqueeze(0), dim=1
                    ).item()
                    top1 = row.argmax().item()
                    top1_val = row.max().item()
                    print(f"  row {r}: cosine={cs:+.4f} top1=token_{top1} val={top1_val:.3f} range=[{row.min():.2f},{row.max():.2f}]")

                # Also show ref top-5
                top5_ref = ref_logits_cpu.topk(5)
                print(f"  ref top-5: {top5_ref.indices.tolist()} values={[f'{v:.3f}' for v in top5_ref.values.tolist()]}")
                print(f"  ref range: [{ref_logits_cpu.min():.3f}, {ref_logits_cpu.max():.3f}]")
                print(f"{'='*60}\n")

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)

        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
            prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0],
            run_time / (step.max().item() + 1)
        ))

        # Dump outputs to json
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            response_text = tokenizer.decode(
                tokens[0, :end_idx], skip_special_tokens=True
            )
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        # Native PyTorch path (without Mirage)
        # DeepSeek V3 requires the model implementation for non-Mirage inference
        try:
            from transformers import AutoModelForCausalLM
            print(f"Loading DeepSeek V3 model from: {args.model_path}")
            with torch.device("cuda"):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to("cuda")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DeepSeek V3 model for native inference: {e}. "
                "For native PyTorch inference, ensure transformers supports the model "
                "or use --use-mirage for Mirage megakernel inference."
            )

        prompt_len = prompt_lengths[0].item()
        output_len = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else (tokens.size(1) - prompt_len)
        )
        output_len = max(0, min(output_len, tokens.size(1) - prompt_len))
        decode_limit = prompt_len + output_len
        prev_pos = 0
        stream = torch.cuda.Stream()

        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:1, prev_pos:cur_pos]
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens[0, cur_pos] = next_token[0]
            prev_pos = cur_pos
            eos_id = config.eos_token_id
            if isinstance(eos_id, list):
                if next_token[0].item() in eos_id:
                    break
            elif next_token[0].item() == eos_id:
                break
            if cur_pos == prompt_len:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:1, :end_idx]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, cur_pos - prompt_len,
                run_time / max(cur_pos - prompt_len, 1)
            )
        )

        # Dump outputs to json
        if save_path and rank == 0:
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(
                    tokens[0, :end_idx], skip_special_tokens=True
                ),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
