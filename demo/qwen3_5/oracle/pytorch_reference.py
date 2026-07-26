"""Plain-torch reference formulas for every dumped Qwen3.5-35B-A3B-FP8 op.

Convention: per `workspace/.claude/skills/test-mode/SKILL.md` ("pytorch_reference.py" file),
one function per in-scope op, importable by both the dump driver (`ref_dump.py`, which calls
the model's OWN bound sub-callables rather than these -- it dumps what HF actually computed)
and the validator (`validate_self_consistency.py`, which calls THESE functions on the dumped
INPUTS and compares against the dumped OUTPUTS). Adapted here from the MPK-kernel-test context
to the HF-oracle context per M2-I3's contract.

Every GDN formula below (`torch_causal_conv1d_update`, `l2norm`, `torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule`) is copied VERBATIM from transformers 5.14.1's
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (the torch fallback used whenever the
optional `fla` / `causal_conv1d` packages are not installed -- confirmed absent in this
project's venv-vllm on 2026-07-25, so this IS the code path that actually ran; see
runtime_diagnostics.json emitted by ref_dump.py for the empirical confirmation on each run).
Reusing the literal HF functions means the GDN self-consistency check is not "a plausible
reimplementation" -- it is the same code, so machine-precision agreement is the expectation,
and any material diff is a real hook-placement bug, not an approximation artifact.

The attention / RMSNorm / RoPE / router / MLP formulas are transcribed line-for-line from the
same file's `Qwen3_5MoeAttention`, `Qwen3_5MoeRMSNorm`, `apply_rotary_pos_emb`,
`Qwen3_5MoeTopKRouter`, `Qwen3_5MoeMLP`, `Qwen3_5MoeExperts` classes.

FP8 dense/expert GEMMs go through HF's `integrations.finegrained_fp8` dispatcher (a real
fp8 x fp8 CUTLASS/Triton kernel with its OWN dynamic per-token-group activation quantization) --
that kernel is not reproducible bit-exactly in plain torch. `fp8_linear_recompute` below instead
dequantizes the checkpoint's weight via its stored `weight_scale_inv` and matmuls against the
dumped (unquantized) activation in fp32. This is a documented, LOOSER-tolerance check: it
verifies the operation's mathematical identity (right weight, right transpose, right scale
application) but does not reproduce the activation-side fp8 rounding noise. See README.md
"Dtype and tolerance policy".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# GDN (Gated DeltaNet) -- verbatim torch-fallback formulas from modeling_qwen3_5_moe.py
# --------------------------------------------------------------------------------------


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Verbatim copy of transformers' `torch_causal_conv1d_update` (decode / single-token-cached
    path). NOTE the original mutates `conv_state` in place via `.copy_()` and returns only `out`
    -- preserved here so a caller who passes the model's live `conv_state` tensor gets identical
    side effects; the validator instead passes a clone and inspects `conv_state` afterwards.

    hidden_states: [B, conv_dim, seq_len] (already transposed to channel-first)
    conv_state:    [B, conv_dim, state_len=3]
    weight:        [conv_dim, kernel_size=4]  (conv1d.weight.squeeze(1))
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def torch_causal_conv1d_prefill(mixed_qkv, conv_weight_3d, bias, seq_len):
    """NOT a literal HF function -- transcribed from the inline `else` branch of
    `Qwen3_5MoeGatedDeltaNet.forward` taken when `causal_conv1d_fn is None` (fla absent) and
    there is no previous cached state (first/only chunked-prefill call):
    `F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])`, i.e. a zero-left-padded
    depthwise conv over the whole chunk, sliced back to the input length, then SiLU.
    `conv_weight_3d`: raw `conv1d.weight`, shape [conv_dim, 1, kernel_size] (unsqueezed layout).
    """
    conv_dim = mixed_qkv.shape[1]
    kernel_size = conv_weight_3d.shape[-1]
    out = F.conv1d(mixed_qkv, conv_weight_3d, bias, padding=kernel_size - 1, groups=conv_dim)
    out = out[:, :, :seq_len]
    return F.silu(out)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """Verbatim copy of transformers' `l2norm` (aligned with the FLA library convention)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    """Verbatim copy of transformers' `torch_recurrent_gated_delta_rule` (the decode-step,
    single-token-per-call path; equivalent to vLLM's `fused_recurrent_gated_delta_rule_packed_decode`).
    Shapes: query/key [B,T,H,Dk], value [B,T,H,Dv], g/beta [B,T,H], initial_state [B,H,Dk,Dv] or None.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Verbatim copy of transformers' `torch_chunk_gated_delta_rule` (the chunked/prefill path)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def gdn_beta(b: torch.Tensor) -> torch.Tensor:
    """beta = sigmoid(b), computed in b's own (native, e.g. bf16) dtype -- matches
    `Qwen3_5MoeGatedDeltaNet.forward`'s `b.sigmoid()` (no `.float()` cast before the sigmoid).
    """
    return b.sigmoid()


def gdn_decay_g(a: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """g = -exp(A_log) * softplus(a + dt_bias), computed in fp32 (matches the `.float()` casts
    in `Qwen3_5MoeGatedDeltaNet.forward`)."""
    return -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def gdn_gated_rmsnorm(hidden_states: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Verbatim copy of `Qwen3_5MoeRMSNormGated.forward` (`linear_attn.norm`; NOT Gemma-style --
    no `+1` on the weight, ones-initialized in the checkpoint)."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)


# --------------------------------------------------------------------------------------
# Full attention -- transcribed from Qwen3_5MoeAttention / Qwen3_5MoeRMSNorm
# --------------------------------------------------------------------------------------


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Verbatim copy of `Qwen3_5MoeRMSNorm.forward` -- Gemma-style `(1+w)`, and note the
    (x*w).to(dtype) ordering (normalize+weight computed in fp32, cast to input dtype LAST)."""
    input_dtype = x.dtype
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Verbatim copy of `apply_rotary_pos_emb` -- partial RoPE: only the first `cos.shape[-1]`
    dims (64 of 256 here) are rotated; the remainder pass through untouched."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention(query, key, value, attention_mask, scaling, num_key_value_groups, dropout=0.0):
    """Copy of `eager_attention_forward`, EXTENDED to reconstruct causal masking when
    `attention_mask is None`.

    Empirically (see runtime_diagnostics.json / README "attn.core_attn_out"), the real forward
    ran with `attention_mask=None` -- HF's mask-construction utilities skip materializing an
    explicit additive mask when the resolved attention backend (sdpa on this box; `fla`/
    flash-attn are absent) enforces causality via an implicit `is_causal=True` flag instead. A
    literal `eager_attention_forward` transcription with `attention_mask=None` would silently
    compute FULL bidirectional attention, which is mathematically wrong for a causal decoder and
    was the actual root cause of the first self-consistency failure on this op (large, non-
    quantization-shaped error, worst in the multi-token prefill case). Reconstructing the causal
    mask here (rather than depending on a dumped mask tensor) is correct regardless of which
    concrete backend produced the real output, since any correct causal decoder must respect it.
    """
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    else:
        q_len, k_len = query.shape[-2], key_states.shape[-2]
        if q_len > 1:
            device = attn_weights.device
            q_pos = torch.arange(k_len - q_len, k_len, device=device).unsqueeze(-1)
            k_pos = torch.arange(k_len, device=device).unsqueeze(0)
            causal = k_pos > q_pos  # True where a key is in the query's future -> masked out
            attn_weights = attn_weights.masked_fill(causal, float("-inf"))
        # q_len == 1 (decode): the single query is always the last position, so every cached key
        # is causally valid -- no masking needed.
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def attn_output_gate(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """out * sigmoid(gate) -- full sigmoid (not SiLU), applied outside the softmax."""
    return attn_output * torch.sigmoid(gate)


# --------------------------------------------------------------------------------------
# MoE -- transcribed from Qwen3_5MoeTopKRouter / Qwen3_5MoeMLP / Qwen3_5MoeExperts / SparseMoeBlock
# --------------------------------------------------------------------------------------


def moe_router(hidden_states: torch.Tensor, gate_weight: torch.Tensor, top_k: int):
    """Verbatim copy of `Qwen3_5MoeTopKRouter.forward`: full-256 fp32 softmax -> top-k
    (torch.topk breaks ties toward the LOWER index by construction) -> renormalize.
    Returns (router_logits, router_probs, topk_ids, topk_weights_renormalized).
    """
    router_logits = F.linear(hidden_states, gate_weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(router_logits.dtype)
    return router_logits, router_probs, topk_ids, topk_weights


def moe_mlp(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of `Qwen3_5MoeMLP.forward` (used for the shared expert): silu(gate(x))*up(x), then down."""
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def moe_expert_weighted(x_tokens: torch.Tensor, gate_up_w_e: torch.Tensor, down_w_e: torch.Tensor,
                         weights_for_tokens: torch.Tensor) -> torch.Tensor:
    """One selected expert's compute + its per-token router weighting, from the inner loop body
    of `Qwen3_5MoeExperts.forward`. `gate_up_w_e`: [2*intermediate, hidden] (packed [gate;up]).
    """
    gate, up = F.linear(x_tokens, gate_up_w_e).chunk(2, dim=-1)
    h = F.silu(gate) * up
    out = F.linear(h, down_w_e)
    return out * weights_for_tokens[:, None]


def moe_shared_gate(hidden_states: torch.Tensor, shared_gate_weight: torch.Tensor, shared_mlp_out: torch.Tensor):
    """sigmoid(x @ shared_expert_gate.W^T) * shared_mlp_out -- gate derived from the PRE-MLP
    hidden state (same input as the router), not from router logits."""
    gate = torch.sigmoid(F.linear(hidden_states, shared_gate_weight))
    return gate, gate * shared_mlp_out


# --------------------------------------------------------------------------------------
# FP8 dense/expert GEMM recompute (dequantized weight, fp32 matmul) -- LOOSE-tolerance check
# --------------------------------------------------------------------------------------


def dequant_fp8_blockwise(weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block_n: int = 128, block_k: int = 128):
    """W_real ~= W_fp8 * weight_scale_inv, tile-expanded. `scale_inv[i,j]` scales the tile
    `W[i*block_n:(i+1)*block_n, j*block_k:(j+1)*block_k]` (vllm-graph.md §3.4)."""
    n, k = weight_fp8.shape
    w = weight_fp8.to(torch.float32)
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_n, dim=0)[:n].repeat_interleave(block_k, dim=1)[:, :k]
    return w * s_full


def fp8_linear_recompute(x: torch.Tensor, weight_fp8: torch.Tensor, scale_inv: torch.Tensor,
                          bias: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize-then-matmul recompute for an FP8 block-quantized nn.Linear. Documented
    LOOSE tolerance: this skips the real kernel's dynamic activation fp8 quantization, so a
    quantization-scale-sized diff is EXPECTED, not a bug (see README dtype/tolerance policy)."""
    w = dequant_fp8_blockwise(weight_fp8, scale_inv)
    out = F.linear(x.to(torch.float32), w, bias.to(torch.float32) if bias is not None else None)
    return out
