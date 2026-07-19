"""
Canonical PyTorch reference implementations for the MLA SM100 layers in this
folder. Used by both kernel-wrapper tests (e.g. ``test_mla_prefill.py``) and
the test_mode tests (``test_*_testmode.py``) so that one definition is the
single source of truth.

Covered (in-scope) layers:
  * mla_prefill_layer        -> mla_prefill_ref
  * mla_kv_gather_layer      -> mla_kv_gather_ref
  * mla_kv_gather_split_layer-> mla_kv_gather_split_ref
  * deepseek_mla_rope_q_fused_layer -> rope_rotate_gptj (tail-64 slice)
  * deepseek_mla_rope_q_split_layer -> rope_rotate_gptj (head pe slice)
  * deepseek_mla_rope_k_layer       -> rope_rotate_gptj (first-64 slice)

Out-of-scope (kept inline in their kernel tests, by design):
  * mla_decode, mla_attention, mla_mtp.
"""

import math

import torch


# ---------------------------------------------------------------------------
# mla_prefill_layer
# ---------------------------------------------------------------------------
def mla_prefill_ref(Q_nope, Q_pe, CKV, KPE, sm_scale):
    """Causal multi-head latent attention over compressed KV (single-batch).

    Shapes match the kernel-wrapper convention used in test_mla_prefill.py:
        Q_nope: [B, S, H, D_CKV]
        Q_pe:   [B, S, H, D_KPE]
        CKV:    [B, S, D_CKV]
        KPE:    [B, S, D_KPE]
    Returns O: [B, S, H, D_V] with D_V = D_CKV (latent dim doubles as V).
    """
    B, S, H, _ = Q_nope.shape
    q = torch.cat([Q_nope.float(), Q_pe.float()], dim=-1)  # [B,S,H,576]
    k = torch.cat([CKV.float(), KPE.float()], dim=-1)      # [B,S,576]
    scores = torch.einsum('bshd,btd->bsht', q, k) * sm_scale
    causal = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1).bool()
    scores.masked_fill_(causal.unsqueeze(0).unsqueeze(2), float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    v = CKV.float()
    o = torch.einsum('bsht,btd->bshd', probs, v)
    return o.to(Q_nope.dtype)


# ---------------------------------------------------------------------------
# Helper: emulate the in-place "append new tokens to paged cache" step that
# both mla_kv_gather variants perform before the gather.
# ---------------------------------------------------------------------------
def _append_new_tokens_to_paged_cache(
    paged_cache,           # [num_pages, page_size, D_K]  (mutated in place)
    c_latent_new,          # [num_new_tokens, D_V]
    k_pe_new,              # [num_new_tokens, K_PE_ROW_STRIDE] (first ROPE_DIM=D_K-D_V are real)
    page_indices,          # 1-D int tensor of pages owned by this request, in seq order
    seq_len,               # int — total seq len AFTER appending
    num_new_tokens,        # int
    d_k, d_v, page_size,
):
    rope_dim = d_k - d_v
    kv_start_pos = seq_len - num_new_tokens
    for tok in range(num_new_tokens):
        seq_pos = kv_start_pos + tok
        page_idx = int(page_indices[seq_pos // page_size].item())
        pos_in_page = seq_pos % page_size
        paged_cache[page_idx, pos_in_page, :d_v] = c_latent_new[tok, :d_v]
        paged_cache[page_idx, pos_in_page, d_v:d_v + rope_dim] = k_pe_new[tok, :rope_dim]


# ---------------------------------------------------------------------------
# mla_kv_gather_layer (concatenated [S, D_K] output)
# ---------------------------------------------------------------------------
def mla_kv_gather_ref(
    c_latent_new,          # [num_new_tokens, D_V]
    k_pe_new,              # [num_new_tokens, K_PE_ROW_STRIDE]
    paged_cache,           # [num_pages, page_size, D_K]  (mutated in place)
    page_indices,          # int32 [num_pages_for_request], in seq order
    seq_len,               # total seq len after appending new tokens
    d_k, d_v, page_size,
):
    """Reference for ``mla_kv_gather_layer``.

    Mirrors ``mla_kv_cache_gather_sm100_task_impl``:
      1. Append the ``num_new_tokens`` rows of c_latent_new + k_pe_new to the
         paged_cache at slots [seq_len - num_new_tokens : seq_len].
      2. Gather the full sequence into a dense ``contiguous_kv`` tensor of
         shape ``[seq_len, D_K]``.

    ``page_indices`` lists the page slots that belong to this request, ordered
    by sequence position (i.e. token at seq_pos lives in
    page_indices[seq_pos // page_size]).
    Returns a new tensor; ``paged_cache`` is mutated in place to reflect the
    appended tokens (matching the kernel side-effect).
    """
    num_new_tokens = c_latent_new.shape[0]
    _append_new_tokens_to_paged_cache(
        paged_cache, c_latent_new, k_pe_new, page_indices,
        seq_len, num_new_tokens, d_k, d_v, page_size,
    )

    contiguous_kv = torch.zeros(
        seq_len, d_k, dtype=paged_cache.dtype, device=paged_cache.device,
    )
    for seq_pos in range(seq_len):
        page_idx = int(page_indices[seq_pos // page_size].item())
        pos_in_page = seq_pos % page_size
        contiguous_kv[seq_pos, :d_k] = paged_cache[page_idx, pos_in_page, :d_k]
    return contiguous_kv


# ---------------------------------------------------------------------------
# mla_kv_gather_split_layer (separate CKV / KPE outputs)
# ---------------------------------------------------------------------------
def mla_kv_gather_split_ref(
    c_latent_new,          # [num_new_tokens, D_V]
    k_pe_new,              # [num_new_tokens, K_PE_ROW_STRIDE]
    paged_cache,           # [num_pages, page_size, D_K]  (mutated in place)
    page_indices,          # int32 [num_pages_for_request], in seq order
    seq_len,               # total seq len after appending new tokens
    d_k, d_v, page_size,
):
    """Reference for ``mla_kv_gather_split_layer``.

    Same semantics as ``mla_kv_gather_ref`` but the gather emits TWO dense
    buffers instead of one:
        ckv_sep [seq_len, D_V]   — latent / value half
        kpe_sep [seq_len, D_K-D_V] — rope half (D_K-D_V == 64 for DeepSeek V3)
    Returns ``(ckv_sep, kpe_sep)``. ``paged_cache`` is mutated in place.
    """
    num_new_tokens = c_latent_new.shape[0]
    rope_dim = d_k - d_v
    _append_new_tokens_to_paged_cache(
        paged_cache, c_latent_new, k_pe_new, page_indices,
        seq_len, num_new_tokens, d_k, d_v, page_size,
    )

    ckv_sep = torch.zeros(
        seq_len, d_v, dtype=paged_cache.dtype, device=paged_cache.device,
    )
    kpe_sep = torch.zeros(
        seq_len, rope_dim, dtype=paged_cache.dtype, device=paged_cache.device,
    )
    for seq_pos in range(seq_len):
        page_idx = int(page_indices[seq_pos // page_size].item())
        pos_in_page = seq_pos % page_size
        ckv_sep[seq_pos, :d_v] = paged_cache[page_idx, pos_in_page, :d_v]
        kpe_sep[seq_pos, :rope_dim] = paged_cache[page_idx, pos_in_page, d_v:d_v + rope_dim]
    return ckv_sep, kpe_sep


# ---------------------------------------------------------------------------
# DeepSeek-V3 MLA RoPE — cos/sin table + GPT-J (interleaved) rotation
# ---------------------------------------------------------------------------
def _yarn_get_mscale(scale=1.0, mscale=1.0):
    """Mirror builder._yarn_get_mscale."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
            / (2 * math.log(base)))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def build_dsv3_yarn_rope_tables(
    max_seq,
    rope_dim=64,
    rope_theta=10000.0,
    factor=40.0,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=1.0,
    original_max_position_embeddings=4096,
    attn_factor=1.0,
    extrapolation_factor=1.0,
    device="cuda",
):
    """Build DeepSeek-V3 YARN cos/sin tables IDENTICALLY to
    ``builder._precompute_rope_embeddings`` (vLLM/SGLang GPT-J / interleaved
    convention): the returned tables are width ``rope_dim`` (=64), where each
    logical frequency value is ``repeat_interleave``-d to occupy two adjacent
    columns (2*pair, 2*pair+1). Both the kernel and the reference index the
    SAME table at ``[pos*rope_dim + 2*pair]``, so the YARN math lives in ONE
    place — correctness is decided by the rotation convention + slice layout,
    not by re-deriving YARN twice.

    Returns (cos, sin) bf16 [max_seq, rope_dim].
    """
    half = rope_dim // 2
    base = float(rope_theta)
    pos_freqs = base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, rope_dim, base, int(original_max_position_embeddings))
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(half, dtype=torch.float32) - low) / (high - low), 0, 1)
    inv_freq_mask = (1 - ramp) * extrapolation_factor
    freqs = (inv_freq_interpolation * (1 - inv_freq_mask)
             + inv_freq_extrapolation * inv_freq_mask)
    m = (_yarn_get_mscale(factor, mscale)
         / _yarn_get_mscale(factor, mscale_all_dim) * attn_factor)

    positions = torch.arange(max_seq, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq, half]
    cos = (angles.cos() * m).repeat_interleave(2, dim=-1).to(
        dtype=torch.bfloat16, device=device)
    sin = (angles.sin() * m).repeat_interleave(2, dim=-1).to(
        dtype=torch.bfloat16, device=device)
    return cos, sin


def rope_rotate_gptj(x, cos, sin):
    """GPT-J / interleaved RoPE rotation matching the kernel
    ``deepseek_mla_rope_sm100_task_impl`` EXACTLY.

    For pair index ``p`` (d0=2p, d1=2p+1) at sequence position ``pos`` the
    kernel reads ``c = cos[pos, d0]``, ``s = sin[pos, d0]`` (the table is
    repeat_interleave-d so cos[d0]==cos[d1]) and computes::

        out[d0] = x[d0]*c - x[d1]*s
        out[d1] = x[d1]*c + x[d0]*s

    All arithmetic is done in float32 (the kernel up-converts bf16->f32, does
    the FMA, then down-converts), then cast back to ``x.dtype``.

    Args:
      x:   [..., S, rope_dim] bf16 (the rope slice to rotate; last-but-one dim
           must be the per-token sequence axis aligned to ``cos``/``sin`` rows).
      cos: [S, rope_dim] (repeat_interleave-d), sin: [S, rope_dim].
    """
    xf = x.float()
    x0 = xf[..., 0::2]
    x1 = xf[..., 1::2]
    c = cos[..., 0::2].float()  # value at even (d0) positions
    s = sin[..., 0::2].float()
    out = torch.empty_like(xf)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x1 * c + x0 * s
    return out.to(x.dtype)
