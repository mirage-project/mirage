"""
Canonical PyTorch reference implementations for the MLA SM100 layers in this
folder. Used by both kernel-wrapper tests (e.g. ``test_mla_prefill.py``) and
the test_mode tests (``test_*_testmode.py``) so that one definition is the
single source of truth.

Covered (in-scope) layers:
  * mla_prefill_layer        -> mla_prefill_ref
  * mla_kv_gather_layer      -> mla_kv_gather_ref
  * mla_kv_gather_split_layer-> mla_kv_gather_split_ref

Out-of-scope (kept inline in their kernel tests, by design):
  * mla_decode, mla_attention, mla_mtp.
"""

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
