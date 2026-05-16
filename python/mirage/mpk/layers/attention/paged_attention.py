"""Paged-KV-cache attention (prefill + decode unified).

This is the catalog counterpart to
:meth:`PersistentKernel.paged_attention_layer`
(``python/mirage/mpk/persistent_kernel.py`` ~line 895). It wraps the
``"paged_attention"`` MPK task — the *production* attention kernel
that qwen3 actually uses for both prefill and decode steps. It is
**not** the same task as the plain ``Attention`` module (which wraps
the decode-only ``single_batch_decoding_kernel`` and is used by tests /
toy demos but not by ``demo/qwen3/demo.py``).

What this kernel actually does
------------------------------

The q/k/v *projections* (``q_proj``, ``k_proj``, ``v_proj``) and the
output projection ``o_proj`` are NOT inside this kernel — they are
separate ``Linear`` modules and the composite ``Qwen3Attention``
orchestrates them. Inside this fused kernel, in this exact order:

1. **Per-request iteration** — each task instance handles ONE request
   (``request_id`` baked into ``task_desc->task_metadata``). The new-Q
   token range is
   ``[qo_indptr_buffer[request_id], qo_indptr_buffer[request_id + 1])``
   and the cached-K/V page range is
   ``[paged_kv_indptr_buffer[request_id], paged_kv_indptr_buffer[request_id + 1])``.
   This is what lets the SAME task graph handle prefill (many new q
   tokens, many cache pages) and decode (1 new q token, prior pages +
   1).
2. **Per-head RMSNorm on q** — using ``q_norm`` weight of shape
   ``(head_dim,)``, eps hard-coded to ``1e-6f`` in
   ``task_register.cc:354``.
3. **Per-head RMSNorm on k** — same with ``k_norm``.
4. **Rotary position embedding** — applied to the q and k tokens being
   processed at positions
   ``[seq_len - num_new_tokens, seq_len)`` (the kernel infers each
   token's absolute position from
   ``num_pages * PAGE_SIZE - PAGE_SIZE + last_page_len`` and the
   token's offset within the new-Q range).
5. **KV-cache append into paged storage** — the new k/v vectors are
   written into the page slot computed via the per-request page table:
   ``paged_kv_indices_buffer[paged_kv_indptr_buffer[request_id] + page_offset_in_request]``
   gives the physical page index in the
   ``(max_num_pages, page_size, num_kv_heads, head_dim)`` slab.
6. **Causal-masked flash-attention** — softmax(q @ K^T / sqrt(head_dim))
   @ V over all keys/values currently in the cache for this request
   (i.e. ``seq_len`` keys; ``seq_len = (num_pages - 1) * PAGE_SIZE +
   last_page_len``). A causal mask is applied per the multitoken
   kernel's design (``valid_lens[i] = seq_len - num_tokens + 1 + i``).
7. **Output write** — per-head outputs are written to
   ``output[first_token_pos : last_token_pos, ...]``.

Prefill vs decode dispatch
--------------------------

There is no Python-side branching. The same task handles both:

* **Decode**: ``qo_indptr_buffer[req + 1] - qo_indptr_buffer[req] == 1``.
  ``num_tokens == 1`` per request. KV-cache append touches 1 slot.
* **Prefill**: ``qo_indptr_buffer[req + 1] - qo_indptr_buffer[req] > 1``.
  ``num_tokens`` new tokens; KV-cache append touches ``num_tokens``
  slots in the last page(s).
* **Chunked prefill / continuous batching**: any mix in a single
  scheduler iteration; each request is handled by its own task
  instance via its ``request_id``.

A compile-time branch in
``multitoken_paged_attention_task_impl`` (line 47) picks between the
``_4_16`` and ``_32_64`` variants based on
``MAX_TOKENS * NUM_QO_HEADS``. ``MAX_TOKENS`` is taken from the QKV
DTensor's first dim (== ``max_num_batched_tokens``); ``NUM_QO_HEADS``
is the GQA group count (``num_q_heads / num_kv_heads``).

Tensor contract
---------------

Let ``T_max`` = ``max_num_batched_tokens`` (compile-time upper bound
on the total number of new-Q tokens across all in-flight requests),
``H`` = ``num_heads`` (Q), ``H_kv`` = ``num_kv_heads``,
``D`` = ``head_dim``, ``P`` = ``page_size``,
``N`` = ``max_num_pages``.

* ``input`` — post-QKV-projection fused tensor.
    * Shape: ``(T_max, H * D + H_kv * D + H_kv * D)``.
    * Layout (per row): ``[ Q (H*D) | K (H_kv*D) | V (H_kv*D) ]``,
      row stride ``= H*D + 2*H_kv*D`` (== ``qkv_stride`` baked at
      code-gen).
    * dtype: ``bfloat16``. ``num_dims == 2``.
    * The kernel reads
      ``qkv_ptr + first_token_pos * qkv_stride`` where
      ``first_token_pos = qo_indptr_buffer[request_id]``.

* ``k_cache`` / ``v_cache`` — paged KV cache slabs.
    * Shape: ``(N, P, H_kv, D)``, ``num_dims == 4``.
    * dtype: ``bfloat16``. Block-structured (NOT contiguous per
      request). Physical pages indexed by
      ``paged_kv_indices_buffer``; each request owns a (variable)
      list of pages identified by
      ``paged_kv_indices_buffer[paged_kv_indptr_buffer[req]
      : paged_kv_indptr_buffer[req + 1]]``.
    * **Phase 2 caller passes the per-layer 4-D slab explicitly.**
      Production driver allocates a 5-D pool
      ``(num_layers, N, P, H_kv, D)`` and slices the ``layer_idx``-th
      4-D slab. **Phase 3 will resolve it via
      ``current_pk().get_kv_cache(self.layer_idx)``** once that helper
      exists; the wiring is flagged as a follow-up because the helper
      is not in PK today.
    * The two persistent-kernel asserts ``k_cache.dim(0) ==
      pk.max_num_pages`` and ``k_cache.dim(1) == pk.page_size`` mean
      the test driver MUST size the cache against the PK's own ``N``
      and ``P``.

* ``q_norm`` / ``k_norm`` — per-head RMSNorm scale vectors.
    * Shape: ``(D,)``, ``num_dims == 1``. ``dtype == bfloat16``.
      eps hard-coded to ``1e-6f`` at codegen.

* ``cos_pos_embed`` / ``sin_pos_embed`` — RoPE tables.
    * Shape: ``(max_seq_len, D)``, ``num_dims == 2``. ``dtype ==
      bfloat16``. The kernel indexes them per absolute position of each
      new-Q token.

* ``output`` — per-head attention outputs (pre-o_proj).
    * Shape: ``(T_max, H * D)``, ``num_dims == 2``. ``dtype ==
      bfloat16``. The kernel writes
      ``output[first_token_pos : last_token_pos, :]`` for this
      request; rows outside that slice are untouched by this task.

Meta-tensors dependencies (read at runtime via ``runtime_config``)
------------------------------------------------------------------

This task reads MORE meta tensors than the plain ``attention`` task:

* ``qo_indptr_buffer`` (``int32[max_num_batched_requests + 1]``) —
  prefix sum of new-Q token counts per request. ``qo_indptr[i + 1] -
  qo_indptr[i]`` is the number of new Q tokens for request ``i``.
* ``paged_kv_indptr_buffer`` (``int32[max_num_batched_requests + 1]``)
  — prefix sum of pages-per-request. ``paged_kv_indptr[i + 1] -
  paged_kv_indptr[i]`` is the page count for request ``i``.
* ``paged_kv_indices_buffer`` (``int32[max_num_pages]``) — flat list
  of physical page indices, indexed via
  ``paged_kv_indptr_buffer``.
* ``paged_kv_last_page_len_buffer``
  (``int32[max_num_batched_requests]``) — length of valid data in
  each request's last page (in the range ``[1, P]``).
* ``prompt_lengths`` (``int32[total_num_requests]``) — read elsewhere
  in the runtime; not directly by this task.
* ``step``, ``tokens`` — read by other tasks but not by this one
  (sequence length is reconstructed from page metadata, not
  ``step[0]``).

**Test mode**: ``_apply_test_mode_meta_defaults`` zero-initialises all
of the above. The test driver MUST set
``qo_indptr_buffer`` and ``paged_kv_indptr_buffer`` and
``paged_kv_indices_buffer`` and ``paged_kv_last_page_len_buffer`` to
non-trivial values for any test that exercises non-empty K/V; the
default zero state means ``num_tokens == 0`` and the kernel returns
immediately.

Page size convention
--------------------

* ``PAGE_SIZE`` is a compile-time template parameter baked from
  ``self.page_size`` at codegen.
* The kernel statically asserts ``PAGE_SIZE % KV_TILE_SIZE == 0`` with
  ``KV_TILE_SIZE == 64`` (see
  ``multitoken_paged_attention_4_16.cuh:246``). So ``page_size`` must
  be a multiple of 64 for the Ampere/Hopper kernel.
* For a single test request, the simplest setup is
  ``num_pages == 1, page_size >= seq_len``.

GQA handling
------------

``num_q_heads / num_kv_heads`` query heads share each kv head. The
kernel template parameter ``NUM_QO_PER_KV`` is baked at codegen from
``params[0] / params[1]`` (see ``task_register.cc:330``). ``NUM_KV_HEADS``
is baked as ``1`` per task instance — i.e. each task handles ONE
kv-head — and the grid parallelises across kv-heads via the y axis.

Block size / parallelism
------------------------

``pk.paged_attention_layer`` asserts
``grid_dim == (max_num_batched_requests, num_kv_heads, *)`` and
the production driver always passes ``(R, H_kv, 1)``. Each task
instance handles one (request, kv-head) pair. ``block_dim`` is
``(128, 1, 1)`` — the multitoken kernel hard-wires ``NUM_THREADS ==
128`` (m16n16k16 mma over 4 warps) on both Ampere and Hopper, so the
default-block-dim override exists exactly as in plain ``Attention``.

``layer_idx`` and Phase-3 KV lookup
-----------------------------------

``__init__`` stores ``layer_idx`` so Phase-3 ``Qwen3Attention.compile()``
can resolve the layer's KV cache slot via
``current_pk().get_kv_cache(self.layer_idx)`` once that helper exists.
At Phase 2 the wiring helper does NOT exist on PK — tests pass
``k_cache`` and ``v_cache`` explicitly via ``compile()`` kwargs.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .._base import MPKModule
from ...context import current_pk

# DTensor is the public Cython class used everywhere in the codebase.
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate-half convention used by Qwen3 / Llama RoPE."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to per-head q/k tensors.

    ``q``: ``(B, T, H, D)``, ``k``: ``(B, T, H_kv, D)``.
    ``cos``/``sin``: ``(T, D)`` — broadcasts over batch and head.
    """
    cos_e = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)
    sin_e = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    q_rot = (q * cos_e) + (_rotate_half(q) * sin_e)
    k_rot = (k * cos_e) + (_rotate_half(k) * sin_e)
    return q_rot, k_rot


def _per_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last axis (head_dim), per head.

    ``x`` shape ``(..., H, D)``, ``weight`` shape ``(D,)``.
    Matches the kernel's per-head RMS computation with ``eps == 1e-6f``
    (hard-coded in ``task_register.cc:354``).
    """
    input_dtype = x.dtype
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(var + eps)
    return (x_normed.to(input_dtype) * weight).to(input_dtype)


class PagedAttention(MPKModule):
    """Paged-KV-cache attention (prefill + decode unified).

    Wraps :meth:`PersistentKernel.paged_attention_layer` 1:1 — the
    production attention kernel used by qwen3 for BOTH prefill and
    decode steps. No internal dispatch; ``Attention``,
    ``PagedAttentionSplitKV``, etc., are separate modules per the
    Phase-2 design decision.

    The kernel reads paged KV cache blocks via the runtime's
    ``paged_kv_indptr_buffer`` / ``paged_kv_indices_buffer`` /
    ``paged_kv_last_page_len_buffer`` meta tensors (block-structured,
    not contiguous). Handles arbitrary new-q lengths per request via
    ``qo_indptr_buffer`` iteration.

    Phase 2: KV cache buffers are passed explicitly to ``compile()``;
    Phase 3 will wire them from
    ``current_pk().get_kv_cache(layer_idx)`` once that helper exists.

    The q/k/v/o projection ``nn.Linear`` layers are NOT in this module —
    the composite ``Qwen3Attention`` orchestrates them around this
    fused kernel.

    Args:
        num_heads: Total number of query heads (``H``). Equals
            ``config.num_attention_heads``.
        num_kv_heads: Number of key/value heads (``H_kv``). For GQA,
            ``num_heads / num_kv_heads`` query heads share each kv
            head. Equals ``config.num_key_value_heads``.
        head_dim: Per-head channel count (``D``). Equals
            ``config.head_dim``.
        layer_idx: Index of this attention layer in the parent model.
            Stored for Phase-3 KV cache resolution via
            ``current_pk().get_kv_cache(layer_idx)``. Phase 2 passes
            ``k_cache``/``v_cache`` explicitly.
        prefix: HF state_dict key prefix (vLLM convention) — e.g.
            ``"model.layers.3.self_attn."``. ``q_norm`` and ``k_norm``
            load from ``{prefix}q_norm.weight`` and
            ``{prefix}k_norm.weight``.

    Attributes:
        q_norm (``nn.Parameter``): ``(head_dim,)`` bf16. Per-head RMS
            scale on q before RoPE. Standard init: ones (matches
            ``Qwen3RMSNorm`` default), overwritten by ``load_state_dict``.
        k_norm (``nn.Parameter``): ``(head_dim,)`` bf16. Per-head RMS
            scale on k before RoPE. Same init.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for GQA"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        # Per-head RMSNorm scales. Standard init = ones (matches
        # Qwen3RMSNorm), overwritten by load_state_dict in production.
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Faithful PyTorch reference for the fused part.

        Models the *eager-PyTorch* attention semantics — NOT the paged
        layout. The compiled path's paged-vs-contiguous re-layout is
        irrelevant to the math, so the reference uses a contiguous
        ``(B, max_seq_len, num_kv_heads, head_dim)`` cache for clarity.
        Only ``compile()`` deals with paged storage.

        Args:
            qkv: ``(B, T, H * D + H_kv * D + H_kv * D)`` fused
                post-projection tensor. Per-row layout:
                ``[ Q (H*D) | K (H_kv*D) | V (H_kv*D) ]``.
            cos: ``(T_max, D)`` cos RoPE table.
            sin: ``(T_max, D)`` sin RoPE table.
            k_cache: ``(B, max_seq_len, H_kv, D)`` cache slab. Updated
                in place at ``positions``.
            v_cache: ``(B, max_seq_len, H_kv, D)`` cache slab. Same.
            positions: ``(B, T)`` int64/int32 absolute positions of the
                NEW Q tokens. Used both for RoPE table lookup and for
                the KV-cache write index.

        Returns:
            ``(B, T, H * D)`` tensor, ready to feed ``o_proj``.
        """
        B, T, _ = qkv.shape
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.head_dim
        groups = H // H_kv

        # 1. Split fused QKV — INTERLEAVED-per-kv-group layout (what
        #    `pk.shuffle_tensors([q,k,v], num_groups=H_kv)` produces and
        #    what `multitoken_paged_attention_sm100_task_impl` expects).
        #    Per row: [g0_q (groups*D) | g0_k (D) | g0_v (D) | g1_q | ...].
        per_group = (groups + 2) * D
        qkv_view = qkv.view(B, T, H_kv, per_group)
        q = qkv_view[..., :groups * D].reshape(B, T, H_kv * groups, D)  # (B,T,H,D)
        k = qkv_view[..., groups * D:(groups + 1) * D].reshape(B, T, H_kv, D)
        v = qkv_view[..., (groups + 1) * D:].reshape(B, T, H_kv, D)

        # 2. Per-head RMSNorm on q and k (NOT v).
        q = _per_head_rmsnorm(q, self.q_norm)
        k = _per_head_rmsnorm(k, self.k_norm)

        # 3. RoPE on q and k, indexed by absolute positions.
        #    positions: (B, T). cos/sin: (T_max, D). We gather per-token.
        #    For a single-batch reference we use positions[0] to slice.
        pos0 = positions[0].to(torch.long)  # (T,)
        cos_slice = cos.index_select(0, pos0).to(q.dtype)  # (T, D)
        sin_slice = sin.index_select(0, pos0).to(q.dtype)  # (T, D)
        q, k = _apply_rotary(q, k, cos_slice, sin_slice)

        # 4. Append k, v into the contiguous cache at the given
        #    positions. Single-batch reference: per-batch index_copy_.
        for b in range(B):
            pos_b = positions[b].to(torch.long)
            k_cache[b].index_copy_(0, pos_b, k[b])
            v_cache[b].index_copy_(0, pos_b, v[b])

        # 5. Compute the per-row attended slice.
        #    Each new-Q token i (at absolute position pos[i]) attends to
        #    cache positions [0, pos[i] + 1) — causal mask.
        #    For simplicity assume contiguous prefill where positions
        #    are consecutive and equal to [seq_len - T, seq_len), i.e.
        #    the prefill case. The decode case (T == 1) collapses to
        #    attending over [0, pos[0] + 1).
        seq_len = int(positions[0, -1].item()) + 1  # # of cached tokens to attend over
        K = k_cache[:, :seq_len]  # (B, S, H_kv, D)
        V = v_cache[:, :seq_len]  # (B, S, H_kv, D)

        # 6. Repeat-interleave KV heads to match Q heads (GQA).
        groups = H // H_kv
        K = K.repeat_interleave(groups, dim=2)  # (B, S, H, D)
        V = V.repeat_interleave(groups, dim=2)  # (B, S, H, D)

        # 7. Causal-masked softmax(q @ K^T / sqrt(D)) @ V.
        q_t = q.transpose(1, 2)  # (B, H, T, D)
        K_t = K.transpose(1, 2)  # (B, H, S, D)
        V_t = V.transpose(1, 2)  # (B, H, S, D)

        # is_causal=True is correct ONLY when the new-Q tokens are
        # exactly the suffix of the attended sequence. That is true for
        # prefill (positions == [seq_len - T, seq_len)) and for decode
        # (T == 1). It is NOT true for arbitrary speculative-decode
        # extends — the kernel applies a per-token mask in that case,
        # but this reference assumes the prefill / decode case.
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_t, K_t, V_t, is_causal=(T > 1), enable_gqa=False,
        )  # (B, H, T, D)
        attn = attn.transpose(1, 2).contiguous().view(B, T, H * D)
        return attn

    # ------------------------------------------------------------------
    # MPK grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, input_dt: DTensor) -> GridDim:
        """Default grid: ``(max_num_batched_requests, num_kv_heads, 1)``.

        Matches ``demo/qwen3/demo.py:593``: each task instance handles
        one (request, kv-head) pair. ``pk.paged_attention_layer``
        asserts ``grid_dim[0] == pk.max_num_batched_requests`` and
        ``grid_dim[1] == num_kv_heads`` so this is the *only* valid
        choice.
        """
        pk = current_pk()
        return (pk.max_num_batched_requests, self.num_kv_heads, 1)

    # default_block_dim: inherited from MPKModule
    # (target_cc < 90 -> (128, 1, 1); target_cc >= 90 -> (256, 1, 1)).
    # The MPK megakernel always launches with WORKER_NUM_THREADS=256 on
    # Hopper/Blackwell anyway; the per-task block_dim metadata is
    # informational. Callers in practice pass block_dim=(128, 1, 1)
    # explicitly per the qwen3 demo convention.

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        cos: DTensor,
        sin: DTensor,
        *,
        output: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> DTensor:
        """Register the ``paged_attention`` task for the current PK.

        Args:
            input:   ``(T_max, H*D + H_kv*D + H_kv*D)`` fused QKV
                     DTensor, bf16. See module docstring.
            k_cache: ``(N, P, H_kv, D)`` paged KV cache slab. Phase 2
                     takes this explicitly; Phase 3 will resolve via
                     ``current_pk().get_kv_cache(self.layer_idx)``.
            v_cache: same shape/role as ``k_cache``.
            cos:     ``(max_seq_len, D)`` RoPE cos table DTensor.
            sin:     ``(max_seq_len, D)`` RoPE sin table DTensor.
            output:  Output buffer routing (same pattern as ``Attention``):

                     * ``None`` — allocate a fresh DTensor via
                       ``pk.new_tensor`` with shape ``(T_max, H * D)``
                       and dtype bf16.
                     * ``torch.Tensor`` — attach via ``pk.attach_input``
                       so the test driver can read back from it.
                     * ``DTensor`` — use directly.
            grid_dim: Override; ``None`` -> ``auto_grid_dim(input)``.
            block_dim: Override; ``None`` -> ``default_block_dim()``.
            name:    Optional unique name for the auto-allocated output.

        Returns:
            The output DTensor of shape ``(T_max, num_heads * head_dim)``.

        Raises:
            RuntimeError: if called outside ``pk.compile_scope()``.
            ValueError: on shape mismatches.
        """
        pk = current_pk()

        if input.num_dims != 2:
            raise ValueError(
                f"PagedAttention.compile expects a 2-D input DTensor "
                f"(fused QKV after the qkv projection); got "
                f"num_dims={input.num_dims}"
            )
        if k_cache.num_dims != 4 or v_cache.num_dims != 4:
            raise ValueError(
                "PagedAttention.compile expects 4-D k_cache and v_cache "
                f"DTensors (max_num_pages, page_size, num_kv_heads, "
                f"head_dim); got num_dims={k_cache.num_dims} / "
                f"{v_cache.num_dims}"
            )

        prefix = self.prefix or "paged_attention."
        num_tokens = input.dim(0)

        # Resolve output DTensor.
        if output is None:
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.new_tensor(
                dims=(num_tokens, self.num_heads * self.head_dim),
                dtype=input.dtype,
                name=out_name,
            )
        elif isinstance(output, torch.Tensor):
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.attach_input(output, name=out_name)
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "PagedAttention.compile output must be None, a "
                f"torch.Tensor, or a DTensor; got {type(output).__name__}"
            )

        # Attach the per-head RMSNorm scales. nn.Parameter is a
        # torch.Tensor subclass, but ``attach_input`` accepts only raw
        # tensors safely — pass ``.data`` to match the existing pattern.
        q_norm_dt = pk.attach_input(
            self.q_norm.data, name=f"{prefix}q_norm"
        )
        k_norm_dt = pk.attach_input(
            self.k_norm.data, name=f"{prefix}k_norm"
        )

        # Resolve grid / block.
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input)
        if block_dim is None:
            block_dim = self.default_block_dim()

        pk.paged_attention_layer(
            input=input,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=q_norm_dt,
            k_norm=k_norm_dt,
            cos_pos_embed=cos,
            sin_pos_embed=sin,
            output=out_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        return out_dt
