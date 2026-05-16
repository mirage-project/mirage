"""Qwen3-variant fused attention kernel (the plain ``attention`` task).

This is the catalog counterpart to :meth:`PersistentKernel.attention_layer`
(``python/mirage/mpk/persistent_kernel.py`` ~line 758). It wraps a *single*
MPK task — the one registered under the name ``"attention"`` — and is
NOT the paged / split-KV / MLA / extend variants (those each get their
own ``MPKModule`` in follow-up PRs, per the locked Phase-2 decision
"every MPK task is its own module").

What this kernel actually does (post-projection fused block)
------------------------------------------------------------

The q/k/v *projections* (``q_proj``, ``k_proj``, ``v_proj`` ``nn.Linear``
layers in the HF Qwen3 reference) and the output projection ``o_proj``
are NOT inside this kernel — they are separate ``Linear`` modules and
the composite ``Qwen3Attention`` orchestrates them. Inside this fused
kernel, in this exact order:

1. **Per-head RMSNorm on q** — using ``q_norm`` weight of shape
   ``(head_dim,)``. The RMS reduction is across the ``head_dim`` axis
   only, *per head*, in float32 accumulation. Matches Qwen3RMSNorm
   applied to ``query_states.view(B, T, num_heads, head_dim)``.
2. **Per-head RMSNorm on k** — same per-head reduction with the
   ``k_norm`` weight, applied to ``key_states.view(B, T, num_kv_heads,
   head_dim)``.
3. **Rotary position embedding** — applied to the q and k tokens being
   processed in this step using ``cos_pos_embed[seq_len - 1, :]`` and
   ``sin_pos_embed[seq_len - 1, :]`` (the kernel hard-wires the
   "current decode position" as ``seq_len - 1``). RoPE is applied to
   the whole ``head_dim`` and uses the rotate-half convention (see
   :func:`apply_rotary_pos_emb` in
   ``demo/qwen3/models/modeling_qwen3.py``).
4. **KV-cache append** — the new k/v vectors are written into
   ``k_cache[batch, seq_len - 1, kv_head, :]`` and
   ``v_cache[batch, seq_len - 1, kv_head, :]`` IN PLACE. Earlier
   positions of the cache (``[:seq_len - 1]``) are read but not
   modified.
5. **FlashAttention-style softmax** — ``softmax(q @ K^T / sqrt(head_dim))
   @ V`` over all ``seq_len`` keys/values in the cache (i.e. no causal
   mask is applied beyond the natural seq_len cap, because this is the
   **decoding step** kernel — one query, all keys).
6. **Output write** — the per-head outputs are written to ``output``
   of shape ``(batch_size, num_heads * head_dim)`` ready to be consumed
   by the subsequent ``o_proj`` Linear.

Sequence length comes from the runtime
``meta_tensors["step"]``: the code-gen in
``src/kernel/task_register.cc:275`` hard-wires ``seq_len`` to
``runtime_config.step[0] + 1``. So if you want the kernel to attend
over ``S`` tokens in the cache, you must set
``meta_tensors["step"][0] = S - 1`` before launching.

Tensor contract
---------------

Let ``B`` = batch_size, ``H``  = ``num_heads``, ``H_kv`` = ``num_kv_heads``,
``D`` = ``head_dim``, ``S`` = ``seq_len`` of the kv cache (``= step[0] + 1``
at runtime).

* ``input`` — post-QKV-projection tensor.
    * Shape: ``(B, H * D + H_kv * D + H_kv * D)`` — i.e. q, k, v
      concatenated along the feature axis (this is the layout
      produced by ``rmsnorm_linear_layer`` with a fused qkv weight,
      see ``demo/qwen3/demo_chat.py:144``).
    * Layout (per row): ``[ q (H*D) | k (H_kv*D) | v (H_kv*D) ]``.
    * dtype: ``bfloat16``. ``num_dims == 2``.
    * The kernel pointer-arithmetics ``d_q``, ``d_k``, ``d_v`` via
      ``HEAD_DIM * NUM_Q_HEADS`` and
      ``HEAD_DIM * (NUM_Q_HEADS + NUM_KV_HEADS)`` offsets — see
      ``single_batch_decoding.cuh:64-69``.

* ``k_cache`` / ``v_cache`` — per-layer contiguous KV cache slabs.
    * Shape: ``(B, max_seq_len, H_kv, D)``, ``num_dims == 4``.
    * dtype: ``bfloat16``.
    * Layout: NHD (token, head, dim).
    * The kernel reads/writes positions ``[0:S]`` (where ``S = step[0]
      + 1``); position ``S - 1`` is written by this call. The driver
      is expected to slice the per-layer cache out of a 5-D pool
      ``(num_layers, B, max_seq_len, H_kv, D)`` and pass the 4-D
      ``layer_idx``-th slice in. At Phase 3 the composite module will
      do ``current_pk().get_kv_cache(self.layer_idx)``. **Phase 2: the
      caller passes the 4-D slice in directly.**

* ``q_norm`` / ``k_norm`` — per-head RMSNorm scale vectors.
    * Shape: ``(D,)``, ``num_dims == 1``. ``dtype == bfloat16``.
    * Applied per-head along the head_dim axis, eps hard-coded to
      ``1e-6f`` in ``task_register.cc:282-283``.

* ``cos_pos_embed`` / ``sin_pos_embed`` — RoPE tables.
    * Shape: ``(max_seq_len, D)``, ``num_dims == 2``. ``dtype ==
      bfloat16``. The kernel indexes them at row ``S - 1``.

* ``output`` — per-head attention outputs (pre-o_proj).
    * Shape: ``(B, H * D)``, ``num_dims == 2``. ``dtype == bfloat16``.

GQA handling
------------

``num_heads / num_kv_heads`` query heads share each kv head. The kernel
template parameter ``NUM_Q_HEADS / NUM_KV_HEADS`` is baked at code-gen
time from ``params[0] / params[1]`` (see ``task_register.cc:266``).
Standard Qwen3 ratios (e.g. 32/8 for Qwen3-8B) are exercised in
production.

Meta-tensors dependencies
-------------------------

The kernel reads the following at runtime via ``runtime_config``:

* ``meta_tensors["step"]`` (``int32[total_num_requests]``) — current
  decode position. ``seq_len = step[0] + 1`` is passed as the size of
  the cache to attend over. **Phase-2 tests MUST set this** to the
  desired ``S - 1`` before ``pk()``; the default is zero, which would
  attend over a single token.

No other meta-tensors are read by the plain ``attention`` task (paged
variants read ``qo_indptr_buffer`` / ``paged_kv_*``, but those route
through different ``pk.*_layer`` methods entirely).

Architecture coverage
---------------------

The Python ``pk.attention_layer`` registers under the single task name
``"attention"``. The code-gen in ``task_register.cc:266`` emits
``kernel::single_batch_decoding_kernel<bfloat16, …>`` unconditionally.
That kernel lives in
``include/mirage/persistent_kernel/tasks/ampere/single_batch_decoding.cuh``
and is the same kernel used on Ampere and Hopper. The Blackwell
attention file ``tasks/blackwell/attention_sm100.cuh`` implements the
*paged* variant and is not reachable through this task. Practically:
the plain ``attention_layer`` works on Ampere and Hopper; paged
variants are required on Blackwell. This module's ``compile()`` defers
to ``pk.attention_layer`` exactly and inherits that arch constraint.

Parallelism / grid
------------------

``pk.attention_layer`` builds the TBGraph with
``new_input(input,  (0, 1, -1), -1, True)`` (partition on
``batch_size`` and a head/kv-group axis) and
``new_input(k_cache, (0, 2, -1), 1, True)`` (partition on
``batch_size`` and ``num_kv_heads``). The conventional launch is

    grid_dim = (batch_size, num_kv_heads, 1)
    block_dim = (128, 1, 1)   # Ampere — kernel hard-wires NUM_THREADS=128

— see ``demo/qwen3/demo_chat.py:150`` for the canonical call. The
kernel is written for ``NUM_THREADS == 128``, so ``block_dim`` is
forced to ``(128, 1, 1)`` regardless of ``self.target_cc``. (RMSNorm
and Linear flip 128/256 on Hopper; this kernel does not.)

``layer_idx`` and Phase-3 KV lookup
-----------------------------------

``__init__`` stores ``layer_idx`` so Phase-3 ``Qwen3Attention.compile()``
can resolve the layer's KV cache slot via
``current_pk().get_kv_cache(self.layer_idx)`` once that helper exists.
At Phase 2 the wiring helper does NOT exist yet — tests pass
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
    # (T, D) -> (1, T, 1, D)
    cos_e = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)
    sin_e = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    q_rot = (q * cos_e) + (_rotate_half(q) * sin_e)
    k_rot = (k * cos_e) + (_rotate_half(k) * sin_e)
    return q_rot, k_rot


def _per_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last axis (head_dim), per head.

    ``x`` shape ``(..., H, D)``, ``weight`` shape ``(D,)``.
    Matches the kernel's per-head RMS computation in
    ``include/mirage/persistent_kernel/tasks/ampere/norm.cuh:46-98``
    with ``eps == 1e-6f`` (hard-coded in ``task_register.cc``).
    """
    input_dtype = x.dtype
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(var + eps)
    return (x_normed.to(input_dtype) * weight).to(input_dtype)


class Attention(MPKModule):
    """Qwen3-variant fused attention kernel.

    Performs (per the kernel) per-head q/k RMSNorm, RoPE on q/k,
    in-place KV-cache append at position ``step[0]``, then
    ``softmax(q @ K^T / sqrt(D)) @ V`` over the cached keys/values.
    Wraps :meth:`PersistentKernel.attention_layer` 1:1 — no internal
    dispatch and no fallback to paged or split-KV kernels.

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
            ``config.head_dim``. Hard-wired to 128 in many of the
            backing kernels, but the Python wrapper takes it as input.
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
        raise RuntimeError(
            "layers.Attention (plain decode, wraps pk.attention_layer) "
            "has no Blackwell-SM100 kernel variant — attention_sm100.cuh "
            "only ships the paged form (multitoken_paged_attention_sm100). "
            "On Blackwell, use layers.PagedAttention instead (one task "
            "per (request, kv-head) pair, handles both prefill and decode "
            "via qo_indptr_buffer-driven iteration). On Ampere/Hopper "
            "this class would be usable, but the catalog is currently "
            "Blackwell-targeted; revisit if Ampere/Hopper support is "
            "needed."
        )
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
        # Per-head RMSNorm scales. Kernel and PyTorch reference both
        # operate on ``(D,)`` vectors. Standard init = ones (matches
        # Qwen3RMSNorm), overwritten by load_state_dict in production.
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Faithful PyTorch reference for the fused part.

        Mirrors what ``single_batch_decoding_kernel`` does end-to-end so
        we can pointwise-diff the compiled output. The HF
        ``Qwen3Attention.forward`` (``modeling_qwen3.py:261-335``) is
        the source of truth for the algebra; here we strip the
        ``q_proj``/``k_proj``/``v_proj``/``o_proj`` calls that live
        outside the kernel.

        Args:
            q_proj: ``(B, T, H * D)`` post-q_proj tensor.
            k_proj: ``(B, T, H_kv * D)`` post-k_proj tensor.
            v_proj: ``(B, T, H_kv * D)`` post-v_proj tensor.
            cos:    ``(T_max, D)`` cos RoPE table. We index ``[:T]`` for
                    the rotary application (matching the kernel's
                    ``cos_ptr + (seq_len - 1) * HEAD_DIM`` for ``T == 1``).
            sin:    ``(T_max, D)`` sin RoPE table.
            k_cache: ``(B, max_seq_len, H_kv, D)``. Updated in place
                     at the active positions (positions ``[seq_len - T,
                     seq_len)`` are written; earlier positions are
                     read).
            v_cache: ``(B, max_seq_len, H_kv, D)``. Same semantics.
            seq_len: Optional explicit total cached length to attend
                     over. Defaults to ``T`` for the prefill case;
                     for single-step decode tests the caller passes
                     the value that matches ``step[0] + 1``.

        Returns:
            ``(B, T, H * D)`` tensor, ready to feed ``o_proj``.
        """
        B, T, _ = q_proj.shape
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.head_dim

        # 1. Reshape to per-head layout.
        q = q_proj.view(B, T, H, D)
        k = k_proj.view(B, T, H_kv, D)
        v = v_proj.view(B, T, H_kv, D)

        # 2. Per-head RMSNorm on q and k (NOT v).
        q = _per_head_rmsnorm(q, self.q_norm)
        k = _per_head_rmsnorm(k, self.k_norm)

        # 3. RoPE on q and k.
        #    The kernel applies cos/sin at position ``seq_len - 1`` per
        #    token; for T==1 decode that matches ``cos[seq_len - 1]``.
        #    For T>1 prefill we'd slice ``cos[seq_len - T:seq_len]``.
        if seq_len is None:
            seq_len = T
        cos_slice = cos[seq_len - T : seq_len]  # (T, D)
        sin_slice = sin[seq_len - T : seq_len]  # (T, D)
        q, k = _apply_rotary(q, k, cos_slice, sin_slice)

        # 4. Append the new k, v into the cache at positions
        #    [seq_len - T, seq_len). Single-batch reference: write
        #    batch 0 only.
        k_cache[:, seq_len - T : seq_len] = k
        v_cache[:, seq_len - T : seq_len] = v

        # 5. Read the full attended slice.
        K = k_cache[:, :seq_len]  # (B, S, H_kv, D)
        V = v_cache[:, :seq_len]  # (B, S, H_kv, D)

        # 6. Repeat-interleave KV heads to match Q heads (GQA).
        groups = H // H_kv
        K = K.repeat_interleave(groups, dim=2)  # (B, S, H, D)
        V = V.repeat_interleave(groups, dim=2)  # (B, S, H, D)

        # 7. Attention: softmax(q @ K^T / sqrt(D)) @ V.
        #    Transpose to (B, H, T, D) and (B, H, S, D) for matmul.
        q_t = q.transpose(1, 2)            # (B, H, T, D)
        K_t = K.transpose(1, 2)            # (B, H, S, D)
        V_t = V.transpose(1, 2)            # (B, H, S, D)

        # scaled_dot_product_attention with causal=False matches the
        # decoding kernel which has no mask (attends over the full
        # cached prefix).
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_t, K_t, V_t, is_causal=False, enable_gqa=False,
        )  # (B, H, T, D)
        attn = attn.transpose(1, 2).contiguous().view(B, T, H * D)
        return attn

    # ------------------------------------------------------------------
    # MPK grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, input_dt: DTensor) -> GridDim:
        """Default grid: ``(batch_size, num_kv_heads, 1)``.

        Matches the convention in ``demo/qwen3/demo_chat.py:150``:
        partition on the batch axis (the kernel's first
        ``new_input(input, (0, 1, -1), -1, True)`` axis) and the
        kv-head axis (the kernel's
        ``new_input(k_cache, (0, 2, -1), 1, True)`` axis). The
        backing CUDA kernel hard-wires ``NUM_KV_HEADS == 1`` per task
        instance (see ``single_batch_decoding.cuh:51``), so each
        kv-head gets its own task.
        """
        batch_size = input_dt.dim(0)
        return (batch_size, self.num_kv_heads, 1)

    # default_block_dim: inherited from MPKModule
    # (target_cc < 90 -> (128, 1, 1); target_cc >= 90 -> (256, 1, 1)).
    # Callers in practice always pass block_dim explicitly per the qwen3
    # demo convention, so the base class default suffices.

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
        """Register the ``attention`` task for the current PK.

        Args:
            input:   ``(B, H*D + H_kv*D + H_kv*D)`` fused QKV DTensor,
                     bf16, row-major. See module docstring for the
                     per-row layout.
            k_cache: ``(B, max_seq_len, H_kv, D)`` per-layer cache slab
                     DTensor. **Phase 2 takes this explicitly** — the
                     test driver builds the slab and attaches it.
                     Phase 3 will resolve it via
                     ``current_pk().get_kv_cache(self.layer_idx)``.
            v_cache: same shape/role as ``k_cache``.
            cos:     ``(max_seq_len, D)`` RoPE cos table DTensor.
            sin:     ``(max_seq_len, D)`` RoPE sin table DTensor.
            output:  Output buffer routing (same pattern as RMSNorm /
                     Linear):

                     * ``None`` — allocate a fresh DTensor via
                       ``pk.new_tensor`` with shape
                       ``(B, H * D)`` and dtype bf16.
                     * ``torch.Tensor`` — attach via ``pk.attach_input``
                       so the test driver can read back from it
                       (canonical test path).
                     * ``DTensor`` — use directly (advanced).
            grid_dim: Override; ``None`` -> ``auto_grid_dim(input)``.
            block_dim: Override; ``None`` -> ``default_block_dim()``,
                       which forces ``(128, 1, 1)`` for this kernel.
            name:    Optional unique name for the auto-allocated output.
                     Defaults to ``f"{prefix}attn_out"``.

        Returns:
            The output DTensor of shape ``(B, num_heads * head_dim)``.

        Raises:
            RuntimeError: if called outside ``pk.compile_scope()``.
            ValueError: on shape / dtype mismatches.
        """
        pk = current_pk()

        if input.num_dims != 2:
            raise ValueError(
                f"Attention.compile expects a 2-D input DTensor "
                f"(fused QKV after the qkv projection); got "
                f"num_dims={input.num_dims}"
            )
        if k_cache.num_dims != 4 or v_cache.num_dims != 4:
            raise ValueError(
                "Attention.compile expects 4-D k_cache and v_cache "
                f"DTensors (B, max_seq_len, num_kv_heads, head_dim); "
                f"got num_dims={k_cache.num_dims} / {v_cache.num_dims}"
            )

        prefix = self.prefix or "attention."
        batch_size = input.dim(0)

        # Resolve output DTensor.
        if output is None:
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.new_tensor(
                dims=(batch_size, self.num_heads * self.head_dim),
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
                "Attention.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        # Attach the per-head RMSNorm scales. nn.Parameter is a
        # torch.Tensor subclass, but ``attach_input`` accepts only raw
        # tensors safely — pass ``.data`` to match the RMSNorm pattern.
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

        pk.attention_layer(
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
