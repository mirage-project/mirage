"""Qwen3 model defined against the new ``mirage.mpk.layers`` catalog.

This is the Phase-3 deliverable of the PyTorch-module refactor: a clean
``nn.Module`` tree that mirrors the wiring of ``demo/qwen3/demo.py``
exactly on the MPK ``compile()`` side, while presenting a normal
PyTorch reference on ``forward()``.

What lives where:

  * ``Qwen3MLP`` owns ``gate_up_proj`` (fused Linear), ``down_proj``
    (LinearWithResidual), and the ``SiluMul`` activation.
  * ``Qwen3Attention`` owns ``qkv_proj`` (fused Linear), ``q_norm`` /
    ``k_norm`` (per-head RMS scales living on the ``PagedAttention``
    leaf — re-exposed here via ``self.attn``), and ``o_proj``
    (LinearWithResidual).
  * ``Qwen3DecoderLayer`` owns the two ``RMSNorm`` instances and
    sequences attention then MLP.
  * ``Qwen3Model`` owns ``embed_tokens`` (``Embed``), the
    ``rotary_emb`` (``RotaryEmbedding``), the layer list, and the
    final ``norm``.
  * ``Qwen3ForCausalLM`` adds ``lm_head`` (``Linear``) and the
    ``ArgmaxPartial`` + ``ArgmaxReduce`` greedy-decode head.

HF state_dict compatibility:

  * Qwen3's checkpoint stores ``q_proj`` / ``k_proj`` / ``v_proj``
    separately. ``Qwen3Attention._load_from_state_dict`` reads all
    three and interleaves them into a single fused
    ``qkv_proj.weight`` using the same kv-group shuffle the existing
    demo applies via ``pk.shuffle_tensors``.
  * Same idea for ``gate_proj`` + ``up_proj`` -> ``gate_up_proj.weight``
    in ``Qwen3MLP``.
  * Qwen3's ``q_norm`` / ``k_norm`` live as ``Qwen3RMSNorm`` modules in
    HF (state_dict key ``...self_attn.q_norm.weight``). The catalog
    ``PagedAttention`` exposes them as raw ``nn.Parameter`` s named
    ``q_norm`` / ``k_norm`` (no ``.weight`` suffix); the override
    in ``Qwen3Attention`` strips the suffix before forwarding.

KV cache:

  * Per the Option-III decision, the *driver* allocates a single
    (num_layers, max_num_pages, page_size, num_kv_heads, head_dim)
    pool for k and v, registers it on the PK via
    ``PersistentKernel(kv_cache={"k_cache": k_pool, "v_cache": v_pool})``,
    and each ``PagedAttention`` layer fetches its slice via
    ``current_pk().get_kv_cache(layer_idx)`` inside ``compile()``.
  * ``Qwen3Model.__init__`` therefore does NOT allocate KV cache; the
    driver does. ``forward()`` (the PyTorch reference) keeps its own
    contiguous in-module cache as ``nn.Buffer`` solely for the
    single-batch eager reference path (see decision #5 in the plan).

What this module DOES NOT handle (in scope of follow-up PRs):

  * Tensor parallelism / world_size > 1. The catalog is unsharded;
    ``Qwen3*`` here assumes ``world_size == 1``.
  * The legacy demo's ``--split-kv-cache`` and ``--spec-decode`` paths.
  * Sampling beyond greedy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...context import current_pk
from ...layers import (
    Argmax,
    ArgmaxPartial,
    ArgmaxReduce,
    Embed,
    Linear,
    LinearWithResidual,
    MPKModule,
    PagedAttention,
    RMSNorm,
    RotaryEmbedding,
    SiluMul,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interleave_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    num_kv_heads: int) -> torch.Tensor:
    """Mirror ``pk.shuffle_tensors([w_q, w_k, w_v], num_groups=num_kv_heads)``.

    Each kv-group g produces a block of rows arranged
    ``[ q[g*qpg : (g+1)*qpg] | k[g*kpg : (g+1)*kpg] | v[g*kpg : (g+1)*kpg] ]``
    where ``qpg = q.shape[0] // num_kv_heads`` and
    ``kpg = k.shape[0] // num_kv_heads``. The result is what the GQA
    attention kernel expects to read row-major.
    """
    qpg = q.shape[0] // num_kv_heads
    kpg = k.shape[0] // num_kv_heads
    blocks = []
    for g in range(num_kv_heads):
        blocks.append(q[g * qpg:(g + 1) * qpg])
        blocks.append(k[g * kpg:(g + 1) * kpg])
        blocks.append(v[g * kpg:(g + 1) * kpg])
    return torch.cat(blocks, dim=0)


def _interleave_gate_up(gate: torch.Tensor, up: torch.Tensor,
                        num_groups: int) -> torch.Tensor:
    """Mirror ``pk.shuffle_tensors([gate, up], num_groups=num_groups)``.

    Output is ``num_groups`` slab-pairs, each of which is
    ``[ gate[g*gpg : (g+1)*gpg] | up[g*upg : (g+1)*upg] ]``. The
    silu_mul kernel reads halved per slab-pair, so this layout is what
    makes the fused MLP correct after a multi-task gate+up linear.
    """
    gpg = gate.shape[0] // num_groups
    upg = up.shape[0] // num_groups
    blocks = []
    for g in range(num_groups):
        blocks.append(gate[g * gpg:(g + 1) * gpg])
        blocks.append(up[g * upg:(g + 1) * upg])
    return torch.cat(blocks, dim=0)


def _grid_for_linear(size: int, use_cutlass: bool = True) -> int:
    """Mirror ``grid_for_rmsnorm_linear_layer`` from demo/qwen3/demo.py.

    Picks the tile divisor that the kernel's task atom expects. Order
    of preference matches the existing demo so the new path emits the
    same task graph.
    """
    if size % 64 == 0 and not use_cutlass:
        return size // 64
    if size / 96 > 400:
        assert size % 256 == 0, f"linear size not supported: {size}"
        return size // 256
    if size % 96 == 0:
        return 96
    if size % 64 == 0:
        return 64
    raise ValueError(f"linear out-dim {size} not divisible by 96 or 64")


# ---------------------------------------------------------------------------
# Qwen3MLP
# ---------------------------------------------------------------------------


class Qwen3MLP(MPKModule):
    """gate_proj + up_proj fused via shuffle_tensors at compile time, then
    silu_mul, then down_proj + residual.

    Matches the OLD demo's wiring exactly: 3 separate ``nn.Parameter``
    weights loaded directly from HF (no Python interleave), fused into a
    single linear via ``pk.shuffle_tensors`` at compile time. The silu_mul
    kernel sees the slab-pair-interleaved layout this produces.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # Three separate parameters loaded directly from HF state_dict.
        # NOT catalog Linear modules (we bypass per-Linear-task path and
        # do the runtime shuffle ourselves).
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Map HF keys ``mlp.gate_proj.weight`` etc. to our parameter names.
        for hf_name, param in [
            ("gate_proj.weight", self.gate_proj_weight),
            ("up_proj.weight", self.up_proj_weight),
            ("down_proj.weight", self.down_proj_weight),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # PyTorch reference: faithful SiLU(gate(x)) * up(x) -> down + residual.
        gate = F.linear(x, self.gate_proj_weight)
        up = F.linear(x, self.up_proj_weight)
        silu_out = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return F.linear(silu_out, self.down_proj_weight) + residual

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, x_dt, residual_dt, *, output=None):
        """Per-layer mlp_mid and silu_mul_out."""
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16
        fused_out = 2 * self.intermediate_size
        num_tasks_linear = _grid_for_linear(fused_out)

        # Attach individual weights
        w_gate_dt = pk.attach_input(
            self.gate_proj_weight, name=f"{self.prefix}gate_proj_weight"
        )
        w_up_dt = pk.attach_input(
            self.up_proj_weight, name=f"{self.prefix}up_proj_weight"
        )
        # Runtime fusion via shuffle_tensors (slab-pair interleave).
        w_gateup_dt = pk.shuffle_tensors(
            inputs=[w_gate_dt, w_up_dt],
            shuffled_dim=0,
            num_groups=num_tasks_linear // 2,
            name=f"{self.prefix}gateup_proj",
        )

        # PER-LAYER mlp_mid
        per_layer_mlp_mid = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, fused_out),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_mlp_mid",
        )
        pk.linear_layer(
            input=x_dt,
            weight=w_gateup_dt,
            output=per_layer_mlp_mid,
            grid_dim=(num_tasks_linear, 1, 1),
            block_dim=(128, 1, 1),
        )
        # PER-LAYER silu_mul_out
        per_layer_silu_mul_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, self.intermediate_size),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_silu_mul_out",
        )
        pk.silu_mul_layer(
            input=per_layer_mlp_mid,
            output=per_layer_silu_mul_out,
            grid_dim=(num_tasks_linear // 2, 1, 1),
            block_dim=(128, 1, 1),
        )

        # down_proj + residual into output DTensor.
        w_down_dt = pk.attach_input(
            self.down_proj_weight, name=f"{self.prefix}down_proj_weight"
        )
        pk.linear_with_residual_layer(
            input=per_layer_silu_mul_out,
            weight=w_down_dt,
            residual=residual_dt,
            output=output,
            grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# Qwen3Attention
# ---------------------------------------------------------------------------


class Qwen3Attention(MPKModule):
    """q/k/v separate weights, fused via shuffle_tensors at compile time,
    PagedAttention, o_proj + residual. Matches OLD demo wiring exactly.

    ``q_norm`` and ``k_norm`` live on the inner ``PagedAttention``.
    """

    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        # Three separate weights loaded directly from HF state_dict.
        # NOT catalog Linear modules (we bypass per-Linear-task path and
        # do the runtime shuffle ourselves).
        self.q_proj_weight = nn.Parameter(
            torch.empty(self.num_heads * self.head_dim, self.hidden_size)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(self.num_kv_heads * self.head_dim, self.hidden_size)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(self.num_kv_heads * self.head_dim, self.hidden_size)
        )
        # The PagedAttention leaf owns q_norm / k_norm parameters.
        self.attn = PagedAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            layer_idx=layer_idx,
            prefix=f"{prefix}",
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.num_heads * self.head_dim)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Map HF keys to our parameters.
        for hf_name, param in [
            ("q_proj.weight", self.q_proj_weight),
            ("k_proj.weight", self.k_proj_weight),
            ("v_proj.weight", self.v_proj_weight),
            ("o_proj.weight", self.o_proj_weight),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))

        # q_norm/k_norm: HF stores them as Qwen3RMSNorm modules with
        # `.weight` inside; the catalog PagedAttention exposes them as
        # raw nn.Parameters named `q_norm` / `k_norm`. Strip the suffix.
        for name in ("q_norm", "k_norm"):
            hf_key = prefix + f"{name}.weight"
            if hf_key in state_dict:
                with torch.no_grad():
                    getattr(self.attn, name).copy_(state_dict.pop(hf_key))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                positions: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # PyTorch reference path with the separate-weight layout.
        bsz, tlen, _ = x.shape
        q = F.linear(x, self.q_proj_weight).view(
            bsz, tlen, self.num_heads, self.head_dim
        )
        k = F.linear(x, self.k_proj_weight).view(
            bsz, tlen, self.num_kv_heads, self.head_dim
        )
        v = F.linear(x, self.v_proj_weight).view(
            bsz, tlen, self.num_kv_heads, self.head_dim
        )
        from ...layers.attention.attention import (
            _apply_rotary as _ar, _per_head_rmsnorm as _phr,
        )
        q = _phr(q, self.attn.q_norm)
        k = _phr(k, self.attn.k_norm)
        q = _ar(q, cos[positions], sin[positions])
        k = _ar(k, cos[positions], sin[positions])
        # Single-batch attention over the new q against the new k/v
        # (cache is empty here — reference is for single-call usage).
        k_full = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v_full = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        scale = self.head_dim ** -0.5
        attn = (q.transpose(1, 2) @ k_full.transpose(1, 2).transpose(-1, -2)) * scale
        if tlen > 1:
            mask = torch.triu(
                torch.full((tlen, tlen), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + mask
        probs = attn.softmax(dim=-1).to(x.dtype)
        ctx = probs @ v_full.transpose(1, 2)
        ctx = ctx.transpose(1, 2).reshape(bsz, tlen, self.num_heads * self.head_dim)
        return F.linear(ctx, self.o_proj_weight) + residual

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, x_dt, cos_dt, sin_dt, *, residual_dt, output=None):
        """Per-layer allocation for attn_in, attn_out, and the fused qkv
        weight. Output (attn_proj_out) is passed in by the caller.
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16
        fused_outdim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        num_tasks_qkv = _grid_for_linear(fused_outdim)

        # Attach individual q/k/v weights, fuse via runtime shuffle_tensors.
        # Matches OLD demo exactly.
        w_q_dt = pk.attach_input(
            self.q_proj_weight, name=f"{self.prefix}q_proj_weight"
        )
        w_k_dt = pk.attach_input(
            self.k_proj_weight, name=f"{self.prefix}k_proj_weight"
        )
        w_v_dt = pk.attach_input(
            self.v_proj_weight, name=f"{self.prefix}v_proj_weight"
        )
        w_qkv_dt = pk.shuffle_tensors(
            inputs=[w_q_dt, w_k_dt, w_v_dt],
            shuffled_dim=0,
            num_groups=self.num_kv_heads,
            name=f"{self.prefix}qkv_proj",
        )

        per_layer_attn_in = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, fused_outdim),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_in",
        )
        pk.linear_layer(
            input=x_dt,
            weight=w_qkv_dt,
            output=per_layer_attn_in,
            grid_dim=(num_tasks_qkv, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Fetch per-layer KV cache via the Option-III PK helper, then
        # attach to PK as inputs for the paged attention task.
        k_cache_torch, v_cache_torch = pk.get_kv_cache(self.layer_idx)
        k_cache_dt = pk.attach_input(
            k_cache_torch, name=f"{self.prefix}k_cache"
        )
        v_cache_dt = pk.attach_input(
            v_cache_torch, name=f"{self.prefix}v_cache"
        )

        per_layer_attn_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, self.num_heads * self.head_dim),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_out",
        )
        self.attn.compile(
            per_layer_attn_in, k_cache_dt, v_cache_dt, cos_dt, sin_dt,
            output=per_layer_attn_out,
            grid_dim=(pk.max_num_batched_requests, self.num_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

        w_o_dt = pk.attach_input(
            self.o_proj_weight, name=f"{self.prefix}o_proj_weight"
        )
        pk.linear_with_residual_layer(
            input=per_layer_attn_out,
            weight=w_o_dt,
            residual=residual_dt,
            output=output,
            grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# Qwen3DecoderLayer
# ---------------------------------------------------------------------------


class Qwen3DecoderLayer(MPKModule):
    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}input_layernorm_",
        )
        self.self_attn = Qwen3Attention(
            config, layer_idx, prefix=f"{prefix}self_attn_"
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}post_attention_layernorm_",
        )
        self.mlp = Qwen3MLP(config, prefix=f"{prefix}mlp_")

    def forward(self, x, cos, sin, positions):
        attn_resid = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin, positions, residual=attn_resid)
        mlp_resid = h
        h2 = self.post_attention_layernorm(h)
        return self.mlp(h2, residual=mlp_resid)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, x_dt, cos_dt, sin_dt):
        """Compile one decoder layer. Every intermediate is allocated
        per-layer (no cross-layer aliasing).
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16
        hidden = self.input_layernorm.hidden_size

        per_layer_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_mlp_out")
        per_layer_attn_proj_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_proj_out")
        per_layer_rmsnorm_attn_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_attn_out")
        per_layer_rmsnorm_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_mlp_out")

        # Attention sub-block: rmsnorm -> qkv linear -> paged attention -> o_proj + residual.
        self.input_layernorm.compile(
            x_dt,
            output=per_layer_rmsnorm_attn_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.self_attn.compile(
            per_layer_rmsnorm_attn_out, cos_dt, sin_dt,
            residual_dt=x_dt,
            output=per_layer_attn_proj_out,
        )
        # MLP sub-block: rmsnorm -> gate_up linear -> silu_mul -> down_proj + residual.
        self.post_attention_layernorm.compile(
            per_layer_attn_proj_out,
            output=per_layer_rmsnorm_mlp_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mlp.compile(
            per_layer_rmsnorm_mlp_out,
            residual_dt=per_layer_attn_proj_out,
            output=per_layer_mlp_out,
        )
        return per_layer_mlp_out


# ---------------------------------------------------------------------------
# Qwen3Model
# ---------------------------------------------------------------------------


class Qwen3Model(MPKModule):
    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.embed_tokens = Embed(
            config.vocab_size, config.hidden_size,
            prefix=f"{prefix}embed_tokens_",
        )
        # Cap precomputed cos/sin to 4096 positions to mirror what the
        # existing demo passes to ``pk.paged_attention_layer`` (it slices
        # the HF rotary table to ``[:4096, :]``). The paged-attention task
        # template bakes cos.dim(0) into its codegen; mismatching against
        # the legacy build causes a runtime illegal-address crash.
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=min(4096, config.max_position_embeddings),
            base=config.rope_theta,
            prefix=f"{prefix}rotary_emb_",
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config, layer_idx=i, prefix=f"{prefix}layers_{i}_"
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}norm_",
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_tokens.shape[-1], device=input_tokens.device
        )
        h = self.embed_tokens(input_tokens)
        cos, sin = self.rotary_emb.cos, self.rotary_emb.sin
        for layer in self.layers:
            h = layer(h, cos, sin, positions)
        return self.norm(h)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, input_tokens_dt):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        cos_dt, sin_dt = self.rotary_emb.compile()
        hidden = self.config.hidden_size
        embed_out_dt = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}embed_out",
        )
        self.embed_tokens.compile(
            input_tokens_dt,
            input_source=1,
            output=embed_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        h_dt = embed_out_dt
        for layer in self.layers:
            h_dt = layer.compile(h_dt, cos_dt, sin_dt)
        # Final RMSNorm output is per-layer (no aliasing).
        final_rmsnorm_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}final_rmsnorm_out",
        )
        self.norm.compile(
            h_dt,
            output=final_rmsnorm_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        return final_rmsnorm_out


# ---------------------------------------------------------------------------
# Qwen3ForCausalLM
# ---------------------------------------------------------------------------


class Qwen3ForCausalLM(MPKModule):
    """Full Qwen3 model + lm_head + split-reduce argmax for greedy decode.

    Drop-in target for the canonical smoke command::

        python demo/qwen3/demo_new.py --model <hf-or-local-path> \\
            --max-num-batched-requests 1

    Driver responsibilities (see ``demo/qwen3/demo_new.py``):
      * Allocate the KV cache pool and pass it to PK via ``kv_cache=``.
      * Allocate meta_tensors (step, tokens, qo_indptr, paged_kv_*).
      * Pre-pad ``lm_head.weight`` to a vocab size divisible by the
        argmax-partial grid (153600 in the demo).
      * Pass the ``output_tokens`` torch tensor through ``model.compile()``
        for argmax-reduce readback.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.model = Qwen3Model(config, prefix=f"{prefix}model_")
        # The lm_head's out_dim is padded to a multiple of the argmax
        # partial-grid; padding lives in the driver, so the param shape
        # below uses the unpadded vocab_size and the driver overwrites
        # self.lm_head.weight after instantiation with the padded
        # weight.
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size,
            prefix=f"{prefix}lm_head_",
        )
        # Greedy-decode head: split-reduce so the large vocab fans out
        # across CTAs (see /home/zepengz/.claude/plans for rationale).
        self.argmax_partial = ArgmaxPartial(
            vocab_size=config.vocab_size,
            num_partial_tasks=1,  # placeholder; overwritten in compile()
            prefix=f"{prefix}argmax_partial_",
        )
        self.argmax_reduce = ArgmaxReduce(
            num_partial_tasks=1,
            prefix=f"{prefix}argmax_reduce_",
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        h = self.model(input_tokens)
        logits = F.linear(h, self.lm_head.weight)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, input_tokens_dt, *, output_tokens=None,
                lm_head_padded_vocab: Optional[int] = None):
        """Build the full task graph.

        Args:
            input_tokens_dt: DTensor produced by
                ``pk.attach_input(input_tokens_torch, name='input_token')``.
            output_tokens: torch.Tensor (max_num_batched_tokens, 1) int64,
                pre-allocated by the driver. Bound as the final
                argmax-reduce output for readback.
            lm_head_padded_vocab: vocab dimension after padding (153600
                in the demo). Must equal ``self.lm_head.weight.shape[0]``;
                the driver pre-pads the weight then passes the size here
                so this module knows the actual GEMM output shape.
        """
        pk = current_pk()
        h_dt = self.model.compile(input_tokens_dt)

        # lm_head: same num_workers-driven grid as the demo.
        logits_dt = self.lm_head.compile(
            h_dt,
            grid_dim=(pk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Argmax split-reduce. Per the demo, partial grid is num_workers,
        # reduce grid is 1.
        # CHUNK_SIZE coupling between partial and reduce is implicit (see
        # the ArgmaxPartial/Reduce module docstrings); partial must
        # compile before reduce within the same scope.
        self.argmax_partial.num_partial_tasks = pk.num_workers
        self.argmax_reduce.num_partial_tasks = pk.num_workers
        part_val_dt, part_idx_dt = self.argmax_partial.compile(
            logits_dt,
            grid_dim=(pk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )
        return self.argmax_reduce.compile(
            part_val_dt, part_idx_dt,
            output=output_tokens,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
