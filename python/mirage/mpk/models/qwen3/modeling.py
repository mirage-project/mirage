from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...context import current_pk
from ...layers import (
    AllReduce,
    ArgmaxPartial,
    ArgmaxReduce,
    ColumnParallelLinear,
    Embed,
    Linear,
    MPKModule,
    PagedAttention,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinearWithResidual,
)
from .._registry import register_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_for_linear(size: int, use_cutlass: bool = True) -> int:
    """Mirror ``grid_for_rmsnorm_linear_layer`` from demo/qwen3/demo.py.

    Picks the tile divisor that the kernel's task atom expects so the
    generated task graph matches the legacy demo's bytes-for-bytes.
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


def _aligned_lm_head_tasks(width: int, max_tasks: int, align: int = 8) -> int:
    for g in range(max_tasks, 0, -1):
        if width % g == 0 and (width // g) % align == 0:
            return g
    return 1


def _remap_qwen3_hf_key(name: str) -> str:
    """Map an HF Qwen3 state_dict key to its MPK catalog named_parameters() path.

    HF stores q_norm/k_norm as RMSNorm modules ('...self_attn.q_norm.weight');
    the catalog stores them as raw params on PagedAttention ('...self_attn.attn.q_norm').
    """
    if name.endswith(".self_attn.q_norm.weight"):
        return name[: -len(".self_attn.q_norm.weight")] + ".self_attn.attn.q_norm"
    if name.endswith(".self_attn.k_norm.weight"):
        return name[: -len(".self_attn.k_norm.weight")] + ".self_attn.attn.k_norm"
    return name


# ---------------------------------------------------------------------------
# Qwen3MLP
# ---------------------------------------------------------------------------


class Qwen3MLP(MPKModule):
    """gate_proj + up_proj fused via shuffle_tensors at compile time, then
    silu_mul, then down_proj + residual + (optional) AllReduce.

    Under TP, gate_proj/up_proj are column-parallel (each rank holds
    ``intermediate_size // tp_size`` rows of the unsharded weight); the
    compile path still calls ``pk.shuffle_tensors`` on the two sharded
    weights — kernels emit one fused linear over the sharded intermediate
    space. down_proj is row-parallel: weight is
    ``(hidden, intermediate_size // tp_size)``; the kernel's
    ``enable_residual`` task param is forced off on non-rank-0, then the
    follow-up AllReduce sums partials and the residual is added once.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        pc = current_pk().parallel_config
        self.tp_size = pc.tp_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.intermediate_size_per_partition = config.intermediate_size // pc.tp_size

        # All three projections are catalog leaves so HF state_dict keys
        # (...mlp.gate_proj.weight etc.) match by named_modules() path.
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            prefix=f"{prefix}gate_proj_",
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            prefix=f"{prefix}up_proj_",
        )
        self.down_proj = RowParallelLinearWithResidual(
            self.intermediate_size, self.hidden_size,
            prefix=f"{prefix}down_proj_",
        )
        if self.tp_size > 1:
            self.allreduce = AllReduce(prefix=f"{prefix}mlp_allreduce_")

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # PyTorch reference. With tp>1, gate_proj/up_proj/down_proj act on
        # local slices; result is this rank's partial. The caller is
        # expected to AllReduce externally — same single-rank-slice
        # semantics as the compile path.
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        silu_out = (F.silu(gate.float()) * up.float()).to(x.dtype)
        # down_proj.forward inherits LinearWithResidual.forward which
        # already includes the residual add.
        return self.down_proj(silu_out, residual)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, x_dt, residual_dt, *, output=None):
        """Compile MLP path. ``output`` is the caller-supplied DTensor for
        the final hidden activation (post-AllReduce when tp>1).
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        fused_out = 2 * self.intermediate_size_per_partition
        num_tasks_linear = _grid_for_linear(fused_out)

        # Per-partition sharded weights, fused at compile via shuffle_tensors.
        w_gate_dt = pk.attach_input(
            self.gate_proj.weight, name=f"{self.prefix}gate_proj_weight",
        )
        w_up_dt = pk.attach_input(
            self.up_proj.weight, name=f"{self.prefix}up_proj_weight",
        )
        w_gateup_dt = pk.shuffle_tensors(
            inputs=[w_gate_dt, w_up_dt],
            shuffled_dim=0,
            num_groups=num_tasks_linear // 2,
            name=f"{self.prefix}gateup_proj",
        )

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

        per_layer_silu_mul_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, self.intermediate_size_per_partition),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_silu_mul_out",
        )
        pk.silu_mul_layer(
            input=per_layer_mlp_mid,
            output=per_layer_silu_mul_out,
            grid_dim=(num_tasks_linear // 2, 1, 1),
            block_dim=(128, 1, 1),
        )

        # down_proj + residual. Under TP this writes a per-rank partial to
        # ``output`` (an NVSHMEM-symmetric tensor when tp>1).
        w_down_dt = pk.attach_input(
            self.down_proj.weight, name=f"{self.prefix}down_proj_weight",
        )
        # On SM100 single-GPU, fuse the residual add into a split-K GEMM:
        # ``splitk_linear_layer`` reduce-adds the matmul partials into
        # ``output`` WITHOUT zeroing, so seeding ``output`` with the residual
        # buffer yields ``residual + silu_out @ W_down`` in a single task.
        # This is the ~3.9ms path and mirrors demo/qwen3/demo.py. TP keeps
        # linear_with_residual because the residual must be added once,
        # post-AllReduce (split-K has no per-rank residual gate).
        if pk.target_cc == 100 and self.tp_size == 1:
            pk.splitk_linear_layer(
                input=per_layer_silu_mul_out,
                weight=w_down_dt,
                output=residual_dt,
                grid_dim=(self.hidden_size // 128,
                          128 * 128 // self.hidden_size, 1),
                block_dim=(256, 1, 1),
            )
            return residual_dt
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
    """q/k/v as separate column-parallel linears, fused at compile time via
    ``shuffle_tensors``; paged attention runs on the per-rank head slice;
    o_proj is row-parallel-with-residual followed by AllReduce (tp>1).
    """

    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.layer_idx = layer_idx
        pc = current_pk().parallel_config
        self.tp_size = pc.tp_size

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads_per_partition = self.num_heads // self.tp_size
        self.num_kv_heads_per_partition = self.num_kv_heads // self.tp_size

        # q/k/v as catalog leaves; HF state_dict keys match named_modules() paths.
        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim,
            prefix=f"{prefix}q_proj_",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim,
            prefix=f"{prefix}k_proj_",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim,
            prefix=f"{prefix}v_proj_",
        )
        self.attn = PagedAttention(
            num_heads=self.num_heads_per_partition,
            num_kv_heads=self.num_kv_heads_per_partition,
            head_dim=self.head_dim,
            layer_idx=layer_idx,
            prefix=f"{prefix}",
        )
        self.o_proj = RowParallelLinearWithResidual(
            self.num_heads * self.head_dim, self.hidden_size,
            prefix=f"{prefix}o_proj_",
        )
        if self.tp_size > 1:
            self.allreduce = AllReduce(prefix=f"{prefix}attn_allreduce_")

    def forward(self, x, cos, sin, positions, residual):
        bsz, tlen, _ = x.shape
        q = self.q_proj(x).view(bsz, tlen, self.num_heads_per_partition, self.head_dim)
        k = self.k_proj(x).view(bsz, tlen, self.num_kv_heads_per_partition, self.head_dim)
        v = self.v_proj(x).view(bsz, tlen, self.num_kv_heads_per_partition, self.head_dim)
        from ...layers.attention.attention import (
            _apply_rotary as _ar, _per_head_rmsnorm as _phr,
        )
        q = _phr(q, self.attn.q_norm)
        k = _phr(k, self.attn.k_norm)
        q = _ar(q, cos[positions], sin[positions])
        k = _ar(k, cos[positions], sin[positions])
        groups = self.num_heads_per_partition // self.num_kv_heads_per_partition
        k_full = k.repeat_interleave(groups, dim=2)
        v_full = v.repeat_interleave(groups, dim=2)
        scale = self.head_dim ** -0.5
        attn = (q.transpose(1, 2) @ k_full.transpose(1, 2).transpose(-1, -2)) * scale
        if tlen > 1:
            mask = torch.triu(
                torch.full((tlen, tlen), float("-inf"), device=x.device),
                diagonal=1,
            )
            attn = attn + mask
        probs = attn.softmax(dim=-1).to(x.dtype)
        ctx = probs @ v_full.transpose(1, 2)
        ctx = ctx.transpose(1, 2).reshape(
            bsz, tlen, self.num_heads_per_partition * self.head_dim,
        )
        # o_proj is RowParallelLinearWithResidual → forward gives partial
        # + residual on local input slice. Single-rank slice semantics.
        return self.o_proj(ctx, residual)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, x_dt, cos_dt, sin_dt, *, residual_dt, output=None):
        """Build the attention sub-block task graph.

        Allocates per-layer ``attn_in`` (fused QKV output, NVSHMEM-symmetric
        when tp>1 is unnecessary since q/k/v are local) and ``attn_out``,
        then fetches per-rank KV-cache pool and runs paged attention with
        per-partition head counts.
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        # Per-partition fused QKV outdim.
        fused_outdim = (
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        ) // self.tp_size
        num_tasks_qkv = _grid_for_linear(fused_outdim)

        # Attach SHARDED q/k/v weights. Each rank holds its own slice;
        # shuffle_tensors fuses them locally by KV-group ordering.
        w_q_dt = pk.attach_input(
            self.q_proj.weight, name=f"{self.prefix}q_proj_weight",
        )
        w_k_dt = pk.attach_input(
            self.k_proj.weight, name=f"{self.prefix}k_proj_weight",
        )
        w_v_dt = pk.attach_input(
            self.v_proj.weight, name=f"{self.prefix}v_proj_weight",
        )
        w_qkv_dt = pk.shuffle_tensors(
            inputs=[w_q_dt, w_k_dt, w_v_dt],
            shuffled_dim=0,
            num_groups=self.num_kv_heads_per_partition,
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

        # Per-rank KV cache slice (dim 0 = layer; remaining dims = pages,
        # page_size, num_kv_heads_per_partition, head_dim).
        k_cache_torch, v_cache_torch = pk.get_kv_cache(self.layer_idx)
        k_cache_dt = pk.attach_input(
            k_cache_torch, name=f"{self.prefix}k_cache",
        )
        v_cache_dt = pk.attach_input(
            v_cache_torch, name=f"{self.prefix}v_cache",
        )

        per_layer_attn_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens,
                  self.num_heads_per_partition * self.head_dim),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_out",
        )
        self.attn.compile(
            per_layer_attn_in, k_cache_dt, v_cache_dt, cos_dt, sin_dt,
            output=per_layer_attn_out,
            grid_dim=(pk.max_num_batched_requests,
                      self.num_kv_heads_per_partition, 1),
            block_dim=(128, 1, 1),
        )

        # o_proj is row-parallel; weight is (hidden, in_per_partition).
        w_o_dt = pk.attach_input(
            self.o_proj.weight, name=f"{self.prefix}o_proj_weight",
        )
        # SM100 single-GPU: fuse the residual add into a split-K GEMM by
        # seeding ``output`` with the residual buffer (see Qwen3MLP.compile).
        if pk.target_cc == 100 and self.tp_size == 1:
            pk.splitk_linear_layer(
                input=per_layer_attn_out,
                weight=w_o_dt,
                output=residual_dt,
                grid_dim=(self.hidden_size // 128,
                          128 * 128 // self.hidden_size, 1),
                block_dim=(256, 1, 1),
            )
            return residual_dt
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
        pc = current_pk().parallel_config
        self.tp_size = pc.tp_size
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}input_layernorm_",
        )
        self.self_attn = Qwen3Attention(
            config, layer_idx, prefix=f"{prefix}self_attn_",
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
        h = self.self_attn(h, cos, sin, positions, attn_resid)
        mlp_resid = h
        h2 = self.post_attention_layernorm(h)
        return self.mlp(h2, mlp_resid)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, x_dt, cos_dt, sin_dt):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16
        hidden = self.input_layernorm.hidden_size

        # On SM100 single-GPU each residual add is fused into a split-K GEMM
        # that writes in place into the residual buffer (the ~3.9ms path; see
        # Qwen3MLP.compile). In that mode o_proj/down_proj produce the residual
        # tensors directly, so the dedicated per-layer projection buffers are
        # only allocated off that path. ``compile()`` returns the buffer that
        # holds its result, which we thread downstream.
        use_splitk = pk.target_cc == 100 and self.tp_size == 1

        # Per-layer intermediate buffers. Under TP the projection
        # outputs cross ranks via NVSHMEM, so those tensors get
        # io_category="nvshmem_tensor".
        nvshmem_kind = "nvshmem_tensor" if self.tp_size > 1 else "cuda_tensor"

        per_layer_rmsnorm_attn_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_attn_out",
        )
        per_layer_rmsnorm_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_mlp_out",
        )

        # Attention sub-block: rmsnorm → qkv linear → paged attn → o_proj+resid
        self.input_layernorm.compile(
            x_dt,
            output=per_layer_rmsnorm_attn_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        per_layer_attn_proj_out = None
        if not use_splitk:
            per_layer_attn_proj_out = pk.new_tensor(
                dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
                name=f"{self.prefix}per_layer_attn_proj_out",
                io_category=nvshmem_kind,
            )
        attn_out_buf = self.self_attn.compile(
            per_layer_rmsnorm_attn_out, cos_dt, sin_dt,
            residual_dt=x_dt,
            output=per_layer_attn_proj_out,
        )

        # AllReduce after attention (tp > 1).
        if self.tp_size > 1:
            attn_allreduce_out = pk.new_tensor(
                dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
                name=f"{self.prefix}per_layer_attn_allreduce_out",
                io_category="nvshmem_tensor",
            )
            allreduce_buf = pk.new_tensor(
                dims=(pk.world_size, pk.max_num_batched_tokens, hidden),
                dtype=_mi_bf16,
                name=f"{self.prefix}per_layer_attn_allreduce_buf",
                io_category="nvshmem_tensor",
            )
            self.self_attn.allreduce.compile(
                input=attn_out_buf,
                buffer=allreduce_buf,
                output=attn_allreduce_out,
                grid_dim=(hidden // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            mlp_input = attn_allreduce_out
            mlp_residual = attn_allreduce_out
        else:
            mlp_input = attn_out_buf
            mlp_residual = attn_out_buf

        # MLP sub-block: rmsnorm → gateup linear → silu_mul → down_proj+resid
        self.post_attention_layernorm.compile(
            mlp_input,
            output=per_layer_rmsnorm_mlp_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        per_layer_mlp_out = None
        if not use_splitk:
            per_layer_mlp_out = pk.new_tensor(
                dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
                name=f"{self.prefix}per_layer_mlp_out",
                io_category=nvshmem_kind,
            )
        mlp_out_buf = self.mlp.compile(
            per_layer_rmsnorm_mlp_out,
            residual_dt=mlp_residual,
            output=per_layer_mlp_out,
        )

        # AllReduce after MLP (tp > 1).
        if self.tp_size > 1:
            mlp_allreduce_out = pk.new_tensor(
                dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
                name=f"{self.prefix}per_layer_mlp_allreduce_out",
                io_category="nvshmem_tensor",
            )
            allreduce_buf_mlp = pk.new_tensor(
                dims=(pk.world_size, pk.max_num_batched_tokens, hidden),
                dtype=_mi_bf16,
                name=f"{self.prefix}per_layer_mlp_allreduce_buf",
                io_category="nvshmem_tensor",
            )
            self.mlp.allreduce.compile(
                input=mlp_out_buf,
                buffer=allreduce_buf_mlp,
                output=mlp_allreduce_out,
                grid_dim=(hidden // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            return mlp_allreduce_out
        return mlp_out_buf


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
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=min(4096, config.max_position_embeddings),
            base=config.rope_theta,
            prefix=f"{prefix}rotary_emb_",
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config, layer_idx=i, prefix=f"{prefix}layers_{i}_",
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}norm_",
        )

    def forward(self, input_tokens):
        positions = torch.arange(
            input_tokens.shape[-1], device=input_tokens.device,
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


@register_model("Qwen3ForCausalLM")
class Qwen3ForCausalLM(MPKModule):
    """Full Qwen3 model + lm_head + split-reduce argmax for greedy decode.

    lm_head is replicated (not vocab-parallel) in v1. ``process_weights``
    pads the lm_head's vocab dim to ``LM_HEAD_PADDED_VOCAB`` so the
    argmax-partial grid divides it evenly (mirrors the legacy demo).
    The driver no longer touches lm_head shape.
    """

    # Padded vocab matches the legacy demo's argmax-partial tile math.
    # See demo/qwen3/demo.py: ``max_factor_leq_n(153600, 96 // ...)``.
    LM_HEAD_PADDED_VOCAB: int = 153600

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.model = Qwen3Model(config, prefix=f"{prefix}model_")
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size,
            prefix=f"{prefix}lm_head_",
        )
        self.argmax_partial = ArgmaxPartial(
            vocab_size=config.vocab_size,
            num_partial_tasks=1,
            prefix=f"{prefix}argmax_partial_",
        )
        self.argmax_reduce = ArgmaxReduce(
            num_partial_tasks=1,
            prefix=f"{prefix}argmax_reduce_",
        )
        # Bound to an alignment-safe task count in process_weights (the live
        # num_workers isn't known until then). Used by compile() for the
        # lm_head linear + argmax_partial grids.
        self._lm_head_tasks: Optional[int] = None

    def resolve_weight(self, name, params):
        return super().resolve_weight(_remap_qwen3_hf_key(name), params)

    def forward(self, input_tokens):
        h = self.model(input_tokens)
        logits = F.linear(h, self.lm_head.weight)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def process_weights(self) -> None:
        """Post-load: pad lm_head + bind argmax_partial num_partial_tasks
        to the live ``num_workers``. ``process_weights`` runs inside
        ``compile_scope`` so ``current_pk()`` is valid.
        """
        super().process_weights()
        pk = current_pk()
        padded = self.LM_HEAD_PADDED_VOCAB
        if self.lm_head.out_features < padded:
            old_w = self.lm_head.weight.data
            padded_weight = torch.zeros(
                padded, old_w.shape[1],
                dtype=old_w.dtype, device=old_w.device,
            )
            padded_weight[: old_w.shape[0]] = old_w
            self.lm_head.weight = nn.Parameter(padded_weight)
            self.lm_head.out_features = padded
            self.argmax_partial.vocab_size = padded
        elif self.lm_head.out_features > padded:
            raise ValueError(
                f"Qwen3ForCausalLM.process_weights: lm_head out_features "
                f"({self.lm_head.out_features}) exceeds LM_HEAD_PADDED_VOCAB "
                f"({padded}); raise the class attribute.",
            )
        # The lm_head linear column-splits the padded vocab across this many
        # tasks and ArgmaxPartial chunks it the same way. Raw num_workers does
        # NOT divide 153600 into 16B-aligned tiles (crashes the lm_head TMA on
        # B200), so pick the largest alignment-safe count <= num_workers.
        self._lm_head_tasks = _aligned_lm_head_tasks(
            self.lm_head.out_features, pk.num_workers
        )
        self.argmax_partial.num_partial_tasks = self._lm_head_tasks
        self.argmax_reduce.num_partial_tasks = self._lm_head_tasks

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite module — see child compile()s")

    def compile(self, input_tokens_dt, *, output_tokens=None):
        pk = current_pk()
        # process_weights sets this to an alignment-safe split of the padded
        # vocab; using raw pk.num_workers here misaligns the lm_head TMA store.
        lm_head_tasks = self._lm_head_tasks
        if lm_head_tasks is None:
            lm_head_tasks = _aligned_lm_head_tasks(
                self.lm_head.out_features, pk.num_workers
            )
        h_dt = self.model.compile(input_tokens_dt)
        logits_dt = self.lm_head.compile(
            h_dt,
            grid_dim=(lm_head_tasks, 1, 1),
        )
        part_val_dt, part_idx_dt = self.argmax_partial.compile(
            logits_dt,
            grid_dim=(lm_head_tasks, 1, 1),
        )
        return self.argmax_reduce.compile(
            part_val_dt, part_idx_dt,
            output=output_tokens,
            grid_dim=(1, 1, 1),
        )
