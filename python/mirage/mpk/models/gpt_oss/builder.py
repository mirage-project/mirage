import torch

from ..graph_builder import GraphBuilder
from ..utils import grid_for_rmsnorm_linear_layer, shuffle_tensors
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, float32, int32, int64

from typing import Optional


# The SM100 linear and the MoE group GEMM both want each task's output column
# slice 16-byte aligned, i.e. a multiple of 8.
def _grid_x(output_size: int, cols_per_task: int = 64) -> int:
    assert output_size % cols_per_task == 0
    assert cols_per_task % 8 == 0
    return output_size // cols_per_task


@register_model_builder("gpt_oss", "GptOss", "openai/gpt-oss-20b")
class GptOssBuilder(GraphBuilder):
    """GPT-OSS-20B: alternating sliding/full attention with per-head sinks, and
    a clamped-alpha SwiGLU MoE. Every projection carries a bias.
    """

    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.rank = mpk.mpi_rank
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.tokenizer = None
        self.eos_token_id = 200002
        self._keep = []  # converted weights must outlive the graph build

    # ---------------------------------------------------------------- loading

    def build_from_model(self, model_name: str, model_path: str | None = None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        source = model_path or model_name
        self.config = AutoConfig.from_pretrained(source)
        assert self.world_size == 1, "GPT-OSS is single-GPU for now"

        # Loaded on the host: the conversions below transpose and
        # de-interleave whole expert tensors, so holding both copies on the
        # GPU would need twice the model. _attach moves each one over.
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.eos_token_id = self.config.eos_token_id

        cfg = self.config
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size
        self.vocab_size = cfg.vocab_size
        # 201728 is a multiple of 256, the slice
        # grid_for_rmsnorm_linear_layer picks at this width.
        self.padded_vocab_size = 201728
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_q_per_kv = self.num_q_heads // self.num_kv_heads
        self.head_dim = cfg.head_dim
        self.num_layers = cfg.num_hidden_layers
        self.num_experts = cfg.num_local_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.layer_types = list(cfg.layer_types)
        self.sliding_window = cfg.sliding_window
        self.swiglu_limit = getattr(cfg, "swiglu_limit", 7.0)
        self.fused_qkv_size = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim

        # RoPE from the model's own module, so YaRN and its attention_scaling
        # are not re-derived here. Its tables are half-width, broadcast against
        # the two rotate-half chunks; MPK's kernel indexes a full head_dim row,
        # so duplicate them.
        dummy = torch.empty(0, dtype=torch.bfloat16)
        positions = torch.arange(self.mpk.max_seq_length).unsqueeze(0)
        cos, sin = model.model.rotary_emb(dummy, positions)
        self.cos_table = torch.cat([cos[0], cos[0]], dim=-1).contiguous().to(torch.bfloat16)
        self.sin_table = torch.cat([sin[0], sin[0]], dim=-1).contiguous().to(torch.bfloat16)

        self.k_cache = torch.zeros(
            (self.num_layers, self.max_num_pages, self.page_size,
             self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16, device="cuda")
        self.v_cache = torch.zeros_like(self.k_cache)

        state_dict = model.state_dict()
        self.build_from_dict(state_dict, with_lm_head=True)

    def build_from_config(self, model_config):
        raise NotImplementedError(
            "GPT-OSS is built from a HuggingFace checkpoint; use build_from_model")

    # ------------------------------------------------------- weight conversion

    def _attach(self, tensor: torch.Tensor, name: str):
        tensor = tensor.contiguous().to("cuda")
        self._keep.append(tensor)
        return self.mpk.attach_input(torch_tensor=tensor, name=name)

    def _fused_qkv(self, sd, prefix, i):
        """Interleave q/k/v by KV head, as the attention kernel's packed layout
        expects, for both the weights and the biases."""
        w = shuffle_tensors(
            [sd[f"{prefix}self_attn.q_proj.weight"],
             sd[f"{prefix}self_attn.k_proj.weight"],
             sd[f"{prefix}self_attn.v_proj.weight"]],
            self.num_kv_heads, 0)
        b = shuffle_tensors(
            [sd[f"{prefix}self_attn.q_proj.bias"],
             sd[f"{prefix}self_attn.k_proj.bias"],
             sd[f"{prefix}self_attn.v_proj.bias"]],
            self.num_kv_heads, 0)
        return (self._attach(w, f"layer_{i}_qkv_proj"),
                self._attach(b.view(1, -1), f"layer_{i}_qkv_bias"))

    def _experts(self, sd, prefix, i):
        """GPT-OSS stores gate_up as [E, hidden, 2I] with gate and up
        INTERLEAVED on the last dim, and down as [E, I, hidden]. MPK wants
        [E, 2I, hidden] with gate as the first half, and [E, hidden, I]."""
        gu = sd[f"{prefix}mlp.experts.gate_up_proj"].permute(0, 2, 1)
        w13 = torch.cat([gu[:, 0::2, :], gu[:, 1::2, :]], dim=1)
        gub = sd[f"{prefix}mlp.experts.gate_up_proj_bias"]
        b13 = torch.cat([gub[:, 0::2], gub[:, 1::2]], dim=1)
        w2 = sd[f"{prefix}mlp.experts.down_proj"].permute(0, 2, 1)
        b2 = sd[f"{prefix}mlp.experts.down_proj_bias"]
        return (self._attach(w13, f"layer_{i}_w13"),
                self._attach(b13, f"layer_{i}_b13"),
                self._attach(w2, f"layer_{i}_w2"),
                self._attach(b2, f"layer_{i}_b2"))

    # ------------------------------------------------------------------ graph

    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        mbt = self.mpk.max_num_batched_tokens
        lm_head = torch.cat(
            [state_dict["lm_head.weight"],
             torch.zeros(self.padded_vocab_size - self.vocab_size,
                         self.hidden_size, dtype=torch.bfloat16)],
            dim=0)

        self.x = self.mpk.attach_input(torch_tensor=self.input_tokens,
                                       name="input_token")
        self.cos_dt = self._attach(self.cos_table, "cos_position_embedding")
        self.sin_dt = self._attach(self.sin_table, "sin_position_embedding")
        # GPT-OSS has no QK-norm; the kernel still takes the two operands.
        self.norm_dummy = self._attach(
            torch.ones(self.head_dim, dtype=torch.bfloat16),
            "qk_norm_dummy")

        h, ii = self.hidden_size, self.intermediate_size
        topk = self.num_experts_per_tok
        self.y = self.mpk.new_tensor(dims=(mbt, h), dtype=bfloat16,
                                     name="embed_out", io_category="cuda_tensor")
        self.rmsnorm_out = self.mpk.new_tensor(dims=(mbt, h), dtype=bfloat16,
                                               name="rmsnorm_out", io_category="cuda_tensor")
        self.attn_in = self.mpk.new_tensor(dims=(mbt, self.fused_qkv_size), dtype=bfloat16,
                                           name="attn_in", io_category="cuda_tensor")
        self.attn_out = self.mpk.new_tensor(dims=(mbt, self.num_q_heads * self.head_dim),
                                            dtype=bfloat16, name="attn_out", io_category="cuda_tensor")
        self.attn_proj_out = self.mpk.new_tensor(dims=(mbt, h), dtype=bfloat16,
                                                 name="attn_proj_out", io_category="cuda_tensor")
        self.attn_biased_out = self.mpk.new_tensor(dims=(mbt, h), dtype=bfloat16,
                                                   name="attn_biased_out", io_category="cuda_tensor")
        self.gate_out = self.mpk.new_tensor(dims=(mbt, self.num_experts), dtype=bfloat16,
                                            name="moe_gate_out", io_category="cuda_tensor")
        self.topk_weight = self.mpk.new_tensor(dims=(mbt, topk), dtype=float32,
                                               name="moe_topk_weight", io_category="cuda_tensor")
        self.routing_indices = self.mpk.new_tensor(dims=(self.num_experts, mbt), dtype=int32,
                                                   name="moe_routing_indices", io_category="cuda_tensor")
        self.moe_mask = self.mpk.new_tensor(dims=(self.num_experts + 1,), dtype=int32,
                                            name="moe_mask", io_category="cuda_tensor")
        self.mlp_mid = self.mpk.new_tensor(dims=(mbt, topk, 2 * ii), dtype=bfloat16,
                                           name="mlp_mid", io_category="cuda_tensor")
        self.swiglu_out = self.mpk.new_tensor(dims=(mbt, topk, ii), dtype=bfloat16,
                                              name="swiglu_out", io_category="cuda_tensor")
        self.mlp_out = self.mpk.new_tensor(dims=(mbt, topk, h), dtype=bfloat16,
                                           name="mlp_out", io_category="cuda_tensor")
        self.mlp_sum_out = self.mpk.new_tensor(dims=(mbt, h), dtype=bfloat16,
                                               name="mlp_sum_out", io_category="cuda_tensor")
        self.argmax_in = self.mpk.new_tensor(dims=(mbt, self.padded_vocab_size), dtype=bfloat16,
                                             name="argmax_in", io_category="cuda_tensor")
        self.argmax_part_value = self.mpk.new_tensor(
            dims=(mbt, self.mpk.num_workers), dtype=bfloat16,
            name="argmax_part_value", io_category="cuda_tensor")
        self.argmax_part_index = self.mpk.new_tensor(
            dims=(mbt, self.mpk.num_workers), dtype=int64,
            name="argmax_part_index", io_category="cuda_tensor")
        argmax_out = self.mpk.attach_input(torch_tensor=self.output_tokens,
                                           name="output_token")

        self.mpk.embed_layer(
            input=self.x,
            weight=self._attach(state_dict["model.embed_tokens.weight"],
                                "embed_tokens"),
            output=self.y, grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
            input_source=1)
        self.x = self.y

        self.build_layers(state_dict)

        self.mpk.rmsnorm_layer(
            input=self.x,
            weight=self._attach(state_dict["model.norm.weight"], "model_norm"),
            output=self.rmsnorm_out,
            grid_dim=(mbt, 1, 1), block_dim=(256, 1, 1))
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=self._attach(lm_head, "lm_head"),
            output=self.argmax_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(self.padded_vocab_size), 1, 1),
            block_dim=(256, 1, 1))
        self.mpk.argmax_partial_layer(
            input=self.argmax_in,
            output=(self.argmax_part_value, self.argmax_part_index),
            grid_dim=(self.mpk.num_workers, 1, 1), block_dim=(256, 1, 1))
        self.mpk.argmax_reduce_layer(
            input=(self.argmax_part_value, self.argmax_part_index),
            output=argmax_out, grid_dim=(1, 1, 1), block_dim=(256, 1, 1))

    def build_layers(self, sd: dict):
        mbt = self.mpk.max_num_batched_tokens
        h, ii = self.hidden_size, self.intermediate_size
        topk = self.num_experts_per_tok

        for i in range(self.num_layers):
            prefix = f"model.layers.{i}."
            w_qkv, b_qkv = self._fused_qkv(sd, prefix, i)
            w13, b13, w2, b2 = self._experts(sd, prefix, i)

            self.mpk.rmsnorm_layer(
                input=self.x,
                weight=self._attach(sd[f"{prefix}input_layernorm.weight"],
                                    f"layer_{i}_input_norm"),
                output=self.rmsnorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(256, 1, 1))
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_qkv, bias=b_qkv,
                output=self.attn_in,
                grid_dim=(_grid_x(self.fused_qkv_size, 80), 1, 1),
                block_dim=(256, 1, 1))

            k_cache = self._attach(self.k_cache[i], f"layer_{i}_k_cache")
            v_cache = self._attach(self.v_cache[i], f"layer_{i}_v_cache")
            sinks = self._attach(
                sd[f"{prefix}self_attn.sinks"].view(self.num_kv_heads,
                                                    self.num_q_per_kv),
                f"layer_{i}_sinks")
            self.mpk.paged_attention_layer(
                input=self.attn_in, k_cache=k_cache, v_cache=v_cache,
                q_norm=self.norm_dummy, k_norm=self.norm_dummy,
                cos_pos_embed=self.cos_dt, sin_pos_embed=self.sin_dt,
                output=self.attn_out,
                grid_dim=(self.mpk.max_num_batched_requests, self.num_kv_heads, 1),
                block_dim=(256, 1, 1),
                enable_qk_norm=False,
                window_size=(self.sliding_window
                             if self.layer_types[i] == "sliding_attention" else 0),
                sinks=sinks)

            # o_proj has two addends and the epilogue one slot. The residual
            # takes it, since it must be added exactly once ahead of a
            # tensor-parallel allreduce; the bias follows as a separate add.
            self.mpk.linear_with_residual_layer(
                input=self.attn_out,
                weight=self._attach(sd[f"{prefix}self_attn.o_proj.weight"],
                                    f"layer_{i}_o_proj"),
                residual=self.x, output=self.attn_proj_out,
                grid_dim=(_grid_x(h, 64), 1, 1), block_dim=(256, 1, 1))
            self.mpk.elementwise_add_layer(
                input_a=self.attn_proj_out,
                input_b=self._attach(
                    sd[f"{prefix}self_attn.o_proj.bias"].view(1, -1).expand(mbt, h),
                    f"layer_{i}_o_bias"),
                output=self.attn_biased_out,
                grid_dim=(mbt, 1, 1), block_dim=(256, 1, 1))
            self.x = self.attn_biased_out

            self.mpk.rmsnorm_layer(
                input=self.x,
                weight=self._attach(sd[f"{prefix}post_attention_layernorm.weight"],
                                    f"layer_{i}_post_attn_norm"),
                output=self.rmsnorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(256, 1, 1))
            self.mpk.linear_layer(
                input=self.rmsnorm_out,
                weight=self._attach(sd[f"{prefix}mlp.router.weight"],
                                    f"layer_{i}_router"),
                bias=self._attach(sd[f"{prefix}mlp.router.bias"].view(1, -1),
                                  f"layer_{i}_router_bias"),
                output=self.gate_out, grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
            self.mpk.moe_topk_softmax_routing_layer(
                input=self.gate_out,
                output=(self.topk_weight, self.routing_indices, self.moe_mask),
                grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
            self.mpk.moe_w13_linear_layer(
                input=self.rmsnorm_out, weight=w13, bias=b13,
                moe_routing_indices=self.routing_indices, moe_mask=self.moe_mask,
                output=self.mlp_mid,
                grid_dim=(10, _grid_x(2 * ii, 128), 1), block_dim=(256, 1, 1))
            self.mpk.moe_clamped_swiglu_layer(
                input=self.mlp_mid, output=self.swiglu_out,
                grid_dim=(mbt, topk, 1), block_dim=(256, 1, 1),
                limit=self.swiglu_limit, alpha=1.702)
            self.mpk.moe_w2_linear_layer(
                input=self.swiglu_out, weight=w2, bias=b2,
                moe_routing_indices=self.routing_indices, moe_mask=self.moe_mask,
                output=self.mlp_out,
                grid_dim=(8, _grid_x(h, 64), 1), block_dim=(256, 1, 1))
            self.mpk.moe_mul_sum_add_layer(
                input=self.mlp_out, weight=self.topk_weight, residual=self.x,
                output=self.mlp_sum_out,
                grid_dim=(mbt, _grid_x(h, 64), 1), block_dim=(256, 1, 1))
            self.x = self.mlp_sum_out

    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

