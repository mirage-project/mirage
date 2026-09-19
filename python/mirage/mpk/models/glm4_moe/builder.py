"""GLM-4.6 (zai-org/GLM-4.6, Glm4MoeForCausalLM) model builder for Mirage MPK.

Architecture (config.json + transformers modeling_glm4_moe.py):
- 92 decoder layers, hidden 5120, 96 Q heads / 8 KV heads, head_dim 128.
- Attention: q/k/v projections WITH bias (attention_bias=true), per-head
  q_norm/k_norm (RMSNorm eps 1e-5) applied before RoPE, partial RoPE
  (partial_rotary_factor 0.5 -> rotary_dim 64 of 128, theta 1e6), causal,
  scores scaled 1/sqrt(head_dim), o_proj without bias.
- MLP: layers 0-2 dense (intermediate 12288); layers 3-91 MoE: 160 routed
  experts (intermediate 1536, top-8) + 1 shared expert (same size).
  Router: topk on sigmoid(logits) + e_score_correction_bias (n_group=1, no
  group limiting); weights = gathered UNBIASED sigmoid scores normalized by
  their sum (norm_topk_prob) * routed_scaling_factor (2.5). The shared
  expert output is added unweighted -- folded here as expert 160, always
  selected at slot 8 with weight 1.0, so the standard moe_w13 / silu_mul /
  moe_w2 / mul_sum_add pipeline runs unchanged.
- Logits: lm_head (untied), vocab 151552 (already a multiple of 256).

v1 limitations: world_size == 1, BF16 weights. Skips the MTP nextn layer
(num_nextn_predict_layers == 1 -> model.layers.92) like the Inkling port.
`num_layers` can be overridden via the state dict contents for smoke tests
(layers are built for indices present in the dict).

Known approximation: the hidden-state RMSNorm task hardcodes eps 1e-6 while
GLM uses 1e-5 (a ~5e-6 relative effect, far below bf16 resolution; the
end-to-end test vs HF shows cosine > 0.9998). The attention q/k norms DO use
the exact 1e-5 via the qk_norm_eps parameter.
"""

from typing import Optional

import torch

from ..utils import grid_for_rmsnorm_linear_layer, shuffle_tensors
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, float32, int32, int64

# ---- GLM-4.6 architecture constants (config.json) ---------------------------
HIDDEN_SIZE = 5120
NUM_Q_HEADS = 96
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_DIM = NUM_Q_HEADS * HEAD_DIM                 # 12288
KV_DIM = NUM_KV_HEADS * HEAD_DIM               # 1024
FUSED_QKV_DIM = Q_DIM + 2 * KV_DIM             # 14336
ROTARY_DIM = 64                                # partial_rotary_factor 0.5
ROPE_THETA = 1e6
RMS_NORM_EPS = 1e-5
NUM_LAYERS = 92
FIRST_K_DENSE = 3                              # first_k_dense_replace
DENSE_INTERMEDIATE = 12288
MOE_INTERMEDIATE = 1536
NUM_ROUTED_EXPERTS = 160
N_SHARED_EXPERTS = 1
NUM_TOTAL_EXPERTS = NUM_ROUTED_EXPERTS + N_SHARED_EXPERTS  # 161
TOPK = 8
K_OUT = TOPK + N_SHARED_EXPERTS                # 9 weight slots per token
ROUTED_SCALING_FACTOR = 2.5
GATE_PADDED = 192                              # 160 -> 192 (24 tasks x 8 rows)
VOCAB_SIZE = 151552
EOS_TOKEN_ID = 151329


@register_model_builder("Glm4Moe", "zai-org/GLM-4.6", "glm4_moe", "glm-4.6")
class Glm4MoeBuilder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.rank = mpk.mpi_rank
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.tokenizer = None
        self.eos_token_id = EOS_TOKEN_ID
        self.hidden_size = HIDDEN_SIZE
        self.head_dim = HEAD_DIM
        self.num_layers = NUM_LAYERS
        self.first_k_dense = FIRST_K_DENSE
        self.vocab_size = VOCAB_SIZE
        self._bufs = {}

    # ------------------------------------------------------------- helpers
    def _pin(self, t: torch.Tensor) -> torch.Tensor:
        return t.contiguous().cuda() if not t.is_cuda else t.contiguous()

    def _attach(self, t: torch.Tensor, name: str):
        return self.mpk.attach_input(torch_tensor=t, name=name)

    def _buf(self, name: str, dims: tuple, dtype=bfloat16):
        """Get-or-create a shared intermediate cuda tensor."""
        if name not in self._bufs:
            self._bufs[name] = self.mpk.new_tensor(
                dims=dims, dtype=dtype, name=name, io_category="cuda_tensor")
        return self._bufs[name]

    @staticmethod
    def _get(state_dict: dict, name: str) -> torch.Tensor:
        assert name in state_dict, f"missing weight: {name}"
        return state_dict[name]

    # ------------------------------------------------------------ entry
    def build_from_config(self, model_config: MirageModelConfig):
        self.num_layers = model_config.num_layers
        self.build_from_dict(model_config.state_dict, model_config.with_lm_head)

    def build_from_model(self, model_name: str, model_path: str | None = None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        src = model_path if model_path is not None else model_name
        config = AutoConfig.from_pretrained(src)
        assert config.model_type == "glm4_moe", config.model_type
        self.num_layers = config.num_hidden_layers
        self.first_k_dense = config.first_k_dense_replace
        with torch.device("cuda"):
            model = AutoModelForCausalLM.from_pretrained(
                src, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.build_from_dict(dict(model.state_dict()), True)

    # ------------------------------------------------- intermediate tensors
    def new_intermediate_tensors(self):
        mbt = self.mpk.max_num_batched_tokens
        self.mbt = mbt
        H = self.hidden_size

        self.embed_out = self._buf("embed_out", (mbt, H))
        self.rmsnorm_out = self._buf("rmsnorm_out", (mbt, H))
        self.attn_in = self._buf("attn_in", (mbt, FUSED_QKV_DIM))
        self.attn_out = self._buf("attn_out", (mbt, Q_DIM))
        self.attn_proj_out = self._buf("attn_proj_out", (mbt, H))

        # dense MLP buffers (layers 0..first_k_dense-1)
        self.dense_mid = self._buf("dense_mid", (mbt, 2 * DENSE_INTERMEDIATE))
        self.dense_silu = self._buf("dense_silu", (mbt, DENSE_INTERMEDIATE))
        self.mlp_out = self._buf("mlp_out", (mbt, H))

        # MoE routing / pipeline buffers (shared across layers)
        self.moe_logits = self._buf("moe_gate_logits", (mbt, GATE_PADDED))
        self.moe_topk_weights = self._buf(
            "moe_topk_weights", (mbt, K_OUT), dtype=float32)
        self.moe_routing_indices = self._buf(
            "moe_routing_indices", (NUM_TOTAL_EXPERTS, mbt), dtype=int32)
        self.moe_active = self._buf(
            "moe_active_experts", (NUM_TOTAL_EXPERTS + 1,), dtype=int32)
        self.moe_mid = self._buf("moe_mid", (mbt, K_OUT, 2 * MOE_INTERMEDIATE))
        self.moe_silu = self._buf("moe_silu", (mbt, K_OUT, MOE_INTERMEDIATE))
        self.moe_down = self._buf("moe_down", (mbt, K_OUT, H))

        # partial-RoPE cos/sin tables [max_seq_len, ROTARY_DIM], theta 1e6
        max_seq = self.mpk.max_seq_length
        inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device="cuda")
            / ROTARY_DIM))
        angles = torch.outer(
            torch.arange(max_seq, dtype=torch.float32, device="cuda"),
            inv_freq)                                   # [max_seq, ROTARY/2]
        emb = torch.cat([angles, angles], dim=-1)       # [max_seq, ROTARY]
        self.cos_pos_embed = self._attach(
            self._pin(emb.cos().to(torch.bfloat16)), "cos_pos_embed")
        self.sin_pos_embed = self._attach(
            self._pin(emb.sin().to(torch.bfloat16)), "sin_pos_embed")

        # KV caches
        self.k_cache_t = torch.zeros(
            (self.num_layers, self.max_num_pages, self.page_size,
             NUM_KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        self.v_cache_t = torch.zeros_like(self.k_cache_t)

        # lm head / argmax
        self.padded_vocab_size = ((self.vocab_size + 255) // 256) * 256
        self.argmax_in = self._buf("argmax_in", (mbt, self.padded_vocab_size))
        self.argmax_part_value = self._buf(
            "argmax_part_value", (mbt, self.mpk.num_workers))
        self.argmax_part_index = self._buf(
            "argmax_part_index", (mbt, self.mpk.num_workers), dtype=int64)

    # ------------------------------------------------------------- layers
    def _build_attention(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        prefix = f"model.layers.{i}.self_attn."

        w_norm = self._attach(
            self._get(state_dict, f"model.layers.{i}.input_layernorm.weight"),
            f"layer_{i}_input_layernorm")
        mpk.rmsnorm_layer(
            input=self.x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        # fused qkv (kv-head-interleaved) linear WITH bias. The bias vector is
        # shuffled with the same row permutation as the weight and tiled to
        # [mbt, FUSED] so it can ride the residual input of the fused linear.
        wq = self._get(state_dict, f"{prefix}q_proj.weight")
        wk = self._get(state_dict, f"{prefix}k_proj.weight")
        wv = self._get(state_dict, f"{prefix}v_proj.weight")
        assert wq.shape == (Q_DIM, H) and wk.shape == (KV_DIM, H) \
            and wv.shape == (KV_DIM, H)
        w_qkv = self._attach(
            self._pin(shuffle_tensors([wq, wk, wv], NUM_KV_HEADS, 0)),
            f"layer_{i}_qkv_proj")
        for n in ("q_proj", "k_proj", "v_proj"):
            state_dict.pop(f"{prefix}{n}.weight", None)
        bq = self._get(state_dict, f"{prefix}q_proj.bias")
        bk = self._get(state_dict, f"{prefix}k_proj.bias")
        bv = self._get(state_dict, f"{prefix}v_proj.bias")
        bias_fused = shuffle_tensors([bq, bk, bv], NUM_KV_HEADS, 0)
        bias_tiled = bias_fused.to(torch.bfloat16).unsqueeze(0).expand(
            self.mbt, FUSED_QKV_DIM)
        b_qkv = self._attach(self._pin(bias_tiled), f"layer_{i}_qkv_bias")
        for n in ("q_proj", "k_proj", "v_proj"):
            state_dict.pop(f"{prefix}{n}.bias", None)
        mpk.linear_with_residual_layer(
            input=self.rmsnorm_out, weight=w_qkv, residual=b_qkv,
            output=self.attn_in,
            grid_dim=(FUSED_QKV_DIM // 64, 1, 1), block_dim=(128, 1, 1))

        # paged attention: per-head qk-norm (eps 1e-5) + partial RoPE (64/128)
        w_q_norm = self._attach(
            self._get(state_dict, f"{prefix}q_norm.weight"),
            f"layer_{i}_q_norm")
        w_k_norm = self._attach(
            self._get(state_dict, f"{prefix}k_norm.weight"),
            f"layer_{i}_k_norm")
        k_cache = self._attach(self.k_cache_t[i], f"layer_{i}_k_cache")
        v_cache = self._attach(self.v_cache_t[i], f"layer_{i}_v_cache")
        mpk.paged_attention_layer(
            input=self.attn_in, k_cache=k_cache, v_cache=v_cache,
            q_norm=w_q_norm, k_norm=w_k_norm,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            output=self.attn_out,
            grid_dim=(mpk.max_num_batched_requests, NUM_KV_HEADS, 1),
            block_dim=(128, 1, 1),
            rotary_dim=ROTARY_DIM, qk_norm_eps=RMS_NORM_EPS)

        wo = self._attach(
            self._get(state_dict, f"{prefix}o_proj.weight"),
            f"layer_{i}_o_proj")
        mpk.linear_with_residual_layer(
            input=self.attn_out, weight=wo, residual=self.x,
            output=self.attn_proj_out,
            grid_dim=(H // 64, 1, 1), block_dim=(128, 1, 1))
        self.x = self.attn_proj_out

    def _build_dense_mlp(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        prefix = f"model.layers.{i}.mlp."
        num_tasks = grid_for_rmsnorm_linear_layer(2 * DENSE_INTERMEDIATE)
        wg = self._get(state_dict, f"{prefix}gate_proj.weight")
        wu = self._get(state_dict, f"{prefix}up_proj.weight")
        w_gateup = self._attach(
            self._pin(shuffle_tensors([wg, wu], num_tasks // 2, 0)),
            f"layer_{i}_gateup_proj")
        state_dict.pop(f"{prefix}gate_proj.weight", None)
        state_dict.pop(f"{prefix}up_proj.weight", None)
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gateup, output=self.dense_mid,
            grid_dim=(num_tasks, 1, 1), block_dim=(128, 1, 1))
        mpk.silu_mul_layer(
            input=self.dense_mid, output=self.dense_silu,
            grid_dim=(num_tasks // 2, 1, 1), block_dim=(128, 1, 1))
        wd = self._attach(
            self._get(state_dict, f"{prefix}down_proj.weight"),
            f"layer_{i}_down_proj")
        mpk.linear_with_residual_layer(
            input=self.dense_silu, weight=wd, residual=self.x,
            output=self.mlp_out,
            grid_dim=(H // 64, 1, 1), block_dim=(128, 1, 1))

    def _build_moe_mlp(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        I = MOE_INTERMEDIATE
        prefix = f"model.layers.{i}.mlp."

        # gate linear: [160, H] padded to [192, H] so grid 24 keeps >=8 bf16
        # rows (16B) per block; router reads stride 192, first 160 columns.
        gate_t = self._get(state_dict, f"{prefix}gate.weight")
        assert gate_t.shape == (NUM_ROUTED_EXPERTS, H), f"gate {gate_t.shape}"
        gate_pad = torch.zeros(GATE_PADDED, H,
                               dtype=gate_t.dtype, device=gate_t.device)
        gate_pad[:NUM_ROUTED_EXPERTS] = gate_t
        w_gate = self._attach(self._pin(gate_pad), f"layer_{i}_moe_gate")
        state_dict.pop(f"{prefix}gate.weight", None)
        gate_grid = min(grid_for_rmsnorm_linear_layer(GATE_PADDED) or
                        GATE_PADDED // 8, GATE_PADDED // 8)
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gate, output=self.moe_logits,
            grid_dim=(gate_grid, 1, 1), block_dim=(128, 1, 1))

        # zero mlp_out before the pipeline (mirrors deepseek_v3 / inkling)
        mpk.tensor_init_layer(
            input=self.mlp_out, dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        bias_t = self._get(state_dict, f"{prefix}gate.e_score_correction_bias")
        assert bias_t.shape == (NUM_ROUTED_EXPERTS,)
        w_bias = self._attach(self._pin(bias_t.float().contiguous()),
                              f"layer_{i}_moe_gate_bias")
        mpk.glm_moe_router_layer(
            logits=self.moe_logits, bias=w_bias,
            output=(self.moe_topk_weights, self.moe_routing_indices,
                    self.moe_active),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            n_shared=N_SHARED_EXPERTS)

        # expert weights, shared expert folded in as expert 160.
        # w13[e] = [gate_proj; up_proj] (silu_mul expects gate block then up
        # block per row), w2[e] = down_proj.
        def expert_w13(p):
            g = self._get(state_dict, f"{p}gate_proj.weight")
            u = self._get(state_dict, f"{p}up_proj.weight")
            assert g.shape == (I, H) and u.shape == (I, H), (g.shape, u.shape)
            return torch.cat([g, u], dim=0)

        w13_list = [expert_w13(f"{prefix}experts.{e}.")
                    for e in range(NUM_ROUTED_EXPERTS)]
        w13_list.append(expert_w13(f"{prefix}shared_experts."))
        w13 = self._attach(self._pin(torch.stack(w13_list, dim=0)),
                           f"layer_{i}_experts_w13")
        w2_list = [self._get(state_dict, f"{prefix}experts.{e}.down_proj.weight")
                   for e in range(NUM_ROUTED_EXPERTS)]
        w2_list.append(
            self._get(state_dict, f"{prefix}shared_experts.down_proj.weight"))
        w2 = self._attach(self._pin(torch.stack(w2_list, dim=0)),
                          f"layer_{i}_experts_w2")
        for e in range(NUM_ROUTED_EXPERTS):
            for n in ("gate_proj", "up_proj", "down_proj"):
                state_dict.pop(f"{prefix}experts.{e}.{n}.weight", None)
        for n in ("gate_proj", "up_proj", "down_proj"):
            state_dict.pop(f"{prefix}shared_experts.{n}.weight", None)

        mpk.moe_w13_linear_layer(
            input=self.rmsnorm_out, weight=w13,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_active, output=self.moe_mid,
            grid_dim=(NUM_TOTAL_EXPERTS, 1, 1), block_dim=(128, 1, 1))
        mpk.moe_silu_mul_layer(
            input=self.moe_mid, output=self.moe_silu,
            grid_dim=(self.mbt, K_OUT, 1), block_dim=(128, 1, 1))
        mpk.moe_w2_linear_layer(
            input=self.moe_silu, weight=w2,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_active, output=self.moe_down,
            grid_dim=(NUM_TOTAL_EXPERTS, 1, 1), block_dim=(128, 1, 1))
        # weighted sum over the 9 slots (8 routed + shared at weight 1.0),
        # plus the outer residual: final = x + sum_k w_k * down_k.
        mpk.moe_mul_sum_add_layer(
            input=self.moe_down, weight=self.moe_topk_weights,
            residual=self.x, output=self.mlp_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

    def _build_mlp(self, i: int, state_dict: dict):
        mpk = self.mpk
        w_norm = self._attach(
            self._get(state_dict,
                      f"model.layers.{i}.post_attention_layernorm.weight"),
            f"layer_{i}_post_attn_layernorm")
        mpk.rmsnorm_layer(
            input=self.x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))
        if i < self.first_k_dense:
            self._build_dense_mlp(i, state_dict)
        else:
            self._build_moe_mlp(i, state_dict)
        # attn writes attn_proj_out, mlp writes mlp_out: alternating shared
        # buffers, same dependency-chaining pattern as the Qwen3 builder.
        self.x = self.mlp_out

    def build_layers(self, state_dict: dict):
        for i in range(self.num_layers):
            self._build_attention(i, state_dict)
            self._build_mlp(i, state_dict)

    # ------------------------------------------------------------- graph
    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        mpk = self.mpk

        # detect layer count from checkpoint if it disagrees with config
        # (also enables 1-2 layer smoke tests with synthetic state dicts)
        import re
        layer_ids = set()
        pat = re.compile(r"model\.layers\.(\d+)\.input_layernorm\.weight")
        for k in state_dict:
            m = pat.match(k)
            if m:
                layer_ids.add(int(m.group(1)))
        if layer_ids:
            detected = max(layer_ids) + 1
            if detected != self.num_layers:
                print(f"[glm4_moe] num_layers {self.num_layers} -> {detected} "
                      f"(from checkpoint)")
                self.num_layers = detected

        self.x = self._attach(self.input_tokens, "input_token")
        self.new_intermediate_tensors()

        # argmax_reduce expects (batch, 1); the test-mode meta tensor is 1-D
        out_tok = self.output_tokens
        if out_tok.dim() == 1:
            out_tok = out_tok.view(-1, 1)
        argmax_out = self._attach(out_tok, "output_token")

        w_embed = self._attach(
            self._get(state_dict, "model.embed_tokens.weight"), "embed_tokens")
        mpk.embed_layer(
            input=self.x, weight=w_embed, output=self.embed_out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
        self.x = self.embed_out

        self.build_layers(state_dict)

        w_norm = self._attach(
            self._get(state_dict, "model.norm.weight"), "model_norm")
        mpk.rmsnorm_layer(
            input=self.x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        if not with_lm_head:
            return

        lm_w = self._get(state_dict, "lm_head.weight")
        assert lm_w.shape[1] == self.hidden_size
        if lm_w.shape[0] != self.padded_vocab_size:
            padded = torch.zeros(self.padded_vocab_size, self.hidden_size,
                                 dtype=lm_w.dtype, device=lm_w.device)
            padded[:lm_w.shape[0]] = lm_w
            lm_w = padded
        w_lm = self._attach(self._pin(lm_w), "lm_head")
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_lm, output=self.argmax_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(self.padded_vocab_size), 1, 1),
            block_dim=(128, 1, 1))
        mpk.argmax_partial_layer(
            input=self.argmax_in,
            output=(self.argmax_part_value, self.argmax_part_index),
            grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
        mpk.argmax_reduce_layer(
            input=(self.argmax_part_value, self.argmax_part_index),
            output=argmax_out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    # ------------------------------------------------------------- tokenizer
    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
