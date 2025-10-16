from .modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os

import mirage as mi
from ..utils import grid_for_rmsnorm_linear_layer, max_factor_leq_n
from ..graph_builder import GraphBuilder
from ...model_registry import register_model_builder


@register_model_builder("Qwen3", "Qwen/Qwen3-8B")
class Qwen3Builder(GraphBuilder):
    model_name: str = "Qwen/Qwen3-8B"
    def __init__(self, mpk: mi.PersistentKernel, weights: dict | None = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        

    def build_from_vllm_graph(self, graph_path):
        raise NotImplementedError("build_from_vllm_graph is not implemented")

    def build_from_model(self, model_path: str | None = None):
        with torch.device("cuda"):
            if model_path is not None:
                print(f"Loading model from model path: {model_path}")
                self.config = AutoConfig.from_pretrained(model_path)
                self.model = Qwen3ForCausalLM(self.config, self.world_size, self.max_num_pages, self.page_size)
                load_model(self.model, f"{model_path}/model{self.rank}-mp{self.world_size}.safetensors")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                self.model = Qwen3ForCausalLM.from_pretrained(self.model_name, world_size=1, max_num_pages=self.max_num_pages, page_size=self.page_size).to("cuda")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.positions = torch.arange(32768).unsqueeze(0).to(self.model.device)
        self.position_embeddings = self.model.model.rotary_emb(self.positions)
        
        self.k_cache = self.model.model.kv_cache[0]
        self.v_cache = self.model.model.kv_cache[1]
        
        self.hidden_size = self.model.config.hidden_size
        self.intermediate_size = self.model.config.intermediate_size
        
        self.vocab_size = 153600
        self.num_q_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.num_local_q_heads = self.num_q_heads // self.world_size
        self.num_local_kv_heads = self.num_kv_heads // self.world_size
        self.head_dim = self.model.config.head_dim
        self.fused_outdim_1 = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.fused_outdim_2 = 2 * self.intermediate_size
        
        self.num_layers = len(self.model.model.layers)
        
        self.build_from_dict(self.model.state_dict())
        
    def new_intermediate_tensors(self):
        self.max_num_batched_tokens = self.mpk.max_num_batched_tokens
        self.y = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        self.rmsnorm_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )
        self.attn_in = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.fused_outdim_1 // self.world_size), # [6, 6144]
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        self.attn_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.num_local_q_heads * self.head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        self.attn_proj_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        )
        self.allreduce_buf = self.mpk.new_tensor(
            dims=(self.world_size, self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="all_reduce_buf",
            io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        )
        self.attn_allreduce_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="attn_allreduce_out",
            io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        )
        self.mlp_mid = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.fused_outdim_2 // self.world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        self.silu_mul_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.intermediate_size // self.world_size),
            dtype=mi.bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        self.mlp_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        )
        self.mlp_final = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="mlp_final",
            io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        )
        self.argmax_in = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.vocab_size),
            dtype=mi.bfloat16,
            name="argmax_in",
            io_category="cuda_tensor",
        )
        self.argmax_part_value = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.mpk.num_workers),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        self.argmax_part_index = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.mpk.num_workers),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        
    def build_layers(self, state_dict):
        # add rmsnorm + linear
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}."
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
                name=f"layer_{i}_input_layernorm",
            )
            if f"{prefix}self_attn.q_proj.weight" in state_dict:
                w_q = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}self_attn.q_proj.weight"], name=f"layer_{i}_q_proj"
                )
                w_k = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}self_attn.k_proj.weight"], name=f"layer_{i}_k_proj"
                )
                w_v = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}self_attn.v_proj.weight"], name=f"layer_{i}_v_proj"
                )
                w_qkv = self.mpk.shuffle_tensors(
                    inputs=[w_q, w_k, w_v],
                    shuffled_dim=0,
                    num_groups=self.num_kv_heads // self.world_size,
                    name=f"layer_{i}_qkv_proj",
                )
            elif f"{prefix}self_attn.qkv_proj.weight" in state_dict:
                w_qkv = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}self_attn.qkv_proj.weight"], name=f"{prefix}qkv_proj"
                )
            else:
                raise ValueError(f"No qkv projection weight found for layer {i}")
            self.mpk.rmsnorm_layer(
                input=self.x,
                weight=w_norm,
                output=self.rmsnorm_out,
                grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out,
                weight=w_qkv,
                output=self.attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )

            # add attention
            w_q_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}self_attn.q_norm.weight"], name=f"layer_{i}_q_norm"
            )
            w_k_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}self_attn.k_norm.weight"], name=f"layer_{i}_k_norm"
            )
            # TODO: KV cache handling
            k_cache = self.mpk.attach_input(
                torch_tensor=self.k_cache[i], name=f"layer_{i}_k_cache"
            )
            v_cache = self.mpk.attach_input(
                torch_tensor=self.v_cache[i], name=f"layer_{i}_v_cache"
            )
            
            # if spec_decode_config:
            #     self.mpk.single_batch_extend_attention_layer(
            #         input=attn_in,
            #         k_cache=k_cache,
            #         v_cache=v_cache,
            #         q_norm=w_q_norm,
            #         k_norm=w_k_norm,
            #         cos_pos_embed=cos_pos_embed,
            #         sin_pos_embed=sin_pos_embed,
            #         output=attn_out,
            #         grid_dim=(1, num_local_kv_heads, 1), #TODO: further divide across batch dim
            #         block_dim=(128, 1, 1),
            #     )
            # else:
            self.mpk.paged_attention_layer(
                input=self.attn_in,
                k_cache=k_cache,
                v_cache=v_cache,
                q_norm=w_q_norm,
                k_norm=w_k_norm,
                cos_pos_embed=self.cos_pos_embed,
                sin_pos_embed=self.sin_pos_embed,
                output=self.attn_out,
                grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, 1),
                block_dim=(128, 1, 1),
            )
            # add linear w/ residual
            self.w = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}self_attn.o_proj.weight"], name=f"layer_{i}_o_proj"
            )
            self.mpk.linear_with_residual_layer(
                input=self.attn_out,
                weight=self.w,
                residual=self.x,
                output=self.attn_proj_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            # reset residual input as x
            self.x = self.attn_proj_out
            # add allreduce if needed
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.attn_proj_out,
                    buffer=self.allreduce_buf,
                    output=self.attn_allreduce_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.attn_allreduce_out
            # add rmsnorm_linear layer
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
                name=f"layer_{i}_post_attn_layernorm",
            )
            if f"{prefix}mlp.gate_proj.weight" in state_dict:
                w_gate_proj = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}mlp.gate_proj.weight"], name=f"layer_{i}_gate_proj"
                )
                w_up_proj = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}mlp.up_proj.weight"], name=f"layer_{i}_up_proj"
                )
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0))
                w_gatedup = self.mpk.shuffle_tensors(
                    inputs=[w_gate_proj, w_up_proj],
                    shuffled_dim=0,
                    num_groups=rmsnorm_num_tasks//2,
                    name=f"layer_{i}_gatedup_proj",
                )
            elif f"{prefix}mlp.gate_up_proj.weight" in state_dict:
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(state_dict[f"{prefix}mlp.gate_up_proj.weight"].dim(0))
                w_gatedup = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}mlp.gate_up_proj.weight"], name=f"layer_{i}_gatedup_proj"
                )
            else:
                raise ValueError(f"No gate or up projection weight found for layer {i}")
            self.mpk.rmsnorm_layer(
                input=self.x,
                weight=w_norm,
                output=self.rmsnorm_out,
                grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out,
                weight=w_gatedup,
                output=self.mlp_mid,
                grid_dim=(rmsnorm_num_tasks, 1, 1),
                block_dim=(128, 1, 1),
            )

            self.mpk.silu_mul_layer(
                input=self.mlp_mid,
                output=self.silu_mul_out,
                grid_dim=(rmsnorm_num_tasks//2, 1, 1),
                block_dim=(128, 1, 1),
            )
            # add silu_mul_linear layer
            self.w = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}mlp.down_proj.weight"], name=f"layer_{i}_down_proj"
            )
            self.mpk.linear_with_residual_layer(
                input=self.silu_mul_out,
                weight=self.w,
                residual=self.x,
                output=self.mlp_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            # reset residual input as x
            self.x = self.mlp_out
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.mlp_out,
                    buffer=self.allreduce_buf,
                    output=self.mlp_final,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.mlp_final
        
    def build_from_dict(self, state_dict):
        # pad vocab_size to facilitate task graph creation
        self.lm_head_weight = torch.cat(
            (
                state_dict["lm_head.weight"],
                torch.full(
                    (153600 - self.model.config.vocab_size, self.hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )
        assert self.lm_head_weight.stride()[0] == self.hidden_size
        # if spec_decode_config and spec_decode_config.method == "promptlookup":
        #     all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
        #     num_tokens_extend = spec_decode_config.spec_length + 1
        # else:
        #     num_tokens_extend = 1
            
        self.x = self.mpk.attach_input(torch_tensor=self.input_tokens, name="input_token")
        self.cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        self.sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )
        
        self.new_intermediate_tensors()
        
        argmax_out = self.mpk.attach_input(torch_tensor=self.output_tokens, name="output_token")

        # add spec tokens layer
        # if spec_decode_config:
        #     spec_tokens = self.mpk.draft_forward_layer_dispatcher(
        #         spec_decode_config = spec_decode_config, 
        #         tokens = all_tokens,
        #         grid_dim=(96, 1, 1),
        #         block_dim=(128, 1, 1),
        #     )
        #     x = spec_tokens
        # Add Embed
        
        self.w = self.mpk.attach_input(
            torch_tensor=state_dict["model.embed_tokens.weight"], name="embed_tokens"
        )
        
        self.mpk.embed_layer(
            input=self.x, 
            weight=self.w, 
            output=self.y, 
            # grid_dim=(max_factor_leq_n(hidden_size, 96 // args.max_num_batched_tokens), total_tokens_per_iter, 1), 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=1,
        )
        self.x = self.y
        
        self.build_layers(state_dict)

        # add rmsnorm_linear layer
        self.w_norm = self.mpk.attach_input(
            torch_tensor=state_dict["model.norm.weight"], name="model_norm_weight"
        )
        self.w_proj = self.mpk.attach_input(torch_tensor=self.lm_head_weight, name="lm_head")
        self.mpk.rmsnorm_layer(
            input=self.x,
            weight=self.w_norm,
            output=self.rmsnorm_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=self.w_proj,
            output=self.argmax_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(self.w_proj.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        # add argmax layer
        # if spec_decode_config and spec_decode_config.method == "promptlookup":
        #     argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
        #                                spec_decode_config.spec_length + 1, 
        #                                1)
        #     argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
        # else:
        argmax_partial_grid_dim = (self.mpk.num_workers, 1, 1)
        argmax_reduce_grid_dim = (1, 1, 1)
        self.mpk.argmax_partial_layer(
            input=self.argmax_in,
            output=(self.argmax_part_value, self.argmax_part_index),
            grid_dim=argmax_partial_grid_dim,
            block_dim=(128, 1, 1),
        )
        self.mpk.argmax_reduce_layer(
            input=(self.argmax_part_value, self.argmax_part_index),
            output=argmax_out,
            grid_dim=argmax_reduce_grid_dim,
            block_dim=(128, 1, 1),
        )
        # if spec_decode_config:
        #     verify_out = self.mpk.verify_layer_dispatcher(
        #         spec_decode_config = spec_decode_config,
        #         spec_tokens = spec_tokens,
        #         target_output = argmax_out,
        #         grid_dim = (1, 1, 1),
        #         block_dim = (128, 1, 1),
        #     )
        
    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)