# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3 model."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from .configuration_qwen3 import Qwen3Config

import mirage as mi

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen3Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        # BC: "rope_type" was originally "type"
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(
        self, position_ids
    ):  # positions = torch.arange(32768).unsqueeze(0).to(model.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type="cuda", enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=torch.bfloat16), sin.to(dtype=torch.bfloat16)
class Qwen3MLP(nn.Module):
    def __init__(self, config, world_size):
        super().__init__()
        self.world_size = world_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.part_inter_size = self.intermediate_size // world_size
        self.gate_proj = nn.Linear(self.hidden_size, self.part_inter_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.part_inter_size, bias=False)
        self.down_proj = nn.Linear(self.part_inter_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        world_size: int,
    ):
        super().__init__()
        self.world_size = world_size
        self.config = config
        self.layer_idx = layer_idx
        assert self.layer_idx is not None
        self.hidden_size = config.hidden_size
        self.qkv_size = config.head_dim * config.num_attention_heads
        self.local_hidden_size = self.hidden_size // self.world_size
        self.local_qkv_size = self.qkv_size // self.world_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_cache, self.value_cache = kv_cache
        assert kv_cache[0].shape == (
            config.num_hidden_layers,
            16,
            4096,
            self.num_key_value_heads // world_size,
            self.head_dim,
        )
        assert kv_cache[1].shape == (
            config.num_hidden_layers,
            16,
            4096,
            self.num_key_value_heads // world_size,
            self.head_dim,
        )
        self.max_position_embeddings = 4096
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads // world_size) * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            (self.num_key_value_heads // world_size) * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            (self.num_key_value_heads // world_size) * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            (self.num_heads // world_size) * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        world_size: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config, kv_cache, layer_idx, world_size)

        self.mlp = Qwen3MLP(config, world_size)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config, world_size: int, max_num_pages: int, page_size: int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # KV cache layout is (L, N, P, H, D) where L is the number of layers, 
        # N is the max number of pages (i.e., 1), 
        # P is the page size (i.e., config.max_embedding_positions), 
        # H is the number of key-value heads, and D is the hidden dim size
        key_cache = torch.empty(
            (
                config.num_hidden_layers,
                max_num_pages,
                page_size,
                config.num_key_value_heads // world_size,
                config.head_dim,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        value_cache = torch.empty(
            (
                config.num_hidden_layers,
                max_num_pages,
                page_size,
                config.num_key_value_heads // world_size,
                config.head_dim,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )

        self.kv_cache = (key_cache, value_cache)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, self.kv_cache, layer_idx, world_size)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.kv_last_page_len = torch.tensor([0], dtype=torch.int32, device="cuda")


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    def __init__(self, config, world_size, max_num_pages, page_size):
        super().__init__(config)
        self.model = Qwen3Model(config, world_size, max_num_pages, page_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)