# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Llama3 model."""

import math
import torch.distributed as dist
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN 
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from .configuration_llama3 import Llama3Config
import time

from .rope import apply_rotary_pos_emb_triton

class Llama3RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Llama3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

class Llama3RotaryEmbedding(nn.Module):
    def __init__(self, config: Llama3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
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

class Llama3MLP(nn.Module):
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

    def forward(self, input_layernorm, hidden_state, stream: torch.cuda.Stream = None):
        hidden_state = input_layernorm(hidden_state)
        output = self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )
        if self.world_size > 1:
            dist.all_reduce(output)

        return output

# same
def naive_attention(
    q,
    key_cache,
    value_cache,
    kv_len,
    layer_idx,
    is_causal=True, 
    enable_gqa=True):
            
    k = key_cache[layer_idx, 0, :kv_len, :, :] # [kv_seq_len, num_kv_heads, head_dim]
    v = value_cache[layer_idx, 0, :kv_len, :, :] # [kv_seq_len, num_kv_heads, head_dim]

    q_for_sdpa = q.permute(1, 0, 2)    # [num_q_heads, 1, head_dim]
    k_for_sdpa = k.permute(1, 0, 2)    # [num_kv_heads, kv_seq_len, head_dim]
    v_for_sdpa = v.permute(1, 0, 2)    # [num_kv_heads, kv_seq_len, head_dim]

    attn_output_sdpa = nn.functional.scaled_dot_product_attention(
        q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=is_causal, enable_gqa=enable_gqa
    )
    attn_output = attn_output_sdpa.permute(1, 0, 2)

    return attn_output


class Llama3Attention(nn.Module):
    def __init__(self, config: Llama3Config, kv_cache: Tuple[torch.Tensor, torch.Tensor], layer_idx: int, world_size: int = 1):
        super().__init__()
        self.world_size = world_size
        self.config = config
        self.layer_idx = layer_idx
        assert self.layer_idx is not None
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.qkv_size = self.head_dim * config.num_attention_heads
        self.local_hidden_size = self.hidden_size // self.world_size
        self.local_qkv_size = self.qkv_size // self.world_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.max_position_embeddings = 4096
        self.key_cache, self.value_cache = kv_cache
        
        num_layers, max_num_pages, page_size, num_kv_heads_per_device, head_dim = kv_cache[0].shape
        assert kv_cache[0].shape == (
            config.num_hidden_layers,
            max_num_pages,
            page_size,
            self.num_key_value_heads // world_size,
            self.head_dim,
        )
        assert kv_cache[1].shape == (
            config.num_hidden_layers,
            max_num_pages,
            page_size,
            self.num_key_value_heads // world_size,
            self.head_dim,
        )

        self.q_proj = nn.Linear(
            config.hidden_size, (config.num_attention_heads // world_size) * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, (config.num_key_value_heads // world_size) * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, (config.num_key_value_heads // world_size) * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            (config.num_attention_heads // world_size) * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = Llama3RotaryEmbedding(config=self.config)

    def forward(
        self,
        input_layernorm,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        hidden_states = input_layernorm(hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_attention_heads // self.world_size, self.head_dim
        )
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim
        )

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        query_states, key_states = apply_rotary_pos_emb_triton(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if q_len > 1:
            self.key_cache[self.layer_idx, 0, :q_len] = key_states[0]
            self.value_cache[self.layer_idx, 0, :q_len] = value_states[0]
        else:
            self.key_cache[self.layer_idx, 0, step] = key_states[0]
            self.value_cache[self.layer_idx, 0, step] = value_states[0]

        q = query_states[0] # Shape: [q_len, num_q_heads, head_dim]

        if q_len > 1:
            attn_output = naive_attention(
                q,
                self.key_cache,
                self.value_cache,
                q_len,
                self.layer_idx,
                True,
                True
            )
        else:
            kv_seq_len = step.item() + 1
            attn_output = naive_attention(
                q,
                self.key_cache,
                self.value_cache,
                kv_seq_len,
                self.layer_idx,
                False,
                True
            )

        attn_output = attn_output.reshape(bsz, q_len, self.local_qkv_size)

        attn_output = self.o_proj(attn_output)
        if self.world_size > 1:
            dist.all_reduce(attn_output)

        return attn_output, None 


class Llama3DecoderLayer(nn.Module):
    def __init__(self, config: Llama3Config, kv_cache: Tuple[torch.Tensor, torch.Tensor], layer_idx: int, world_size: int = 1):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Llama3Attention(config=config, kv_cache=kv_cache, layer_idx=layer_idx, world_size=world_size)

        self.mlp = Llama3MLP(config, world_size)
        self.input_layernorm = Llama3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            input_layernorm=self.input_layernorm,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            step=step,
            stream=stream,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm, hidden_states, stream=stream)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        return outputs


class Llama3PreTrainedModel(PreTrainedModel):
    config_class = Llama3Config

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

class Llama3Model(Llama3PreTrainedModel):
    def __init__(self, config: Llama3Config, world_size: int = 1,
        max_num_pages: int = 1, page_size: int = 4096):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
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
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Llama3DecoderLayer(config, self.kv_cache, layer_idx, world_size) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Llama3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama3RotaryEmbedding(config=config)

        self.post_init()
        
        self.kv_last_page_len = torch.tensor([0], dtype=torch.int32, device="cuda")
    
    def get_input_embeddings(self):
        return self.embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        causal_mask = None

        hidden_states = inputs_embeds

        # decoder layers
        self.kv_last_page_len.copy_(step + 1)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                step=step,
                stream=stream,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return (hidden_states,)

class Llama3ForCausalLM(Llama3PreTrainedModel, GenerationMixin):
    def __init__(self, config, world_size=1, max_num_pages=1, page_size=4096):
        super().__init__(config)
        self.model = Llama3Model(config, world_size, max_num_pages, page_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
        num_logits_to_keep: int = 0,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            step=step,
            stream=stream,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        
        return logits
