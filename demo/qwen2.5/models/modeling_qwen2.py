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
"""PyTorch Qwen2 model."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from .configuration_qwen2 import Qwen2Config

import flashinfer
import mirage as mi

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        #hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance)
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        # BC: "rope_type" was originally "type"
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, position_ids):

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type="cuda", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=torch.bfloat16), sin.to(dtype=torch.bfloat16)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.enable_mirage = False

    def fuse_weights(self):
        self.fused_weight = torch.transpose(torch.cat((self.gate_proj.weight, self.up_proj.weight), 0), 0, 1)

    def superoptimize_kernels(self):
        self.enable_mirage = True
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(1, self.hidden_size), dtype=mi.bfloat16)
        G = graph.new_input(dims=(1, self.hidden_size), dtype=mi.bfloat16)
        W = graph.new_input(dims=(self.hidden_size, 2*self.intermediate_size), strides=(1, self.hidden_size), dtype=mi.bfloat16)
        D = graph.rms_norm(X, normalized_shape=(self.hidden_size,))
        D = graph.mul(D, G)
        O = graph.matmul(D, W)
        graph.mark_output(O)
        self.kernel1 = graph.superoptimize(config="mlp")

        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(1, self.intermediate_size), dtype=mi.bfloat16)
        Y = graph.new_input(dims=(1, self.intermediate_size), dtype=mi.bfloat16)
        W = graph.new_input(dims=(self.intermediate_size, self.hidden_size), strides=(1, self.intermediate_size), dtype=mi.bfloat16)
        D = graph.mul(graph.silu(X), Y)
        O = graph.matmul(D, W)
        graph.mark_output(O)
        self.kernel2 = graph.superoptimize(config="mlp")

    def forward(self, input_layernorm, hidden_state, stream: torch.cuda.Stream = None):
        if hidden_state.shape[-2] == 1 and self.enable_mirage:
            # use mirage kernels for decoding
            output = self.kernel1(inputs=(hidden_state, input_layernorm.weight, self.fused_weight), stream=stream)[0]
            gate_output, up_output = torch.chunk(output, 2, -1)
            output = self.kernel2(inputs=(gate_output, up_output, self.down_proj.weight), stream=stream)[0]
        else:
            # use the original for prefilling
            hidden_state = input_layernorm(hidden_state)
            output = torch.matmul(hidden_state, self.fused_weight)
            gate_output, up_output = torch.chunk(output, 2, -1)
            output = self.down_proj(self.act_fn(gate_output) * up_output)

        return output

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, kv_cache: Tuple[torch.Tensor, torch.Tensor], layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        assert self.layer_idx is not None
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_cache, self.value_cache = kv_cache
        assert kv_cache[0].shape == (config.num_hidden_layers, 1, config.max_position_embeddings, self.num_key_value_heads, self.head_dim)
        assert kv_cache[1].shape == (config.num_hidden_layers, 1, config.max_position_embeddings, self.num_key_value_heads, self.head_dim)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.enable_mirage = False

    def fuse_weights(self):
        self.fused_weight = torch.transpose(torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), 0), 0, 1)
        self.fused_bias = torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), 0)

    def superoptimize_kernels(self):
        self.enable_mirage = True
        graph = mi.new_kernel_graph()
        self.fused_outdim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        X = graph.new_input(dims=(1, self.hidden_size), dtype=mi.bfloat16)
        G = graph.new_input(dims=(1, self.hidden_size), dtype=mi.bfloat16)
        W = graph.new_input(dims=(self.hidden_size, self.fused_outdim), strides=(1, self.hidden_size), dtype=mi.bfloat16)
        D = graph.rms_norm(X, normalized_shape=(self.hidden_size,))
        D = graph.mul(D, G)
        O = graph.matmul(D, W)
        graph.mark_output(O)
        self.kernel = graph.superoptimize(config="mlp")

    def forward(
        self,
        input_layernorm,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        decode_wrapper = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if q_len == 1 and self.enable_mirage:
            # use mirage kernels for decoding
            xqkv = self.kernel(inputs=(hidden_states, input_layernorm.weight, self.fused_weight), stream=stream)[0]
            xqkv = xqkv.view(bsz, q_len, self.fused_outdim)
        else:
            # use the original for prefilling
            hidden_states = input_layernorm(hidden_states)
            xqkv = torch.matmul(hidden_states, self.fused_weight)
        xqkv = xqkv + self.fused_bias
        query_states = xqkv[:, :, : (self.num_heads * self.head_dim)]
        xkv = xqkv[:, :, (self.num_heads * self.head_dim) :]
        key_states, value_states = xkv.chunk(2, -1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if q_len > 1:
            self.key_cache[self.layer_idx,0,:q_len]=key_states[0]
            self.value_cache[self.layer_idx,0,:q_len]=value_states[0]
        else:
            self.key_cache[self.layer_idx, 0, step]=key_states[0]
            self.value_cache[self.layer_idx, 0, step]=value_states[0]

        if q_len > 1:
            fl_attn_output = flashinfer.single_prefill_with_kv_cache(query_states[0], self.key_cache[self.layer_idx,0,:q_len,:,:], self.value_cache[self.layer_idx,0,:q_len,:,:], causal=True, kv_layout="NHD")
        else:
            fl_attn_output = decode_wrapper.run(query_states[0], (self.key_cache[self.layer_idx], self.value_cache[self.layer_idx]))

        attn_output = fl_attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, kv_cache: Tuple[torch.Tensor, torch.Tensor], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config, kv_cache, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def fuse_weights(self):
        self.mlp.fuse_weights()
        self.self_attn.fuse_weights()

    def superoptimize_kernels(self):
        self.mlp.superoptimize_kernels()
        self.self_attn.superoptimize_kernels()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        decode_wrapper = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        #hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            input_layernorm = self.input_layernorm,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            decode_wrapper=decode_wrapper,
            step=step,
            stream=stream,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        #hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(self.post_attention_layernorm, hidden_states, stream=stream)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config

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

class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # KV cache layout is (L, N, P, H, D) where L is the number of layers, N is the max number of pages (i.e., 1), P is the page size (i.e., config.max_embedding_positions), H is the number of key-value heads, and D is the hidden dim size
        key_cache = torch.empty((config.num_hidden_layers, 1, config.max_position_embeddings, config.num_key_value_heads, config.hidden_size // config.num_attention_heads)) 
        value_cache = torch.empty((config.num_hidden_layers, 1, config.max_position_embeddings, config.num_key_value_heads, config.hidden_size // config.num_attention_heads))
        self.kv_cache = (key_cache, value_cache)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, self.kv_cache, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

        # Create flashinfer decode wrapper
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        kv_page_indices = torch.arange(1).int().to("cuda")
        kv_page_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        self.kv_last_page_len = torch.tensor([0], dtype=torch.int32, device="cuda")
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                kv_layout="NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=kv_page_indptr,
                paged_kv_indices_buffer=kv_page_indices,
                paged_kv_last_page_len_buffer=self.kv_last_page_len
                )
        self.decode_wrapper.plan(
            kv_page_indptr,
            kv_page_indices,
            self.kv_last_page_len,
            28, # num_qo_heads,
            4, # num_kv_heads,
            128, # head_dimension
            32768, # page_size
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16)

    def fuse_weights(self):
        for decoder_layer in self.layers:
            decoder_layer.fuse_weights()

    def superoptimize_kernels(self):
        for decoder_layer in self.layers:
            decoder_layer.superoptimize_kernels()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,):
        inputs_embeds = self.embed_tokens(input_ids)
        
        causal_mask = None

        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = None
        self.kv_last_page_len.copy_(step+1)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                decode_wrapper=self.decode_wrapper,
                step=step,
                stream=stream,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return (hidden_states,)

class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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

    def fuse_weights(self):
        self.model.fuse_weights()

    def superoptimize_kernels(self):
        self.model.superoptimize_kernels()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: torch.cuda.Stream = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,):

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
