# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The Mirage Team.
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
"""Whole-prompt GPU prefill followed by the existing MPK decode graph.

This is a PyTorch CUDA backend with eager or SDPA full attention, not a
persistent-kernel prefill graph.
It shares the decode graph's weights and writes its recurrent/KV buffers; no
HF model, HF cache, tokenizer, or checkpoint reload participates at runtime.
Keep prefill and decode backend names separate when reporting performance.

The chunk-delta operation order is adapted from Transformers 5.14.1's
``models/qwen3_5/modeling_qwen3_5.py::torch_chunk_gated_delta_rule``. Padding,
FP32 intermediates, BF16 boundaries and chunk size are numerical contracts.
"""
import torch
import torch.nn.functional as F

from .attention_reference import partial_rope
from .reference import gated_rms_norm, rms_norm


def chunk_gated_delta(query, key, value, g, beta, initial_state=None):
    """BF16 [1, time, heads, dim] -> BF16 outputs and a new FP32 state.

Use 64-token chunks even for short prompts. Do not substitute repeated
single-token updates: that changes FP32 reductions before BF16 rounding.
Inputs, including an optional initial recurrent state, are never modified.
"""
    chunk_size = 64
    dtype = query.dtype
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    batch, heads, length, key_dim = key.shape
    value_dim = value.shape[-1]
    padding = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, padding)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, padding)) for x in (beta, g)]
    query = query * (1 / (query.shape[-1] ** 0.5))
    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, value, key_beta, value_beta = [
        x.reshape(batch, heads, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, key_beta, value_beta)
    ]
    g = g.reshape(batch, heads, -1, chunk_size).cumsum(-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device))
    correction = -((key_beta @ key.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for row_idx in range(1, chunk_size):
        row = correction[..., row_idx, :row_idx].clone()
        previous = correction[..., :row_idx, :row_idx].clone()
        correction[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * previous).sum(-2)
    correction = correction + torch.eye(chunk_size, dtype=correction.dtype, device=correction.device)
    value = correction @ value_beta
    key_decay = correction @ (key_beta * g.exp().unsqueeze(-1))
    state = (torch.zeros(batch, heads, key_dim, value_dim, device=value.device, dtype=value.dtype)
             if initial_state is None else initial_state.to(value))
    output = torch.zeros_like(value)
    for chunk in range((length + padding) // chunk_size):
        q, k, v = query[:, :, chunk], key[:, :, chunk], value[:, :, chunk]
        attention = (q @ k.transpose(-1, -2)) * decay[:, :, chunk]
        residual_value = v - key_decay[:, :, chunk] @ state
        previous_output = (q * g[:, :, chunk, :, None].exp()) @ state
        output[:, :, chunk] = previous_output + attention @ residual_value
        state = (state * g[:, :, chunk, -1, None, None].exp()
                 + (k * (g[:, :, chunk, -1, None] - g[:, :, chunk]).exp()[..., None]).transpose(-1, -2)
                 @ residual_value)
    output = output.reshape(batch, heads, -1, value_dim)[:, :, :length]
    return output.transpose(1, 2).contiguous().to(dtype), state


def _gdn(hidden, tensors, config, past):
    batch, length = hidden.shape[:2]
    kh, vh, d = config.linear_num_key_heads, config.linear_num_value_heads, config.linear_key_head_dim
    qkv = F.linear(hidden, tensors["qkv_weight"]).transpose(1, 2)
    z = F.linear(hidden, tensors["z_weight"]).reshape(batch, length, vh, d)
    # Separate projections match HF's GEMM shapes. The decode graph stores the
    # two matrices in one padded allocation; these slices share that storage.
    a = F.linear(hidden, tensors["ab_weight"][:vh])
    b = F.linear(hidden, tensors["ab_weight"][vh:2*vh])
    if past:
        # HF stores four conv inputs; only the last three affect continuation.
        # Recreate the fourth (unused) slot to preserve the conv GEMM shape.
        context = F.pad((tensors["conv_state"].unsqueeze(0) if batch == 1 else tensors["conv_state"]), (1, 0))
        qkv = torch.cat((context, qkv), dim=-1)
    new_conv = F.pad(qkv, (max(0, 3 - qkv.shape[-1]), 0))[:, :, -3:].contiguous()
    mixed = F.silu(F.conv1d(qkv, tensors["conv_weight"].reshape(config.conv_dim, 1, 4),
                           padding=3, groups=config.conv_dim)[:, :, :qkv.shape[-1]])
    if past:
        mixed = mixed[:, :, -length:]
    q, k, v = mixed.transpose(1, 2).split((kh*d, kh*d, vh*d), dim=-1)
    q, k = [x.reshape(batch, length, kh, d).repeat_interleave(vh // kh, dim=2) for x in (q, k)]
    v = v.reshape(batch, length, vh, d)
    parameters = tensors["parameters"]
    g = -parameters[0].exp() * F.softplus(a.float() + parameters[1])
    raw, state = chunk_gated_delta(q, k, v, g, b.sigmoid(),
                                  (tensors["state"].unsqueeze(0) if batch == 1 else tensors["state"]) if past else None)
    gated = gated_rms_norm(raw.reshape(-1, 128), tensors["gated_norm_weight"], z.reshape(-1, 128),
                          config.rms_norm_eps).reshape(batch, length, vh * d)
    output = F.linear(gated, tensors["out_weight"])
    return output, {"state": state[0] if batch == 1 else state,
                    "conv_state": new_conv[0] if batch == 1 else new_conv}


def _attention(hidden, mixer, config, past):
    t = mixer.tensors
    batch, length = hidden.shape[:2]
    qh, kvh, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    qg = F.linear(hidden, t["qg_weight"]).reshape(batch, length, qh, 2*d)
    query, gate = qg.chunk(2, dim=-1)
    query = rms_norm(query, t["q_norm"], config.rms_norm_eps).transpose(1, 2)
    key = F.linear(hidden, t["k_weight"]).reshape(batch, length, kvh, d)
    key = rms_norm(key, t["k_norm"], config.rms_norm_eps).transpose(1, 2)
    value = F.linear(hidden, t["v_weight"]).reshape(batch, length, kvh, d).transpose(1, 2)
    rope = mixer.rope_table[past:past + length]
    query, key = partial_rope(query, key, rope[:, 0].unsqueeze(0), rope[:, 1].unsqueeze(0))
    if past:
        key = torch.cat(((t["key_cache"].unsqueeze(0) if batch == 1 else t["key_cache"])[:, :past].transpose(1, 2), key), dim=2)
        value = torch.cat(((t["value_cache"].unsqueeze(0) if batch == 1 else t["value_cache"])[:, :past].transpose(1, 2), value), dim=2)
    # Preserve HF repeat_kv layout for eager and for masked SDPA chunks.
    def repeat(x):
        return x[:, :, None].expand(batch, kvh, qh // kvh, past + length, d).reshape(batch, qh, past + length, d)
    if mixer.attn_implementation == "sdpa":
        # Match HF 5.14.1's SDPA dispatch for unpadded single requests. A fresh
        # prompt uses the causal flag; a cached multi-token chunk needs an
        # offset mask and repeated KV heads (GQA + mask selects another path).
        if past and length > 1:
            positions = torch.arange(past, past + length, device=hidden.device)
            mask = torch.arange(past + length, device=hidden.device)[None, :] <= positions[:, None]
            attended = F.scaled_dot_product_attention(query, repeat(key), repeat(value),
                attn_mask=mask[None, None], is_causal=False, dropout_p=0., scale=256 ** -0.5)
        else:
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=length > 1,
                dropout_p=0., scale=256 ** -0.5, enable_gqa=True)
    else:
        scores = (query @ repeat(key).transpose(-1, -2)) * (256 ** -0.5)
        positions = torch.arange(past, past + length, device=hidden.device)
        causal = torch.arange(past + length, device=hidden.device)[None, :] > positions[:, None]
        mask = torch.zeros(length, past + length, dtype=hidden.dtype, device=hidden.device)
        mask.masked_fill_(causal, torch.finfo(hidden.dtype).min)
        probabilities = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attended = probabilities @ repeat(value)
    attended = attended.transpose(1, 2).contiguous().reshape(batch, length, qh*d)
    output = F.linear(attended * gate.reshape(batch, length, qh*d).sigmoid(), t["out_weight"])
    return output, {"key_cache": key[0].transpose(0, 1) if batch == 1 else key.transpose(1, 2),
                    "value_cache": value[0].transpose(0, 1) if batch == 1 else value.transpose(1, 2)}


@torch.inference_mode()
def block_prefill(model, tokens):
    """Compute one contiguous prompt chunk, then commit its state atomically.

The model validates lifecycle, token range, length and mode before calling.
One-token calls use the already validated MPK step instead of this function.
All allocations and computations precede writes to caller-owned cache state.
"""
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Exact prefill requires torch.backends.cuda.matmul.allow_tf32=False")
    c, past = model.config, model.position
    batch = model.batch_size
    end = past + (len(tokens) if batch == 1 else len(tokens[0]))
    # Do not let an enclosing autocast context narrow the GDN FP32 matmuls.
    with torch.autocast(device_type="cuda", enabled=False):
        ids = torch.tensor([tokens] if batch == 1 else tokens, dtype=torch.int64, device=model.tensors["embedding"].device)
        hidden = F.embedding(ids, model.tensors["embedding"])
        embedding = hidden[:, -1].clone()
        pending, last_hidden = [], []
        for i, layer in enumerate(model.layers):
            t = layer.tensors
            normalized = rms_norm(hidden, t["input_norm_weight"], c.rms_norm_eps)
            if i in c.gdn_layers:
                mixed, state = _gdn(normalized, t, c, past)
            else:
                mixed, state = _attention(normalized, layer.mixer, c, past)
            residual = hidden + mixed
            normalized = rms_norm(residual, t["post_norm_weight"], c.rms_norm_eps)
            gate = F.linear(normalized, t["gate_up_weight"][:c.intermediate_size])
            up = F.linear(normalized, t["gate_up_weight"][c.intermediate_size:])
            hidden = residual + F.linear(F.silu(gate) * up, t["down_weight"])
            pending.append(state)
            last_hidden.append(hidden[:, -1].clone())
        normalized = rms_norm(hidden, model.tensors["norm_weight"], c.rms_norm_eps)
        logits = F.linear(normalized[:, -1:], model.tensors["lm_head"])[:, 0]
        next_token = logits.argmax(-1)
    # A failed computation must not leave a mixture of old/new layer states.
    torch.cuda.synchronize()
    for layer, state, output in zip(model.layers, pending, last_hidden):
        for name, value in state.items():
            destination = layer.tensors[name]
            if name in ("key_cache", "value_cache"):
                destination = destination[:end] if batch == 1 else destination[:, :end]
            destination.copy_(value)
        layer.tensors["decoder_output"].copy_(output)
        layer.mixer.tensors["position"].fill_(end - 1)
    model.tensors["hidden"].copy_(embedding)
    model.tensors["normalized"].copy_(normalized[:, -1])
    model.tensors["logits"].copy_(logits)
    model.tensors["next_token"].copy_(next_token)
    model.tensors["token"].copy_(ids[:, -1])
    torch.cuda.synchronize()
    model.position = end
    for layer in model.layers:
        layer.mixer.position = end
    return model.tensors["logits"]
