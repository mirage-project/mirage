# vLLM decode compute graph — `Qwen/Qwen3.5-35B-A3B-FP8`

Primary reference for the MPK port. Everything here is read off source at a pinned commit; every
claim carries a `file:line`. Where this document contradicts the earlier scouting note
(`design/scouting/vllm-qwen35-graph.md`) it says so explicitly in §7.

**Sources pinned**

| what | pin |
|---|---|
| vLLM | commit `0ba2aa35a81dcc3246b26291368b53fa2389c7d7` (2026-07-25, *"Stabilize GPU memory teardown between ROCm CI tests (#49242)"*). All `path:line` cites are relative to the vLLM repo root at this commit. |
| checkpoint | `Qwen/Qwen3.5-35B-A3B-FP8`, `config.json` + `model.safetensors.index.json` + safetensors headers read from the HF `main` revision on 2026-07-25. Cites of the form `config.json:<field>` / `index` refer to these. |
| target | single B200 (sm_100, device capability family 100), TP=1, EP=1, DP=1, text-only, greedy decode, no MTP/spec-decode, no prefix caching beyond what vLLM forces. |

**Scope**: the decode step (one token per running sequence). Prefill is described only where the
decode path shares state with it. Vision tower and MTP are out of scope and are flagged where they
would otherwise leak into the graph.

---

## 0. TL;DR — one decode step

```
h = embed_tokens[tok]                                     # [B,2048] bf16, no scale
×40 layers, layer_types = [LIN,LIN,LIN,FULL] repeating:

  (x, r) = GemmaRMSNorm(h, r)                             # y = x·rsqrt(mean x²+1e-6)·(1+w)

  LIN (30 layers, i mod 4 != 3):
    qkvz = x·Wqkvz[12288,2048]      # FP8 blockwise
    ba   = x·Wba[64,2048]           # BF16  (checkpoint refuses to quantize a/b)
    (qkv[8192], z[32,128]) = split(qkvz) ; (b[32], a[32]) = split(ba)
    qkv = silu(conv1d_update(qkv, W[8192,4], state[3,8192]))          # depthwise, w=4, BF16
    q,k = L2norm(qkv[0:2048]), L2norm(qkv[2048:4096])   (16 heads×128)
    v   = qkv[4096:8192]                                (32 heads×128)
    g = -exp(A_log[hv])·softplus(a[hv]+dt_bias[hv]) ; β = σ(b[hv])
    S[hv] = S[hv]·e^g ;  S[hv] += β·(v − S[hv]k)⊗k ;  o[hv] = S[hv]·(q/√128)   # S is FP32
    y = (RMSNorm_128(o)·w) ⊙ silu(z)  → flatten → ·Wout[2048,4096]    # FP8 blockwise

  FULL (10 layers, i ≡ 3 mod 4):
    qkv = x·Wqkv[9216,2048]  → [q|gate](8192) ‖ k(512) ‖ v(512)      # FP8 blockwise
    q,gate = per-head chunk of 512 → 16×[q(256)|gate(256)]
    q,k = GemmaRMSNorm_256 ; RoPE on dims[0:64] only (partial_rotary_factor=0.25)
    o = PagedAttn(q[16,256], k[2,256], v[2,256], scale=1/16, causal)   # BF16 KV
    y = (o ⊙ σ(gate)) · Wo[2048,4096]                                 # FP8 blockwise

  (x, r) = GemmaRMSNorm(y, r)
  MoE (all 40 layers):
    p    = softmax_256(x·Wg[256,2048]) ; (w,ids) = top8(p) ; w /= Σw   # router GEMM is BF16
    shd  = σ(x·Wsg[1,2048]) · (SiluMul(x·W13s[1024,2048])·W2s[2048,512])   # FP8 blockwise
    y    = shd + Σ_{j<8} w_j·(SiluMul(x·w13[id_j][1024,2048])·w2[id_j][2048,512])  # FP8 blockwise
h = GemmaRMSNorm(y, r) ; logits = h · Wlm[248320,2048]ᵀ               # lm_head is BF16
```

Weight footprint on device: **31.31 GiB fp8 + 1.94 GiB bf16 + 8.2 MB fp32 scales = 33.26 GiB**
(text path only; derivation in §5.4).

---

## 1. Config, as vLLM actually resolves it

`Qwen3_5MoeForConditionalGeneration` is registered at `vllm/model_executor/models/registry.py:574-577`
→ `vllm/model_executor/models/qwen3_5.py`. Qwen3.5 is a thin specialization of Qwen3-Next: it reuses
`Qwen3NextAttention`, `Qwen3NextDecoderLayer`, `Qwen3NextModel`, `Qwen3NextSparseMoeBlock` from
`vllm/model_executor/models/qwen3_next.py`, and the GDN mixer
`QwenGatedDeltaNetAttention` from `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:342`.

```
Qwen3_5MoeForConditionalGeneration            qwen3_5.py:614
└── language_model : Qwen3_5MoeForCausalLM    qwen3_5.py:381
    ├── model : Qwen3_5Model                  qwen3_5.py:210   (subclasses Qwen3NextModel)
    │   ├── embed_tokens : VocabParallelEmbedding(248320, 2048)   qwen3_5.py:238
    │   ├── layers[0..39] : Qwen3_5DecoderLayer                   qwen3_5.py:113
    │   └── norm : GemmaRMSNorm(2048)                             qwen3_5.py:258
    └── lm_head : ParallelLMHead(248320, 2048)   (tie_word_embeddings=False)
└── visual : Qwen3_VisionTransformer          (unused in text decode)
```

The multimodal wrapper's `forward` goes straight to `self.language_model.model(...)`
(`qwen3_5.py:506`), so the text graph is exactly `Qwen3NextModel.forward` (`qwen3_next.py:648-714`).

### 1.1 Values from the shipped `config.json` (not the class defaults)

The vLLM config class is `Qwen3_5MoeTextConfig` (`vllm/transformers_utils/configs/qwen3_5_moe.py`,
`model_type = "qwen3_5_moe_text"`, ctor at `:44-128`). **Read the shipped file, not the defaults** —
three fields differ from the class defaults and all three change the graph:

| field | class default | **shipped value** | consequence |
|---|---|---|---|
| `max_position_embeddings` | 32768 (`:52`) | **262144** | RoPE cache size |
| `rope_parameters.rope_theta` | 10000 (`rotary_embedding/__init__.py:64`) | **1e7** | different cos/sin table |
| `mamba_ssm_dtype` | absent | **`"float32"`** | GDN recurrent state is **fp32**, not bf16 (§2.1.5) |

Fields that match the class defaults and are load-bearing:

| field | value | source |
|---|---|---|
| `vocab_size` | 248320 | `qwen3_5_moe.py:46` |
| `hidden_size` | 2048 | `:47` |
| `num_hidden_layers` | 40 | `:48` |
| `num_attention_heads` / `num_key_value_heads` | 16 / 2 | `:49-50` |
| `head_dim` | 256 | `:60` |
| `hidden_act` | `"silu"` | `:51` |
| `rms_norm_eps` | 1e-6 | `:54` |
| `tie_word_embeddings` | False | `:56`, `:128` |
| `attn_output_gate` | **true** (explicit in `config.json`) | default `True` at `qwen3_next.py:265` |
| `partial_rotary_factor` | **0.25 — present in the checkpoint** under `rope_parameters` | see §6, gotcha 2 |
| `linear_conv_kernel_dim` | 4 | `:61` |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | `:62-63` |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 | `:64-65` |
| `moe_intermediate_size` | 512 | `:66` |
| `shared_expert_intermediate_size` | 512 | `:67` |
| `num_experts_per_tok` / `num_experts` | 8 / 256 | `:68-69` |
| `layer_types` | explicit 40-entry list in `config.json`, identical to the derived pattern | derivation at `:95-102` |

`layer_types[i] = "linear_attention" if bool((i+1) % 4) else "full_attention"`
(`qwen3_5_moe.py:97-102`, `full_attention_interval = 4`) ⇒ **layers 3, 7, 11, …, 39 are full
attention (10 layers); the other 30 are GDN linear attention.** Layer 0 is linear.

Derived dims (TP=1):

```
key_dim    = 128 * 16 = 2048
value_dim  = 128 * 32 = 4096
conv_dim   = 2*key_dim + value_dim = 8192          (qwen_gdn_linear_attn.py:389)
q_size     = 16 * 256 = 4096      kv_size = 2 * 256 = 512
rotary_dim = int(256 * 0.25) = 64                  (rotary_embedding/__init__.py:69-72)
```

`layer_scale` is `getattr(config, "layer_scale", False)` → **False** (`qwen3_5.py:182`), so there
are no per-layer scale multiplies. There is **no embedding scale**: `Qwen3_5Model.__init__` builds a
bare `VocabParallelEmbedding` (`qwen3_5.py:238`), and `LogitsProcessor` uses `scale=1.0`,
`soft_cap=None` (`vllm/model_executor/layers/logits_processor.py:37-55`). `vocab_size = 248320 =
3880 × 64` is already a multiple of `DEFAULT_VOCAB_PADDING_SIZE = 64`
(`vocab_parallel_embedding.py:32`) ⇒ **no vocab padding**.

### 1.2 Normalization: Gemma `(1+w)` everywhere except the GDN gated norm

`qwen3_5.py:39` — `from ...layernorm import GemmaRMSNorm as Qwen3_5RMSNorm`; same alias in
`qwen3_next.py:28`. Consequently `input_layernorm` (`qwen3_5.py:175`),
`post_attention_layernorm` (`qwen3_5.py:178`), the final `model.norm` (`qwen3_5.py:258`) and the
full-attention `q_norm`/`k_norm` (`qwen3_next.py:319-320`) are all `GemmaRMSNorm`:

```
y = x * rsqrt(mean(x²) + eps) * (1.0 + w)          # layernorm.py:151-160
```

with `self.weight = nn.Parameter(torch.zeros(hidden_size))` (`layernorm.py:148`) and
`weight = self.weight.float() + 1.0` (`layernorm.py:157`). The class docstring
(`layernorm.py:133-138`) names the two deltas vs plain RMSNorm: the `(1+w)` and the
`(x*w).to(orig_dtype)` ordering.

**The one exception** is the GDN output gated norm `linear_attn.norm`, a `RMSNormGated`
(`layernorm.py:172`) constructed at `qwen_gdn_linear_attn.py:459-466`. Its weight is
`torch.empty(...)` then `torch.nn.init.ones_` (`layernorm.py:212`, `:218-219`) and it is used
directly — `out = x_normed * weight` at `layernorm.py:257`, **no `+1`**.

### 1.3 Residual structure (`qwen3_next.py:492-570`)

```
# layer i
if residual is None:                       # i == 0
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
else:
    hidden_states, residual = input_layernorm(hidden_states, residual)   # fused add+norm
                                            # returns (norm(h+r), h+r)
hidden_states = mixer(hidden_states)        # linear_attn(h) or self_attn(pos, h)
hidden_states, residual = post_attention_layernorm(hidden_states, residual)
hidden_states = mlp(hidden_states)
return hidden_states, residual
# after the loop:
hidden_states, _ = model.norm(hidden_states, residual)   # qwen3_next.py:711
```

The classic vLLM deferred-residual form: the add is folded into the *next* norm.

---

## 2. Layer-type dataflow

### 2.1 GDN linear-attention layer (×30)

Class `QwenGatedDeltaNetAttention` (`qwen_gdn_linear_attn.py:342`), constructed with
`gqa_interleaved_layout=False` for Qwen3.5 (`qwen3_5.py:142`) — Qwen3-Next passes `True`
(`qwen3_next.py:439`). That flag is the single biggest structural difference; see §6 gotcha 5.

#### 2.1.1 Modules

| module | class | shape (TP=1) | precision |
|---|---|---|---|
| `in_proj_qkvz` | `MergedColumnParallelLinear`, `output_sizes=[2048,2048,4096,4096]` (`:503-514`) | W `[12288, 2048]` | **FP8 block** |
| `in_proj_ba` | `MergedColumnParallelLinear`, `output_sizes=[32,32]` (`:528-538`) | W `[64, 2048]` | **BF16** |
| `conv1d` | `ColumnParallelLinear(in=4, out=8192)` then `.unsqueeze(1)` (`:390-396`) | W `[8192, 1, 4]` | **BF16** |
| `dt_bias` | `nn.Parameter(ones(32))` (`:439`) | `[32]` | BF16 in ckpt |
| `A_log` | `nn.Parameter(empty(32, dtype=float32))` (`:442-447`) | `[32]` | **FP32** |
| `norm` | `RMSNormGated(128, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu")` (`:459-466`) | W `[128]` | FP32 in ckpt |
| `out_proj` | `RowParallelLinear(4096, 2048, bias=False)` (`:468-471`) | W `[2048, 4096]` | **FP8 block** |

`create_qkvz_proj` (`:490-514`): with `gqa_interleaved_layout=False` the output sizes are the four
independent shards `[key_dim, key_dim, value_dim, value_dim]`, i.e. the fused weight is laid out
`[q | k | v | z]` contiguously — *not* interleaved by key-head group. `create_ba_proj` (`:516-538`):
`[num_v_heads] * 2 = [32, 32]`, layout `[b | a]`. All biases are `False`; `conv1d.bias is None`.

`conv1d` is built **without a `quant_config` argument** (`:390-395`) so it can never be quantized,
independent of the checkpoint's skip list. `in_proj_ba` *is* given the quant config (`:535`) — it
stays bf16 only because the checkpoint lists both of its shards as unconvertible (§3.2). The code
comment at `:411` says so directly: *"ba_proj doesn't support blockwise fp8 quantization."* The
mechanical reason: its output shard size is 32, and 32 is not a multiple of `block_n = 128`.

#### 2.1.2 Forward, part 1 — input projection (`forward_cuda`, `:829-887`)

```python
mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)   # [T,2048] -> [T,12288]      :843
ba,         _ = self.in_proj_ba(hidden_states)     # [T,2048] -> [T,   64]      :844

# Qwen3.5 branch (:855-863): pure splits, no rearrange
qkv_size = 2*key_dim + value_dim = 8192
z_size   = value_dim             = 4096
mixed_qkv, z = mixed_qkvz.split([8192, 4096], dim=-1)   # views, no copy
z = z.reshape(T, 32, 128)
b, a = self.split_ba(ba)      # ba.chunk(2,-1) -> b [T,32], a [T,32]            :558-566
b = b.contiguous(); a = a.contiguous()

core_attn_out = torch.zeros((T, 32, 128), dtype=bf16)   # deliberately zeros    :870
torch.ops.vllm.qwen_gdn_attention_core(mixed_qkv, b, a, core_attn_out, layer_name)  # :876
```

The `zeros` (not `empty`) is deliberate; the comment at `:868-869` references vLLM PR #28182.
The custom op dispatches to `_forward_core` (`:1180`).

#### 2.1.3 Decode dispatch (`_forward_core`, `:1180-1218`)

```python
if (self.enable_packed_recurrent_decode          # envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        and attn_metadata.spec_sequence_masks is None
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0):
    return self._forward_core_decode_non_spec(...)
```

`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` defaults to **`True`** (`vllm/envs.py:124`). For a
pure-decode batch with no speculative decoding this packed path is therefore the default, and it is
what MPK should target. The generic path (`:1220-1496`, three ops instead of two) runs only for
mixed prefill+decode batches or under MTP/spec.

#### 2.1.4 Packed decode path (`_forward_core_decode_non_spec`, `:1564-1616`)

```python
conv_state = kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1,-2)  # :1579-1583
ssm_state  = kv_cache[1]                                                                 # :1584
conv_weights = self.conv1d.weight.view(8192, 4)                                          # :1591-1593

mixed_qkv = causal_conv1d_update(
    mixed_qkv[:T], conv_state, conv_weights,
    self.conv1d.bias,          # None
    self.activation,           # "silu"
    conv_state_indices=non_spec_state_indices_tensor[:T],
    validate_data=False)                                                                 # :1594-1602

fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv=mixed_qkv,                 # [T, 8192]
    a=a, b=b,                            # [T, 32] each
    A_log=self.A_log, dt_bias=self.dt_bias,
    scale=self.head_k_dim**-0.5,         # 128**-0.5 = 0.08838834764831845
    initial_state=ssm_state,             # [num_blocks, 32, 128, 128]  FP32
    out=core_attn_out[:T].unsqueeze(1),  # [T, 1, 32, 128]
    ssm_state_indices=non_spec_state_indices_tensor[:T],
    use_qk_l2norm_in_kernel=True)                                                        # :1604-1615
```

The packed kernel does its own q/k/v unpacking — there is no separate split/rearrange op.

**Causal conv1d update** — kernel `_causal_conv1d_update_kernel`
(`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:749-1060`), launcher `causal_conv1d_update`
(`:1069`). Per channel `d ∈ [0,8192)`, `state_len = width-1 = 3`, `seqlen = 1`:

```
window   = [S[d,0], S[d,1], S[d,2], x[n,d]]
y[n,d]   = silu( Σ_{j=0..3} W[d,j] * window[j] )        # bias is None
S'[d,:]  = [S[d,1], S[d,2], x[n,d]]
```

The state store (`tl.store(conv_state_ptrs_target, ...)`, `:933`) happens **before** the
accumulation loop (`:972`). The accumulator is fp32 (`:943`); `x` is cast to `conv_state.dtype` on
entry (`:1131`). The kernel **writes its output over its input** (`:1162-1163`), so the q/k/v slice
of the `in_proj_qkvz` output is mutated in place; the `z` columns are untouched.

**Gated delta-rule recurrence** — kernel `fused_recurrent_gated_delta_rule_packed_decode_kernel`
(`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:255-336`), launcher `:339-478`.
Grid `(NV, B*HV) = (4, T*32)`, `BK=128`, `BV=32`, `num_warps=1`, `num_stages=3`. `H=16` key heads,
`HV=32` value heads, `K=V=128`. Head mapping `i_h = i_hv // 2` (`:284`) — **GVA: two value heads
share one key/query head.**

Offsets into the packed 8192-wide conv output (`:306-308`):

```
q : mixed_qkv[n,          i_h*128  : +128]      # base 0
k : mixed_qkv[n, 2048  + i_h*128   : +128]      # base H*K   = 2048
v : mixed_qkv[n, 4096  + i_hv*128  : +128]      # base 2*H*K = 4096
```

Math, all fp32 (`:313-336`):

```
q ← q / sqrt(Σq² + 1e-6) ;  k ← k / sqrt(Σk² + 1e-6) ;  q ← q * 128^-0.5
x        = a[n,hv] + dt_bias[hv]
softplus = log(1+exp(x))  if x <= 20 else x           # SOFTPLUS_THRESHOLD = 20
g        = -exp(A_log[hv]) * softplus                 # A_log is fp32 -> decay exp(g) ∈ (0,1]
beta     = sigmoid(b[n,hv])                           # round-tripped through bf16, :325
S  ←  S * exp(g)          # S: [V=128, K=128] per v-head
δ  ←  (v - S @ k) * beta
S  ←  S + outer(δ, k)
o  ←  S @ q
```

i.e. `S_t = S_{t-1}·e^{g_t} + β_t (v_t − S_{t-1}e^{g_t} k_t) k_tᵀ`, `o_t = S_t q_t`. **The decay
`exp(g)` is a scalar per (token, v-head)**, not per-channel. `beta` at `:325` is
`tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)` — the sigmoid is round-tripped through
bf16; the equivalent kernel in `fused_sigmoid_gating.py:136` does *not* do this. Bit-exact
reproduction must match whichever path it emulates.

Stores: `o → out[n,0,hv,:]`; `S → ssm_state[state_idx, hv, :, :]` in place (`ht = initial_state`,
`fused_recurrent.py:457-458`). If `ssm_state_indices[n] <= 0` (`NULL_BLOCK_ID = 0`) the kernel
zero-fills the output and returns (`:296-299`).

#### 2.1.5 Recurrent state shapes and dtypes

`MambaStateShapeCalculator.gated_delta_net_state_shape`
(`vllm/model_executor/layers/mamba/mamba_utils.py:246-268`) with
`(tp=1, num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4,
num_spec=0)`:

```
conv_dim             = 8192
conv_state_shape     = (3, 8192)      # SD layout (default, mamba_utils.py:26-48)
temporal_state_shape = (32, 128, 128)
```

The conv kernels want `(..., dim, state_len)`, so under SD the layer takes a transposed view
`kv_cache[0].transpose(-1,-2)` (`qwen_gdn_linear_attn.py:1579-1583`). **Under SD the innermost
contiguous axis of the conv state is the 8192-wide channel axis** — physically
`[num_blocks][3][8192]`. `VLLM_SSM_CONV_STATE_LAYOUT` defaults to `None → "SD"` (`envs.py:227`).

**Dtypes — the checkpoint overrides the default.** `_mamba_state_dtype`
(`mamba_utils.py:96-108`) returns `(conv_state_dtype, temporal_state_dtype)`; with
`mamba_cache_dtype="auto"` the conv state is the model dtype (**bf16**), and the temporal state
follows only if `mamba_ssm_cache_dtype == "auto"`. It is not:
`Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`
(`vllm/model_executor/models/config.py:744-768`) copies the HF field `mamba_ssm_dtype` into
`cache_config.mamba_ssm_cache_dtype` whenever the CLI value is `"auto"` (`:754-757`), and the
shipped `config.json` sets **`mamba_ssm_dtype: "float32"`**.

⇒ **The recurrent state `S` is FP32 on this checkpoint**: `32·128·128·4 = 2 MiB` per sequence per
linear layer, plus `3·8192·2 = 48 KiB` conv. **2096 KiB per sequence per layer × 30 layers =
61.4 MiB per sequence.** (The scouting note treated this as a hypothetical; it is the actual
configuration.)

#### 2.1.6 Forward, part 3 — gated output norm + out proj (`_output_projection`, `:774-791`)

```python
core_attn_out = core_attn_out.reshape(-1, 128)     # [T*32, 128]
z             = z.reshape(-1, 128)
core_attn_out = self.norm(core_attn_out, z)        # RMSNormGated
core_attn_out = core_attn_out.reshape(T, 32, 128).flatten(-2)   # [T, 4096]
output, _     = self.out_proj(core_attn_out)       # [T,4096] -> [T,2048]
```

`RMSNormGated` semantics (`layernorm.py:221-269`, `forward_static`), `norm_before_gate=True`:

```
out = ( x * rsqrt(mean(x²) + 1e-6) * w ) * silu(z)
```

computed in fp32 (`x=x.float(); weight=weight.float(); z=z.float()`, `:243-246`), cast back at the
end (`:269`). The weight is a plain ones-initialized vector of length **128 shared across all 32
v-heads** — not Gemma-style. `activation` comes from `getattr(config, "output_gate_type", "silu")`
with `"swish"→"silu"` (`qwen_gdn_linear_attn.py:452-457`). CUDA path is FLA's `rmsnorm_fn`
(`vllm/third_party/flash_linear_attention/ops/layernorm_guard.py:319-331`).

#### 2.1.7 Op table — one GDN layer, B tokens (one per sequence)

GEMMs are `[M,K] × [K,N]`; vLLM stores weights `[N,K]` and applies `x @ W.T`.

| # | op | in → out | dtype in → out | kernel | cite |
|---|---|---|---|---|---|
| 1 | `input_layernorm` (Gemma, fused add) | `h,r [B,2048]` → `(x[B,2048], r[B,2048])` | bf16 → bf16 (fp32 interior) | `rms_norm` / `fused_add_rms_norm` custom op | `layernorm.py:151-160` |
| 2 | quantize activation | `[B,2048]` bf16 → `q [B,2048]` fp8 + `As [B,16]` fp32 | bf16 → e4m3 | `QuantFP8`, per-token-group of 128 | `BlockScaledMMLinearKernel.py:119-122` |
| 3 | `in_proj_qkvz` | `[B,2048] × [2048,12288]` → `[B,12288]` | e4m3×e4m3 → bf16 | `cutlass_scaled_mm` block-scaled (§3.5) | `cutlass.py:311-326` |
| 4 | `in_proj_ba` | `[B,2048] × [2048,64]` → `[B,64]` | bf16 → bf16 | plain `torch.nn.functional.linear` (Unquantized) | `linear.py:275-276` |
| 5 | split | `qkvz → mixed_qkv[B,8192] view, z[B,32,128] view`; `ba → b,a [B,32]` + 2 `contiguous()` | — | view/copy | `qwen_gdn_linear_attn.py:855-863` |
| 6 | `causal_conv1d_update` | `x[B,8192]` (row-stride 12288, **strided**), `W[8192,4]`, state `[·,3,8192]` → `[B,8192]` **in place** | bf16 → bf16 (fp32 acc) | `_causal_conv1d_update_kernel`, grid `(B,32)`, `BLOCK_N=256` | `causal_conv1d.py:749-1060` |
| 7 | `fused_recurrent_gated_delta_rule_packed_decode` | per `(token, hv∈32)`: `S[128,128]` R+W, `q,k[128]`, `v[128]` → `o[128]` | fp32 state, bf16 in/out | `fused_recurrent_..._packed_decode_kernel`, grid `(4, B*32)`, `BK=128,BV=32`, warps=1 | `fused_recurrent.py:255-336` |
| 8 | `RMSNormGated(128)` | `x,z [B*32,128]`, `w[128]` → `[B*32,128]` | bf16 → bf16 (fp32 interior) | FLA `rmsnorm_fn` | `layernorm_guard.py:319-331` |
| 9 | quantize activation | `[B,4096]` bf16 → fp8 + `As [B,32]` | bf16 → e4m3 | `QuantFP8` | as (2) |
| 10 | `out_proj` | `[B,4096] × [4096,2048]` → `[B,2048]` | e4m3×e4m3 → bf16 | `cutlass_scaled_mm` block-scaled | `cutlass.py:311-326` |

Per-layer weight bytes: **33.55 MB fp8** (`in_proj_qkvz` 25.17 + `out_proj` 8.39) + **0.33 MB bf16**
(`in_proj_ba` 0.26, `conv1d` 0.066) + 8 KB scales. Per-token state traffic: **4.29 MB**
(2 MiB `S` read + 2 MiB `S` write + 48 KiB conv read + 48 KiB conv write). GEMM FLOPs ≈
`2·B·(12288·2048 + 64·2048 + 2048·4096) = 67.4 MFLOP × B`; recurrence FLOPs ≈ 3.7 MFLOP × B.
**Arithmetic intensity of the recurrence is ≈ 0.9 FLOP/byte at fp32 state — the single most
memory-bound op in the model, and the biggest fusion opportunity.**

### 2.2 Full-attention layer (×10)

Class `Qwen3NextAttention` (`qwen3_next.py:231-400`).

#### 2.2.1 Modules

```python
self.attn_output_gate = getattr(config, "attn_output_gate", True)      # :265  -> True
self.qkv_proj = QKVParallelLinear(
    hidden_size        = 2048,
    head_size          = 256,
    total_num_heads    = 16 * (1 + True) = 32,      # <-- the gate rides in the Q shard
    total_num_kv_heads = 2,
    bias               = getattr(config,"qkv_bias",False) = False)      # :267-275
self.o_proj  = RowParallelLinear(16*256 = 4096, 2048, bias=False)      # :277-284
self.q_norm  = GemmaRMSNorm(256, eps=1e-6)                             # :319
self.k_norm  = GemmaRMSNorm(256, eps=1e-6)                             # :320
self.scaling = 256**-0.5 = 0.0625                                      # :261
self.attn    = Attention(16, 256, 0.0625, num_kv_heads=2, attn_type=DECODER)  # :302-317
```

`qkv_proj` weight is `[32·256 + 2·256 + 2·256, 2048] = [9216, 2048]`, splitting as
`[q_gate: 8192 | k: 512 | v: 512]` (`:345-347`, `:368-370`).

#### 2.2.2 The output gate is the second half of each Q head

```python
q_gate = q_gate.view(*orig_shape, self.num_heads, -1)   # [T,16,512]      :372
q, gate = torch.chunk(q_gate, 2, dim=-1)                # each [T,16,256] :373
```

so the 8192-wide `q_gate` block is `[h0_q(256) | h0_gate(256) | h1_q(256) | h1_gate(256) | …]`.
Independently confirmed by the fused kernel's addressing:
`in_base = q_gate_ptr + token*stride + local_head*2*head_dim` (`fused_qk_norm_rope.py:54`) and
`gate_in_base = in_base + head_dim` (`:111`), documented in the launcher docstring
(`fused_qk_norm_rope.py:133`: *"q_gate: (n_tokens, num_q_heads * 2 * head_dim) -- per head:
[q|gate]"*).

#### 2.2.3 QK norm + partial RoPE — and the mRoPE wrinkle

`get_rope(head_size=256, max_position=262144, rope_parameters=config.rope_parameters)`
(`qwen3_next.py:286-291`) → `rotary_dim = int(head_size * partial_rotary_factor) = int(256*0.25) =
64` (`rotary_embedding/__init__.py:69-72`). **Only the first 64 of the 256 head dims are rotated;
dims `[64,256)` are RMSNorm'd but pass through un-rotated.** NeoX style
(`is_neox_style=True` default, `__init__.py:36`): halves are `[0:32]` and `[32:64]`.

The shipped `rope_parameters` carries `mrope_section: [11,11,10]` and `mrope_interleaved: true`, so
`get_rope` builds an **`MRotaryEmbedding`**, not a plain `RotaryEmbedding`
(`rotary_embedding/__init__.py:101-112`: `if "mrope_section" in rope_parameters`). The assertion
`sum(mrope_section) == rotary_dim // 2` holds (`11+11+10 = 32 = 64//2`, `mrope.py:249`).
`Qwen3_5MoeTextConfig` deliberately exempts these keys from RoPE validation
(`qwen3_5_moe.py:103-106`).

For **text-only** decode this is mathematically a no-op relative to standard partial NeoX RoPE:
when `positions.ndim == 1` the section logic is skipped entirely (`mrope.py:277`, guarded by
`if positions.ndim == 2`), and when positions arrive as `(3,T)` the T/H/W rows are identical for
text tokens, so `apply_interleaved_rope` (`mrope.py:190-198`) selects identical values from all
three rows. **MPK can implement plain partial NeoX RoPE with `theta = 1e7` over dims `[0:64]`.**

Two implementations, selected at construction (`qwen3_next.py:326-331`):

```python
self.use_fused_qk_norm_rope_gate = (self.attn_output_gate
                                    and getattr(self.rotary_emb, "is_neox_style", False)
                                    and current_platform.is_cuda()
                                    and text_only)
text_only = mm_config is None or mm_config.language_model_only      # :325
```

* **Fused** — `fused_qk_rmsnorm_rope_gate` (`fused_qk_norm_rope.py:117-201`): one Triton launch,
  grid `(T, num_q_heads + num_kv_heads) = (T, 18)` (`:171`). Split → RMSNorm (variance over the full
  256, `:62`) → partial NeoX RoPE on `[0:64]` (`:79-107`) → verbatim gate copy (`:110-114`). The
  Gemma `+1` is applied by the caller: it is passed `self.q_norm.weight.float() + 1.0`
  (`qwen3_next.py:355-356`). It deliberately round-trips the normalized value through bf16 before
  RoPE (`:67`) to bit-match the unfused path.
* **Eager** (`qwen3_next.py:367-387`): `chunk` → `q_norm(q.view(-1,16,256))` →
  `k_norm(k.view(-1,2,256))` → `self.rotary_emb(positions, q, k)`.

⚠ For `Qwen3_5MoeForConditionalGeneration` a multimodal config exists and `language_model_only`
defaults to **False** (`vllm/config/multimodal.py:78`; CLI `--language-model-only`,
`vllm/engine/arg_utils.py:1255`), so **the fused kernel is OFF unless the server is started with
`--language-model-only`**; the eager split+norm+MRoPE path runs. Both compute the same function.

#### 2.2.4 Attention + output gate (`forward`, `:389-400`)

```python
qkv, _ = self.qkv_proj(hidden_states)          # [T,2048] -> [T,9216]      :394
q, k, v, gate = self._project_qkv_gate(qkv, positions)                    # :395
attn_output = self.attn(q, k, v)               # [T,4096]                  :396
if gate is not None:
    attn_output = attn_output * torch.sigmoid(gate)                        # :397-398
output, _ = self.o_proj(attn_output)           # [T,4096] -> [T,2048]      :399
```

**The output gate is applied OUTSIDE the attention kernel**, on the flat `[T, 16·256]` tensor, as
`out * sigmoid(gate)` — a full sigmoid, not SiLU, and not an attention "sink" (`sinks` is `None`
for a plain `Attention` layer). `Attention` reshapes `q→[T,16,256]`, `k,v→[T,2,256]`,
`out→[T,16,256]` (`vllm/model_executor/layers/attention/attention.py:536-541`).

#### 2.2.5 The current token's KV-cache write — a separate op, ordered before the read

`self.attn(q, k, v)` is not one op. `FlashInferBackend.forward_includes_kv_cache_update = False`
(`v1/attention/backends/flashinfer.py:498`), so `Attention.forward` issues **two** ops, the write
first:

```python
if (not self.attn_backend.forward_includes_kv_cache_update
        and self.kv_sharing_target_layer_name is None
        and key is not None and value is not None):
    kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)   # attention.py:546-553
unified_attention_with_output(query, key, value, output, self.layer_name,
                              kv_cache_dummy_dep=kv_cache_dummy_dep)            # attention.py:554-561
```

(`attention.py:543-561` for the direct-call path, `:562-581` for the `torch.ops.vllm.*` path.) The
returned `kv_cache_dummy_dep` is a zero-element tensor whose only purpose is to make the data
dependency explicit so **torch.compile cannot reorder the read ahead of the write** — the docstring
says exactly that (`attention.py:779-782`, return at `:798`). **So the current token is written into
the paged cache before the decode kernel reads it, and its own causal attention sees it.** The
`seq_lens` handed to the kernel count the current token, so position `t` attends over `[0..t]` with
no special-casing of the diagonal.

`unified_kv_cache_update` (`attention.py:774-798`) pulls the per-layer `slot_mapping` out of the
forward context (`:766-770`) and calls `FlashInferImpl.do_kv_cache_update`
(`flashinfer.py:2251-2288`), which re-views the packed cache and calls the copy kernel:

```python
# (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))     zero-copy views
k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)   # flashinfer.py:2274-2278
torch.ops._C_cache_ops.reshape_and_cache_flash(
    key, value, k_cache, v_cache, slot_mapping,
    self.kv_cache_dtype, layer._k_scale, layer._v_scale)                    # flashinfer.py:2279-2288
```

i.e. the HND storage `(num_blocks, 2, 32, 512)` is presented to the copy kernel as two
`(num_blocks, 32, 2, 256)` NHD-shaped views over the same bytes; K occupies `[..., :256]` and V
`[..., 256:]` of the packed last dim.

**Slot addressing.** `reshape_and_cache_flash_kernel`
(`csrc/libtorch_stable/cache_kernels.cu:315-345`) is one CUDA block per token:

```
slot = slot_mapping[token];  if (slot < 0) return;      // PAD_SLOT_ID for cuda-graph padding
block_idx    = slot / block_size;      // block_size = the KERNEL block, 32
block_offset = slot % block_size;
dst = cache + block_idx*block_stride + block_offset*page_stride;
```

`slot_mapping` is built by `_compute_slot_mapping_kernel` (`v1/worker/block_table.py:346-409`).
At DCP=1 it reduces to a **two-level** index, because the KV-manager block (1056) and the kernel
block (32) differ (§4.3) — `BlockTable.__init__` sets `self.kv_cache_block_size = block_size`
(1056, `block_table.py:56`), `self.block_size = kernel_block_size` (32, `:75`) and
`self.blocks_per_kv_block = 1056 // 32 = 33` (`:76`):

```
kv_blk   = pos // 1056                                   # which KV-manager block
kv_off   = pos %  1056
row      = kv_blk * 33 + kv_off // 32                    # column in the (expanded) block table
phys     = block_table[req, row]                         # physical kernel-block id
slot     = phys * 32 + (kv_off % 32)                      # block_table.py:399-407
```

Padded rows get `PAD_SLOT_ID` and the copy kernel skips them (`cache_kernels.cu:328-331`).

**For an MPK port** none of the 1056/33/32 machinery is required — it exists only to make mamba and
attention pages byte-identical inside one allocator (§4.7). What *is* required is the invariant:
**append `k_t`,`v_t` for the current token into the paged cache before the decode attention kernel
runs, and include position `t` in the kernel's sequence length.** Only 2 KV heads × 256 dims × 2
bytes × 2 tensors = **2 KiB per token per full-attn layer** is written (20 KiB per token per step
across all 10 layers) — negligible next to the 2.47 GB of weight traffic, but skipping it silently
breaks autoregression.

#### 2.2.6 Op table — one full-attention layer, B tokens

| # | op | in → out | dtype in → out | kernel | cite |
|---|---|---|---|---|---|
| 1 | `input_layernorm` (Gemma, fused add) | `[B,2048]` → `[B,2048]` ×2 | bf16 → bf16 | `fused_add_rms_norm` | `layernorm.py:151-160` |
| 2 | quantize activation | `[B,2048]` → fp8 + `As[B,16]` | bf16 → e4m3 | `QuantFP8` (1,128) | `BlockScaledMMLinearKernel.py:119-122` |
| 3 | `qkv_proj` | `[B,2048] × [2048,9216]` → `[B,9216]` | e4m3 → bf16 | `cutlass_scaled_mm` block-scaled | `cutlass.py:311-326` |
| 4 | split | `[B,9216]` → `q_gate[B,8192] ‖ k[B,512] ‖ v[B,512]` | — | view | `qwen3_next.py:368-370` |
| 5 | per-head chunk | `q_gate.view(B,16,512)` → `q[B,16,256]`, `gate[B,16,256]` | — | view/chunk | `qwen3_next.py:372-373` |
| 6 | `q_norm` (Gemma, per head) | `[B,16,256]`, `w[256]` → same | bf16 → bf16 (fp32) | `GemmaRMSNorm`, used as `1+w` | `qwen3_next.py:380` |
| 7 | `k_norm` (Gemma, per head) | `[B,2,256]`, `w[256]` → same | bf16 → bf16 | `GemmaRMSNorm` | `qwen3_next.py:383` |
| 8 | partial NeoX RoPE | rotate dims `[0:64]` only; `cos_sin_cache[pos,0:64]` | bf16 → bf16 | `MRotaryEmbedding` (text ⇒ plain RoPE) | `mrope.py:262-320` |
| — | *(5–8 collapse into one Triton launch `fused_qk_rmsnorm_rope_gate`, grid `(B,18)`, only with `--language-model-only`)* | | | | `fused_qk_norm_rope.py:117-201` |
| **9** | **KV-cache write (current token)** — **separate op, MUST precede op 10** | `k[B,2,256]`, `v[B,2,256]`, `slot_mapping[B]` int64 → in-place scatter into the paged cache at `blk = slot/32`, `off = slot%32` | bf16 → bf16 (no scaling: `kv_cache_dtype="auto"`) | `torch.ops._C_cache_ops.reshape_and_cache_flash`, one CUDA block per token | `flashinfer.py:2279-2288`; kernel `cache_kernels.cu:315-345`; ordering `attention.py:546-561` |
| 10 | paged attention decode (reads the row op 9 just wrote) | `q[B,16,256]`; KV cache `(N,2,32,512)` bf16 HND, K‖V packed; `scale=0.0625`; causal; `seq_lens` includes position `t`; `sinks=None` → `out[B,16,256]` | bf16 → bf16 | FlashInfer `trtllm_batch_decode_with_kv_cache` (trtllm-gen) | `v1/attention/backends/flashinfer.py:2214-2236` |
| 11 | `out * sigmoid(gate)` | `[B,4096]` → `[B,4096]` | bf16 → bf16 | eager elementwise | `qwen3_next.py:397-398` |
| 12 | quantize activation | `[B,4096]` → fp8 + `As[B,32]` | bf16 → e4m3 | `QuantFP8` | as (2) |
| 13 | `o_proj` | `[B,4096] × [4096,2048]` → `[B,2048]` | e4m3 → bf16 | `cutlass_scaled_mm` block-scaled | `cutlass.py:311-326` |

Per-layer weight bytes: **27.26 MB fp8** + 1 KB bf16 (`q_norm`,`k_norm`) + 6.7 KB scales.
GEMM FLOPs = `2·B·(9216·2048 + 2048·4096) = 54.5 MFLOP × B`.

### 2.3 MoE block (×40 — every layer)

Class `Qwen3NextSparseMoeBlock` (`qwen3_next.py:103-228`). Qwen3.5 uses it for **every** layer
(`qwen3_5.py:159-163` keys only on `model_type == "qwen3_5_moe_text"`; unlike Qwen3-Next it does not
consult `decoder_sparse_step` or `mlp_only_layers`).

#### 2.3.1 Construction (`qwen3_next.py:140-195`)

```python
self.gate               = ReplicatedLinear(2048, 256, bias=False, quant_config=None)  # :140-146
self.shared_expert_gate = ReplicatedLinear(2048,   1, bias=False, quant_config=None)  # :148-154
self.shared_expert      = Qwen3NextMLP(hidden=2048, intermediate=512, act="silu",
                                       quant_config=quant_config,          # <-- FP8
                                       reduce_results=False,
                                       expert_gate=self.shared_expert_gate)  # :167-176
self.experts            = FusedMoE(shared_experts=self.shared_expert, gate=self.gate,
                                   num_experts=256, top_k=8, hidden_size=2048,
                                   intermediate_size=512,
                                   renormalize=getattr(config,"norm_topk_prob",True),
                                   quant_config=quant_config,              # <-- FP8
                                   n_shared_experts=None, shared_expert_gate=None)  # :178-195
```

Both `gate` and `shared_expert_gate` pass **`quant_config=None` explicitly**, so they are bf16 by
construction — no `is_layer_skipped` lookup is involved (`LinearBase.__init__`:
`if quant_config is None: self.quant_method = UnquantizedLinearMethod()`, `linear.py:275-276`).
**There is no fp32 upcast of the router GEMM** either: `gate` is a plain `ReplicatedLinear`, not a
`GateLinear`, so `router_logits` is bf16 `[B,256]`.

⚠ **At this commit `FusedMoE` is a factory function, not an `nn.Module`**
(`vllm/model_executor/layers/fused_moe/layer.py:100`); it returns a `MoERunner`
(`fused_moe/runner/moe_runner.py:236+`). Routing lives in `fused_moe/router/`, weights in
`fused_moe/routed_experts.py`, kernel choice in `fused_moe/oracle/`.

Because `gate=` was passed in, `MoERunner.is_internal_router` is **True**
(`moe_runner.py:317-319`), so `Qwen3NextSparseMoeBlock.forward` takes `:210-214` and passes
`router_logits=hidden_states` as a *placeholder* — the real router GEMM happens inside the runner
(`moe_runner.py:814-819`). `norm_topk_prob` is **not** defined in `Qwen3_5MoeTextConfig` and not in
the shipped `config.json`, so `renormalize=True` comes from the `getattr` default at
`qwen3_next.py:185`.

#### 2.3.2 Router: softmax over all 256, THEN top-8, THEN renormalize

Two code paths compute the same function.

**(a) Inside the fused MoE kernel.** Routing is driven by
`routing_method_type = RoutingMethodType.RenormalizeNaive (= 4)`, documented at
`fused_moe/config.py:112-113` as *"RenormalizeNaive: Softmax -> TopK -> Renormalize"*. It is
resolved by `get_routing_method_type(scoring_func="softmax", renormalize=True, …)`
(`config.py:165-169`) from `FusedTopKRouter.routing_method_type`
(`router/fused_topk_router.py:145-154`).
*Do not confuse with `RoutingMethodType.Renormalize (= 1)`, which is TopK → Softmax
(`config.py:105-106`) — that is a different model.*

**(b) Triton / reference path.** `FusedTopKRouter._compute_routing`
(`router/fused_topk_router.py:156-174`) → `fused_topk` (`scoring_func="softmax"` default from
`layer.py:118`) → `vllm_topk_softmax` → `torch.ops._moe_C.topk_softmax`
(`vllm/_custom_ops.py:2385-2413`), CUDA kernel
`csrc/libtorch_stable/moe/topk_softmax_kernels.cu`. With `num_experts = 256` the launcher picks the
fused `topkGating` kernel, `case 256: LAUNCH_TOPK(256, WARPS_PER_TB, …)` (`:704-706`).

```
l   = x @ W_gate.T                                   # [B,2048]@[2048,256] -> [B,256] bf16
p_j = exp(l_j - max l) / Σ_{j=0..255} exp(l_j - max l)     # FULL 256-wide softmax, fp32
S   = argtop8(p)                                     # ties -> LOWER expert index wins
w_i = p_i / Σ_{i'∈S} p_{i'}   for i ∈ S              # renormalize
```

The kernel computes the whole softmax row rather than only the top-k, *"to closer match torch"*
(comment at `topk_softmax_kernels.cu:441-444`; `row_sum` at `:425`, `reciprocal_row_sum` at
`:445-450`). Tie-breaking: *"We want lower indices to 'win' in every thread so we break ties this
way"* (`:536-537`). Renormalization: `selected_sum` accumulated at `:561`, applied as
`scale /= denom` at `:581-590`. NaN/Inf are clamped to 0 (`:466-471`). Outputs:
`topk_weights` **fp32 `[B,8]`**, `topk_ids` **int32 `[B,8]`**. `VLLM_MOE_SKIP_PADDING` defaults
**True** (`envs.py:196`), so padded rows get `topk_ids = -1`.

No sigmoid, no expert groups, no `e_score_correction_bias`, no routing simulation, no EPLB remap.

#### 2.3.3 Shared expert

`Qwen2MoeMLP.forward` (`vllm/model_executor/models/qwen2_moe.py:112-120`):

```python
gate_up, _ = self.gate_up_proj(x)        # [B,2048] -> [B,1024]     FP8
out        = SiluAndMul()(gate_up)       # silu(gu[:, :512]) * gu[:, 512:] -> [B,512]
out, _     = self.down_proj(out)         # [B,512]  -> [B,2048]     FP8
out        = sigmoid(self.expert_gate(x)[0]) * out                # :118
```

**Always on for every token** (no top-k), scaled by `sigmoid(x @ shared_expert_gate.Wᵀ)` — a scalar
per token derived from the *pre-MLP hidden state*, **not** from the router logits, and applied
**after** `down_proj`. `SiluAndMul` is `silu(x[..., :d]) * x[..., d:]` with `d = 512`
(`vllm/model_executor/layers/activation.py:118-143`). `reduce_results=False`; at TP=1 there is no
all-reduce at all.

**Where it runs at decode:** never fused into the expert kernel on NVIDIA (the ROCm AITER "FSE" path
that would append it as expert slot 256 requires `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()`,
`layer.py:87-94`). Instead `SharedExperts._determine_shared_experts_order`
(`runner/shared_experts.py:89-109`) picks **`MULTI_STREAM_OVERLAPPED`** whenever
`is_cuda() and aux_stream is not None and M <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`
(default **256**, `envs.py:272`) — i.e. always at decode for B ≤ 16. The whole shared MLP is
enqueued on an **auxiliary CUDA stream** (`shared_experts.py:131-142`) and runs concurrently with
gate + router + routed experts, with a `wait_stream` join before the add. This is independent of
quantization.

#### 2.3.4 Routed-expert runtime layout

Runtime tensors (see §5 for the checkpoint→runtime mapping):

```
w13_weight : [256, 1024, 2048]     # rows [0:512) = gate/w1, [512:1024) = up/w3
w2_weight  : [256, 2048,  512]
```

Gate and up are **packed, not interleaved** (`routed_experts.py:495-500`). No biases. Per layer that
is `256×(1024·2048 + 2048·512) = 805 M` params = **768 MiB at fp8** (1.5 GiB at bf16).

Expert compute per selected `(token, expert)`:

```
h  = x @ w13[e].T                 # [1,2048] x [2048,1024] -> [1,1024]
h  = silu(h[:512]) * h[512:]      # -> [1,512]
y  = h @ w2[e].T                  # [1,512]  x [512,2048]  -> [1,2048]
out += topk_weight * y            # topk_weight is fp32 (see below on WHERE it is applied)
```

**Where the router weight is applied — what is verified and what is not.** `topk_weights` is fp32
`[B,8]` (§2.3.2), and the reduce is over the 8 selected experts. Two different levels of evidence:

* **Triton backend — verified.** `MUL_ROUTED_WEIGHT=True` multiplies the fp32 weight into the fp32
  GEMM2 accumulator *before* the single output cast, and the source says so explicitly:
  *"This multiplication MUST be performed in float32 before any precision conversion to ensure
  numerical stability"* — `accumulator *= moe_weight[:, None]` then `accumulator.to(compute_type)`
  (`fused_moe.py:563-578`), followed by `moe_sum` over the top-k axis (`triton_moe.py:539-542`).
* **The FlashInfer TRT-LLM kernel that actually runs — NOT verified.** `trtllm_fp8_block_scale_moe`
  is a prebuilt cubin; vLLM only passes it `routing_logits` and lets it do routing, permute, both
  GEMMs, SwiGLU and the weighted reduce internally (`trtllm_fp8_moe.py:414-443`). **Its internal
  accumulate width, the point at which the router weight is applied, and the intermediate rounding
  between GEMM1/SwiGLU/GEMM2 are not observable from this repo** — the statement above is *inferred
  by analogy with the Triton path*, not established.

  For AC-3 this matters: if MPK is bit-compared against a vLLM-FP8 run it will differ wherever the
  cubin's accumulate/cast points differ, and the discrepancy is not resolvable by reading vLLM.
  Compare against the **HF `transformers` fp32-accumulate reference** (which AC-3 already pins) and
  treat any vLLM-vs-MPK mismatch here as expected, not as an MPK bug. If a numeric comparison
  against the FlashInfer path is genuinely needed, read the kernel in the pinned
  `flashinfer-python==0.6.15.post1` wheel rather than this tree.

Combine: `result = shared_output + fused_output` (`runner/moe_runner.py:722-723`), before any
all-reduce (`MoEPrepareAndFinalizeNoDPEP*.output_is_reduced()` is False,
`prepare_finalize/no_dp_ep.py:54-55`). At TP=1/EP=1 there is no all-reduce.

#### 2.3.5 Op table — one MoE block, B tokens

Ops 4–7 run concurrently on the aux stream (§2.3.3). Ops 2–3 and 8–12 are typically fused into a
single kernel invocation by the selected MoE backend (§3.6); the table lists them separately because
that is the decomposition MPK must reproduce.

| # | op | in → out | dtype in → out | kernel | cite |
|---|---|---|---|---|---|
| 1 | `post_attention_layernorm` (Gemma, fused add) | `[B,2048]` → `[B,2048]` ×2 | bf16 → bf16 | `fused_add_rms_norm` | `layernorm.py:151-160` |
| 2 | `gate` (router GEMM) | `[B,2048] × [2048,256]` → `[B,256]` | **bf16 → bf16** | plain linear (`quant_config=None`) | `qwen3_next.py:140-146` |
| 3 | softmax(256) → top-8 → renorm | `[B,256]` → `w[B,8]` fp32, `ids[B,8]` int32 | bf16 → fp32/int32 | `topkGating` (or in-kernel `RenormalizeNaive`) | `topk_softmax_kernels.cu:704-706` |
| 4 | `shared_expert.gate_up_proj` | `[B,2048] × [2048,1024]` → `[B,1024]` | e4m3 → bf16 | block-scaled fp8 GEMM — **aux stream** | §3.5 |
| 5 | `SiluAndMul` | `[B,1024]` → `[B,512]` | bf16 → bf16 | `SiluAndMul` | `activation.py:118-143` |
| 6 | `shared_expert.down_proj` | `[B,512] × [512,2048]` → `[B,2048]` | e4m3 → bf16 | block-scaled fp8 GEMM | §3.5 |
| 7 | `shared_expert_gate` | `[B,2048] × [2048,1]` → `[B,1]`; `σ(·)` × step 6 | **bf16 → bf16** | plain GEMV (`quant_config=None`) | `qwen3_next.py:148-154` |
| 8 | expert align / scatter | `B·8` (token,expert) pairs → per-expert blocks | int32 | `moe_align_block_size` (skipped for tiny M on the Triton path) | `fused_moe.py:1485-1531` |
| 9 | routed **w13** grouped GEMM | `[B·8, 2048] × [2048, 1024]` per expert → `[B·8,1024]` | e4m3 → bf16 | §3.6 |  |
| 10 | `SiluAndMul` | `[B·8,1024]` → `[B·8,512]` | bf16 → bf16 (fused in kernel) | §3.6 | |
| 11 | routed **w2** grouped GEMM | `[B·8, 512] × [512, 2048]` per expert → `[B·8,2048]` | e4m3 → bf16 | §3.6 | |
| 12 | weighted reduce + add shared | `out[B,2048] = Σ_j w_j·y_j + shared` | fp32 → bf16 | `moe_sum` / kernel epilogue | `moe_runner.py:722-723` |

Distinct experts touched per layer is `≤ min(256, 8B)`; each expert is `3.15 MB` at fp8
(6.29 MB at bf16).

### 2.4 Model prologue / epilogue

| op | in → out | dtype | cite |
|---|---|---|---|
| `embed_tokens` gather | `tok[B]` → `[B,2048]` | bf16, no scale | `qwen3_5.py:238` |
| final `model.norm` (Gemma, fused add) | `(h,r)[B,2048]` → `[B,2048]` | bf16 | `qwen3_next.py:711` |
| `lm_head` | `[B,2048] × [2048,248320]` → `[B,248320]` | **bf16 → bf16** | `logits_processor.py:37-55` |

`lm_head` is a `ParallelLMHead(VocabParallelEmbedding)` (`vocab_parallel_embedding.py:505`), which
is **not** a `LinearBase`; `Fp8Config.get_quant_method` returns `None` for it (`fp8.py:220`) and the
layer falls back to `UnquantizedEmbeddingMethod` (`vocab_parallel_embedding.py:276-280`). It is
therefore bf16 regardless of the skip list — 1.02 GB of weight traffic per decode step.

---

## 3. FP8 — what actually runs in fp8 on this checkpoint

### 3.1 The checkpoint's quantization contract

From the shipped `config.json`:

```json
"quantization_config": {
  "quant_method": "fp8", "activation_scheme": "dynamic",
  "weight_per_tensor": false, "act_per_tensor": false,
  "weight_block_size": [128, 128],
  "modules_to_not_convert": [ ...287 fully-qualified module names... ]
}
```

**`modules_to_not_convert` contains fully-qualified literal names, not globs.** Per language-model
layer the entries are:

| layer kind | entries |
|---|---|
| linear-attention layer `i` (30 of them) | `…layers.{i}.linear_attn.conv1d`, `…linear_attn.in_proj_a`, `…linear_attn.in_proj_b`, `…mlp.gate`, `…mlp.shared_expert_gate` |
| full-attention layer `i` (10 of them) | `…mlp.gate`, `…mlp.shared_expert_gate` |
| global | `lm_head`, `model.language_model.embed_tokens`, 112 `model.visual.*` entries, `mtp.fc`, `mtp.layers.0.mlp.gate`, `mtp.layers.0.mlp.shared_expert_gate` |

(2 + 170 + 112 + 3 = 287; the 170 language-layer entries are 30 `conv1d` + 30 `in_proj_a` +
30 `in_proj_b` + 40 `mlp.gate` + 40 `mlp.shared_expert_gate`.)

This matters: had they been globs, vLLM would silently ignore them — `is_layer_skipped` does **no**
wildcard matching (§3.2).

Confirmed against the safetensors index and headers (`model.safetensors.index.json`, 64 196 tensors,
`total_size = 37,454,799,072`). Every quantized tensor has a `<name>.weight_scale_inv` companion;
every unquantized one does not:

| checkpoint tensor | dtype | shape | scale tensor | scale dtype/shape |
|---|---|---|---|---|
| `layers.{i}.linear_attn.in_proj_qkv.weight` | **F8_E4M3** | `[8192,2048]` | `…in_proj_qkv.weight_scale_inv` | **BF16** `[64,16]` |
| `layers.{i}.linear_attn.in_proj_z.weight` | **F8_E4M3** | `[4096,2048]` | ✓ | BF16 `[32,16]` |
| `layers.{i}.linear_attn.out_proj.weight` | **F8_E4M3** | `[2048,4096]` | ✓ | BF16 `[16,32]` |
| `layers.{i}.linear_attn.in_proj_a.weight` | BF16 | `[32,2048]` | — | — |
| `layers.{i}.linear_attn.in_proj_b.weight` | BF16 | `[32,2048]` | — | — |
| `layers.{i}.linear_attn.conv1d.weight` | BF16 | `[8192,1,4]` | — | — |
| `layers.{i}.linear_attn.norm.weight` | **F32** | `[128]` | — | — |
| `layers.{i}.linear_attn.A_log` | F32 | `[32]` | — | — |
| `layers.{i}.linear_attn.dt_bias` | BF16 | `[32]` | — | — |
| `layers.{i}.self_attn.q_proj.weight` | **F8_E4M3** | `[8192,2048]` | ✓ | BF16 `[64,16]` |
| `layers.{i}.self_attn.k_proj.weight` / `v_proj` | **F8_E4M3** | `[512,2048]` | ✓ | BF16 `[4,16]` |
| `layers.{i}.self_attn.o_proj.weight` | **F8_E4M3** | `[2048,4096]` | ✓ | BF16 `[16,32]` |
| `layers.{i}.self_attn.q_norm.weight` / `k_norm` | BF16 | `[256]` | — | — |
| `layers.{i}.mlp.gate.weight` | BF16 | `[256,2048]` | — | — |
| `layers.{i}.mlp.shared_expert_gate.weight` | BF16 | `[1,2048]` | — | — |
| `layers.{i}.mlp.shared_expert.{gate,up}_proj.weight` | **F8_E4M3** | `[512,2048]` | ✓ | BF16 `[4,16]` |
| `layers.{i}.mlp.shared_expert.down_proj.weight` | **F8_E4M3** | `[2048,512]` | ✓ | BF16 `[16,4]` |
| `layers.{i}.mlp.experts.{e}.{gate,up}_proj.weight` | **F8_E4M3** | `[512,2048]` | ✓ | BF16 `[4,16]` |
| `layers.{i}.mlp.experts.{e}.down_proj.weight` | **F8_E4M3** | `[2048,512]` | ✓ | BF16 `[16,4]` |
| `{input,post_attention}_layernorm.weight`, `model.language_model.norm.weight` | BF16 | `[2048]` | — | — |
| `model.language_model.embed_tokens.weight`, `lm_head.weight` | BF16 | `[248320,2048]` | — | — |

Two surprises worth internalizing: **the checkpoint stores `weight_scale_inv` as BF16**, and
`linear_attn.norm.weight` is **F32**. Scale shapes are exactly `[ceil(N/128), ceil(K/128)]`.

### 3.2 How vLLM decides, layer by layer

`Fp8Config` — `vllm/model_executor/layers/quantization/fp8.py:95`.
`from_config` (`:155-173`) reads `ignored_layers` first and falls back to `modules_to_not_convert`
only if that is falsy (`:160-166`); both feed `self.ignored_layers`.
Block-quant validation at `:115-131` raises `ValueError` if the checkpoint is not fp8-serialized, if
`len(weight_block_size) != 2`, or if `activation_scheme != "dynamic"`. **There is no assertion that
the block size is `[128,128]`** — a different block size simply falls through the kernel list (§3.5).

`apply_vllm_mapper` (`fp8.py:151-153`) rewrites `ignored_layers` through
`hf_to_vllm_mapper.get_unstacked_mapper()` (`models/utils.py:163-170`, which drops
`orig_to_new_stacked` so `in_proj_qkv` survives as itself). Invoked at
`model_loader/utils.py:304-312` (or `models/interfaces.py:1063-1071` for `SupportsQuant` models).
The Qwen3.5 mapper is inherited from `Qwen3VLForConditionalGeneration`
(`models/qwen3_vl.py:1705-1711`):

```
"model.visual."         -> "visual."
"lm_head."              -> "language_model.lm_head."
"model.language_model." -> "language_model.model."
```

so `model.language_model.layers.0.linear_attn.in_proj_a` becomes
`language_model.model.layers.0.linear_attn.in_proj_a`, which is exactly the runtime module prefix.

`is_layer_skipped` — `vllm/model_executor/layers/quantization/utils/quant_utils.py:510-572`.
Called without `skip_with_substr`, so the matcher is `prefix_full_match` = **exact `in` membership**
(`:517-518`). There is **no fnmatch/glob** anywhere on the fp8 path (globbing exists only in
`modelopt.py:178` and `quark/quark.py`). Fused names are expanded through
`packed_modules_mapping` (`qwen3_5.py:288-298`, `:403-406`):

```python
{"qkv_proj": ["q_proj","k_proj","v_proj"],
 "gate_up_proj": ["gate_proj","up_proj"],
 "in_proj_qkvz": ["in_proj_qkv","in_proj_z"],
 "in_proj_ba": ["in_proj_b","in_proj_a"]}
```

and every shard must agree, else a hard `ValueError("Detected some but not all shards of {prefix}
are quantized …")` (`quant_utils.py:551-559`). Resolution for this checkpoint:

| runtime prefix | route | result |
|---|---|---|
| `…linear_attn.in_proj_qkvz` | shards `in_proj_qkv`, `in_proj_z` — neither listed | **`Fp8LinearMethod`** |
| `…linear_attn.in_proj_ba` | shards `in_proj_b`, `in_proj_a` — **both** listed → consistent | `UnquantizedLinearMethod` |
| `…linear_attn.out_proj` | plain match, not listed | **`Fp8LinearMethod`** |
| `…linear_attn.conv1d` | never reaches `get_quant_method` (`quant_config` not passed, `qwen_gdn_linear_attn.py:390-395`) | `UnquantizedLinearMethod` |
| `…self_attn.qkv_proj` | shards `q/k/v_proj` — none listed | **`Fp8LinearMethod`** |
| `…self_attn.o_proj` | not listed | **`Fp8LinearMethod`** |
| `…mlp.gate`, `…mlp.shared_expert_gate` | `quant_config=None` at construction (`qwen3_next.py:144`, `:152`) | `UnquantizedLinearMethod` |
| `…mlp.shared_expert.gate_up_proj` | shards `gate_proj`, `up_proj` — not listed | **`Fp8LinearMethod`** |
| `…mlp.shared_expert.down_proj` | not listed | **`Fp8LinearMethod`** |
| `…mlp.experts` (`RoutedExperts`) | `"experts" in prefix` branch (`quant_utils.py:560-567`); no ignored entry contains `"experts"` | **`Fp8MoEMethod`** |
| `…embed_tokens`, `language_model.lm_head` | not a `LinearBase` → `get_quant_method` returns `None` (`fp8.py:220`) | `UnquantizedEmbeddingMethod` |

`get_quant_method` dispatch table (`fp8.py:175-220`): `LinearBase` + skipped →
`UnquantizedLinearMethod` (`:178-184`); `LinearBase` + fp8-serialized → `Fp8LinearMethod(self)`
(`:193-196`); `RoutedExperts` + skipped → `UnquantizedFusedMoEMethod` (`:197-203`); `RoutedExperts`
+ fp8-serialized → `Fp8MoEMethod(self, layer)` (`:210-211`); `Attention` → `Fp8KVCacheMethod`
(`:218-219`); otherwise `None` (`:220`).

Note the redundancy: of the checkpoint's 5 per-layer skip entries, only `in_proj_a` / `in_proj_b`
actually change vLLM's behaviour. `conv1d`, `mlp.gate` and `mlp.shared_expert_gate` are already
unquantizable by construction, and `lm_head` / `embed_tokens` never enter the linear path. Also note
that the `"lm_head"` entry would *not* be rewritten by the mapper (`orig_to_new_prefix` has
`"lm_head."` **with** a trailing dot, and `_map_name` requires `key.startswith(prefix)`,
`models/utils.py:121-126`) — harmless here for exactly that reason.

**`Fp8KVCacheMethod` for `Attention` does not quantize the KV cache.** It only registers
`q_scale`/`k_scale`/`v_scale`/`prob_scale` parameters so scales *can* be loaded from a checkpoint
(`quantization/kv_cache.py:42-69`; subclass at `fp8.py:859`). The cache dtype comes from
`kv_cache_dtype_str_to_dtype(kv_cache_dtype, model_config)`
(`layers/attention/attention.py:323-326`), driven by `cache_dtype`, which defaults to `"auto"`
(`vllm/config/cache.py:76`) ⇒ model dtype. This checkpoint ships no `k_scale`/`v_scale` tensors, so
the paged KV cache is **bf16** (§4.2).

### 3.3 FP8 vs BF16 — the complete GEMM inventory

| GEMM | shape `[N,K]` | ×count | precision | why |
|---|---|---|---|---|
| `linear_attn.in_proj_qkvz` | `[12288, 2048]` | 30 | **FP8 block 128×128** | both shards quantized in ckpt |
| `linear_attn.in_proj_ba` | `[64, 2048]` | 30 | BF16 | both shards in `modules_to_not_convert`; N=32 per shard < block_n=128 |
| `linear_attn.conv1d` (depthwise, not a GEMM) | `[8192, 4]` | 30 | BF16 | built without `quant_config` |
| `linear_attn.out_proj` | `[2048, 4096]` | 30 | **FP8 block** | |
| `self_attn.qkv_proj` | `[9216, 2048]` | 10 | **FP8 block** | |
| `self_attn.o_proj` | `[2048, 4096]` | 10 | **FP8 block** | |
| `mlp.gate` (router) | `[256, 2048]` | 40 | BF16 | `quant_config=None` |
| `mlp.shared_expert_gate` | `[1, 2048]` | 40 | BF16 | `quant_config=None` |
| `mlp.shared_expert.gate_up_proj` | `[1024, 2048]` | 40 | **FP8 block** | |
| `mlp.shared_expert.down_proj` | `[2048, 512]` | 40 | **FP8 block** | |
| routed experts `w13` | `[256, 1024, 2048]` | 40 | **FP8 block** | |
| routed experts `w2` | `[256, 2048, 512]` | 40 | **FP8 block** | |
| `lm_head` | `[248320, 2048]` | 1 | BF16 | not a `LinearBase` |
| attention QK^T / PV | — | 10 | BF16 | KV cache dtype `auto` |

Every block-quantized weight has `N % 128 == 0` and `K % 128 == 0`, so no `ceil` padding occurs
anywhere and all the kernels' divisibility preconditions hold (see §3.7 for what would happen
otherwise).

### 3.4 Scale storage, layout and consumption

**Weight scales.** `Fp8LinearMethod` creates the weight as a `ModelWeightParameter` fp8e4m3 `[N,K]`
row-major (`fp8_utils.py:1247-1264`, registered at `fp8.py:352-355`) and the scale as a
`BlockQuantScaleParameter` (`vllm/model_executor/parameter.py:397`) named **`weight_scale_inv`**
(`fp8.py:366-379`) with

```python
data = torch.empty((ceil(N/block_n), ceil(K/block_k)), dtype=torch.float32)
input_dim = 1, output_dim = 0
```

(`fp8_utils.py:1283-1296`; `dtype` defaults to fp32 at `fp8_utils.py:1276` and is fp8e8m0 only if
`is_scale_e8m0`, which is False for a plain `Fp8Config` — `fp8.py:282`, `:376`; init to
`finfo(float32).min`, `fp8_utils.py:1305-1306`). The checkpoint's BF16 scale tensors are up-cast
into these fp32 params on load. `input_scale` is **not** created — activations are dynamic
(`fp8.py:382-385` is gated on `act_q_static`). Block-shape validation runs at `fp8.py:340-350` →
`validate_fp8_block_shape` (`fp8_utils.py:1196-1244`); at TP=1 only the merged-GEMM `N % block_n`
check can fire, and every merged shard here is 128-aligned.

Semantics: `weight_scale_inv[i,j]` is the multiplicative dequant scale for the weight tile
`W[i*128:(i+1)*128, j*128:(j+1)*128]`, i.e. `W_real ≈ W_fp8 * weight_scale_inv`.

Shard fusion is a plain row concatenation because every shard's N is a multiple of 128:

| runtime param | built from | scale rows |
|---|---|---|
| `in_proj_qkvz.weight_scale_inv` `[96,16]` | `in_proj_qkv` `[64,16]` ‖ `in_proj_z` `[32,16]` | 8192/128 = 64, 4096/128 = 32 |
| `qkv_proj.weight_scale_inv` `[72,16]` | `q` `[64,16]` ‖ `k` `[4,16]` ‖ `v` `[4,16]` | 8192/128, 512/128, 512/128 |
| `shared_expert.gate_up_proj.weight_scale_inv` `[8,16]` | `gate` `[4,16]` ‖ `up` `[4,16]` | |

**Post-load processing.** `Fp8LinearMethod.process_weights_after_loading` (`fp8.py:398-445`) does
nothing for the block path beyond `if self.block_quant: assert not self.act_q_static`
(`:412-413`) and `layer.input_scale = None` (`:441-443`), then delegates to the selected kernel
(`:445`). The generic block base
(`kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:75-95`) calls
`process_fp8_weight_block_strategy` (`fp8_utils.py:1371-1405`), which on CUDA does only
`_maybe_pad_fp8_weight` — a no-op unless ROCm (`fp8_utils.py:1179-1188`). **On the selected B200
kernel (§3.5) there is no transpose, no padding, and no UE8M0 requantization; the weights and their
fp32 block scales stay in `[N,K]` / `[N/128, K/128]` row-major.**

**Activation scales.** `activation_scheme: dynamic` + block ⇒ per-token-group-of-128 quantization,
performed inside `apply_weights` on the 2-D view of the input
(`BlockScaledMMLinearKernel.py:116-122`) by `QuantFP8`
(`vllm/model_executor/layers/quantization/input_quant_fp8.py:30`). The CUTLASS kernel constructs it
as `QuantFP8(static=False, group_shape=(1,128), num_token_padding=None, use_ue8m0=False,
column_major_scales=True)` (`cutlass.py:279-285`). Scales are **fp32**, logical shape `[M, K/128]`
stored column-major (`fp8_utils.py:609-632`); the kernel is `torch.ops._C.per_token_group_fp8_quant`
(`fp8_utils.py:669-684`) with a Triton fallback (`:653-692`). Preconditions:
`x.shape[-1] % 128 == 0` (`:596-599`) and `x.stride(-1) == 1` (`:600`).

Because `use_ue8m0=False` is passed explicitly, the UE8M0 short-circuit at the top of
`QuantFP8.forward_cuda` (`input_quant_fp8.py:93-103`) is skipped and control reaches
`per_token_group_quant_fp8` (`fp8_utils.py:567`).

**Numerical contract — MPK must reproduce this primitive exactly.** For each group `g` of 128
contiguous elements of one token's activation row:

```
absmax = max( max_{i∈g} |x_i| , eps )          # eps = 1e-10, fp8_utils.py:570
scale  = absmax / fp8_max                      # fp8_max = 448.0 = finfo(float8_e4m3fn).max
                                               #   (quant_utils.py:27-35; the 224.0 branch is ROCm-fnuz only)
x_q    = clamp( x_i / scale, fp8_min, fp8_max ).to(float8_e4m3fn)   # fp8_min = -448.0
store scale (fp32) at [g, token]               # column-major, fp8_utils.py:625-629
```

Both implementations agree: Triton `_per_token_group_quant_fp8_colmajor`
(`fp8_utils.py:504-563`, quant math at `:556-563`) and the CUDA kernel used on a contiguous input
(`csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`: `local_absmax` seeded to
`eps` at `:47`, reduced at `:54`/`:66`, `y_s = local_absmax / max_8bit` at `:68`,
`q = fminf(fmaxf(x / y_s, min_8bit), max_8bit)` at `:85`).

Four points that silently break bit-parity if implemented differently:

1. **`eps` seeds the max, it does not clamp the scale.** An all-zero group gets
   `scale = 1e-10/448 ≈ 2.2e-13`, not `0` and not `1.0`; every element then quantizes to `0`
   anyway, but the *stored scale* differs from a `scale=1` convention and will propagate into the
   dequant.
2. **Division by the scale, not multiplication by a reciprocal.** The kernels compute `x / y_s`;
   an `x * (1/y_s)` formulation differs in the last ulp.
3. **Saturating clamp before the cast.** `clamp` to `[-448, 448]` happens in fp32 *before* the
   `to(float8_e4m3fn)`, so an out-of-range value saturates rather than becoming `inf`/`NaN`. With
   `scale = absmax/448` the clamp is normally inactive, but it is load-bearing for the ±448 endpoint.
4. **Round-to-nearest-even on the fp32→e4m3 cast** — the hardware `cvt` default in both the Triton
   `.to()` and the CUDA `__nv_fp8` conversion. Neither path uses stochastic or truncating rounding.

The same primitive (`group_size=128`, dynamic, fp32 scales) quantizes the MoE activation before the
FlashInfer call, via `_quantize_input` in `prepare_finalize/no_dp_ep.py:117-125`; the only
difference is that its scales are transposed to `[K/128, M]` at `trtllm_fp8_moe.py:405`.

The GEMM itself:

```python
def apply_block_scaled_mm(self, A, B, As, Bs):
    return ops.cutlass_scaled_mm(A, B.T, out_dtype=self.config.out_dtype,
                                 scale_a=As, scale_b=Bs.T)     # cutlass.py:311-326
```

so `A` is `[M,K]` fp8, `B.T` is `[K,N]` fp8, `As` is the per-token-group activation scale and `Bs.T`
the transposed weight block scale; the output is **bf16** (`out_dtype = torch.get_default_dtype()`,
`fp8.py:284`), with the bias add and the final `.to(out_dtype).view(output_shape)` at
`BlockScaledMMLinearKernel.py:139-141`.

### 3.5 Dense-linear kernel selection on B200 — and the Qwen3.5 DeepGEMM auto-disable

Selection is **static, at `create_weights()` time**: `Fp8LinearMethod.create_weights` calls
`init_fp8_linear_kernel(...)` (`fp8.py:387-394`), which for a per-group activation scale takes the
block branch (`kernels/linear/__init__.py:597-601`) and runs `choose_scaled_mm_linear_kernel`
(`__init__.py:508`) over `_POSSIBLE_FP8_BLOCK_KERNELS` (`__init__.py:355-365`), CUDA priority order:

```
[ FlashInferFp8DeepGEMMDynamicBlockScaledKernel,   # __init__.py:359
  DeepGemmFp8BlockScaledMMKernel,                  # :360
  CutlassFp8BlockScaledMMKernel,                   # :361
  MarlinFP8ScaledMMLinearKernel,                   # :362
  TritonFp8BlockScaledMMKernel,                    # :363
  HummingFP8ScaledMMLinearKernel ]                 # :364
```

Evaluation for `Qwen3.5-35B-A3B-FP8` on B200, default flags:

| candidate | verdict | reason |
|---|---|---|
| FlashInfer block-scale | **rejected** | `has_flashinfer_fp8_blockscale_gemm()` requires `current_platform.is_device_capability(90)` — Hopper-**exact** (`vllm/utils/flashinfer.py:938-944`). Never true on sm100. |
| **DeepGEMM** | **rejected** | `should_auto_disable_deep_gemm(model_type)` — see below |
| **CUTLASS block-scaled** | **SELECTED** | `is_supported` needs only `CUTLASS_BLOCK_FP8_SUPPORTED` (`cutlass.py:288-294`), which for cc ≥ 100 requires the build to have `CUDA_VERSION >= 12080` (`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:161-174`); `can_implement` needs only activation `GroupShape(1,128)` (`cutlass.py:303-309`) |
| Marlin | rejected | on cc ≥ 89 requires `VLLM_TEST_FORCE_FP8_MARLIN=1` (default `"0"`), `marlin.py:46-55`. Note this env var only restores *eligibility* — the list order is fixed and CUTLASS is still evaluated first, so setting it does not move Marlin ahead of CUTLASS on the dense path |
| Triton block | (reachable floor) | `is_supported` = CUDA-alike (`triton.py:160-164`); would win only if CUTLASS were unavailable |
| Humming | last | `humming.py:26-42` |

**The DeepGEMM rejection is model-specific and load-bearing for benchmark fairness.**

```python
_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}                                                   # vllm/utils/deep_gemm.py:27-30

def should_auto_disable_deep_gemm(model_type):       # vllm/utils/deep_gemm.py:33-46
    if model_type is None: return False
    if not (current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)): return False
    return model_type in _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES
```

with the docstring *"Returns True if the model is known to have accuracy degradation with DeepGemm's
E8M0 scale format on Blackwell GPUs (SM100+)"*, consumed by
`DeepGemmFp8BlockScaledMMKernel.can_implement` at
`kernels/linear/scaled_mm/deep_gemm.py:73-76`. Our `hf_text_config.model_type` is
`"qwen3_5_moe_text"` and the device family is 100 ⇒ **DeepGEMM is disabled for every dense fp8
linear in this model on B200**, and with it the UE8M0 activation-scale path
(`input_quant_fp8.py:93-103`) and the `requant_weight_ue8m0_inplace` weight rewrite
(`fp8_utils.py:1113-1114`). vLLM runs CUTLASS with plain fp32 scales instead. This is *deliberate
accuracy protection*, not a bug — but it means **"vLLM FP8 on B200" for this model is CUTLASS
block-scaled, not DeepGEMM**, and MPK should be compared against that.

Env defaults referenced above: `VLLM_USE_DEEP_GEMM=True` (`envs.py:185`),
`VLLM_USE_DEEP_GEMM_E8M0=True` (`envs.py:187`),
`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=True` (`envs.py:197`), `VLLM_BATCH_INVARIANT=False`
(`envs.py:89`), `VLLM_DISABLED_KERNELS=[]` (`envs.py:123`), `VLLM_TEST_FORCE_FP8_MARLIN="0"`
(`envs.py:1066-1067`).

### 3.6 MoE FP8 at 256 experts / intermediate 512

The routed experts go through `Fp8MoEMethod` (`vllm/model_executor/layers/quantization/fp8.py:492`),
reached from `Fp8Config.get_quant_method` at `fp8.py:210-211` via `RoutedExperts._get_quant_method`
(`fused_moe/routed_experts.py:199`). It is an **entirely separate oracle** from the dense-linear one
of §3.5 — different priority list, different winner.

#### 3.6.1 Weights and scales

`Fp8MoEMethod.__init__` sets `weight_key = kFp8Static128BlockSym`,
`activation_key = kFp8Dynamic128Sym` (`fp8.py:515-517`) and immediately resolves the backend
(`fp8.py:527-532`). `create_weights` (`fp8.py:534-672`) validates:

```python
if intermediate_size_per_partition % block_n != 0:     # 512 % 128 == 0  -> OK
    raise ValueError("The output_size of gate's and up's weight = ... "
                     "is not divisible by weight quantization block_n = ...")
if tp_size > 1 and intermediate_size_per_partition % block_k != 0:   # skipped at TP=1
    raise ValueError("The input_size of down's weight = ...")        # fp8.py:562-574
```

then creates plain `torch.nn.Parameter(requires_grad=False)` tensors (**not**
`ModelWeightParameter`/`BlockQuantScaleParameter`, so no `input_dim`/`output_dim` attributes; loading
is driven by `RoutedExperts.weight_loader` plus the `quant_method: "block"` attr set at
`fp8.py:647-653`):

| param | code | shape | dtype |
|---|---|---|---|
| `w13_weight` | `fp8.py:577-586` | `[256, 1024, 2048]` | `float8_e4m3fn` (forced at `fp8.py:548`) |
| `w2_weight` | `fp8.py:589-598` | `[256, 2048, 512]` | `float8_e4m3fn` |
| `w13_weight_scale_inv` | `fp8.py:627-632`, registered `:642` | `[256, 2·⌈512/128⌉, ⌈2048/128⌉] = [256, 8, 16]` | `float32` |
| `w2_weight_scale_inv` | `fp8.py:633-638`, registered `:643` | `[256, ⌈2048/128⌉, ⌈512/128⌉] = [256, 16, 4]` | `float32` |
| `w13_input_scale` / `w2_input_scale` | `fp8.py:670-672` | not created (dynamic) | — |

The `_inv` suffix comes from `self.weight_scale_name = "weight_scale_inv" if self.block_quant`
(`fp8.py:510-512`).

#### 3.6.2 Backend selection — the winner is FlashInfer TRT-LLM

`select_fp8_moe_backend` (`fused_moe/oracle/fp8.py:271`), priority list
`_get_priority_backends` (`oracle/fp8.py:69`, base ordering `:80-95`):

```
[AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, DEEPGEMM, VLLM_CUTLASS, TRITON,
 MARLIN, HUMMING, BATCHED_DEEPGEMM, BATCHED_VLLM_CUTLASS, BATCHED_TRITON, XPU, CPU, HPC]
```

None of the three re-ordering hooks fires: the sm100 hook needs `use_deepep_v2_kernels`
(`oracle/fp8.py:103-110`), the "prefer Triton for TP / FI-CUTLASS for EP on Hopper" hook needs
`is_device_capability(90)` (`:112-122`), and XPU/CPU do not apply (`:124-131`).
`VLLM_CUTLASS`/`BATCHED_VLLM_CUTLASS` are removed because `Fp8MoEMethod` passes
`allow_vllm_cutlass=False` (`fp8.py:531`, removal at `oracle/fp8.py:390-392`). The loop at
`oracle/fp8.py:394-408` returns the first `k_cls.is_supported_config(...)` that passes.

| # | backend | verdict on B200 / this shape | why |
|---|---|---|---|
| 1 | `AITER` | fail | ROCm only |
| 2 | **`FLASHINFER_TRTLLM`** | **WINNER** → `TrtLlmFp8ExpertsMonolithic` | device `is_cuda() ∧ is_device_capability_family(100) ∧ has_flashinfer_trtllm_fused_moe()` (`experts/trtllm_fp8_moe.py:95-103`) ✓; quant scheme `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` ✓; activation SILU (`:111-117`) ✓; parallel — no all2all, no EPLB, no SP (`:120-127`) ✓; routing `RenormalizeNaive` in the allow-list ✓; router-logits dtype bf16 ✓ |
| 3 | `FLASHINFER_CUTLASS` | fail | block-fp8 is admitted **only** with `p.is_device_capability(90)` — Hopper-exact (`experts/flashinfer_cutlass_moe.py:166-174`) |
| 4 | `DEEPGEMM` | would pass | `TritonOrDeepGemmExperts`; the runner-up |
| 6 | `TRITON` | would pass | `TritonExperts` (`experts/triton_moe.py:94-138`) |
| 7 | `MARLIN` | would pass | but as **W8A16** (fp8 weights, bf16 activations), `oracle/fp8.py:594-604` |

`backend_to_kernel_cls` lists `TrtLlmFp8ExpertsMonolithic` **before** `TrtLlmFp8ExpertsModular`
(`oracle/fp8.py:139-145`), so the router-fused monolithic variant wins. `MoERunner` then takes the
monolithic branch — `if self.routed_experts.quant_method.is_monolithic: forward_monolithic(x,
router_logits, ...)` (`runner/moe_runner.py:564-570`) — so **the router GEMM's output goes straight
into the kernel and vLLM's own `topk_softmax` is never called**. The shared expert is still launched
on the aux stream by the `_maybe_apply_shared_experts` calls that bracket the branch
(`moe_runner.py:560-562`, `:588-591`).

Note the asymmetry with §3.5: `should_auto_disable_deep_gemm` is consulted **only** by the dense
linear kernel (`kernels/linear/scaled_mm/deep_gemm.py:75`) and by
`VllmConfig.__post_init__`, which sets `quant_config.use_deep_gemm = False` and logs *"Auto-disabled
DeepGemm for model_type=%s on Blackwell … Falling back to CUTLASS"* (`vllm/config/vllm.py:1002-1019`).
The MoE oracle keys its DeepGEMM removal on the **env var** `VLLM_USE_DEEP_GEMM` /
`VLLM_MOE_USE_DEEP_GEMM` (`oracle/fp8.py:358-371`), not on `quant_config.use_deep_gemm`. It does not
matter at default settings because FlashInfer TRT-LLM wins ahead of DeepGEMM anyway — but it does
mean **the MoE and the dense linears end up on different vendors' kernels**.

#### 3.6.3 Weight relayout for the winning kernel

`process_weights_after_loading` (`fp8.py:720-763`) skips FNUZ conversion, the static-activation
branch, and the per-tensor requant (`if not self.block_quant:`, `fp8.py:754`), then calls
`_setup_kernel` (`:761`) → `convert_to_fp8_moe_kernel_format` (`fused_moe/oracle/fp8.py:457`) →
`prepare_fp8_moe_layer_for_fi(..., is_trtllm=True)`
(`quantization/utils/flashinfer_utils.py:430`). For block quant that does exactly three things:

1. **No FI alignment padding** of the 512 intermediate — the `align_moe_weights_for_fi` call is
   guarded by `if not block_quant:` (`flashinfer_utils.py:481-490`).
2. **W13 → W31 swap** of both weights and scales — FlashInfer wants `[up; gate]`, vLLM stores
   `[gate; up]` (`flashinfer_utils.py:493-496`, `swap_w13_to_w31` at `:39-43`). The swap is
   ```python
   x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
   ```
   — the leading expert axis is folded into the `-1`, so **the expert axis is untouched** and the
   `flip` exchanges the two halves of the **W13 output axis** (`x.shape[-2]`), per expert:

   | tensor | axis that flips | before → after |
   |---|---|---|
   | `w13_weight [256, 1024, 2048]` | axis 1 (output rows) | rows `[0:512)`=gate,`[512:1024)`=up → `[0:512)`=up,`[512:1024)`=gate |
   | `w13_weight_scale_inv [256, 8, 16]` | axis 1 (output **block** rows) | blocks `[0:4)`=gate,`[4:8)`=up → `[0:4)`=up,`[4:8)`=gate |

   Do **not** read this as a permutation of the first (expert) axis — that would mis-assign every
   expert's scales. `w2`/`w2_scale` are not swapped (single output block, `is_act_and_mul` guard at
   `:493`).
3. **BlockMajorK 4-D reshuffle of the weights only** (`flashinfer_utils.py:499-500` →
   `_shuffle_deepseek_fp8_moe_weights`, `:319-352`): per expert, `shuffle_matrix_a(...,
   epilogue_tile_m=64)` then `convert_to_block_layout(..., block_k=128)`, producing
   `(E, K/128, Mn, 128)`:

   ```
   w13_weight : [256, 1024, 2048] -> [256, 16, 1024, 128]   fp8e4m3
   w2_weight  : [256, 2048,  512] -> [256,  4, 2048, 128]   fp8e4m3
   ```

4. **Scales are clamped, not relayouted**: `w13_scale.clamp_(min=1e-10)`,
   `w2_scale.clamp_(min=1e-10)` (`flashinfer_utils.py:514-519`) — a guard against near-zero block
   scales on dead experts producing NaN. They keep their fp32 3-D shapes `[256,8,16]` /
   `[256,16,4]`; the only change is the gate/up half-swap along `w13_scale`'s **middle** axis from
   step 2. **No TMA column-major transform, no UE8M0 requantization** on this path. (The DeepGEMM
   MoE path *would* do both — `fp8_utils.py:1113-1114` and `:1080-1087` — which is another reason
   the two backends are not interchangeable numerically.)

#### 3.6.4 The kernel call

`TrtLlmFp8ExpertsMonolithic._apply_block_scale` (`experts/trtllm_fp8_moe.py:356-445`):

```python
assert self.topk <= global_num_experts          # 8 <= 256                    :383
assert global_num_experts % 4 == 0              # 256                         :384
assert self.quant_config.block_shape in [[128, 128], [1, 32]]                 :385
assert global_num_experts <= 512   # "#experts <= #threads 512"               :387
assert a1q_scale is not None                                                  :389
# block-fp8 (non-MXFP8) branch:
assert self.topk <= 32                                                        :401
fp8_quant_type      = Fp8QuantizationType.DeepSeekFp8
use_shuffled_weight = True
weight_layout       = WeightLayout.BlockMajorK                                :403-404
hidden_states_scale = a1q_scale.t().contiguous()      # -> [K/128, M]         :405

result = flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
    routing_logits=router_logits,          # bf16 [M,256], straight from mlp.gate
    routing_bias=None,
    hidden_states=hidden_states,           # fp8e4m3 [M,2048]
    hidden_states_scale=hidden_states_scale,
    gemm1_weights=w1, gemm1_weights_scale=w13_weight_scale_inv,   # [256,8,16] fp32
    gemm2_weights=w2, gemm2_weights_scale=w2_weight_scale_inv,    # [256,16,4] fp32
    num_experts=256, top_k=8, n_group=0, topk_group=0,
    intermediate_size=512, local_expert_offset=0, local_num_experts=256,
    routing_method_type=RoutingMethodType.RenormalizeNaive,   # = 4
    use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK,
    fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
    tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config))               :414-443
```

`activation_type` is deliberately **not** passed for plain-SiLU block-fp8 (`:441-442`).

The activation is quantized before the call by
`MoEPrepareAndFinalizeNoDPEPMonolithic.prepare` → `_quantize_input(a1, quant_config)`
(`prepare_finalize/no_dp_ep.py:117-125`) with `block_shape=[128,128]`, giving fp32 per-token-group
scales that the transpose at `:405` turns into `[K/128, M]` column-major.

So one FlashInfer call performs: router softmax-256 → top-8 → renormalize → permute → grouped
block-fp8 GEMM1 → SwiGLU → grouped block-fp8 GEMM2 → weighted reduce. Ops 2–3 and 8–12 of the
§2.3.5 table are all inside it.

#### 3.6.5 Triton tuning table (only reached if FlashInfer is unavailable)

The repo ships
`fused_moe/configs/E=256,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json`
— an exact match for our shape. `try_get_optimal_moe_config` (`fused_moe.py:1363-1392`) derives
`E, _, N = w2_shape` (so N = 512, `:1378`) and picks the nearest M key
(`configs[min(configs.keys(), key=lambda x: abs(x - M))]`, `:1388`):

| M | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_M | warps | stages |
|---|---|---|---|---|---|---|
| 1 | 16 | 128 | 128 | 1 | 4 | 4 |
| 2 / 4 / 8 | 16 | 128 | 128 | 1 | 4 | 3 |
| 16 | 16 | 128 | 128 | **64** | 4 | 3 |

**This file is dead weight on the default path** — FlashInfer TRT-LLM never consults it. It is still
the best available prior for tile selection in an MPK expert GEMM at these shapes. If the file were
missing, `get_default_config` (`fused_moe.py:1240-1294`) would give `BLOCK_M=16, BLOCK_N=64`
(because `M ≤ 8 ∧ block_n % 64 == 0`), `BLOCK_K=128`, `GROUP_M=1`, `warps=4`,
`stages = 4 if M ≤ 4 else 3`, after logging *"Using default MoE config. Performance might be
sub-optimal!"* (`fused_moe.py:1108-1112`).

### 3.7 Where vLLM would fall off FP8 — the fairness list

AC-4/AC-5 require comparing mpk-FP8 against vLLM-FP8. The list below is what to assert about the
baseline before trusting a number. It splits into conditions that **crash** (safe — you will notice)
and conditions that **silently change precision or kernel** (dangerous).

#### 3.7.1 There is no silent dequant-to-bf16 for this checkpoint

* **Dense linear.** The only bf16-dequant branch in `Fp8LinearMethod.apply`
  (`fp8.py:446-488`) is gated on `envs.VLLM_BATCH_INVARIANT` **and** explicitly routes block-quant
  straight back to the fp8 kernel (`fp8.py:454-461`); the dequant at `:466-487` is reachable only for
  non-block per-tensor/channel fp8. If no block kernel can be constructed, vLLM raises
  `ValueError("Failed to find a kernel that can implement the ScaledMM linear layer. Reasons: …")`
  (`kernels/linear/__init__.py:574-577`) rather than falling back.
* **MoE.** `Fp8MoeBackend.EMULATION` (the dequant backend, `oracle/fp8.py:58-60`) is MXFP8-only and
  appears in neither `_AVAILABLE_BACKENDS` (`:80-95`) nor `map_fp8_backend` (`:252-262`). If nothing
  supports the config, `raise NotImplementedError("No FP8 MoE backend supports the deployment
  configuration.")` (`oracle/fp8.py:414-417`).

So a vLLM run that starts and serves this checkpoint **is** doing fp8 GEMMs for every GEMM listed as
FP8 in §3.3. The risk is not precision loss; it is **which fp8 kernel**.

#### 3.7.2 Hard errors (loud)

| condition | cite |
|---|---|
| `weight_block_size` present but checkpoint not fp8-serialized / `len != 2` / `activation_scheme != "dynamic"` | `fp8.py:115-131` |
| Only *some* shards of a fused module listed in `modules_to_not_convert` | `quant_utils.py:551-559` |
| `intermediate_size_per_partition % 128 != 0` (would need TP ≥ 8: 512/8 = 64) | `fp8.py:562-567` |
| `--linear-backend=X` where X has no block kernel | `kernels/linear/__init__.py:557-564` |
| `--moe-backend=cutlass` (vLLM CUTLASS is disabled for fp8 MoE) | `oracle/fp8.py:342-352` |
| `--moe-backend=emulation` (not in the fp8 mapping) | `oracle/fp8.py:250-268` |
| `--moe-backend=flashinfer_cutlass` on sm100 with block-fp8 | `_return_or_raise`, `oracle/fp8.py:313-327` |

#### 3.7.3 Silent kernel demotion — dense linear (§3.5 tree)

| trigger | effect | default |
|---|---|---|
| `should_auto_disable_deep_gemm("qwen3_5_moe_text")` on sm100/sm120 | DeepGEMM rejected → **CUTLASS**. *Always active for us.* | n/a (`utils/deep_gemm.py:27-46`) |
| not Blackwell-exact-90 | FlashInfer block-scale rejected (needs `is_device_capability(90)`) | n/a (`utils/flashinfer.py:938-944`) |
| build without CUDA ≥ 12.8 | `CUTLASS_BLOCK_FP8_SUPPORTED` False → falls to **Triton** `w8a8_triton_block_scaled_mm` | `scaled_mm_entry.cu:161-174` |
| `VLLM_TEST_FORCE_FP8_MARLIN=1` | **Makes Marlin *eligible*; does not select it.** The env var only lifts the cc ≥ 89 disqualification inside `MarlinFP8ScaledMMLinearKernel.is_supported` (`marlin.py:46-55`). There is no reorder or force-select on the dense path: `choose_scaled_mm_linear_kernel` walks `_POSSIBLE_FP8_BLOCK_KERNELS` in fixed order (`__init__.py:566-572`) and CUTLASS precedes Marlin (`__init__.py:361` vs `:362`). Marlin wins **only if CUTLASS also fails** — i.e. a build without CUDA ≥ 12.8 — so on a stock B200 build the selection is unchanged | `"0"` (`envs.py:1066-1067`) |
| `VLLM_BATCH_INVARIANT=1` | Marlin excluded (`marlin.py:44-45`); block path still fp8, still CUTLASS | `False` (`envs.py:89`) |
| `VLLM_DISABLED_KERNELS` lists a kernel class name | that candidate skipped | `[]` (`envs.py:123`) |
| `--linear-backend != auto` | list filtered to that backend | `auto` |
| running fp16 instead of bf16 | DeepGEMM's `out_dtype != bfloat16` check rejects it (moot here — already rejected) | model dtype bf16 |
| **N % 64 != 0 or K % 128 != 0** | DeepGEMM rejected (`utils/deep_gemm.py:700-719`). Every weight in §3.3 satisfies both, so this never fires for us | — |

#### 3.7.4 Silent kernel demotion — MoE (§3.6 tree)

| trigger | effect | default |
|---|---|---|
| `flashinfer` not importable | TRTLLM out → **DEEPGEMM** | pinned `flashinfer-python==0.6.15.post1` (`requirements/cuda.txt:17-18`) |
| flashinfer present but no cubins and no `nvcc` | `has_flashinfer()` False → DEEPGEMM | `VLLM_HAS_FLASHINFER_CUBIN=False` (`envs.py:241`) |
| flashinfer missing **any one** of `trtllm_fp8_block_scale_moe`, `trtllm_fp8_per_tensor_scale_moe`, `trtllm_fp4_block_scale_moe`, `trtllm_mxint4_block_scale_moe`, `trtllm_bf16_moe` | `has_flashinfer_trtllm_fused_moe()` False → DEEPGEMM. **Availability is symbol presence, not a version assert** | `utils/flashinfer.py:244-259` |
| not sm100 family | TRTLLM out; on Hopper the priority hook moves **TRITON** (TP) or **FI-CUTLASS** (EP) to the front | `oracle/fp8.py:112-122` |
| EPLB, sequence-parallel MoE, or all2all (DP/EP) enabled | `TrtLlmFp8ExpertsBase._supports_parallel_config` fails (`trtllm_fp8_moe.py:120-127`); the loop continues down `_AVAILABLE_BACKENDS` and the winner becomes **DEEPGEMM** (`TritonOrDeepGemmExperts`) for plain EPLB/SP, or **BATCHED_DEEPGEMM** once a batched all2all format is in play. Which one actually wins depends on the remaining config — **assert it from the run's backend log line, do not assume** | off |
| LoRA enabled | TRTLLM has no `LoRAExpertsMixin` → next passing backend, in practice **DEEPGEMM**, and **TRITON** at decode via the runtime sub-fallback below | off |
| `VLLM_USE_DEEP_GEMM=1` **explicitly set** | force-selects **DEEPGEMM**, bypassing the priority loop and the faster TRTLLM (`oracle/fp8.py:358-371`) | unset (`envs.py:185`) |
| `VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0` explicitly set | DEEPGEMM + BATCHED_DEEPGEMM removed from the list (`oracle/fp8.py:360-362`); TRTLLM still wins at defaults, so this only matters when TRTLLM is already out, where the winner becomes **TRITON** | unset |
| `VLLM_TEST_FORCE_FP8_MARLIN=1` | **force-selects MARLIN**, bypassing the priority loop (`oracle/fp8.py:373-378` returns immediately). Unlike the dense path this *is* a genuine force-select, and Marlin is **W8A16** — fp8 weights, **bf16 activations**. The one setting that changes the MoE's arithmetic precision | `"0"` |
| `VLLM_BATCH_INVARIANT=1` | only `TritonExperts` declares batch invariance (`triton_moe.py:147-149`) → **TRITON**, plus `get_moe_configs` forced to `None` (`fused_moe.py:1068-1070`) | `False` |
| DeepEP-LL / NIXL-EP | batched activation format → **BATCHED_DEEPGEMM**, or **BATCHED_TRITON** if DeepGEMM is out | off |

Every row above changes *which* kernel runs; only the Marlin row changes the *precision*. When any
of them is live the winner is whatever the priority loop reaches next, which depends on the rest of
the config — so §3.7.5 asks for the backend to be read out of the run log rather than inferred.

**Runtime sub-fallback inside DEEPGEMM** (relevant only if TRTLLM is unavailable):
`TritonOrDeepGemmExperts._select_experts_impl` (`experts/triton_deep_gemm_moe.py:77-84`) is

```python
if is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2):
    return self.experts            # DeepGEMM
return self.fallback_experts       # Triton
```

and `_valid_deep_gemm` (`experts/deep_gemm_moe.py:58-121`) would reject our shape twice over:
`align <= M` fails for every decode batch M ∈ {1..16} (`:53-55`), and there is an explicit
**`elif N <= 512: return False`** with the log *"DeepGemm disabled for N ≤ 512 … we will fallback to
triton"* (`:88-97`) — our `N` is exactly 512. But the `is_deep_gemm_e8m0_used()` short-circuit
(default **True** on B200: `VLLM_USE_DEEP_GEMM_E8M0=True`, `envs.py:187`, and
`support_deep_gemm()` True for sm100, `platforms/cuda.py:665-671`) **bypasses both checks**, because
the weights were already UE8M0-requantized in place at load and Triton would be numerically wrong.
Set `VLLM_USE_DEEP_GEMM_E8M0=0` and every decode step falls to Triton.

#### 3.7.5 What the benchmark must pin

To claim "mpk-FP8 beat vLLM-FP8 at its best standard config", record from the vLLM run and assert:

1. `flashinfer` importable **and** `has_flashinfer_trtllm_fused_moe()` True — otherwise the MoE ran
   on DeepGEMM/Triton, which is not vLLM's best config on B200.
2. The selected MoE backend from vLLM's own log line (`logger.info_once(_make_log_backend(backend))`,
   `oracle/fp8.py:404`) reads `FLASHINFER_TRTLLM`.
3. The dense-linear kernel is `CutlassFp8BlockScaledMMKernel`, and vLLM emitted the
   *"Auto-disabled DeepGemm for model_type=qwen3_5_moe_text on Blackwell"* warning
   (`config/vllm.py:1013-1019`) — its absence means the DeepGEMM exclusion list changed and the
   comparison basis moved.
4. `VLLM_BATCH_INVARIANT`, `VLLM_TEST_FORCE_FP8_MARLIN`, `VLLM_USE_DEEP_GEMM`,
   `VLLM_MOE_USE_DEEP_GEMM`, `VLLM_DISABLED_KERNELS`, `--moe-backend`, `--linear-backend`,
   `--kv-cache-dtype` all left at defaults; no LoRA; TP=EP=DP=1.
5. `--language-model-only` **not** set (that would enable the fused QK-norm+RoPE+gate kernel and make
   vLLM slightly faster than the config we measured) — or set it on both sides and say so.
6. `mamba_ssm_cache_dtype` resolved to `float32` (from the checkpoint), not overridden on the CLI.
7. The KV cache is bf16 (`--kv-cache-dtype auto`). FP8 KV cache would be a *different* precision
   contract from AC-1 and must not be enabled on either side.

---

## 4. KV / state cache and the hybrid allocator

The model declares itself `IsHybrid` (`qwen3_5.py:399`) and supplies the mamba spec via classmethods
(`qwen3_5.py:522-557`). Two coexisting cache kinds.

### 4.1 GDN state — 30 layers

| tensor | stored shape | as the kernels see it | dtype | bytes |
|---|---|---|---|---|
| conv state | `(3, 8192)` (SD, default) | `(8192, 3)` via `.transpose(-1,-2)` | bf16 | 48 KiB |
| recurrent state `S` | `(32, 128, 128)` | same | **fp32** (`mamba_ssm_dtype`) | **2 MiB** |

`num_spec > 0` (MTP) widens the conv state to `(3 + num_spec, 8192)` (`mamba_utils.py:257-261`).

### 4.2 Full-attention paged KV — 10 layers

`Attention(num_heads=16, head_size=256, scale=256**-0.5, num_kv_heads=2)`
(`qwen3_next.py:302-317`) → `FullAttentionSpec(block_size, num_kv_heads=2, head_size=256,
head_size_v=256, dtype=bf16)` (`layers/attention/attention.py:686-691`). The FlashInfer cache packs
K and V in the last dim (`v1/attention/backends/flashinfer.py:396-408`):

```
(num_blocks, num_kv_heads, kernel_block_size, 2*head_size) = (N, 2, 32, 512)  bf16
                                                              K = [...,:256], V = [...,256:]
```

On sm100 FlashInfer forces the HND layout with an identity stride order (`flashinfer.py:410-433`,
`:492-496`); K/V are split zero-copy at `flashinfer.py:1851`. `head_size=256` is supported:
`get_supported_head_sizes() -> [64,128,256,512]` (`flashinfer.py:456-458`).

### 4.3 Page-size alignment — the block size is 1056, not 544

The hybrid allocator requires identical `page_size_bytes` across all KV cache groups
(`v1/core/kv_cache_utils.py:1169-1198`). A mamba page does not scale with `block_size`, so
`Platform._align_hybrid_block_size` (`platforms/interface.py:764-934`) raises the attention block
size until it matches, then pads the mamba page up to it. **With the fp32 recurrent state:**

```
attn_page_size_1_token  = 1 * 2 * (256+256) * 2 B                       = 2,048 B
mamba_page_size         = 3*8192*2  +  32*128*128*4                     = 2,146,304 B
kernel_block_alignment  = max(min(backend.get_supported_kernel_block_sizes()),
                              cache_config.block_size)                  = 16      # :874-882
attn_block_size         = 16 * cdiv(2_146_304, 16*2048)                 = 1056    # :898-901
cache_config.mamba_block_size = cache_config.block_size                 = 1056    # :911-912
mamba_page_size_padded  = 1056 * 2048                                   = 2,162,688 B (0.76 % pad)
```

| ssm dtype | mamba page | KV-manager `block_size` | padded page | pad |
|---|---|---|---|---|
| **fp32 (this checkpoint)** | **2,146,304 B** | **1056** | **2,162,688 B** | **0.76 %** |
| bf16 (hypothetical) | 1,097,728 B | 544 | 1,114,112 B | 1.49 % |

On B200 FlashInfer advertises large pages — `get_supported_kernel_block_sizes()` returns
`[16,32,64,128,256,512,1024]` because `num_qo_heads//num_kv_heads = 8 > 1`, device family 100, and
`can_use_trtllm_attention` (`flashinfer.py:353-373`). `select_common_block_size`
(`v1/worker/utils.py:250-315`) then picks the largest advertised **divisor** of 1056; `1056 = 2⁵·3·11`
so 1024/512/256/128/64 all fail and the answer is **kernel_block_size = 32**, giving
`num_blocks_per_kv_block = 1056/32 = 33` (`v1/worker/gpu_model_runner.py:7384-7387`).

**Grouping:** with 10 full-attn and 30 GDN layers, `get_kv_cache_groups`
(`kv_cache_utils.py:1224-1258`) yields **4 KV cache groups** — one attention group of 10 layers plus
three GDN groups of 10 layers each — so one attention layer and three GDN layers share each physical
slab, disambiguated by per-group block tables. Mamba layers get a raw
`[num_blocks, 1, 1, page_size_bytes]` **int8** view (`gpu_model_runner.py:7429-7440`) which
`MambaBase.bind_kv_cache` (`layers/mamba/abstract.py:29-43`) slices into `(num_blocks, 3, 8192)` and
`(num_blocks, 32, 128, 128)`; the pad bytes are never touched.

### 4.4 `mamba_cache_mode` — `align` is the only supported mode

`MambaCacheMode = Literal["all","align","none"]` (`vllm/config/cache.py:38`).
`MambaModelConfig.verify_and_update_config` (`models/config.py:558-602`) resolves
`none → "all" if the model supports mamba prefix caching else "align"`; Qwen3.5 does not implement
`SupportsMambaPrefixCaching`, so it lands on **`align`**, and `qwen3_5.py:307-311` hard-rejects
`"all"`. Align keeps only `page_size_bytes * (2 + num_speculative_blocks)` resident per request
(`v1/kv_cache_interface.py:709-730`), with earlier block-table entries nulled to
`NULL_BLOCK_ID = 0`.

**Decode cost:** a decode step copies one mamba page (block N → N+1) only when the sequence crosses a
`block_size = 1056` boundary; otherwise `preprocess_mamba`
(`v1/worker/mamba_utils.py:992-1055`, called from `gpu_model_runner.py:4313-4339`) is a no-op and the
GDN kernels update state in place.

### 4.5 Decode attention backend

`CudaPlatform._get_backend_priorities` (`platforms/cuda.py:143-155`) for capability major 10,
non-MLA, causal returns `[FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]`.
FlashInfer passes every gate (head_size 256, bf16, `kv_cache_dtype="auto"`, sm100), so
**`FlashInferBackend` is selected**. The decode kernel is the **TRT-LLM-gen** one:
`can_use_trtllm_attention(num_qo_heads=16, num_kv_heads=2)` is True
(`vllm/utils/flashinfer.py:411-419`), `_get_flashinfer_trtllm_api_decode_kernel` returns
`TRTLLM_GEN` on sm100, and forward dispatches to

```python
trtllm_batch_decode_with_kv_cache(query, kv_cache, workspace, block_tables, seq_lens,
                                  max_seq_len, bmm1_scale, bmm2_scale, window_left,
                                  sinks=None, kv_layout="HND", backend="trtllm-gen",
                                  q_len_per_req)          # flashinfer.py:2214-2236
```

`flash_attn_varlen_func` is not used in this configuration. Cascade attention is disabled
unconditionally (`flashinfer.py:1518-1525`). The pinned FlashInfer is
`flashinfer-python==0.6.15.post1` (`requirements/cuda.txt:17`).

### 4.6 `GDNAttentionMetadata` — how the batch is split

`v1/attention/backends/gdn_attn.py:41-79` (fields), `:168-513` (builder). Decode-relevant:

| field | decode meaning |
|---|---|
| `num_prefills`, `num_prefill_tokens` | 0 in a pure-decode step |
| `num_decodes`, `num_decode_tokens` | non-spec decodes; equal (1 token each) |
| `num_actual_tokens` | includes CUDA-graph padding |
| `non_spec_state_indices_tensor` | `[batch − num_spec_decodes]` — **the per-request state block id used by both decode kernels** |
| `spec_sequence_masks` | `None` ⇒ no spec decode ⇒ fast path |

The batch is reordered decode-first (`reorder_batch_threshold = 1`, `gdn_attn.py:85`). For `align`
mode `mamba_get_block_table_tensor` (`v1/attention/backends/utils.py:927-965`) gathers columns
`[(seq_len−1)//block_size … +num_spec]`, so `non_spec_state_indices_tensor = block_table[:,0]` is the
current running-state block. CUDA-graph padding fills the tail with `NULL_BLOCK_ID = 0` and both
decode kernels skip rows with `state_idx <= 0`. `_cudagraph_support =
AttentionCGSupport.UNIFORM_BATCH` (`gdn_attn.py:83`); capture is decode-only. If both non-spec and
spec decodes are present, the non-spec decodes are reclassified as prefills (`gdn_attn.py:243-251`),
so the mixed case never hits the fast path.

### 4.7 Notes for the megakernel port

* The recurrent state is the dominant per-token memory traffic in the GDN layers: **2 MiB read +
  2 MiB write per token per linear layer** at fp32 — with the conv state that is 4.29 MB/layer,
  ×30 layers = **122.8 MiB (128.8 MB) per token per step, even at B=1**. Persisting `S` in shared
  memory/registers across the delta-rule update is the single
  biggest win on this side of the model. (This is 2× the number the scouting note gave, because the
  state is fp32.)
* The state update is *in place* on the same buffer (`ht = initial_state`,
  `fused_recurrent.py:457-458`), so no double-buffering is required for non-spec decode.
* The conv1d update writes its output over its input (`causal_conv1d.py:1162-1163`), so the q/k/v
  slice of the `in_proj_qkvz` output is mutated in place; the `z` columns are untouched.
* MPK does not have to reproduce vLLM's hybrid page-alignment machinery — it exists only so mamba
  and attention pages have identical byte size inside one allocator. A standalone runtime can size
  the two caches independently (attention `block_size=16`, one GDN state slot per sequence) and skip
  the 1056/32 dance entirely.
* Likewise the align-mode state snapshotting is a prefix-caching feature. Without prefix caching,
  one `(conv, S)` slot per running sequence per linear layer suffices and `preprocess_mamba`
  disappears.

---

## 5. Weights → runtime mapping

### 5.1 Name rewriting

Two mappers compose. First the VL prefix mapper, inherited from
`Qwen3VLForConditionalGeneration.hf_to_vllm_mapper` (`models/qwen3_vl.py:1705-1711`):

```
"model.visual."          -> "visual."
"lm_head."               -> "language_model.lm_head."
"model.language_model."  -> "language_model.model."
```

Then `Qwen3_5Model.hf_to_vllm_mapper` (`qwen3_5.py:213-220`) — the Qwen3-Next mapper
(`qwen3_next.py:593-604`) **plus the Qwen3.5-specific GDN fusion**:

| checkpoint suffix | vLLM param | shard |
|---|---|---|
| `.q_proj` / `.k_proj` / `.v_proj` | `.qkv_proj` | `"q"` / `"k"` / `"v"` |
| `.mlp.gate_proj` / `.mlp.up_proj` | `.mlp.gate_up_proj` | 0 / 1 |
| `.shared_expert.gate_proj` / `.up_proj` | `.shared_expert.gate_up_proj` | 0 / 1 |
| **`.in_proj_qkv`** | **`.in_proj_qkvz`** | **(0,1,2)** |
| **`.in_proj_z`** | **`.in_proj_qkvz`** | **3** |
| **`.in_proj_b`** | **`.in_proj_ba`** | **0** |
| **`.in_proj_a`** | **`.in_proj_ba`** | **1** |

The tuple shard `(0,1,2)` means the checkpoint tensor is treated as *already fused* over
`output_sizes[0:3]` and copied into rows `[0:8192]` of `in_proj_qkvz.weight`
(`layers/linear.py:758-792`). `packed_modules_mapping` (`qwen3_5.py:288-298`, `:403-406`) declares
the same four fusions for the quantization skip logic.

> **The single biggest checkpoint-layout difference vs Qwen3-Next:** Qwen3-Next ships one fused
> `in_proj_qkvz` / `in_proj_ba` with **GQA-interleaved** per-key-group layout; Qwen3.5 ships
> **four/two separate tensors in plain `[q|k|v|z]` and `[b|a]` order** and vLLM fuses them at load
> time. That is why `gqa_interleaved_layout=False` (`qwen3_5.py:142`) and why the runtime path is a
> plain `split` (`qwen_gdn_linear_attn.py:855-863`) instead of the `fix_query_key_value_ordering`
> unpack (`:568-617`).

### 5.2 Per-layer checkpoint keys → runtime tensors

Checkpoint prefix `model.language_model.layers.{i}.`; runtime prefix
`language_model.model.layers.{i}.`.

**Every layer:**

| key | shape | dtype | runtime |
|---|---|---|---|
| `input_layernorm.weight` | `[2048]` | BF16 | same, used as `1+w` |
| `post_attention_layernorm.weight` | `[2048]` | BF16 | same |

**Linear-attention layer** (i ∈ {0,1,2,4,5,6,…}, 30 of them), prefix `linear_attn.`:

| key | shape | dtype | runtime tensor |
|---|---|---|---|
| `in_proj_qkv.weight` | `[8192, 2048]` | F8_E4M3 | `in_proj_qkvz.weight[0:8192]` |
| `in_proj_qkv.weight_scale_inv` | `[64, 16]` | BF16 | `in_proj_qkvz.weight_scale_inv[0:64]` (fp32) |
| `in_proj_z.weight` | `[4096, 2048]` | F8_E4M3 | `in_proj_qkvz.weight[8192:12288]` |
| `in_proj_z.weight_scale_inv` | `[32, 16]` | BF16 | `in_proj_qkvz.weight_scale_inv[64:96]` (fp32) |
| `in_proj_b.weight` | `[32, 2048]` | BF16 | `in_proj_ba.weight[0:32]` |
| `in_proj_a.weight` | `[32, 2048]` | BF16 | `in_proj_ba.weight[32:64]` |
| `conv1d.weight` | `[8192, 1, 4]` | BF16 | `conv1d.weight`, materialized `[8192,4]` then `.unsqueeze(1)` (`qwen_gdn_linear_attn.py:396`); used as `.view(8192,4)` (`:1591-1593`) |
| `A_log` | `[32]` | **F32** | `A_log` (fp32 param, `:442-447`) |
| `dt_bias` | `[32]` | BF16 | `dt_bias` |
| `norm.weight` | `[128]` | **F32** | `norm.weight`, ones-init, **no `+1`** |
| `out_proj.weight` | `[2048, 4096]` | F8_E4M3 | `out_proj.weight` |
| `out_proj.weight_scale_inv` | `[16, 32]` | BF16 | `out_proj.weight_scale_inv` (fp32) |

The `[q | k | v]` order inside `in_proj_qkv` matches the packed conv order — `q[0:2048]`,
`k[2048:4096]`, `v[4096:8192]` — as consumed by the recurrence kernel (`fused_recurrent.py:306-308`).

**Full-attention layer** (i ∈ {3,7,…,39}, 10 of them), prefix `self_attn.`:

| key | shape | dtype | runtime tensor |
|---|---|---|---|
| `q_proj.weight` | **`[8192, 2048]`** | F8_E4M3 | `qkv_proj.weight[0:8192]` — 32 head-slots of 256 = 16 × `[q(256)|gate(256)]` |
| `q_proj.weight_scale_inv` | `[64, 16]` | BF16 | `qkv_proj.weight_scale_inv[0:64]` |
| `k_proj.weight` | `[512, 2048]` | F8_E4M3 | `qkv_proj.weight[8192:8704]` |
| `k_proj.weight_scale_inv` | `[4, 16]` | BF16 | `qkv_proj.weight_scale_inv[64:68]` |
| `v_proj.weight` | `[512, 2048]` | F8_E4M3 | `qkv_proj.weight[8704:9216]` |
| `v_proj.weight_scale_inv` | `[4, 16]` | BF16 | `qkv_proj.weight_scale_inv[68:72]` |
| `q_norm.weight` | `[256]` | BF16 | Gemma (`1+w`), per head over head_dim |
| `k_norm.weight` | `[256]` | BF16 | same |
| `o_proj.weight` | `[2048, 4096]` | F8_E4M3 | `o_proj.weight` |
| `o_proj.weight_scale_inv` | `[16, 32]` | BF16 | `o_proj.weight_scale_inv` |

No qkv/attention biases (`attention_bias: false` in `config.json`; `qkv_bias` absent).

**MoE block**, prefix `mlp.`:

| key | shape | dtype | runtime tensor |
|---|---|---|---|
| `gate.weight` | `[256, 2048]` | BF16 | `experts.gate.weight` (router, unquantized) |
| `shared_expert_gate.weight` | `[1, 2048]` | BF16 | `shared_expert_gate.weight` |
| `shared_expert.gate_proj.weight` (+`_scale_inv [4,16]`) | `[512, 2048]` | F8_E4M3 | `shared_expert.gate_up_proj.weight[0:512]` |
| `shared_expert.up_proj.weight` (+`_scale_inv [4,16]`) | `[512, 2048]` | F8_E4M3 | `shared_expert.gate_up_proj.weight[512:1024]` |
| `shared_expert.down_proj.weight` (+`_scale_inv [16,4]`) | `[2048, 512]` | F8_E4M3 | `shared_expert.down_proj.weight` |
| `experts.{e}.gate_proj.weight` (+`_scale_inv [4,16]`) | `[512, 2048]` | F8_E4M3 | `experts.w13_weight[e, 0:512, :]` (shard `w1`) |
| `experts.{e}.up_proj.weight` (+`_scale_inv [4,16]`) | `[512, 2048]` | F8_E4M3 | `experts.w13_weight[e, 512:1024, :]` (shard `w3`) |
| `experts.{e}.down_proj.weight` (+`_scale_inv [16,4]`) | `[2048, 512]` | F8_E4M3 | `experts.w2_weight[e]` (shard `w2`) |

### 5.3 Routed experts: this checkpoint ships the **per-expert 2-D** form

`RoutedExperts.build_expert_params_mapping` (`fused_moe/routed_experts.py:1012-1102`) with
`ckpt_names = ("gate_proj", "down_proj", "up_proj")` (`layer.py:131`) emits both a `fused_mapping`
(3-D pre-stacked, `:1064-1084`) and a `per_expert_mapping` (2-D, `:1086-1100`); the loader picks by
tensor rank — `is_fused = loaded_weight.dim() == 3` (`:907`).

**The `-FP8` checkpoint ships the 2-D per-expert form.** The safetensors index contains
`model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` (10 240 = 40×256 of
each) plus their `weight_scale_inv` companions, and contains **no**
`experts.gate_up_proj` / `experts.down_proj` 3-D tensors at all. (The scouting note asserted the
pre-stacked 3-D form based on the HF `base_model_tp_plan` at `qwen3_5_moe.py:31-32`; that plan
describes the *transformers* module layout, not the serialized tensors. See §7.)

Either way vLLM materializes:

```
experts.w13_weight           : [256, 1024, 2048]  fp8e4m3    # [0:512) = gate, [512:1024) = up
experts.w13_weight_scale_inv : [256,    8,   16]  fp32
experts.w2_weight            : [256, 2048,  512]  fp8e4m3
experts.w2_weight_scale_inv  : [256,   16,    4]  fp32
```

`Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = True` (`qwen3_5.py:618`) is a **LoRA-only
flag** — it selects the 3-D MoE LoRA A/B packing (`vllm/lora/utils.py:397`,
`vllm/lora/model_manager.py:130-133`) and has **no effect on base-weight loading**.

### 5.4 Model-level keys and totals

| key | shape | dtype |
|---|---|---|
| `model.language_model.embed_tokens.weight` | `[248320, 2048]` | BF16 |
| `model.language_model.norm.weight` | `[2048]` | BF16 (Gemma) |
| `lm_head.weight` | `[248320, 2048]` | BF16 (`tie_word_embeddings: false`) |
| `model.visual.*` | vision tower (333 tensors, all BF16) | **not needed for text decode** |
| `mtp.*` | MTP draft layer (~1 540 tensors, fp8 + scales) | skipped by the main model (`skip_prefixes=["mtp."]`, `qwen3_5.py:372`, `:515-520`) |

Parameter count and resident bytes for the **text decode path** (no vision, no MTP):

| quantity | value |
|---|---|
| total params | 34.66 B = 30·33.72 M + 10·27.26 M + 40·808.98 M + 2·248320·2048 + norms |
| fp8 weight bytes | **33,617,346,560 B = 31.31 GiB** |
| bf16 weight bytes | **2,086,537,856 B = 1.94 GiB** (embed 0.95 + lm_head 0.95 + routers 0.04 + GDN a/b/conv1d + norms) |
| fp32 block-scale bytes | **8,207,360 B = 8.2 MB** |
| **total resident** | **35,712,091,776 B = 33.26 GiB** |
| active params / token | 2.95 B (incl. lm_head) → *"35B-A3B"* |
| dense (non-routed-expert) weight bytes / step | **2.47 GB** |
| GDN state traffic / step | **B × 128.8 MB** (fp32 `S`; 4.29 MB × 30 layers) |

Routed-expert weight traffic per decode step (worst case, all `8B` selections distinct):

| B | ≤ distinct experts / layer | expert bytes / layer | expert bytes / step (×40) | total weight bytes / step |
|---|---|---|---|---|
| 1 | 8 | 25.2 MB | 1.01 GB | 3.48 GB |
| 2 | 16 | 50.3 MB | 2.01 GB | 4.49 GB |
| 4 | 32 | 101 MB | 4.03 GB | 6.50 GB |
| 8 | 64 | 201 MB | 8.05 GB | 10.53 GB |
| 16 | 128 | 403 MB | 16.11 GB | 18.58 GB |

**MoE weight streaming dominates the decode step for every B ≥ 2** — at B=16 half of each layer's
768 MiB expert bank is touched — so an expert-major megakernel schedule that visits each resident
expert once and gathers its assigned tokens is the right shape. (Compared with a bf16 build, fp8
halves this traffic; the vLLM baseline gets the same halving, so the *relative* target is unchanged
but the absolute bandwidth headroom is doubled.)

Layer order for one full step:

```
embed_tokens[B]                                  # gather, [B,2048]
for i in 0..39:
    if (i+1) % 4 != 0:  LINEAR-ATTN block  (§2.1)   # i = 0,1,2, 4,5,6, ...  (30x)
    else:               FULL-ATTN  block   (§2.2)   # i = 3,7,11,...,39      (10x)
    MoE block (§2.3)                                # every layer            (40x)
model.norm(h, residual)                             # GemmaRMSNorm, fused add
logits = h @ lm_head.weight.T                       # [B,2048] @ [2048,248320]
```

---

## 6. Gotchas

The five that were re-verified on source for this document, plus what turned up alongside them.

1. **Every trunk RMSNorm is Gemma-style `(1+w)`, and the weights are stored zero-centred.**
   That includes `input_layernorm`, `post_attention_layernorm`, `model.norm`, and the
   full-attention `q_norm`/`k_norm`. A plain `x·rsqrt(…)·w` implementation silently produces
   near-zero activations. **The only exception is the GDN `linear_attn.norm`** (`RMSNormGated`,
   ones-initialized, used as-is) — and note it is stored as **F32** in this checkpoint while every
   Gemma norm weight is BF16.
   *Cites:* `qwen3_5.py:39`, `layernorm.py:148` (zeros) and `:157` (`+1.0`) vs
   `layernorm.py:212`,`:218-219` (ones) and `:257` (no `+1`); construction at
   `qwen_gdn_linear_attn.py:459-466`.

2. **RoPE is partial at 0.25 with head_dim 256, so only dims `[0:64]` rotate — but on THIS
   checkpoint the factor comes from `config.json`, not the config class.** The scouting note said
   the factor is injected by vLLM's config class and is absent from the checkpoint. That is only
   half right: `qwen3_5_moe.py:92` uses `kwargs.setdefault("partial_rotary_factor", 0.25)`, which is
   a *default*, and the shipped `config.json` **does** carry
   `text_config.rope_parameters.partial_rotary_factor = 0.25`. It is easy to miss because it lives
   inside `rope_parameters`, not at the top level of `text_config`. Either way `rotary_dim = 64`
   (`rotary_embedding/__init__.py:69-72`) and dims `[64:256)` are RMSNorm'd but not rotated.
   **Also from `rope_parameters`: `rope_theta = 1e7` (not 1e4), and `mrope_section=[11,11,10]` +
   `mrope_interleaved=true` cause `get_rope` to build an `MRotaryEmbedding`
   (`rotary_embedding/__init__.py:101-112`). For text-only tokens that reduces exactly to standard
   partial NeoX RoPE (§2.2.3), but the code path is not the plain `RotaryEmbedding`.**

3. **The attention output gate is the second half of each Q head inside `q_proj`.** `q_proj` is
   `[8192, 2048]` = 32 head-slots of 256 laid out `[h0_q | h0_gate | h1_q | h1_gate | …]`, because
   `qkv_proj` is built with `total_num_heads = 16 * (1 + attn_output_gate) = 32`. A naive
   `q_proj → 16×256` reader gets garbage. The gate is applied *outside* the attention kernel as
   `out * sigmoid(gate)` on the flat `[T,4096]` tensor — a full sigmoid, not SiLU, and not a sink.
   *Cites:* `qwen3_next.py:265`, `:267-275`, `:372-373`, `:397-398`; kernel addressing
   `fused_qk_norm_rope.py:54`, `:111`, docstring `:133`.

4. **The router is softmax-over-all-256 → top-8 → renormalize, not top-8 → softmax.** vLLM encodes
   this as `RoutingMethodType.RenormalizeNaive (= 4)`, explicitly distinct from `Renormalize (= 1)`
   which is the other order. The CUDA kernel computes the full 256-wide softmax in fp32 with
   **lower-index tie-breaking**, then divides the selected weights by their sum. And
   `norm_topk_prob` is in neither the config class nor `config.json` — `renormalize=True` comes from
   a `getattr` default.
   *Cites:* `fused_moe/config.py:112-113` and `:105-106`, `:165-169`;
   `topk_softmax_kernels.cu:425`, `:441-450` (full-row softmax "to closer match torch"), `:536-537`
   (tie-break), `:561`,`:581-590` (renormalize), `:704-706` (E=256 launch);
   `qwen3_next.py:185` (`getattr` default).

5. **Qwen3.5's GDN projections are NOT the Qwen3-Next layout.** Qwen3-Next ships one fused,
   GQA-interleaved `in_proj_qkvz`; Qwen3.5 ships separate `in_proj_qkv` / `in_proj_z` / `in_proj_b`
   / `in_proj_a` in plain `[q|k|v|z]` and `[b|a]` order, and vLLM fuses them at load time. Copying
   the Qwen3-Next `fix_query_key_value_ordering` unpack into a Qwen3.5 port produces a silently
   permuted q/k/v. Confirmed on both sides: `gqa_interleaved_layout=False` at `qwen3_5.py:142` vs
   `True` at `qwen3_next.py:439`; the branch at `qwen_gdn_linear_attn.py:855-863`
   (split) vs `:846-854` (unpack); the shard construction at `:490-514` and `:516-538`; and the
   checkpoint itself, which contains four separate `in_proj_*` tensors per linear layer.

6. **The recurrent state is FP32 on this checkpoint, not bf16.** `config.json` sets
   `text_config.mamba_ssm_dtype = "float32"`, which `models/config.py:754-757` copies into
   `cache_config.mamba_ssm_cache_dtype`. Everything downstream changes: 2 MiB per sequence per layer
   instead of 1 MiB, 61.4 MiB/sequence total, mamba page 2,146,304 B, KV-manager block size **1056**
   (not 544), FlashInfer kernel block 32 with 33 blocks per KV block. Size any buffer off the fp32
   number.

7. **`in_proj_ba` stays BF16 and it is not optional.** Its two shards are 32 rows each and
   `block_n = 128`, so a `[128,128]` block quantization cannot represent them cleanly; the code says
   so at `qwen_gdn_linear_attn.py:411` and the checkpoint lists both shards in
   `modules_to_not_convert`. If only *one* of `in_proj_a`/`in_proj_b` were listed,
   `is_layer_skipped` would raise `ValueError` (`quant_utils.py:551-559`) rather than silently
   mis-quantizing. MPK must keep `a`/`b` in bf16.

8. **`modules_to_not_convert` is matched by exact string equality — no globs.** `is_layer_skipped`
   uses `prefix in ignored_layers` (`quant_utils.py:517-518`); fnmatch exists only on the
   ModelOpt/Quark paths. A checkpoint that wrote `"*in_proj_a*"` would be silently ignored and the
   layer would be quantized. This one writes all 287 names out in full, so it works — but any
   downstream tooling that rewrites the list must preserve exact names.

9. **DeepGEMM is auto-disabled for Qwen3.5 on Blackwell.** `should_auto_disable_deep_gemm`
   (`vllm/utils/deep_gemm.py:33-46`) excludes `model_type ∈ {"qwen3_5_text", "qwen3_5_moe_text"}`
   on sm100/sm120 because of *"accuracy degradation with DeepGemm's E8M0 scale format"*. The dense
   fp8 linears therefore run **CUTLASS block-scaled MM with plain fp32 scales**, and no UE8M0
   requantization happens at load. Any "vLLM FP8 on B200" baseline for this model is a CUTLASS
   baseline. See §3.5 and §3.7.

10. **Runners-up.** The decay `exp(g)` is a scalar per (token, v-head), not per-channel, and `g` is
    computed from an **fp32** `A_log`. q/k are L2-normalized *inside* the recurrence kernel with
    `eps = 1e-6` and *then* scaled by `128^-0.5`. GVA maps 32 value heads onto 16 key heads as
    `i_h = i_hv // 2`. `beta = sigmoid(b)` is round-tripped through bf16 in the packed decode kernel
    (`fused_recurrent.py:325`) but **not** in the generic kernel
    (`fused_sigmoid_gating.py:136`) — pick the path you intend to bit-match.
    The fused QK-norm+RoPE+gate Triton kernel is **off by default** for this model because
    `language_model_only` defaults to False (`config/multimodal.py:78`).
    `mamba_cache_mode="all"` is hard-rejected (`qwen3_5.py:307-311`).
    `weight_scale_inv` is stored **BF16** in the checkpoint but held **fp32** at runtime.

---

## 7. Deltas vs the scouting note

`design/scouting/vllm-qwen35-graph.md` was read at the same vLLM commit and its vLLM-side cites
check out. It was written for a **bf16, unquantized** target and against the *class defaults* rather
than the shipped `-FP8` checkpoint, so the following claims do not hold for our target:

| # | scouting note | corrected |
|---|---|---|
| 1 | *"`partial_rotary_factor` … is injected by the config class, not the checkpoint … easy to miss when reading `config.json`"* | The shipped `config.json` **does** contain it, under `text_config.rope_parameters`. The class `setdefault` is only a fallback. §6 gotcha 2. |
| 2 | *"if the shipped `config.json` carries `mrope_section`, `get_rope` builds an `MRotaryEmbedding`"* — framed as conditional | It **does** carry `mrope_section: [11,11,10]` and `mrope_interleaved: true`. `MRotaryEmbedding` is the actual path. Also `rope_theta = 1e7`, not the 1e4 default. §2.2.3. |
| 3 | *"the checkpoint **may** set `mamba_ssm_dtype: float32` … check this before sizing anything"* | It **does**. `S` is fp32; state traffic and page sizes double. §2.1.5, §4.3, §6 gotcha 6. |
| 4 | KV-manager `block_size = 544`, `num_blocks_per_kv_block = 17` | **1056** and **33**, because the mamba page is 2,146,304 B with an fp32 `S`. §4.3. |
| 5 | GDN state = 31.4 MiB/sequence; 1 MiB `S` read + 1 MiB write per token per layer; 60 MiB/token/step | **61.4 MiB/sequence**; 2 MiB + 2 MiB; **122.8 MiB (128.8 MB) per token per step**. §4.7. |
| 6 | *"Pre-stacked 3-D form — **this is what Qwen3.5 actually ships**"* (`experts.gate_up_proj [256,2048,1024]`) | The `-FP8` checkpoint ships the **per-expert 2-D form** (`experts.{e}.{gate,up,down}_proj.weight`) and contains no 3-D expert tensors. The `base_model_tp_plan` the note cited describes the transformers module layout, not the serialized tensors. §5.3. |
| 7 | Winner MoE kernel `FLASHINFER_TRTLLM → TrtLlmBf16ExpertsMonolithic` chosen by `select_unquantized_moe_backend` (`oracle/unquantized.py:193`); expert weights kept as raw `[E,2I,H]`/`[E,H,I]` by the Triton backend | That oracle is the **unquantized** one and is never reached — `Fp8Config.get_quant_method` returns `Fp8MoEMethod` (`fp8.py:210-211`), which runs `select_fp8_moe_backend` (`oracle/fp8.py:271`) with a *different* priority list. The winner is still FlashInfer TRT-LLM but the kernel is `TrtLlmFp8ExpertsMonolithic` calling `trtllm_fp8_block_scale_moe`, and the weights **are** shuffled into BlockMajorK `[256,16,1024,128]` / `[256,4,2048,128]` with a W13→W31 swap. §3.6. |
| 11 | *"A megakernel port should target the raw `[E,2I,H]`/`[E,H,I]` layout and ignore the FlashInfer shuffle"* | Still the right advice for MPK — but note the *checkpoint* order is `[gate; up]` while the kernel FlashInfer actually runs consumes `[up; gate]` in BlockMajorK. Do not copy a dump of vLLM's post-`process_weights_after_loading` tensors and expect `[E,2I,H]`. §3.6.3. |
| 12 | Per-layer expert bank *"805 M params = 1.61 GiB bf16"* | 805 M params = **768 MiB at fp8**; per expert 3.15 M params = **3.15 MB**. §2.3.4. |
| 8 | Whole-step weight bytes 69.3 GB bf16; dense bytes/step 3.87 GB; expert bytes/step 2.0–32.2 GB | fp8: **33.26 GiB resident**, dense **2.47 GB/step**, experts **1.01–16.11 GB/step**. §5.4. |
| 9 | `attn_page_size_1_token` / hybrid-alignment table listed fp32 as a hypothetical row | fp32 is the actual row. §4.3. |
| 10 | No FP8 coverage at all | §3. |

Everything else in the scouting note that this document re-derives — the Gemma-norm exception, the
q\|gate packing, the `RenormalizeNaive` router, the non-interleaved GDN layout, the GDN decode
kernel math, the attention backend selection, the `align` cache mode — was verified and stands.
