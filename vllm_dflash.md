> **Status: CURRENT implementation.** This documents DFlash speculative decoding as it
> exists on vLLM v1 (`method="dflash"`, parallel drafting, TP>=1), for the combination
> **target = a captured-hidden-state model** + **draft = `DFlashDraftModel`** (Qwen3-style
> dense drafter, e.g. `z-lab/Qwen3-8B-DFlash-b16`; the `z-lab/Kimi-K2.6-DFlash` draft is
> the same architecture paired with a Kimi-K2.6 target).
>
> Scope: **single linear chain only (no tree).** The draft produces a whole block of `B`
> tokens in one non-causal forward; verify accepts the longest matching prefix via the
> generic rejection sampler. `num_speculative_tokens = B - 1`.
>
> Notation: `[tensor]` `{weight}` `★KV write★`.
>
>   - `B`   = block_size = `1 + num_speculative_tokens` (b16 config: 16; so `s = 15`)
>   - `K`   = number of captured target layers (`dflash_config.target_layer_ids`, b16: 5)
>   - `L`   = draft layers (b16: 5)
>   - `H_t` = target hidden_size per captured layer (Qwen3-8B: 4096; **K2.6: 7168**)
>   - `H_d` = draft hidden_size (b16: 4096)
>   - `n_q / n_kv / d` = draft q-heads / kv-heads / head_dim (b16: 32 / 8 / 128)
>   - `q_size / kv_size` = `n_q·d` / `n_kv·d` (b16: 4096 / 1024)
>   - `I`   = draft intermediate (b16: 12288), `V` = vocab (b16: 151936)
>   - `MASK` = `dflash_config.mask_token_id` (b16: 151669; id must fall inside the
>     target vocab — vLLM does not resize target embeddings)
>   - `bs` = batch size, `n` = per-request prefix len, `S = Σn` = total context tokens
>   - dtype: bf16 throughout (`"dtype": "bfloat16"`); RMSNorm / RoPE / softmax internal fp32
>   - TP>1: `q_size→q_size/TP`, `kv_size→kv_size/TP`, heads sharded; `fc`/`embed`/`lm_head`
>     are Replicated / VocabParallel.
>
> Launch (illustrative):
>
> ```bash
> vllm serve <target-model> \
>     --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-8B-DFlash-b16","num_speculative_tokens":15}' \
>     --tensor-parallel-size 8
> ```
>
> Core files:
> [v1/spec_decode/dflash.py](vllm/v1/spec_decode/dflash.py) (proposer / orchestration) ·
> [model_executor/models/qwen3_dflash.py](vllm/model_executor/models/qwen3_dflash.py) (draft model + KV materialization) ·
> [v1/spec_decode/utils.py](vllm/v1/spec_decode/utils.py) (`copy_and_expand_dflash_inputs_kernel`) ·
> [v1/spec_decode/llm_base_proposer.py](vllm/v1/spec_decode/llm_base_proposer.py) (sampling base) ·
> [config/speculative.py](vllm/config/speculative.py) (`use_dflash`, method detection) ·
> [transformers_utils/configs/speculators/algos.py](vllm/transformers_utils/configs/speculators/algos.py) (`update_dflash`)

---

## Workflow (orchestration / control flow)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 0 · Init (once at startup)                                            │
│  Load draft model (DFlashQwen3) → load_weights → _build_fused_kv_buffers     │
│  Pre-stack: _fused_kv_weight[L·2·kv_size, H_d] · _k_norm_weights[L] · RoPE   │
│  draft / target share one KV pool (scheduler reserves +1 lookahead slot)     │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │  request arrives
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 · Prefill (1 target forward)                                        │
│  prompt ─► target (capture K aux-hidden layers)                              │
│           ├─► sample → bonus t₀                 [bs]                          │
│           └─► aux hidden concat → [S, K·H_t]    ← fed to the proposer         │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  ▼
                    ╔══════════ DFlashProposer.propose ══════════╗
                    ║  §2 combine_hidden_states (fc)              ║
                    ║       [S,K·H_t] → ctx_hidden [S,H_d]        ║
                    ║  §3 copy_and_expand_dflash_inputs_kernel    ║  ← no forward
                    ║       → input_ids/positions/slot/sample_idx ║
                    ║  §4 precompute_and_store_context_kv         ║  ← no forward
                    ║       ctx_hidden ─► per-layer K/V ─►★draftKV★║     (proj/norm/RoPE/write)
                    ╚══════════════════════╤═════════════════════╝
              ┌──────────────────────────┤◄───────────────────────────┐
              │            decode loop (yields 1~B tokens per round)    │
              ▼                                                        │
┌───────────────────────────────────────────────────────────────┐     │
│  STAGE 3 · Draft a block (1 non-causal draft forward)          │     │
│  embed([t₀,MASK×(B-1)]) ─► L layers (non-causal attn,          │     │
│                            KV = context + this block)          │     │
│  final_hidden[sample_idx] ─► TP-safe argmax (draft lm_head)    │     │
│  → draft chain [t₀, d₁ … d_{B-1}]   (linear, no tree)          │     │
└──────────────────────────────┬────────────────────────────────┘     │
                               ▼                                       │
┌───────────────────────────────────────────────────────────────┐     │
│  STAGE 4 · Target verify (1 target forward)                    │     │
│  [t₀ .. d_{B-1}] ─► target (chain causal mask, capture on)     │     │
│  RejectionSampler: accept longest prefix + new bonus t₀'       │     │
│       draft : t₀  d₁  d₂  d₃  d₄ …                             │     │
│       target: ✓   ✓   ✓   ✗      → commit_len = 4             │     │
│  rejected-position KV freed; capture K-layer hidden of commits │     │
└──────────────────────────────┬────────────────────────────────┘     │
                               ▼                                       │
┌───────────────────────────────────────────────────────────────┐     │
│  STAGE 5 · Re-materialize & state update (no forward)          │     │
│  committed hidden ─► §2/§4-style materialize ─► ★draft KV★      │     │
│  update bonus = t₀'; advance positions / seq_lens              │     │
└──────────────────────────────┬────────────────────────────────┘     │
                               │  not EOS / under max_tokens           │
                               └───────────────────────────────────────┘
                               │  termination reached
                               ▼
                        request finishes, KV released
```

How to read this:

- **Only three places do heavy compute**: Stage 1 (target prefill), Stage 3 (draft
  L-layer forward), Stage 4 (target verify). Stages 2/5 are projections + cache writes.
- **One decode round = Stage 3 → 4 → 5**: two model forwards buy 1~B tokens; the average
  yield is the accept length (≈ 3.6–4.9 on K2.6 per the model card).
- **Materialization happens in two places**: after prefill (whole prompt, §4) and after
  verify (the 1~B committed tokens, Stage 5). The block's own 16-token KV in Stage 3 is
  written by the draft forward and discarded — that is **not** materialization.
- **vs SGLang**: vLLM inlines §2/§3/§4 inside `DFlashProposer.propose` (one unified
  `GPUModelRunner`), whereas SGLang runs a separate draft worker + target worker; vLLM
  verify uses the generic `RejectionSampler`, SGLang uses a DFlash-specific accept kernel.

---

## Dataflow (op-by-op, shapes + dtype)

Concrete numbers = `z-lab/Qwen3-8B-DFlash-b16`, TP=1. K2.6 substitutions noted inline.

### §1 — Boundary inputs (from the target model)

```
next_token_ids (bonus t₀)   [bs]            int64     ← token sampled by target
target_hidden_states        [S, K·H_t]      bf16      ← K=5 captured layers concatenated
                                                          (K2.6: [S, 5·7168 = 35840])
target_positions            [S]             int64     ← context absolute positions
CommonAttentionMetadata { block_table [bs,max_blk] int32, query_start_loc [bs+1] int32,
                          seq_lens [bs] int32, ... }
```

### §2 — fc projection: `combine_hidden_states` (qwen3_dflash.py)

```
in : target_hidden_states  [S, K·H_t] = [S,20480]   bf16
op : fc = ReplicatedLinear(K·H_t → H_d), weight [4096,20480] bf16, no bias   (replicated on every rank)
out: ctx_hidden            [S, H_d] = [S,4096]       bf16
```
> `hidden_norm` is NOT here — it runs inside §4.

### §3 — Input prep: `copy_and_expand_dflash_inputs_kernel` (one fused Triton kernel)

grid `(bs, num_blocks)`, produces all outputs at once:

```
out_input_ids            [bs·B] = [bs·16]   int32   = per req [t₀, MASK×15]   (is_bonus ? t₀ : 151669)
context_positions        [S]                int64   = copy of target_positions
query_positions          [bs·B]             int64   = last_valid_pos + 1 + offset(0..15)
context_slot_mapping     [S]                int64   = block_table lookup: blk_id·block_size + pos%block_size
query_slot_mapping       [bs·B]             int64   = same lookup (query positions)
token_indices_to_sample  [bs·s] = [bs·15]   int32   = the 15 MASK slots only (bonus slot skipped)
```

### §4 — KV materialization: `precompute_and_store_context_kv` (no draft forward)

Goal: project the target's context hidden directly into each draft layer's K/V and write
to the draft KV cache, without running the draft over context tokens. Six ops:

```
① hidden_norm  (ops.rms_norm, CUDA, fp32 internal)
   in : ctx_hidden [S,4096] bf16 ; weight _hidden_norm_weight [4096] bf16 ; eps=1e-6
   out: normed     [S,4096] bf16

② fused KV projection (F.linear, one big GEMM for all L layers)
   weight _fused_kv_weight = concat of each layer's qkv_proj.weight[q_size:]
        → [L·2·kv_size, H_d] = [10240,4096] bf16
   in : normed [S,4096]              out: all_kv_flat [S, 10240] bf16

③ reshape / permute (no compute)
   all_kv_flat.view(S, L, 2, n_kv, d) = [S,5,2,8,128].permute(2,1,0,3,4).contiguous()
   → all_k [L,S,n_kv,d] = [5,S,8,128] bf16 ,  all_v [5,S,8,128] bf16

④ per-layer k_norm  (for i in L: ops.rms_norm, CUDA)
   in : all_k[i] [S,8,128] ; weight _k_norm_weights[i] [128] bf16
   out: all_k_normed [5,S,8,128] bf16

⑤ fused RoPE over all layers  (ops.rotary_embedding, in-place, K passed as "query", key=None)
   view all_k_normed → [L·S, kv_size] = [5·S,1024]
   positions_repeated = context_positions.repeat(L)  [5·S] int64
   cos_sin_cache [max_pos, rotary_dim] cast→bf16 ; head_size=128 ; neox
   out: all_k_flat rotated [5·S,1024] bf16

⑥ per-layer cache write  (for i in L: attn.impl.do_kv_cache_update)
   all_k_final = view [L,S,8,128]
   write K=all_k_final[i], V=all_v[i] → layer-i draft KV cache @ context_slot_mapping[S]
```
> **K2.6 delta**: `fc` input becomes `K·7168 = 35840`; ②/⑤ math unchanged. If K2.6 RoPE has
> `rotary_dim < head_dim` or is non-neox, confirm `ops.rotary_embedding` pass-through matches.

### §5 — Draft non-causal forward (1 pass, query tokens only)

```
embed: input_embeds = embed_tokens(out_input_ids)        ← VocabParallelEmbedding(V,H_d)
       in [bs·16] int32 → out [bs·16,4096] bf16
positions = query_positions [bs·16] int64
hidden_states = input_embeds ; residual = None
```

Per DecoderLayer (×L=5):

```
input_layernorm (RMSNorm, fused add+norm; first layer residual=hidden)
    [bs·16,4096] bf16 → hidden, residual [bs·16,4096] bf16

self_attn (DFlashQwen3Attention.forward; KV cache already holds context):
    qkv = F.linear(h, qkv_proj.weight [q_size+2·kv_size = 6144, 4096])   → [bs·16,6144] bf16
    split → q[bs·16,4096]  k[bs·16,1024]  v[bs·16,1024]
    q_norm: view[bs·16,32,128] RMSNorm(d=128) view back  → [bs·16,4096]
    k_norm: view[bs·16, 8,128] RMSNorm(d=128) view back  → [bs·16,1024]
    rotary_emb(positions, q, k)                          → rotated q,k
    attn(q,k,v):                                          ← Attention, attn_type=DECODER
        · write this block's k,v to cache @ query_slot_mapping
        · NON-CAUSAL: 16 queries see [context S + this block 16] fully (softmax fp32)
        → attn_output [bs·16, 4096 (q_size)] bf16
    o_proj = RowParallelLinear(q_size → H_d) [4096,4096]  → [bs·16,4096] bf16

post_attention_layernorm (RMSNorm fused add+norm)         → hidden, residual

mlp (Qwen3MLP, SiLU gated):
    gate_up = F.linear(h, [2·I = 24576, 4096])            → [bs·16,24576]
    split gate[12288] up[12288] ; act = silu(gate)*up     → [bs·16,12288]
    down = F.linear(act, [4096,12288])                    → [bs·16,4096] bf16
→ (hidden, residual)
```

Final: `hidden, _ = norm(hidden, residual)` (RMSNorm fused) → `final_hidden [bs·16,4096] bf16`

> Non-causal requires a capable backend: vLLM asserts attn metadata `causal is False`
> (`dflash.py build_per_group_and_layer_attn_metadata`), use FLASH_ATTN. RoPE distinguishes
> the 16 MASKs by absolute position (same embedding, different positions). Slot j's output
> predicts the token AT slot j (fill-in-the-mask, no next-token shift).

### §6 — Draft token selection (llm_base_proposer.py)

```
sample_hidden = final_hidden[token_indices_to_sample]   → [bs·15,4096] bf16   (15 MASK slots only)
_greedy_sample:
  use_local_argmax_reduction:  model.get_top_tokens(sample_hidden)            ← TP-safe sharded argmax
  else:    logits = lm_head(sample_hidden) [bs·15, V=151936] → argmax         ← ParallelLMHead(draft_vocab,H_d)
  (if draft_vocab ≠ target: remap via draft_id_to_target_id; b16 vocab equal → no remap)
draft_token_ids → reshape [bs,15]
draft chain / req = [t₀, d₁ … d₁₅]   (16)
```

### §7 — Target verify (target side, not draft)

```
target forward over [t₀,d₁..d₁₅] (chain causal mask, hidden capture on)
vLLM: generic RejectionSampler → accept longest prefix + new bonus t₀'
      capture committed tokens' K-layer hidden → back to §2/§4 (re-materialize) → next round §5
```

---

## Implementation checklist

| Stage | What to implement | Key points |
|---|---|---|
| §2 fc | `combine_hidden_states` | `ReplicatedLinear(K·H_t→H_d)`; K2.6 input = 35840 |
| §3 input prep | one fused kernel (eager torch is fine to bring up first) | emits 6 tensors; slot_mapping looked up from block_table inside the kernel |
| §4 materialize | `precompute_and_store_context_kv` + `_build_fused_kv_buffers` | concat-weight single GEMM + per-layer rms_norm + **one RoPE over all layers** (positions repeat L) + per-layer `do_kv_cache_update` |
| §5 draft fwd | `DFlashQwen3Attention/DecoderLayer/Model.forward` | KV pre-filled, forward only processes the 16 query tokens; non-causal backend; `@support_torch_compile` |
| §6 sample | `_greedy_sample` / `get_top_tokens` | TP-safe sharded argmax; d2t remap |
| §7 verify | reuse the framework `RejectionSampler` | DFlash only produces draft tokens |

### Precision gotchas (most likely cross-impl divergence)

1. **§4-⑤ RoPE**: vLLM rotates all layers in one call (positions repeated L times) vs
   SGLang's per-layer fp32 fused kernel; watch `rotary_dim < head_dim` / non-neox pass-through.
2. **RMSNorm**: `ops.rms_norm` computes variance in fp32, I/O bf16; keep `hidden_norm`
   (§4-①) and `q_norm/k_norm` (§5, §4-④) eps consistent (1e-6).
3. **KV cache dtype**: bf16 by default; if fp8-quantized, §4-⑥ writes and §5 reads must
   carry the scale.

---

## Cross-reference

Companion doc for the SGLang implementation of the same algorithm:
[sgl_dflash.md](sgl_dflash.md). The algorithm (materialize-KV + non-causal mask-block draft
+ chain verify) is identical; the differences are in kernel decomposition (input prep,
materialization fusion axis), the draft's embedding/lm_head source (vLLM has its own +
d2t remap; SGLang shares the target's), and the verify path (generic RejectionSampler vs
DFlash-specific accept kernel).
