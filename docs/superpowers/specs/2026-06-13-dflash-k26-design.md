# DFlash (Kimi K2.6) on MPK — Design Spec

Date: 2026-06-13 · Branch: `dflash-k26` (forked from `mpk`)

Goal: implement Kimi-K2.6 DFlash speculative-decoding **draft** support in MPK and complete
three alignment stages — **kernel → layer → e2e** — single-card first, TP=8 as the final phase.

## Oracles & environments
- **MPK dev env**: conda `mirage00` (torch 2.11+cu130, transformers 4.57.1, mirage installed).
- **vLLM oracle**: conda `vllm`, source checkout at `/home/letianr/vllm` with real DFlash
  (`vllm/v1/spec_decode/dflash.py`, `model_executor/models/qwen3_dflash.py`). This is the
  e2e oracle.
- **HF reference**: `/raid/catalyst/models/Kimi-K2.6-DFlash-tmp/dflash.py` (`DFlashDraftModel`),
  the per-op / per-layer numerical oracle (run with real sliced draft weights).
- **Draft weights**: `/raid/catalyst/models/Kimi-K2.6-DFlash-tmp` (downloaded via
  `huggingface_hub` from `SubSir/Kimi-K2.6-DFlash-tmp`; git-lfs not installed on host).
- **Target** (`/raid/catalyst/models/Kimi-K2.6`, ~1T MLA+MoE): **out of scope**. The draft
  consumes *dumped* `target_hidden`; alignment never runs the target in MPK.

## K2.6 draft config (authoritative — differs from the b16 example in `vllm_dflash.md`)
`B`(block)=8, `s`=7, `K`(captured layers)=6 `[1,12,24,35,47,58]`, `L`(draft layers)=6,
`H_d`=`H_t`=7168, `n_q`/`n_kv`=64/8, `d`=128, `q_size`=8192, `kv_size`=1024, `I`=18432,
`MASK`=163838, vocab=163840, `rms_norm_eps`=**1e-5**, RoPE=**YaRN** (theta=50000, factor=64,
orig_max_pos=4096, β_fast=32/β_slow=1, mscale=1), attention = **sliding_window=2048 on layers
0–4, full on layer 5**. dtype bf16; norm/rope/softmax internal fp32. q/k/v projections are
**separate** in the checkpoint (fuse to MPK's QKV at load). Draft has **no embed/lm_head/d2t**
— it shares the target's embedding + lm_head.

## Reference math (from `dflash.py`)
Model-level once: `ctx = hidden_norm(fc(target_hidden_concat[S, K·H_t]))`  → `ctx[S,H_d]`.
Per draft layer i (×L), with `h` = block hidden (`bs·B` tokens):
```
r=h; h=input_layernorm(h)
q=q_norm(q_proj(h));  k_noise=k_proj(h); v_noise=v_proj(h)
k_ctx=k_proj(ctx);    v_ctx=v_proj(ctx)
k=k_norm(cat([k_ctx,k_noise])); v=cat([v_ctx,v_noise]);  RoPE(q,k)   # YaRN
attn: q (bs·B) attends to [ctx_len + bs·B] NON-CAUSAL; sliding_window on L0–4, full L5
h = r + o_proj(attn);  r=h; h=post_attention_layernorm(h); h = r + mlp_silu(h)
```
Final: `final_hidden = norm(h)`. Token = argmax(target_lm_head(final_hidden[mask_slots])).

KV strategy = **vLLM-style materialize** (chosen): `k_ctx/v_ctx` are precomputed once from
`ctx` (k_norm + YaRN-RoPE at context positions) and written to the draft paged KV cache; the
draft forward computes only block KV and reads context KV from cache. Math identical to the
per-layer recompute above.

## MPK pipeline boundary (alignment interface)
- **Input**: `noise_embedding [bs·B, H_d]` (pre-embedded `[t₀, MASK×s]`) + `target_hidden
  [S, K·H_t]` + `target_positions[S]` + `query_positions[bs·B]` + paged-KV metadata.
- **Output**: `final_hidden [bs·B, H_d]`. Token selection (lm_head+argmax+d2t) is a thin
  wrapper aligned separately (uses target lm_head weight, dumped).

## Section 1 — New kernels (TP-aware from the start)
| # | kernel | inputs | outputs | notes |
|---|---|---|---|---|
| K1 | `dflash_input_prep` | `t₀[bs]`, `target_positions[S]`, `block_table`, `query_start_loc`, `seq_lens`, `block_size` | `out_input_ids[bs·B]`, `query/context_positions`, `query/context_slot_mapping`, `token_indices_to_sample[bs·s]` | eager-torch first, then kernel |
| K2 | `dflash_kv_store` | normed+RoPE'd `K,V [L,S,n_kv,d]`, `context_slot_mapping` | writes draft paged KV cache | first standalone KV-write in MPK |
| K3 | `dflash_noncausal_attn` | `q[bs·B,n_q,d]`, block `k/v`, context KV (cache), `slot_mapping`, `sliding_window` | `attn_out[bs·B,q_size]` | non-causal; sliding L0–4 + full L5; biggest new kernel |
| K4 | `dflash_gather_sample` | `final_hidden[bs·B,H_d]`, `token_indices_to_sample` | `sample_hidden[bs·s,H_d]` | small index-gather |

## Section 2 — Reused kernels (verified, not trusted)
RMSNorm (**eps 1e-5**), Linear (fc/qkv/o/gate_up/down), silu_mul, argmax, target_verify,
concat, copy_layer. Plus **YaRN RoPE** — MPK rotary may be plain neox; verify/extend (risk).
Each gets a quick numeric check vs the PyTorch op in Phase A.

## Section 3 — Layers (each aligned to the reference's corresponding op)
- `L_fc_norm`: fc → hidden_norm → `ctx`.
- `L_materialize` (§4): per-layer k/v_proj(ctx) → k_norm → YaRN-RoPE → **K2** store. Align:
  cache == reference `k_ctx/v_ctx` after norm+rope.
- `L_draft_layer` ×6: input_norm → qkv → q_norm/k_norm → RoPE → **K3** → o_proj+res →
  post_norm → silu MLP+res. Align: per-layer hidden.
- `L_final_norm` → `final_hidden`. Align: full-model hidden.
- `L_sample`/`L_verify`: K4 gather → target lm_head → argmax → d2t → target_verify.
- Python `DFlashBuilder` mirrors `Eagle3Builder` but a **single non-causal forward** (not the
  Eagle3 K-step autoregressive loop).

## Section 4 — Alignment harness (`demo/qwen3/dflash_correctness/`)
- `ref_dump.py`: load `dflash.py` + sliced single-layer weights, register forward hooks, dump
  every op/layer tensor for fixed inputs (`.pt` files).
- Per-kernel test (MPK test-mode, same inputs, `torch.allclose` at bf16 tol ~2e-2 rel).
- Per-layer test (1 DecoderLayer), then full L=6.
- e2e: feed dumped `target_hidden`/`noise_embedding`; match draft tokens to HF reference and,
  where runnable, to a vLLM dflash dump from `/home/letianr/vllm`.

## Section 5 — Phasing (tasks #1–#5)
- **P0** setup: branch, weights, `ref_dump.py`, single-layer slice. *(in progress)*
- **PA** kernels: K1–K4 + YaRN + reused re-tests — each aligned (stage 1).
- **PB** layers: materialize-KV, 1 draft layer, full L=6 — each aligned (stage 2).
- **PC** e2e: sample+verify, draft-token match vs reference/vLLM (stage 3).
- **PD** TP=8: shard heads, replicate fc, vocab-parallel lm_head, allreduce; TP8==TP1.

## Risks
1. **YaRN RoPE** parity with MPK rotary (most likely divergence — `vllm_dflash.md` §gotcha 1).
2. **Sliding-window** in the non-causal attention kernel (new mask logic).
3. eps mismatch (1e-5 vs 1e-6) silently degrading norm alignment.
4. Standalone KV-cache write (K2) — first of its kind in MPK; paged-cache layout must match
   what K3 reads.
