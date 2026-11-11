# SGLang v0.5.22 (ROCm, MI355X) DSA decode contract for GLM-5.3-Flash

(Written 2026-11-07 by an Explore agent from the SGLang source in the v0.5.22 ROCm image; file:line citations are relative to
sglang/python/sglang. Settled since from the checkpoint headers (/dev/shm/glm53f): q_a_proj, kv_a_proj_with_mqa, q_b_proj,
o_proj are block-FP8 (e4m3 + fp32 weight_scale_inv per 128 x 128); kv_b_proj and the indexer's wq_b / wk / weights_proj /
index_kpool_compress_gate are bf16; config mhc = true (the mHC boundary is outside this attention module). kv_b bf16 ->
the absorbs take the torch.bmm branch (section 3), to be confirmed in the server trace.)

# One plain-decode step of a GLM-5.3-Flash DSA attention layer in SGLang v0.5.22 on ROCm MI355X (fp8 KV, tilelang DSA, kpool=4)

All citations are `path:line` under `SGL=` an SGLang v0.5.22 checkout's `python/sglang` (for example `$SGL/srt/models/deepseek_v2.py:2055`). Everything below comes from reading the code. Two things are config-dependent and I could not resolve them:
- **Checkpoint quantization format.** I could not find the model's `config.json` on this machine. Section 3 lists the variants (block-fp8, online per-tensor, or bf16 left unquantized).
- **`config.mhc`.** I assume it is off.

---

## 0. Which code path runs (decode, `--attention-backend dsa`, HIP)

| Step | Branch taken and why | Cite |
|---|---|---|
| DSA layer module | Glm5Next builds `DeepseekV2AttentionMLA(..., skip_rope=True)` for non-KDA layers | `$SGL/srt/models/glm5_next.py:720-739` |
| Indexer class | `IndexerKPool`, because `index_kpool=4 > 1` | `$SGL/srt/models/deepseek_v2.py:2115-2139` |
| Every layer runs its own indexer | `indexer_types` is all "full", so `skip_topk=False` and `next_skip_topk=False` | `$SGL/srt/configs/model_config.py:410-418`, `$SGL/srt/models/deepseek_v2.py:2110-2111` |
| Indexer uses `forward_cuda` | `IndexerKPool` defines no `forward_hip`, so HIP falls back to `forward_cuda` | `$SGL/kernels/fused_op.py:153` |
| Forward method | Decode uses `handle_attention_dsa`. `use_mha=False` for decode, so the method is MLA, remapped on HIP to **MLA_ROCM**. That means `forward_absorb_rocm_prepare` / `forward_absorb_rocm_core` | `$SGL/srt/models/deepseek_common/attention_backend_handler.py:206-217, 32-43`; `$SGL/srt/layers/attention/dsa_backend.py:4016-4017`; `$SGL/srt/models/deepseek_v2.py:2421-2429, 2491-2492` |
| No RoPE anywhere in the MLA part | `skip_rope=True` and `qk_rope_head_dim=0`, so `rotary_emb=None` and `scaling` keeps no yarn mscale | `$SGL/srt/models/deepseek_v2.py:2161-2176` |
| No RoPE in the indexer | `apply_rope = not skip_rope and rope_head_dim>0` is False | `$SGL/srt/layers/attention/dsa/dsa_indexer_kpool.py:667` |
| Single stream | The model's `alt_stream` is None on ROCm unless `SGLANG_ROCM_USE_MULTI_STREAM` is set (default False), so there is no dual-stream overlap | `$SGL/srt/models/glm5_next.py:1038-1046`; `$SGL/srt/environ.py:972` |
| Core branch | The `rotary_emb is None` test fails at `forward_mla_rocm.py:831`, so the code takes the plain `attn_mqa(q_nope_out, k_nope, k_nope, ..., q_rope=q_pe, k_rope=k_pe, topk_indices=...)` call (no fused rope/cache kernels). `save_kv_cache=True` | `$SGL/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:828-941` |
| Backend decode implementation | `dsa_decode_impl="tilelang"`. `_resolve_kpool_tail_backend` leaves tilelang unchanged | `$SGL/srt/layers/attention/dsa_backend.py:2529-2530, 2645-2658`; `$SGL/srt/layers/attention/dsa/dsa_backend_kpool.py:61-77` |
| Fused top-k is on | `SGLANG_DSA_FUSE_TOPK=True` and not under PD, so `use_fused_topk=True`. The transform method is PAGED (it is only RAGGED for EXTEND with flashmla) | `$SGL/srt/environ.py:1831`; `$SGL/srt/layers/attention/dsa/utils.py:181-196`; `$SGL/srt/layers/attention/dsa_backend.py:651, 4038-4062` |

---

## 1. Decode dataflow, per GPU (TP4 → 16 local heads; T = batch rows)

### 1.0 Input
- `x` is the **output of `input_layernorm`** (RMSNorm), applied in `attn_boundary.prepare` (`$SGL/srt/models/glm5_next.py:774-825, 966-968`).
- No `quant_format` is passed there, so `x` is plain bf16 `[T,4096]`. The same `x` goes to the MLA projections and to the indexer (`forward_mla_rocm.py:643-649`).

### 1.1 Fused q/kv down-projection (replicated)
- `fused_qkv_a_proj_with_mqa` is a `ReplicatedLinear(4096 → 1536+512+0 = 2048)` (`deepseek_v2.py:2066-2072`).
- It is called through `prepare_qkv_latent`; the min-latency fused_a_gemm is CUDA-only (`$SGL/kernels/ops/gemm/fused_a_gemm.py:35-42`).
- It is fetched lazily in `fetch_qkv_latent` (`$SGL/srt/layers/layer_boundary/adapters/attention.py:61-70`; `deepseek_v2.py:2506-2531`).
- It is split into `q[T,1536]` and `latent_cache[T,512]`; `k_nope = latent_cache[..., :512]` (`forward_mla_rocm.py:513-521`).

### 1.2 Norms
- **Which norm kernel runs.** The block-fp8 fused path `fused_rms_fp8_group_quant` needs `_is_block_scale_fp8(q_b_proj)`. That function checks for a 2-D `weight_scale` attribute (`$SGL/srt/models/deepseek_common/utils.py:79-93`). Fp8Config block layers only register `weight_scale_inv` (`$SGL/srt/layers/quantization/fp8.py:681,688`; the only `weight_scale =` sets are on non-block paths at 1134/1140/1216). So under `--quantization fp8` this is False and the code takes `elif _use_aiter:` → **aiter `fused_qk_rmsnorm`** (quant_type=No) (`forward_mla_rocm.py:584-592, 104-124`).
- **Output.** bf16 `q = q_a_layernorm(q)` `[T,1536]` and bf16 `k_nope = kv_a_layernorm(k_nope)` `[T,512]`.
- `q_lora = q`, the normed bf16, which the indexer uses (`forward_mla_rocm.py:598-600`).
- `k_nope` is then unsqueezed to `[T,1,512]` (`forward_mla_rocm.py:631`).

### 1.3 q up-projection (head-split)
- `q_b_proj` is a `ColumnParallelLinear(1536 → 64*256)`, which is **16×256 = 4096 per GPU**. Output is viewed `[T,16,256]` bf16 (`deepseek_v2.py:2074-2081, 2533-2545`).
- The min-latency path is CUDA-only, and the weight shape is not in the verified set anyway (`deepseek_v2.py:2244-2251`).
- `q_nope = q` (all 256 dims). `q_pe` and `k_pe` have **width 0** (`$SGL/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:196-203`).

### 1.4 Indexer (`IndexerKPool._forward_cuda_impl`, all heads replicated on every GPU)
Code: `$SGL/srt/layers/attention/dsa/dsa_indexer_kpool.py:1700-1871`. Single-stream decode means the branch at `:1830-1835` and then `_get_topk_paged` (`:1840-1848`).

1. **Query projection.** `query = wq_b(q_lora)`: `ReplicatedLinear(1536 → 32*128)`, quantized like the other linears. Rearranged to `[T,32,128]` bf16 (`:207-213, 705-709`).
2. **Key projection and norm.**
   - `key = wk(x)`: `ReplicatedLinear(4096 → 128)` (`:215-221, 710`).
   - `key = k_norm(key)`, where `k_norm` is `LayerNorm(128, dtype=fp32, eps=1e-6, with bias)` (`:231, 711`).
   - On HIP, x is bf16 but the module dtype is fp32, so it runs `forward_native`: cast to fp32, `F.layer_norm`, cast **back to bf16** (`$SGL/srt/layers/layernorm.py:1013-1029, 1044-1075`).
3. **Hadamard on the query only.** `query = rotate_activation(query)` uses `fast_hadamard_transform.hadamard_transform(scale=128^-0.5)` on HIP; the output is bf16 (`dsa_indexer_kpool.py:722-723`; `$SGL/srt/layers/attention/dsa/dsa_indexer.py:241-253`). The key is **not** rotated here; it is rotated after pooling (step 6).
4. **Query quantization.** `q_fp8, q_scale = act_quant(query, 128, "ue8m0")` runs the TileLang `act_quant_kernel` (`scale_fmt="ue8m0"` comes from `deepseek_v2.py:2127`):
   - One group per head (N=128).
   - `amax = max(|x|, 1e-4)`.
   - **Power-of-two scale** `2^ceil(log2(amax/448))`.
   - `y = clamp(x/s, ±448)` stored as fp8 e4m3fn; `q_scale` is fp32 `[T,32,1]`.
   - Code: `dsa_indexer_kpool.py:1710, 1832`; `$SGL/kernels/ops/attention/dsa/tilelang_kernel.py:58-71, 91-169`.
5. **Compression gate.** `gate_score = F.linear(x, index_kpool_compress_gate)`. The gate weight is a plain bf16 Parameter `[128,4096]`, so the GEMM is unquantized and the output is bf16 `[T,128]` (`dsa_indexer_kpool.py:196-198, 611-612`).
6. **Pool compression and index-cache write.** `_compress_write` → `_compress_write_decode` → `pool.kpool_decode_update_index_cache` → Triton `_kpool_decode_update_and_maybe_write_cache_kernel`, one program per row (`dsa_indexer_kpool.py:599-630, 348-376`; `$SGL/srt/mem_cache/memory_pool.py:5492-5528`; `$SGL/srt/layers/attention/dsa/kpool_fp8_index.py:1015-1103, 1250-1417`). The kernel:
   - Sets `slot = pos % 4` and `phys = pos % TAIL_SIZE`, with `TAIL_SIZE = 4 + tail_extra_slots`, which is 4 without speculative decoding.
   - Treats the row as valid only if `0 ≤ req < REQ_POOL_SIZE`, `out_cache_loc != 0`, and `0 ≤ pos < seq_len` (`kpool_fp8_index.py:1290-1301`).
   - **If `slot == 3` (the pool closes on this token):**
     - For each of the 4 slots it takes the tail ring entry, or the current key/score for the current slot, and adds `ape[slot]` (fp32 `[4,128]`).
     - It computes a **per-dimension softmax over the 4 slots** and the weighted sum `acc/denom` in fp32.
     - It rounds the result to **bf16**, applies `_hadamard128` (butterfly, ×1/√128), and rounds to **bf16** again.
     - It quantizes one scale per 128-vector: `absmax ≥ 1e-4`, scale `2^ceil(log2(absmax/448))` (ROUND_SCALE because `scale_fmt` is not None), values clamped to ±448.
     - It stores fp8 and the fp32 scale into the index-K page (`kpool_fp8_index.py:1314-1409`; layout in §2.2).
   - **Always:** it writes the raw bf16 `key` and bf16 `score_current` into the tail ring at `[req, pos%4]` (`:1411-1417`).
7. **Head-gate weights.** `_get_logits_head_gate` is `torch.compile`d:
   - `weights_proj(x.float())`: an **fp32**, unquantized `ReplicatedLinear(4096 → 32)`.
   - `× 32^-0.5`, then `× q_scale × 128^-0.5`, giving fp32 `[T,32,1]`, later squeezed to `[T,32]`.
   - Code: `dsa_indexer_kpool.py:222-230, 246-251, 1835, 1021`.
8. **Pooled metadata.** Recomputed in torch **for every layer** on HIP, because `pooled_*` stays None (see §4):
   - `pool_seqlens = cache_seqlens // 4`.
   - `pool_block_tables = real_page_table[:, ::4] // 4`.
   - Code: `dsa_indexer_kpool.py:852-876`; `kpool_fp8_index.py:59-69`.
9. **Paged MQA logits over pooled keys.** `aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits(Preshuffle=True, KVBlockSize=64)`. The backend is forced to AITER on HIP.
   - Output: fp32 `logits[T, pool_block_tables.shape[1]*64]`.
   - Code: `dsa_indexer_kpool.py:1013-1062`; `$SGL/kernels/ops/attention/dsa/paged_mqa_logits.py:107-135`; `$SGL/srt/layers/attention/dsa/paged_mqa_logits_backend.py:24-31`.
   - Semantics, from the torch reference: `logit[t,j] = k_scale[j] · Σ_h w[t,h] · relu(q_fp8[t,h]·k_fp8[j])` (`$SGL/srt/layers/attention/dsa/utils.py:463-492`; same as the TileLang version at `tilelang_kernel.py:1438-1442`). I could not read the aiter source (aiter is not installed here).
10. **Top-k with the tail rule.** `topk_from_pooled_history_logits(kpool=4, topk=2048)`:
    - `group_topk = 512`. topk-v2 is CUDA-only, so it uses the **JIT kernel `fast_kpool_topk_transform_fused`** (`$SGL/kernels/jit/csrc/dsa/kpool_topk_transform.cuh`; it has HIP includes) (`kpool_fp8_index.py:764-840, 593-594`; `$SGL/kernels/ops/attention/dsa/kpool_topk_transform.py`).
    - Per row (`kpool_topk_transform.cuh:295-355`):
      - `length = pool_seqlens` is the number of closed pools.
      - If `length ≤ 512`, **all pools are taken in order**; otherwise a radix top-512 over the pool logits (`:62-278`).
      - Each selected pool g expands to tokens `4g..4g+3`; `history_len = min(length*4, 2048)`.
      - Then **tail tokens** `length*4 + i` for `i < seq_len % 4` are appended. These are the still-open pool, and include the current token when `pos%4 != 3`.
      - Every raw token index is mapped through `page_table_1[row, tok]`, i.e. `req_to_token`, to a **physical KV loc**. The rest is padded with −1.
    - Output: int32 `[T, 2048+3 = 2051]`.
    - Note: if `length > 512`, the pool just closed by the current token is not guaranteed to be selected.

### 1.5 q absorption (W_UK, head-split)
- `w_kc` is `[16,256,512]` (column-major storage) from kv_b_proj (`$SGL/srt/models/deepseek_common/deepseek_weight_loader.py:734-758`).
- **If `w_kc` is fp8 e4m3fn:** `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` (aiter Triton). It quantizes q_nope to fp8 internally per token per group of 128, uses the **per-tensor** weight scale `attn.w_scale`, and outputs bf16 `[16,T,512]`, transposed to `[T,16,512]` (`forward_mla_rocm.py:196-246, 723-728`; tiles at `:156-182`).
- **Otherwise:** `torch.bmm` in bf16 (`:241-245`).

### 1.6 MLA KV write, then sparse attention (`DeepseekSparseAttnBackend.forward_decode`)
1. **Latent write.** `set_mla_kv_buffer(layer, out_cache_loc, k_nope[T,1,512], k_rope[T,1,0])` (`dsa_backend.py:2554-2567`) reaches `DSATokenToKVPool._write_mla_kv_buffer`. On HIP with DSA and fp8 this is `set_mla_kv_buffer_triton_fp8_quant`: a **plain bf16 → fp8 e4m3fn cast, no scale**; writes to loc 0 are skipped (`$SGL/srt/mem_cache/memory_pool.py:4996-5006`; `$SGL/kernels/ops/kvcache/mla_buffer.py:332-427`).
2. **Query.** `q_all = torch.cat([q_nope, q_rope(width 0)])` gives a contiguous `[T,16,512]` bf16 tensor (`dsa_backend.py:2571-2580, 2650-2651`; `$SGL/kernels/ops/attention/utils.py:210-214`).
3. **Indices.** `page_table_1 = topk_indices` unchanged, because the fused top-k with the sgl-kernel backend already returns physical locs (`dsa_backend.py:2600-2601, 992-1000`).
4. **`_forward_tilelang`.** It right-pads the indices with −1 to a multiple of 64 (2051 → 2112) (`dsa_backend.py:3406-3435`), then calls `tilelang_sparse_fwd` (`tilelang_kernel.py:1277-1344`):
   - **`q.to(fp8)` is an unscaled cast** of the absorbed bf16 query.
   - Partial kernel `sparse_mla_fwd_decode_partial_fp8`, with `d_v=512`, `d_tail=0`, `block_I=64`, 256 threads, 16 heads per block, `inner_iter` from `_pick_inner_iter(T, 33, 256, 2)` (`:1299-1314, 74-88, 1000-1274`). Per index block it:
     - computes the fp8×fp8 QKᵀ score over 4×128 tiles in fp32;
     - masks invalid (−1) indices to −inf;
     - runs online softmax with `exp2(s·sm_scale·log2e)`;
     - multiplies P by 448, casts P to fp8, computes P·V with V = the same 512-dim latent rows, and multiplies by 1/448 (`:1017-1023, 1216-1234`);
     - writes the partial O as **bf16** and the LSE as fp32.
   - Combine kernel `sparse_mla_fwd_decode_combine` produces bf16 output `[T,16,512]` (`:914-997`).
   - **Softmax scale:** `layer.scaling = qk_head_dim^-0.5 = 256^-0.5 = 1/16` (`deepseek_v2.py:2055, 2179-2188`).
   - **Positions attended:** the 2048 pooled-history tokens plus up to 3 tail tokens.

### 1.7 o absorption (W_UV) and o_proj
1. **W_UV absorb.** `w_vc` is `[16,512,256]` (`deepseek_weight_loader.py:755-758`).
   - **If `w_kc` is fp8:** the same aiter Triton a8w8 batched GEMM (per-128-group activation quant, per-tensor weight scale), writing a bf16 `(T,16,256)` buffer.
   - **Otherwise:** a bf16 `torch.bmm`.
   - The o_proj is not "block-scale" by `_is_block_scale_fp8`, so the result is just `flatten` → bf16 `[T,4096]` (`forward_mla_rocm.py:249-350, 1074`).
2. **o_proj.** `RowParallelLinear(64*256 → 4096)` takes 4096 inputs per GPU and has `reduce_results=False`. Its output is a **partial sum** `[T,4096]` bf16 (`deepseek_v2.py:2150-2158`; `forward_mla_rocm.py:1088`).
   - The return value is `(output, None)` because `next_skip_topk=False` (`:1090-1097`).
   - The TP all-reduce happens later in `attn_boundary.finish` (`glm5_next.py:984`).

### bf16 rounding and fp8 quantization points
- **Linear layers:**
  - **Fp8 activation quant:** inside every fp8 GEMM (qkv_a, q_b, wq_b, wk, o_proj).
  - **bf16 outputs:** the qkv_a latent, the normed q and k_nope, q, query, key.
- **Indexer query:** Hadamard output is bf16; `act_quant` gives fp8 with a pow2 per-head scale.
- **Compression gate:** gate_score is bf16, and so is the tail ring.
- **Pooled index key:**
  - bf16 → Hadamard → bf16;
  - fp8 with a pow2 per-128 scale.
- **Absorb GEMMs:** the q/v absorb GEMMs quantize their activations per 128-group to fp8 and output bf16.
- **MLA latent and attention query:**
  - The MLA latent is stored as an unscaled fp8 cast.
  - The query is cast unscaled to fp8 inside `tilelang_sparse_fwd`.
- **Inside the attention kernel:**
  - P is scaled by 448 and cast to fp8.
  - Partial O is bf16; the combined output is bf16.

---

## 2. Cache state read and written per DSA layer

The pool is `HybridLinearKVPool` (KDA plus DSA layers). It holds `full_kv_pool = DSATokenToKVPool` and maps a global layer_id to a dense DSA index via `_transfer_full_attention_id` (`$SGL/srt/mem_cache/kv_cache_configurator.py:2397-2502`; `memory_pool.py:4304-4330, 4465-4470, 4614-4625, 4656-4734`). `pool.page_size = 256`.

### 2.1 MLA latent KV (read and written)
- **Dimension and dtype.**
  - `kv_cache_dim = 512`: on HIP with tilelang, `calculate_mla_kv_cache_dim` returns `kv_lora + rope = 512+0` with no scale bytes (`kv_cache_configurator.py:3232-3238`).
  - Therefore `dsa_kv_cache_store_fp8=False` (`memory_pool.py:4791-4802`).
  - The dtype is `torch.float8_e4m3fn` (fp8_dtype on gfx950, non-fnuz) (`$SGL/srt/mem_cache/kv_cache_dtype.py:56-58`; `$SGL/kernels/ops/quantization/fp8_kernel.py:102-106`).
- **Buffer.** `kv_buffer[l]` has shape `[size + 256, 1, 512]` and is **stored as uint8**, viewed as fp8 by `get_key_buffer` (`memory_pool.py:4869-4876, 4902-4905, 2054-2057`).
- **Per-token cost.** 512 B per token, no per-token scale; the value is a direct cast (scale 1). The FlashMLA-style 528-B scaled layout is not used.
- **Row 0** is a dummy pad slot; writes to loc 0 are skipped (`mla_buffer.py:362-364, 398-412`).
- **Writer:** Triton `set_mla_kv_buffer_fp8_quant_kernel`, grid `(T, 4)`, BLOCK=128 (`mla_buffer.py:332-427`).
- **Reader:** the TileLang partial kernel, gathering `kv[idx, 0, :512]` for both K and V (`tilelang_kernel.py:1146-1152`).

### 2.2 Indexer pooled-key cache (read and written)
- **Buffer** (`$SGL/srt/mem_cache/index_key_cache.py:16-49`):
  - `IndexKeyCache.buffer[l]` is uint8 `[num_pages, 64*(128+4) = 8448]`.
  - `num_pages = (index_buf_size + 256 + 1)//256`.
  - `index_buf_size = size` with no DCP (`memory_pool.py:5342-5349`).
  - `index_page_size = page_size/kpool = 64` (`memory_pool.py:5339`).
  - The kernel view has the same shape (`kernel_pages_per_row = 256/4/64 = 1`).
- **What a page holds.** One row, i.e. one 256-token KV page, holds **64 pooled keys**.
- **Layout per row (preshuffled; tile 16).** HIP requires preshuffle, otherwise the pool asserts page size 1 (`memory_pool.py:5361-5369`; `$SGL/srt/layers/attention/dsa/utils.py:150`).
  - **Bytes [0, 8192): fp8 values.** Offset = `(slot//16)*2048 + (col//16)*256 + (slot%16)*16 + col%16`, where `slot` is the pooled key in the page (0..63) and `col` is 0..127 (`kpool_fp8_index.py:28-52`).
  - **Bytes [8192, 8448): fp32 scales.** 64 values, one per pooled key; the scale is at fp32 index `page*8448/4 + 2048 + slot` (`kpool_fp8_index.py:1402-1409`; `S_OFFSET_NBYTES_IN_PAGE = 64*128` at `:1099`).
  - Each scale is a power of 2 (ue8m0-style ceil) per 128-dim pooled key. The values are Hadamard-rotated.
- **Location of pooled key g = pos//4** for request row b (`kpool_fp8_index.py:1383-1393`):
  - `page = real_page_table[b, (g//64)*4] // 4`. That is `req_to_token[r, 256*(g//64)] // 256`, the 256-token KV page of that token block.
  - `slot = g % 64`.
- **Read side.** `_get_topk_paged` reinterprets the buffer as `[num_pages, 64, 1, 132]` for aiter (`dsa_indexer_kpool.py:1013-1019`).

### 2.3 Tail ring for the open pool (read and written)
- **Buffers.** `_compress_tail_k[l]` and `_compress_tail_score[l]` are each **bf16 `[R, 4, 128]`** (`memory_pool.py:5425-5464`).
  - `R = req_to_token.shape[0] + 1`.
  - The width is `index_kpool + tail_extra_slots`, which is 4 without speculative decoding (`kv_cache_configurator.py:2441-2445`).
- **Indexing.** Row = `req_pool_indices[b]`, slot = `pos % 4`.
- **Contents.** The raw post-k_norm key (pre-Hadamard) and the raw gate score (pre-ape) of tokens in the open pool.
- **Read and write.** The decode kernel reads 3 entries and writes 1 (`kpool_fp8_index.py:1300-1301, 1317-1363, 1411-1417`).

### 2.4 Mapping a position to a cache location
- **req_to_token.** `req_to_token_pool.req_to_token` is int32 `[size+1, max_context_len]`; row 0 is padding (`memory_pool.py:398-406`).
- **Decode allocation.** It writes `req_to_token[req, old_seq_len] = out_cache_loc` with 256-aligned paged slots, then seq_lens is incremented (`$SGL/srt/mem_cache/allocation.py:625-665`).
- **Positions.** `positions = clamp(seq_lens-1, 0)` (`$SGL/srt/model_executor/forward_batch_info.py:1182-1184`).
- **Writers in a decode step:**
  - The Triton fp8 cast kernel writes the MLA latent.
  - `_kpool_decode_update_and_maybe_write_cache_kernel` writes the tail ring and, on pool close, the pooled index key.
  - There is no other index-K writer in decode.

---

## 3. Kernels on ROCm, in order (single stream)

| # | Step | Kernel or library | Source |
|---|---|---|---|
| 1 | qkv_a GEMM (4096→2048) | **Block-fp8 ckpt:** `aiter_w8a8_block_fp8_linear` = `aiter_per1x128_quant` + CK `gemm_a8w8_blockscale[_bpreshuffle]` or `triton_gemm_a8w8_blockscale`. **bf16 ckpt + online quant:** per-tensor `apply_fp8_linear` | `$SGL/srt/layers/quantization/fp8_utils.py:1145-1163, 1561-1680`; `fp8.py:1105-1134, 1328-1402` |
| 2 | q_a / kv_a RMSNorm | aiter `fused_qk_rmsnorm` (bf16) | `forward_mla_rocm.py:104-124, 584-592` |
| 3 | q_b_proj (1536→4096 local) | same GEMM family as #1 | `deepseek_v2.py:2533-2545` |
| 4 | wq_b (1536→4096), wk (4096→128) | same GEMM family (replicated) | `dsa_indexer_kpool.py:207-221` |
| 5 | k_norm | torch `F.layer_norm` in fp32 | `layernorm.py:1044-1058` |
| 6 | Hadamard(query) | `fast_hadamard_transform` | `dsa_indexer.py:241-253` |
| 7 | act_quant(query) | TileLang `act_quant_kernel` | `tilelang_kernel.py:91-169` |
| 8 | gate GEMM | `F.linear` bf16 (hipBLASLt) | `dsa_indexer_kpool.py:612` |
| 9 | pool compress, tail ring, index-K write | Triton `_kpool_decode_update_and_maybe_write_cache_kernel` | `kpool_fp8_index.py:1250-1417` |
| 10 | head weights | `torch.compile` of an fp32 linear plus elementwise ops | `dsa_indexer_kpool.py:246-251` |
| 11 | pool_seqlens / pooled block table | torch div / index / floordiv | `dsa_indexer_kpool.py:862-867`; `kpool_fp8_index.py:59-69` |
| 12 | paged MQA logits | aiter Triton/Gluon `deepgemm_fp8_paged_mqa_logits` (Preshuffle) | `paged_mqa_logits.py:107-135` |
| 13 | top-512 pools, expand to 2048, tail, map to loc | JIT `kpool_topk_transform.cuh` (radix, 1024 threads) | `$SGL/kernels/jit/csrc/dsa/kpool_topk_transform.cuh:295-466` |
| 14 | W_UK absorb | aiter Triton `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` (fp8 w_kc) or `torch.bmm` | `forward_mla_rocm.py:196-246` |
| 15 | MLA latent write | Triton `set_mla_kv_buffer_fp8_quant_kernel` | `mla_buffer.py:332-427` |
| 16 | q cat, index pad, q→fp8 cast | torch | `dsa_backend.py:2651, 3419-3427`; `tilelang_kernel.py:1294-1295` |
| 17 | sparse MLA | TileLang `sparse_mla_fwd_decode_partial_fp8` + `sparse_mla_fwd_decode_combine` | `tilelang_kernel.py:914-1274` |
| 18 | W_UV absorb | same aiter Triton batched GEMM, or `torch.bmm` | `forward_mla_rocm.py:249-350` |
| 19 | o_proj | GEMM family #1; all-reduce later | `forward_mla_rocm.py:1088` |

**kv_b_proj → w_kc/w_vc** (`deepseek_weight_loader.py:607-702, 724-767`):
- **Block-fp8 checkpoint on HIP:** `block_quant_to_tensor_quant` gives fp8 e4m3fn with a per-tensor `w_scale`, so the aiter a8w8 batched GEMMs are used.
- **Unquantized bf16 checkpoint:** `w_kc` stays bf16 (the GLM bf16→fp8 requant only matches arch `GlmMoeDsaForCausalLM`, at `:726-732`), so `torch.bmm` is used.

---

## 4. Batch-dependent state and where it lives

**`forward_batch`:**
- `seq_lens` `[bs]`, including the current token.
- `req_pool_indices` `[bs]`.
- `out_cache_loc` `[bs]`; 0 for cuda-graph pad rows.
- `positions` `[bs]` = seq_lens−1.

**Backend `forward_metadata` (`DSAMetadata`, `dsa_backend.py:241-311`):**
- **Eager build:** `init_forward_metadata`, decode branch (`:1083-1150, 1320-1395`).
- **Graph capture:** `_build_forward_metadata_cuda_graph` (`:1597-1765`).
- **Graph replay:** `_apply_cuda_graph_metadata` (`:1767-2083`). On HIP with kpool>1 the fused metadata kernel is **not** used, because `experimental_kpool_metadata_fusion` is CUDA-only (`$SGL/srt/layers/attention/dsa/dsa_metadata_manager.py:39-48`). Replay does torch copies instead (`:1837-1851, 2055-2059`).

Fields that matter:
- `cache_seqlens_int32 [bs]` = seq_lens. The indexer reads it via `get_seqlens_int32` for pool lengths, the tail count, and validity (`$SGL/srt/layers/attention/dsa/dsa_indexer_metadata.py:102-109`).
- `page_table_1 [bs, W]` int32 = `req_to_token[req_pool_indices, :W]`, with `W = req_to_token.shape[1]` in graphs. It is the static buffer `decode_cuda_graph_metadata["page_table"][:bs]`. HIP never drops the wide table (`dsa_backend.py:1496-1522, 1624, 1843-1844`).
- `real_page_table [bs, ceil(W/64)]` = `page_table_1[:, ::64] // 64` (64-token page ids). Built at `:1054-1062, 1380, 1728, 2055-2059`.
- `dsa_cache_seqlens_int32` (`compute_dsa_seqlens`) is computed but not read by the tilelang decode path (`$SGL/srt/layers/attention/dsa/utils.py:157-167`).
- `kpool_write_plan`: `req`, `write_start = seq_lens−1`, `write_loc`. It is refreshed each step by the Triton `update_kpool_write_plan_cuda_graph` (`$SGL/srt/layers/attention/dsa/dsa_backend_kpool.py:125-135, 222-245`; `$SGL/srt/layers/attention/dsa/kpool_plan.py:753-868`; `$SGL/srt/layers/attention/dsa/kpool_fp8_index.py:306`). The plain-decode compress path does **not** read it, and `pool_seqlens_per_q` is None in decode.
- `pooled_cache_seqlens_int32` and `pooled_real_page_table` stay **None on HIP**: `init_pooled_paged_mqa_metadata` returns early when not `is_cuda()` (`kpool_plan.py:588-593`). So each layer recomputes them (`dsa_indexer_kpool.py:857-867`).

**Pool-side state:**
- `req_to_token`.
- The tail rings, indexed by `req_pool_indices`.
- The index-K and MLA buffers, indexed by loc/page as in §2.

**Index-K buffer size:** `num_pages = (index_buf_size + 256 + 1)//256` rows per DSA layer, with `index_buf_size = size` (`memory_pool.py:5342-5349`; `index_key_cache.py:16-49`).

