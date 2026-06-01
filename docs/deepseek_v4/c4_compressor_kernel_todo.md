# DeepSeek V4 C4 Compressor Kernel TODO

This note tracks the implementation decisions for the MPK
`dsv4_c4_compress_sm100` task. The branch now contains a correctness-first CUDA
implementation for the DeepSeek V4 Flash Base CSA C4 compressor/cache insert.
It is deliberately not a final performance kernel yet.

## Target

- Model: `deepseek-ai/DeepSeek-V4-Flash-Base`.
- Attention component: CSA C4 KV compressor and cache insert.
- First supported architecture: SM100 / Blackwell.
- First supported constants:
  - `compress_ratio = 4`
  - `overlap = true`
  - `head_dim = 512`
  - `rope_head_dim = 64`
  - `nope_head_dim = 448`
  - `coff = 2`
  - `kv_score_dim = 2048`

## Source References

- DeepSeek official Hugging Face inference:
  - `DeepSeek-V4-Flash/inference/model.py::Compressor`
  - `DeepSeek-V4-Flash/inference/model.py::Indexer`
  - `DeepSeek-V4-Flash/inference/model.py::Attention`
- SGLang DeepSeek V4 branch:
  - `python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh`
  - `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/compress.cuh`
  - `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh`
  - `python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh`
- vLLM DeepSeek V4 support:
  - `vllm/model_executor/layers/deepseek_compressor.py`
  - `vllm/model_executor/layers/deepseek_v4_attention.py`
  - `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`
- FlashMLA:
  - sparse attention index format
  - future FP8 KV cache format

## Current V1 Contract

The MPK task consumes seven inputs to stay within `MAX_INPUTS_PER_TASK = 7`:

1. `kv_score`: float32 `[max_num_batched_tokens, 2048]`
2. `token_meta`: int32 `[max_num_batched_tokens, 2]`
3. `state_cache`: float32 `[max_requests, 8, 2048]`
4. `c4_cache`: bf16 `[num_c4_pages, c4_page_size, 512]`
5. `ape`: float32 `[8, 512]`
6. `norm_weight`: float32 `[512]`
7. `rope_cos_sin`: float32 `[max_seq_len, 64]`

`token_meta[:, 0]` is absolute sequence position. `token_meta[:, 1]` is the
physical C4 cache slot, or `-1` when that token does not emit a compressed KV.

`state_cache` stores raw kv/score rows, without APE added. The kernel adds APE
only at compression time. This mirrors SGLang's state representation while
matching the official Hugging Face `Compressor.forward` math.

## Implemented In This Branch

- Per-request CTA dispatch using `task_metadata.request_id = blockIdx.x`.
- Unified prefill/decode task:
  - prefill consumes each request's `qo_indptr_buffer` token range
  - decode is represented by one-token ranges and can reuse the same compiled
    graph across positions by updating the attached input tensors
- Current-row state update:
  - token at absolute position `p` writes row `4 + p % 4`
- C4 boundary handling:
  - `token_meta[:, 1] < 0` updates state only
  - `token_meta[:, 1] >= 0` emits one compressed C4 cache row
- Eight-slot pooling:
  - overlap slots read previous block KV and score from rows `0..3`
  - current slots read current block KV and score from rows `4..7`
  - first C4 block masks overlap slots with score `-inf` and KV zero
  - stable softmax is computed independently for every hidden dimension
- Post-processing:
  - fp32 weighted sum
  - RMSNorm over 512 dimensions with epsilon `1e-6`
  - GPT-J/interleaved RoPE on the last 64 dimensions at position
    `absolute_position + 1 - 4`
  - bf16 cache write to flattened physical C4 slot
- State transition:
  - after a C4 write, current rows shift into overlap rows
  - for prefill windows, final current rows are rewritten to only the trailing
    `seqlen % 4` remainder, matching the official start_pos==0 behavior

## Implementation TODO

- Add a high-level `MPKModule` leaf after PR #695's API lands on `mpk`.
- Add runtime dtype variants if MPK's fused projection path produces bf16 or
  fp8 `kv_score`. The current branch intentionally fixes fp32 for correctness.
- Decide whether production builders should attach raw HF APE `[4, 1024]` and
  split in CUDA. The current layer wrapper pre-packs `[8, 512]`.
- Decide cache format:
  - correctness-first branch writes BF16 `[512]`
  - performance branch should consider FlashMLA FP8-with-scale cache
- Decide C4 page size. This implementation defaults to `128`; final DeepSeek V4
  metadata may want a page size tied to raw-token pages.
- Optimize prefill separately. The current unified kernel is intentionally
  simple and serializes the tokens of each request inside one CTA.
- Add stronger debug validation for malformed `token_meta`, including C4 writes
  not aligned to `(absolute_position + 1) % 4 == 0`.

## Follow-up Kernel Order

1. Complete `dsv4_c4_compress_sm100`.
2. Add `dsv4_c128_compress_sm100` for HCA.
3. Add indexer C4 compressor with `head_dim=128` and rotate/quant path.
4. Add indexer logits/top-k.
5. Add sparse index conversion and SWA/C4 combine.
6. Add FlashMLA-style sparse attention decode/prefill tasks.
