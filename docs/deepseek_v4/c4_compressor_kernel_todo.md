# DeepSeek V4 C4 Compressor Kernel TODO

This draft note tracks the implementation decisions for the MPK
`dsv4_c4_compress_sm100` task. The current branch intentionally wires a
compile-safe skeleton first; the CUDA math is left as TODOs so the follow-up PR
can implement one kernel at a time.

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

## Current Skeleton Contract

The MPK task consumes seven inputs to stay within `MAX_INPUTS_PER_TASK = 7`:

1. `kv_score`: `[max_num_batched_tokens, 2048]`
2. `token_meta`: int32 `[max_num_batched_tokens, 2]`
3. `state_cache`: float32 `[max_requests, 8, 2048]`
4. `c4_cache`: bf16 `[num_c4_pages, c4_page_size, 512]`
5. `ape`: float32 `[8, 512]`
6. `norm_weight`: `[512]`
7. `rope_cos_sin`: float32 `[max_seq_len, 64]`

`token_meta[:, 0]` is absolute sequence position. `token_meta[:, 1]` is the
physical C4 cache slot, or `-1` when that token does not emit a compressed KV.

## Implementation TODO

- Confirm whether MPK will produce `kv_score` as fp32 or bf16. DeepSeek's
  official reference performs compression in fp32.
- Decide whether checkpoint `ape` should be prepacked in the Python builder as
  `[8, 512]` or attached raw as `[4, 1024]` and split inside CUDA.
- Implement prefill semantics:
  - `cutoff = seqlen - (seqlen % 4)`
  - compressed blocks only for full C4 groups
  - remainder tokens are saved in state
  - first block invalid overlap is masked with score `-inf` and KV zero
- Implement decode semantics:
  - write current token into `state_cache[request, 4 + abs_pos % 4]`
  - emit compressed KV only when `(abs_pos + 1) % 4 == 0`
  - shift current state to overlap state after a cache write
- Implement C4 weighted pooling:
  - add APE before softmax
  - stable softmax over the eight C4 overlap/current slots per hidden dim
  - weighted KV sum in fp32
- Implement post-processing:
  - RMSNorm over 512 dims
  - RoPE on the last 64 dims using position `abs_pos + 1 - 4`
  - write BF16 result into `c4_cache[c4_slot]`
- Decide cache format:
  - correctness-first branch writes BF16 `[512]`
  - performance branch should consider FlashMLA FP8-with-scale cache
- Decide C4 page size. This skeleton defaults to `128`; final DeepSeek V4
  metadata may want a page size tied to raw-token pages.
- Decide whether prefill and decode remain one task. SGLang separates some
  paths; MPK may keep one runtime-gated task to match existing MLA patterns.

## Follow-up Kernel Order

1. Complete `dsv4_c4_compress_sm100`.
2. Add `dsv4_c128_compress_sm100` for HCA.
3. Add indexer C4 compressor with `head_dim=128` and rotate/quant path.
4. Add indexer logits/top-k.
5. Add sparse index conversion and SWA/C4 combine.
6. Add FlashMLA-style sparse attention decode/prefill tasks.
