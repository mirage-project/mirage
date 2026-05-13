# QKV-a fusion bug ROOT-CAUSED + FIXED 2026-05-13

## TL;DR

The bug was NOT in the FP8 dense GEMM kernel.

The bug was in `per_token_group_quantize_fp8_task_impl`
(`include/mirage/persistent_kernel/tasks/blackwell/per_token_group_quantize_fp8.cuh`):
the kernel used **the same stride (`GLOBAL_STRIDE`) for both INPUT reads and
OUTPUT writes**. For the QKV-a fused path where the input is a column slice
of a wider parent buffer (input stride 2176) but the output is sized for
the slice width (1536), the output writes went out of bounds for
`batch_idx >= 90`, corrupting whatever was adjacent in MPK's buffer pool.

I introduced this bug when I added the `input_stride_override` /
`hidden_size_override` params to `quantize_fp8_layer` (and the corresponding
codegen in `task_register.cc`) for QKV-a fusion: I let the kernel template
take `GLOBAL_STRIDE = input_stride_override` (= 2176) but the kernel used
that same stride to address the smaller output buffer (1536-wide), causing
the overflow.

The fix: added an `OUTPUT_STRIDE` template param defaulting to
`GLOBAL_STRIDE` (preserves all legacy callers exactly). The codegen passes
`OUTPUT_STRIDE = hidden_size` when slicing.

## Repro / ablation chain that found it

Ran 4 progressive ablations (see `feedback_ablation_design`):

| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | Dump artifact only — final hidden is correct | Compare `layer_00_residual.pt` fused vs baseline | FALSE: cos=0.966 at L0, degrades to 0.889 by L3. Real bug. |
| H3 | Bug in a downstream consumer of qkv_a_out (rmsnorm/rope) | Move dump task BEFORE rmsnorm via builder edit; rerun fused smoke | FALSE: pre-RMSnorm dump shows EXACT SAME rows-1..71-zero pattern across all 3 slices (q_a, c_latent, k_pe). Rules out consumer-side bug. |
| H6 | Pure kernel logic bug at N=2176 | Standalone test: call `fp8_gemm_dense_smallm_sm100_task_impl<128,3>` with REAL MPK weight (qkv_a_proj.weight from `/tmp/dpskv3_v8_weight_cache_qkva_fused_2176/`) + REAL MPK input (`layer0_input_norm.pt`) | FALSE: standalone passes (cos=1.000, 0 zero rows). Kernel is correct. |
| H8 | MPK quantize task overflows output buffer for the slice case | Inspect codegen variants in generated kernel.cu; trace template args | TRUE: variant 2 (q_b's prefill quantize) has `<128, 1536, 128, 2176, 1, ...>` — HIDDEN_SIZE=1536, GLOBAL_STRIDE=2176. Kernel writes at `batch * 2176 + col` into a 196608-byte buffer; batches >= 90 overflow. Verified by standalone test `test_quantize_slice_fix.py` showing 0 canary corruption + 0 zero rows + cos=1.000 vs ref AFTER fix. |

The "rows 1..71 zero, rows 72..127 correct" pattern is the SIGNATURE of
this overflow corrupting qkv_a_out's first ~72 rows (depending on buffer
pool layout). The exact rows-1..71 boundary corresponds to where the
overflow lands relative to qkv_a_out's start; rows 72..127 escape
corruption because the qkv_a GEMM rewrites them post-overflow.

## The fix

`per_token_group_quantize_fp8.cuh`:
```cpp
template <int BATCH_SIZE, int HIDDEN_SIZE, int GROUP_SIZE,
          int GLOBAL_STRIDE,   // input stride (per-row of parent buffer)
          int GROUP_TILES,
          typename T, typename DST_T, bool SCALE_UE8M0,
          int OUTPUT_STRIDE = GLOBAL_STRIDE,    // NEW: output buffer row stride
          typename SCALE_PACKED_T = ...>
__device__ void per_token_group_quantize_fp8_task_impl(...) {
  int const input_row_base  = batch_idx * GLOBAL_STRIDE;
  int const output_row_base = batch_idx * OUTPUT_STRIDE;
  // ... reads use input_row_base, writes use output_row_base ...
}
```

`src/kernel/task_register.cc`:
```cpp
int output_stride = has_slice_override ? hidden_size : input_stride;
code.e("kernel::per_token_group_quantize_fp8_task_impl<$, $, $, $, $,",
       batch_size, hidden_size, GROUP_SIZE, input_stride, group_tiles);
code.e("    cute::bfloat16_t, __nv_fp8_e4m3, $, $>(",
       scale_ue8m0 ? "true" : "false", output_stride);
```

Defaults preserve all legacy callers (they pass `input_stride == hidden_size`
implicitly via `has_slice_override == false`).

## Standalone verification scripts (kept as regression artefacts)

- `tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_quantize_slice_fix.py`
  — direct test: quantize a [128, 1536] slice of a [128, 2176] BF16 buffer
  with `OUTPUT_STRIDE=1536`. Asserts (a) no canary bytes are written past
  the (128, 1536) output region, (b) all 128 output rows are non-zero,
  (c) FP8 output matches PyTorch ref with cos=1.000.
- `tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_smallm_n2176_bug.py`
  — direct multi-CTA launch of the dense GEMM at the bug-shape (M=128,
  N=2176, K=7168, nw=128) with random data. PASSES — proves the GEMM is OK.
- `tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_smallm_real_bytes.py`
  — same as above but uses REAL MPK qkv_a_proj.weight + real input. PASSES.

## End-to-end (MPK) verification

Pending — needs 4 free GPUs for TP=4 EP=2 smoke. Will dump `layer0_q_a_out.pt`
in fused mode and verify rows-1..71-zero pattern is gone.

## Lessons

- **Add output_stride params explicitly when adding stride-aware code paths.**
  My QKV-a fusion work modified 5 downstream kernels for the wider qkv_a_out
  buffer. For 4 of them (rmsnorm, deepseek_mla_rope, mla_kv_gather,
  elementwise_add) the output happens to share the same stride as the input
  (in-place ops on the same buffer, or output is a separate fixed-stride
  cache). For the 5th (quantize), the input is a slice but the output is a
  smaller contiguous buffer — and that's exactly where the bug fires.
  Always check this pattern when adding stride params.

- **A standalone test with REAL bytes is much stronger evidence than random
  data.** My first standalone test (random data) passed and made me
  hypothesize an MPK runtime context bug. The actual root cause was nothing
  exotic — it was a stride mismatch in my own task_register codegen, fully
  reproducible in isolation. Use real bytes early.

- **Always audit ALL downstream kernels when introducing a wider parent
  buffer.** The bug is in a kernel I didn't suspect at first because I was
  focused on the GEMM (which had the rows-1..71-zero pattern in its output).
  But the actual culprit was a SIBLING task (variant 2 of quantize for q_b)
  that corrupted the GEMM's output buffer through pool overflow. Cross-task
  memory aliasing in the buffer pool is hard to reason about; the only
  reliable defense is to make every stride-aware kernel exhaustively
  verified.
