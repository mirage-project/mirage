# QKV-a Fusion Bug — FIXED (2026-05-13)

## Summary

The QKV-a fusion in fused mode (`MPK_DSV3_QKV_A_FUSED=1`) had a multi-iteration
buffer-sharing bug that caused `qkv_a_out` rows 1..71 to be zero in dumps and
in real attention output. **Fixed by giving the QKV-a quantize a dedicated
FP8 input buffer (not shared with other quantize tasks).**

## Mechanism

Pre-fix, `_fp8_mbt_buffers_for_reduction_f32scale(reduction_size)` cached
buffers by `reduction_size` only, so EVERY caller with `reduction_size=7168`
shared `fp8_input_v2_7168_shared`:
- Input quantize (feeds qkv_a fused GEMM)
- Post-attn quantize (feeds dense gate_up GEMM)
- MoE gate / shared expert quantize (for MoE layers)

MPK persistent megakernel iterates the task graph many times: 1 prefill iter +
`(max_seq_length - prompt_length)` decode iters. With `max_seq_length=256` and
`prompt_length=128` (a typical demo config), that's **129 iterations** in a
single megakernel launch.

`quantize_fp8_layer` has an **early-exit on `request_id >= active_rows`** where
`active_rows = qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]`. In decode iters,
`active_rows = 1` (only one new token), so quantize ONLY writes row 0; rows
1..127 retain whatever the previous-iter writer left in the shared buffer.

The corruption chain:
1. **PREFILL iter** (active_rows=128):
   - Input quantize writes correct FP8/scale for all 128 rows of `fp8_input_v2_7168_shared`.
   - qkv_a fused GEMM reads correct FP8 → produces correct `qkv_a_out`.
   - Post-attn rmsnorm runs on `attn_proj_out`. For rows where the magnitude is small (typical for residual streams with low-norm contributions), the rmsnorm output is small.
   - **Post-attn quantize OVERWRITES `fp8_input_v2_7168_shared`** with its own
     scale-per-row. For small rmsnorm output rows, `group_max ≤ 1e-10` triggers
     the eps clamp, and `scale = 1e-10 / 448 = 2.232e-13` (the fallback).
   - End of PREFILL: shared FP8 buffer has rows 1..71 = FALLBACK (poisoned).
2. **DECODE iter 1..127** (active_rows=1):
   - Input quantize fires only for row 0. Rows 1..127 unchanged.
   - **qkv_a fused GEMM** reads `fp8_input_v2_7168_shared`. For rows 1..71,
     scale = fallback = 2.232e-13. FP8 = orig / fallback >> 448 → saturates to
     448. Dequant = `448 × 2.232e-13 ≈ 1e-10` ≈ 0.
   - qkv_a_out rows 1..71 get overwritten with ~0 **in every decode iter**.
3. **Final dump** captures the last decode iter's state: `qkv_a_out` rows 1..71 = 0.

Why this is fused-only: in unfused mode there are 3 separate `_fp8_linear`
calls (q_a, kv_a_latent, kv_a_rope) each with their own quantize. After all
three fire in PREFILL, the buffer holds the LAST one's output. But each of
those was operating on the SAME input rmsnorm_out — so the post-prefill
buffer content is still "input rmsnorm quantize content" for all rows. Then
post-attn quantize fires and overwrites. But the **3 quantize calls fully
re-establish all 128 rows before post-attn**, so when decode iters' GEMMs
read, they're reading INPUT rmsnorm content (correct), not POST-ATTN
content. (Plus the wrong-but-self-consistent fp8+scale property documented
in scratch/qkva_fusion_bug_2026_05_13_FINAL.md applies.)

## The fix

`python/mirage/mpk/models/deepseek_v3/builder.py`:

1. `_fp8_mbt_buffers_for_reduction_f32scale` now takes a `tag: str = "shared"`
   parameter. Cache key is `(reduction_size, tag)` instead of just
   `reduction_size`. Buffer names become `fp8_input_v2_{rs}_{tag}`.
2. `_fp8_linear` / `_fp8_linear_v2` accept and forward `fp8_buf_tag`.
3. The QKV-a fused call in `_build_mla_attention_layer` passes
   `fp8_buf_tag="qkv_a"` so the QKV-a quantize uses
   `fp8_input_v2_7168_qkv_a` instead of the shared buffer.

After the fix, the QKV-a input quantize's FP8 buffer is **never overwritten by
any other task**. In decode iters, the early-exit still skips rows >= 1, but
the buffer's rows 1..127 retain PREFILL'S INPUT QUANTIZE content (which is
correct, not fallback), so the GEMM reads correct data and produces correct
output.

## Verification

Before fix: `qkv_a_out` has 71 zero rows (rows 1..71).
After fix:  `qkv_a_out` has 0 zero rows.

Layer-residual cos vs unfused baseline:
- Before fix: L0/L1/L2/L3 cos = 0.97745 / 0.97299 / 0.96540 / 0.95638
- After fix:  L0/L1/L2/L3 cos = 0.97839 / 0.97389 / 0.96618 / 0.95724

Small improvement consistent with the GEMM output actually being correct now.

Qwen3 sanity (TP=4, max-new-tokens=10): coherent text output, no regression.

## Reproducer

```bash
# Bug repro: pre-fix or with FP8_BUF_ATTACH=0 fallback to shared buffer
MPK_DSV3_QKV_A_FUSED=1 mpirun -np 4 ... demo.py \
  --layers 0-3 --max-seq-length 256 --prompt-length 128 ...

# Verify post-fix
python -c "
import torch
qa = torch.load('outputs/.../dump/layer0_q_a_out.pt', weights_only=True).float()
zero_rows = (qa.abs().sum(dim=1)==0).sum().item()
assert zero_rows == 0, f'expected 0 zero rows, got {zero_rows}'
print('PASS: qkv_a_out has no zero rows')
"
```

## Related — why colleague-side discussion is short

The bug is **NOT** in any kernel algorithm:
- Standalone quantize kernel test passes 100% (7168/7168 scale match vs PyTorch ref).
- Standalone fp8_gemm_dense_smallm test passes (cos=1.000 on real MPK bytes).
- The rmsnorm_hopper kernel writes are correct (verified via in-kernel printf).

The bug is **purely in the BUILDER's buffer cache strategy**, which is
DeepSeek-V3 builder code. Other models (Qwen3) don't hit this because they
don't share `fp8_input_v2_7168_shared` between two different rmsnorm tasks
in the same layer (Qwen3 has only one rmsnorm-then-quantize-then-GEMM chain
per layer).

## Code that changed (3 files, ~15 lines)

- `python/mirage/mpk/models/deepseek_v3/builder.py`:
  - `_fp8_mbt_buffers_for_reduction_f32scale`: add `tag` param + key cache by `(rs, tag)` + rename buffers `fp8_*_v2_{rs}_{tag}`
  - `_fp8_linear_v2`: add and forward `fp8_buf_tag` param
  - `_fp8_linear`: add and forward `fp8_buf_tag` param
  - Fused qkv_a call site: pass `fp8_buf_tag="qkv_a"`
