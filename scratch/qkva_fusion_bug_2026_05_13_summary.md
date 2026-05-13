# QKV-a Fusion Bug — Investigation Summary (2026-05-13 afternoon)

## Status: NARROWED BUT NOT RESOLVED

## Bug
- `MPK_DSV3_QKV_A_FUSED=1` on TP=4 EP=2 produces `qkv_a_out` rows 1..71 = EXACTLY ZERO.
- Row 0 and rows 72..127 NORMAL.
- L0 cos vs baseline = 0.977 (instead of expected 0.999).
- Reproducible with **single layer** (layers=0-0), so NOT a cascading bug.

## Confirmed Facts
1. `fp8_scale_v2_7168` rows 1..71 = exactly `2.232e-13` = `1e-10 / 448` = the kernel's
   *fallback scale* (produced when max_abs in the row ≤ 1e-10). So qkv_a quantize
   for rows 1..71 saw **effectively zero input** at its execution moment.
2. `layer0_input_norm.pt` (an `elementwise_add` dump of rmsnorm_out, dep_event=3,
   same event as the quantize) shows rows 1..71 with **norm ~4.5** (correct input
   rmsnorm output).
3. With `MPK_DSV3_RMSNORM_OUT_ATTACH=1` + sentinel 9999.0 prefill:
   - Final state of rmsnorm_out has all 128 rows ≠ sentinel → rmsnorm DID write
     every row.
   - Final state norm ≈ 30 = post-attn rmsnorm output (the last writer).
   - qkv_a_out **still** has rows 1..71 zero.
4. Standalone `fp8_gemm_dense_smallm` kernel test with the saved MPK input bytes
   (cos=1.000) confirms the GEMM kernel is correct on correct input.
5. Task-graph JSON inspection:
   - All quantize tasks have correct `request_id` (0..127), correct `kv_idx` (0..3).
   - All read from `rmsnorm_out @ offset 0`, base buffer.
   - Event 3 has `num_triggers=128` matching the 128 input-rmsnorm producer tasks.
   - Event chain is topologically correct.
6. `tensor_init` task DOES NOT write `rmsnorm_out` — only `output_ptrs[0]` (the target)
   is written; `rmsnorm_out` appears only as output_ops[1] (dummy dep edge), and
   the kernel ignores it.
7. **`membar.gl` at the END of `rms_norm_hopper_impl` did NOT fix the bug** — so it's
   not a producer-side missing-fence issue (reverted that change).
8. `MPK_DSV3_QKV_A_OUT_ATTACH=1` and `MPK_DSV3_RMSNORM_OUT_ATTACH=1` (attaching the
   buffers as standalone torch tensors, bypassing the MPK buffer pool) both DO NOT
   fix the bug — so it's not buffer-pool aliasing on those two tensors.

## Open Hypothesis
At qkv_a quantize execution moment, `rmsnorm_out` rows 1..71 have values in the
≤1e-10 range (effectively zero). At `layer0_input_norm` dump execution moment,
the same rows have norm ~4.5. Both consumers have `dep_event=3` (same dependency).

For this to happen, **something writes near-zero values to rmsnorm_out rows 1..71
between the rmsnorm task and the qkv_a quantize task**, then it is restored before
the dump fires.

No code path in static analysis appears to do this. The fp8_input_v2_7168 bytes
in row 1..71 sum to ~23552 (not 0), suggesting the input wasn't pure zero — more
like noisy near-zero values. Possible:

a) **Cache coherence** between SMs — the quantize CTA reads stale L1/L2 lines
   from BEFORE rmsnorm wrote. But `membar.gl` on the producer side didn't help.
b) **Some kernel I haven't audited** writes near-zero to rmsnorm_out (e.g., embed
   kernel for some rows? rope kernel? something else with the wrong offset).
c) **Indexing bug in the quantize kernel** that's data-dependent on certain row
   values (very unlikely — rows 0+72..127 succeed with identical code path).

## Diagnostic Tools Built
- `MPK_DSV3_QKV_A_FUSED=1` — fused mode (env-gated).
- `MPK_DSV3_FP8_BUF_ATTACH=1` — attach FP8 input/scale as torch tensors so we can
  see post-megakernel state.
- `MPK_DSV3_QKV_A_OUT_ATTACH=1` — attach qkv_a_out as torch tensor.
- `MPK_DSV3_RMSNORM_OUT_ATTACH=1` + `MPK_DSV3_RMSNORM_SENTINEL=<value>` — pre-fill
  rmsnorm_out with sentinel; check post-megakernel which rows are still sentinel.
- `scratch/scan_free_gpus.sh` — find 4 free GPUs.
- `tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_smallm_real_bytes.py` —
  standalone kernel test on saved MPK bytes.

## Recommended Next Steps
1. **Add a SECOND elementwise_add dump** that reads `rmsnorm_out` BUT has its
   dep_event = the quantize's trigger_event (event 4). This captures rmsnorm_out
   AT QUANTIZE COMPLETION. Diff vs the existing layer0_input_norm dump to see
   exactly what content the quantize observed.
2. **Add a forced ordering: rmsnorm → identity_layer → quantize**. The identity
   layer would do `out = in + 0`, serialising the chain. If quantize then sees
   correct input, race condition between rmsnorm and quantize is confirmed (and
   the fix is to add this identity bridge or strengthen the event chain).
3. Audit the embed kernel and the rope kernels for any path that writes near-zero
   to `embed_out` or `rmsnorm_out` rows 1..71 (unlikely but worth ruling out).
