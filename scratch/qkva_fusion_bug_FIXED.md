# QKV-a Fusion Bug — FIXED (2026-05-13, v2: kernel-side)

## Summary

The QKV-a fusion in fused mode (`MPK_DSV3_QKV_A_FUSED=1`) had a bug where
`qkv_a_out` rows 1..71 ended up zero in dumps and propagated through
attention. The root cause is an **asymmetric early-exit semantics** between
`quantize_fp8_layer` (which respects `active_rows` and skips non-active row
writes) and `fp8_gemm_dense_smallm/mediumm` (which always wrote all M=128
rows, even when most were "padding" in decode iters).

**Fixed by making the FP8 dense GEMM also respect `active_rows`**: the
`runtime_m_mode=0` codegen path in `task_register.cc::register_fp8_gemm_dense_variant`
now passes `min(compile-time M, qo_indptr_buffer[max_req])` as the runtime
M, so the kernel's existing per-row write check `if (mi < M)` automatically
filters writes to active rows only.

## Why the bug happened

MPK's persistent megakernel iterates the task graph N times in a single
launch (N = 1 prefill + `max_seq_length − prompt_length` decode iters; typically
~128 iters). `quantize_fp8_layer` has an `active_rows` early-exit:

```cpp
int active_rows_ = runtime_config.qo_indptr_buffer[MAX_NUM_BATCHED_REQUESTS];
if (task_desc->task_metadata.request_id >= active_rows_) return;
```

In decode iters (active_rows=1), only row 0's quantize fires. Rows 1..127
of `fp8_input_v2_7168_shared` retain **whatever the previous writer left**.
For DSv3 layer 0 with QKV-a fused, that previous writer was PREFILL's
post-attn quantize, whose input (post-attn rmsnorm output) is small enough
for many rows that the eps clamp kicks in and `scale = 1e-10/448 = 2.232e-13`
(the fallback).

Then in each decode iter, `fp8_gemm_dense_smallm` (which had `M=128`
compile-time, no `active_rows` awareness) read the stale fallback-scale rows
and produced near-zero output. The kernel WROTE those near-zero rows into
`qkv_a_out`, overwriting prefill's correct values. After ~128 decode iters
the final dump shows `qkv_a_out` rows 1..71 = 0.

The user-facing symptom was a dump artifact — the actual end-to-end model
output had a related but smaller drift (L0..L3 residual cos vs unfused
~0.977/0.973/0.965/0.956 instead of ~0.999).

## The fix (one file, ~12 lines)

`src/kernel/task_register.cc::register_fp8_gemm_dense_variant`:

Before (runtime_m_mode=0): always passed compile-time M to the kernel.
After (runtime_m_mode=0): emit

```cpp
int active_rows_ = runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
int runtime_m_  = active_rows_ < M ? active_rows_ : M;
if (runtime_m_ <= 0) return;
// ... kernel call with runtime_m_ as the M argument
```

The kernel itself is unchanged. It already uses M for:
1. Task tile distribution: `total = ((M+BM-1)/BM) * nn`
2. Per-row write check: `if (mi < M)`

Both honour the runtime-computed `min(M, active_rows)`, so decode iters'
GEMMs only write rows 0..active_rows-1 — rows 1..127 keep prefill's correct
content.

This change is **semantically correct for every caller of
`fp8_gemm_dense_smallm/mediumm`** — there's no legitimate reason to compute
output rows that the upstream `quantize_fp8_layer` already skipped.
`runtime_m_mode=1` (kv_b prefill decompression) keeps its existing
`runtime_m = (lp - fp - 1)*PAGE_SIZE + last_page_len` logic — that branch
already overrode M.

## Why this is better than the builder-side fix (tag'd buffers)

The earlier dedicated-buffer fix (commit `7bfa392f`) was reverted in favour
of this approach because:
- **One change at the right layer.** The bug is a contract mismatch between
  quantize (active_rows-aware) and GEMM (not). Fixing the GEMM side
  restores symmetry without splitting buffers.
- **Generalises automatically.** Every existing fp8_gemm_dense call benefits;
  no per-call-site `fp8_buf_tag` plumbing required.
- **No memory pressure.** No extra FP8 buffers allocated.
- **Matches the user's intuition.** Early-exit / active-row semantics
  should preserve data integrity end-to-end, not just at one task's output.

## Verification

```
Before fix:  qkv_a_out has 71 zero rows (rows 1..71)
After fix:   qkv_a_out has 0 zero rows
             Row norms: r0=7.470, r1=24.025, r35=23.728, r71=23.926, r72=24.868, r127=24.704
```

Layer residual cos vs unfused baseline:
```
Before fix: 0.97745 / 0.97299 / 0.96540 / 0.95638
After fix:  0.97713 / 0.97273 / 0.96509 / 0.95622
```
(Very small noise-level difference; same precision regime as unfused
baseline, which itself differs from BF16-PyTorch ref by similar margins
because the FP8 fused weight is a byte-concat of three separate FP8
weights and has slightly different block-scale boundaries.)

Qwen3 TP=4 max-new-tokens=10: full coherent paragraph generated, no
regression.

## Reproducer (post-fix verification)

```bash
cd /home/muhengl/mirage
git checkout dev-v8-rope-prefill-main          # contains the fix
.venv/bin/python -m uv pip install -e .          # rebuild C++ library
MPK_DSV3_QKV_A_FUSED=1 MPK_DSV3_QKV_A_FUSED_N=2176 \
mpirun -np 4 [NVSHMEM/MPI env] \
  /home/muhengl/mirage/.venv/bin/python demo/deepseek_v3/demo.py \
  --model-path /raid/catalyst/models/DeepSeek-V3 --use-mirage \
  --max-num-batched-tokens 128 --max-num-batched-requests 1 \
  --page-size 128 --max-num-pages 2 --max-seq-length 256 \
  --prompt-length 128 --ignore-eos --max-new-tokens 1 \
  --layers 0-3 --mtp 0 --ep-size 2 \
  --output-dir outputs/repro/build --dump-hidden-dir outputs/repro/dump

python -c "
import torch
qa = torch.load('outputs/repro/dump/layer0_q_a_out.pt', weights_only=True).float()
n = (qa.abs().sum(dim=1)==0).sum().item()
assert n == 0, f'expected 0 zero rows, got {n}'
print('PASS')
"
```

## Followups for colleague review

1. **Should `runtime_m_mode=0` be renamed?** The default behaviour now uses
   runtime active_rows, so the "0" no longer means "compile-time M". Could
   rename to `runtime_m_mode=0/active_rows` for clarity.
2. **Same fix for group GEMMs?** `fp8_group_gemm_smallm/largem` (MoE path)
   has the same structure: do they need the same `min(M, active_rows)` fix?
   Probably yes, but MoE has expert routing on top so the active-row count
   is computed differently. Worth checking after this fix lands.
3. **Same fix for BF16 splitk?** `splitk_linear` family used by the router
   gate has the same M=mbt issue conceptually but a different runtime
   shape. Worth a parallel audit.
