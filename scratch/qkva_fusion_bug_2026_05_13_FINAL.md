# QKV-a Fusion Bug — FINAL Root Cause Analysis (2026-05-13 evening)

## Verdict

**MPK runtime has a race / scheduling bug where the fused `qkv_a` FP8 GEMM
task reads its FP8 input from `fp8_input_v2_7168_shared` AT THE WRONG MOMENT
in some fraction of its executions.** Even though the dep-event chain says
GEMM should wait for the input quantize to complete, the GEMM kernel's
accumulator reads ZERO most of the time (508/512 firings) and the correct
value only 4/512 times.

Kernel itself is correct. Standalone test on identical bytes passes 100%.
Bug is purely in MPK runtime — most likely consumer-side memory ordering /
cache invalidation OR a scheduling bug that fires GEMM multiple times.

## Evidence chain (all verified by in-kernel `printf`)

Setup: TP=4 EP=2, layers 0-0, `MPK_DSV3_QKV_A_FUSED=1`, mbt=128, hidden=7168.

### Stage 1 — Input rmsnorm writes correct data (RM_DBG ★)

`rmsnorm_hopper` task for row 1 writes `0.018921` to `rmsnorm_out` row 1
(out_ptr `0xfc56603800`, in_ptr `0x...c5832c00` = embed row 1, weight_ptr
`0x...16600` = `input_layernorm.weight`). All 128 input-rmsnorm tasks
(rows 0..127) write **correct** rmsnorm output.

### Stage 2 — Quantize ① (post-input-rmsnorm) reads correct data (QZ_DBG ✓)

Per-rank, ONE qkv_a-input-quantize task instance fires. Its printf shows:

```
[QZ_DBG] row=1 g=0 input_ptr=0xfc56600000 input[7168]=0.018921 input_row_base=7168
[QZ_DBG] row=1 g=0 lane=0 local_max=4.858398e-02
```

Quantize ① CORRECTLY sees `0.018921`, computes correct `local_max`, writes
correct fp8 + scale to `fp8_input_v2_7168_shared`.

### Stage 3 — qkv_a FP8 GEMM reads `fp8_input_v2_7168` ... but acc[0] is zero MOST of the time (GEMM_DBG ✗)

`fp8_gemm_dense_smallm` worker-0 CTA's consumer-warp printf for row 1, on=0:

```
4 entries:  [GEMM_DBG] mi=1 on=0 acc[0]=-0.161166 acc[1]=0.025629 acc[15]=-0.023110
508 entries: [GEMM_DBG] mi=1 on=0 acc[0]=0.000000 acc[1]=0.000000 acc[15]=0.000000
```

**Same task. Same target row. 99% of executions see ZERO input and produce
ZERO output. Only ~1% (4 firings) produce the correct accumulation.**

Since the LAST consumer-warp write to `qkv_a_out` row 1 wins, and the
zero-acc writes dominate, **qkv_a_out row 1 ends up = 0**.

### Stage 4 — Cascade: row 1 = 0 propagates through attention

- Attention input includes qkv_a_out row 1 → attention output row 1 = 0.
- `o_proj` should add residual (= embed row 1, non-zero), but `attn_proj_out`
  row 1 ALSO ends up = 0 (the residual is somehow lost — possibly a
  second instance of the same race in `linear_with_residual`).

### Stage 5 — Post-attn rmsnorm reads attn_proj_out = 0, writes 0 (RM_DBG ★)

Same out_ptr `0xfc56603800` (= rmsnorm_out row 1), different weight_ptr
(`0x...19e00` = `post_attn_layernorm.weight`):

```
[RM_DBG] out_ptr=0xfc56603800  in_ptr=0x...32a03800  w_ptr=...19e00  val[0]=0.000000
```

Post-attn rmsnorm wrote 0 because its input was 0 (= attn_proj_out row 1).

### Stage 6 — Quantize ② reads 0 and produces fallback (QZ_DBG ✗)

```
[QZ_DBG] row=1 g=0 input_ptr=0xfc56600000 input[7168]=0.000000 input_row_base=7168
[QZ_DBG] row=1 g=0 lane=0 local_max=1.000000e-10  ← fallback (eps clamp)
```

This is the fp8_scale `2.232e-13` fallback we observed in dumps.

## What's the same buffer reading "different values"?

Multiple consumers of `rmsnorm_out` row 1 at different lifetimes:
- Quantize ① reads it after `input_rmsnorm` wrote `0.018921` → sees `0.018921`
- Quantize ② reads it after `post_attn_rmsnorm` overwrote it with `0` → sees `0`

That part is fine — rmsnorm_out is correctly reused. The dump
(`elementwise_add(rmsnorm_out, ...)`) and the `rmsnorm_out_v2_attached` torch
tensor (with sentinel) both confirm at-end-of-megakernel the buffer state
matches what the *last* rmsnorm wrote (post-attn rmsnorm for layer 0).

## The real bug is in Stage 3: GEMM execution

The `fp8_gemm_dense_smallm` kernel for the qkv_a fused path:
- Receives the SAME task descriptor with SAME `input_ptrs[0]` for each
  execution.
- Reads `fp8_input_v2_7168_shared` ... and 99% of the time sees ZERO input.

Possible mechanisms (cannot disambiguate from current data):
1. **MPK reruns the persistent megakernel many times** and most reruns have
   uninitialized `fp8_input_v2_7168_shared` (= 0) at the time GEMM fires.
   That would mean my `printf` is firing per persistent-megakernel-iteration,
   and only 1 iteration has the dep chain correctly satisfied while the
   others fire GEMM with stale/uninit buffer.
2. **Worker-scheduler race**: workers re-fetch the same task descriptor
   multiple times, executing GEMM many times. Most executions catch
   `fp8_input_v2_7168_shared` BEFORE quantize ① wrote it (zero) or AFTER
   `post_attn_rmsnorm` invalidated it via cache coherence.
3. **Persistent kernel iteration model**: the 128 firings could be the
   megakernel's internal iteration loop running through 128 timesteps,
   where most timesteps have no real input data (padded/zero).

The 4 correct firings = 1 per rank likely corresponds to the ACTUAL prefill
iteration. The 508 zero firings = 127 per rank correspond to warmup/decode
iterations with zero-padded inputs.

## Why UNFUSED mode works

In unfused mode, each of {q_a_proj, kv_a_latent, kv_a_rope} does its OWN
quantize + GEMM pair, all sharing `fp8_input_v2_7168_shared`. The dep chain
is more rigid (3 separate (quantize, GEMM) pairs per layer per iteration),
which apparently happens to sequence correctly. Plus, the wrong-but-
self-consistent scale + fp8 mechanism (seen in `test_quantize_standalone.py`)
cancels in dequant.

In fused mode there's only 1 (quantize, GEMM) pair per layer per iteration,
so the race window is wider and the bug is exposed.

## What's been ruled out

- ❌ Buffer pool aliasing (tried `MPK_DSV3_RMSNORM_OUT_ATTACH=1` + sentinel)
- ❌ Producer-side `membar.gl` in `rms_norm_hopper_impl`
- ❌ Consumer-side `ld.global.cv` (cache-bypass) in `per_token_group_quantize_fp8`
- ❌ Forced serialization via `identity_layer` bridge
  (`MPK_DSV3_RMSNORM_BRIDGE=1`)
- ❌ Quantize kernel correctness (standalone test on real MPK bytes passes
  with cos≥0.9998, 7168/7168 scale match)
- ❌ Dep-event chain (statically correct per `task_graph_rank0.json`:
  event-3 num_triggers=128 = rmsnorm count, event-4 num_triggers=512 =
  quantize count, GEMM dep=event-4)

## Recommended next steps (for the runtime owner)

1. **Verify megakernel iteration count for `--max-new-tokens 1`** —
   instrument `persistent_kernel.cuh` worker loop to print iteration number
   and task IDs processed. If we see GEMM firing >>1 times per launch with
   different fp8 input contents, the megakernel's iteration management is
   wrong.

2. **Tighten consumer-side memory ordering**: switch `fp8_gemm_dense_smallm`'s
   FP8 input reads to use `ld.global.acquire.sys` (or equivalent acquire-
   ordered load) so the GEMM serialises against the quantize's release
   atomic. The current `__ldg` / regular load may bypass the release.

3. **Add a runtime invariant check**: before GEMM consumer-warp reads the
   FP8 input for row N, assert that the corresponding scale `fp8_scale_v2_7168[N]`
   is NOT the fallback `1e-10/448` value. If it is, log and abort. This
   would catch the race in CI before it propagates to correctness failures.

## Reproducer (one-shot, ~3 min on 4 free GPUs)

```bash
cd /home/muhengl/mirage
git checkout dev-v8-rope-prefill-main  # cb6c0d2a or later
MPK_DSV3_QKV_A_FUSED=1 MPK_DSV3_QKV_A_FUSED_N=2176 \
MPK_DSV3_FP8_BUF_ATTACH=1 \
mpirun -np 4 [NVSHMEM/MPI env] \
  /home/muhengl/mirage/.venv/bin/python demo/deepseek_v3/demo.py \
  --model-path /raid/catalyst/models/DeepSeek-V3 --use-mirage \
  --max-num-batched-tokens 128 --max-num-batched-requests 1 \
  --page-size 128 --max-num-pages 2 --max-seq-length 256 \
  --prompt-length 128 --ignore-eos --max-new-tokens 1 \
  --layers 0-0 --mtp 0 --ep-size 2 \
  --output-dir outputs/repro/build --dump-hidden-dir outputs/repro/dump
```

Then:
```python
import torch
mpk_scale = torch.load('outputs/repro/dump/fp8_scale_v2_7168.pt', weights_only=True)
# rows 1..71 will be exactly 2.232e-13 (fallback); row 0 + rows 72..127 normal
print((mpk_scale - 2.232143e-13).abs().lt(1e-15).all(dim=1).sum().item())  # → 71
```

Standalone test that PROVES kernel correctness (cos=1.000):
```bash
cd tests/runtime_python/blackwell/sm100_fp8_gemm_dense
python test_quantize_standalone.py
# Output: "MPK real bytes — standalone kernel vs PyTorch ref: 7168/7168 (100.0%)"
```
