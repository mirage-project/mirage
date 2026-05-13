# Bug: `fp8_gemm_dense_smallm_sm100` drops output rows 1..71 at N=2176

**Reporter**: QKV-a fusion work (2026-05-12).
**Component**: `include/mirage/persistent_kernel/tasks/blackwell/fp8_gemm_dense_sm100_common.cuh` (smallm and mediumm variants both fail).
**Severity**: Correctness — silently zeros 71/128 output rows.

## Symptom

Calling `fp8_gemm_dense_smallm_sm100_task_impl<BN=128, NS=3>` from the persistent kernel with

```
M = 128         (batch / mbt)
K = 7168        (hidden_size)
N = 2176        (q_lora 1536 + kv_lora 512 + k_pe padded 128)
num_workers = 128
```

produces an output buffer where **rows 1..71 are all-zero (cudaMalloc-init)** while rows 72..127 contain the correct GEMM result. The pattern is identical across all 17 N-tile column bands (verified by dumping the whole `(128, 2176)` output and checking each `[bn*128 : (bn+1)*128)` band).

Same kernel + `fp8_gemm_dense_mediumm_sm100_task_impl<BN=128, NS=3>` (NE=4 instead of NE=2) reproduces the SAME row pattern. So it's not specific to the NE template parameter.

## Repro

Branch tip after these commits:
- `52800d05` — row-swap fuse-Q (correctness verified, no perf gain at scale)
- `99e90a4b` — QKV-a fusion infra (stride-aware downstream kernels)
- this commit — diagnostic dumps + N-pad env override

```
MPK_DSV3_QKV_A_FUSED=1 MPK_DSV3_QKV_A_FUSED_N=2176 \
    bash scratch/run_qkva_smoke.sh repro fused
python3 -c "
import torch
f = torch.load('outputs/dpskv3_qkva_fused_repro/dump/layer0_q_a_out.pt',
               weights_only=True).float()
print('Per-row q_a (cols [:1536]) norms:')
for r in [0, 1, 8, 32, 64, 71, 72, 96, 127]:
    print(f'  row{r:3d}: {f[r,:1536].norm().item():.3f}')
"
```

Expected output: row 0 nonzero (decode-overwrite artifact), rows 1..71 all 0.000, rows 72..127 nonzero.

## What works vs what fails

All shapes have M=128, K=7168, num_workers=128 in the SAME persistent kernel:

| Caller | N | nn = N/BN | Status |
|---|---|---|---|
| baseline k_pe (kv_a rope) | 128 | 1 | ✅ all 128 rows written |
| baseline c_latent (kv_a latent) | 512 | 4 | ✅ |
| baseline q_a | 1536 | 12 | ✅ |
| QKV-a fused **(this case)** | **2176** | **17** | ❌ rows 1..71 zero |
| diag pad to 2304 | 2304 | 18 | ❌ rows 1..71 zero (SAME pattern) |
| baseline q_b absorbed | 4096 | 32 | ✅ |
| row-swap q_b_unabsorbed | 6144 | 48 | ✅ |

So nn∈{17, 18} reproduces. nn∈{1, 4, 12, 32, 48} works.

## What I tried and ruled out

- **Builder/demo wiring**: weight bytes verified byte-equal to the corresponding slices of the working baseline weights (q_a_proj.weight, kv_a_proj_with_mqa.weight). Per-task input/output offsets in the task graph JSON are correct. Same FP8 quantize task graph as baseline (event-chain-equivalent).
- **NE=2 vs NE=4**: forcing `fp8_gemm_dense_mediumm` (NE=4) produces the SAME row pattern.
- **`mb_init(bte, 128)` count**: matches block_dim=256 → 128 consumer threads (wid 4..7). Correct.
- **N alignment** (multiple of 4*BN=512): both N=2176 and N=2304 hit the bug; padding to N=4096 (which is 4*BN-aligned) is known-good but unrelated.

## Where I think the bug is

In `task_impl_tpl<BN, NS, NE>` (`fp8_gemm_dense_sm100_common.cuh:97-347`), the consumer-warp section at **lines 229-339**:

```cpp
} else if (wid >= 4) {
    int const et = tid - 128, ew = wid - 4;
    int gki = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) break;
      int bm = bidx / nn, bn = bidx % nn;
      int om = bm * BM, on = bn * BN;
      int mi = om + et;
      float acc[BN]; ...
      for (int ki = 0; ki < nk; ki++, gki++) {
        ...
        int ai = gki % NE;
        int ap = (gki / NE) & 1;
        mb_wait(btf + ai * 8, ap);                       // line 257
        ...
        for (int i = 0; i < BN / 16; i++) {
          uint32_t ta_ = taddr + ((ew * 32) << 16) + ai * BN + i * 16;
          ...
          tcgen05.ld.32x32b.x16.b32 → v[16]              // lines 265-284
          ...
          for (int j = 0; j < 16; j++)
            acc[i * 16 + j] += v[j] * sf;
        }
        ...
        mb_arrive(bte + ai * 8);                         // line 293
      }
      if (mi < M) {                                      // line 296
        // store acc to output
      }
    }
}
```

The **128 consumer threads** map to:
- wid 4 (lanes 0..31): writes output rows 0..31 (et 0..31, mi=et)
- wid 5: rows 32..63
- wid 6: rows 64..95
- wid 7: rows 96..127

Observed: rows 0..71 are zero (everywhere in the tile). That's **wid 4 entirely, wid 5 entirely, and lanes 0..7 of wid 6** — total 72 lanes producing zero output, the remaining 56 lanes (wid 6 lanes 8..31 + all of wid 7) producing correct output.

The 72-vs-56 split is the same on every tile and on every nn∈{17, 18} call. The boundary `et = 72` is suspicious because `M - nk = 128 - 56 = 72` (with nk=K/BK=7168/128=56), but baseline configurations with the same K/nk work fine.

My best guess is that one of the following is wrong for THIS specific shape:

1. **The `mb_wait(btf + ai * 8, ap)` at line 257** — the first time consumer lanes reach this for a given tile, they read uninitialized TMEM (the "phantom prologue") because the barrier phase starts at the right state for them to skip waiting. For nn=17/18 specifically, the issuer's commit-to-btf may be racing with the consumer in a way that some lanes still see the uninitialized state on iteration `gki=NE` and beyond, never seeing the MMA result.
2. **The TMEM read at lines 263-284** — the address `taddr + ((ew * 32) << 16) + ai * BN + i * 16` puts ew=0..3 in the high half of the 32-bit TMEM address. For some ew values combined with this nn, the read may land outside the TCA=256-column allocation and return zero.
3. **An issuer-side issue at lines 184-228** that causes the MMA never to fire for some `gki`, leaving TMEM zero for ai stages that those lanes later read.

A printf at line 296 logging `(tid, mi, et, acc[0])` per task (and a corresponding printf at the issuer's MMA commit) on a single-CTA repro should pin it down.

## Slower-but-correct workaround (for in-tree use until fixed)

None landed yet — the bug only fires for the QKV-a fused GEMM, which is **env-gated default OFF** via `MPK_DSV3_QKV_A_FUSED=1`. The default path (3 separate GEMMs into `q_a_out / c_latent_out / k_pe_out`) is unaffected and validated.

If a runtime workaround is wanted before the kernel fix lands, options are:
- Pad N to 4096 or 6144 (known-good nn) by adding more zero rows after k_pe. Wasteful (doubles N-tile count) but bypasses the bug.
- Fall back to the 3 separate baseline GEMMs when `MPK_DSV3_QKV_A_FUSED=1` is set. Removes the GEMM-fusion benefit but keeps the input-quantize sharing.

## Repro tools committed alongside

- `scratch/run_qkva_smoke.sh` — env-tunable repro driver.
- `MPK_DSV3_QKV_A_FUSED_N=<N>` — pad fused N to any 128-multiple ≥ 2112.
- `scratch/fp8_dense_smallm_n2176_bug.md` (this file) — full repro story.

Once the kernel team fixes this, flipping `MPK_DSV3_QKV_A_FUSED=1` should give 1 GEMM + 1 quantize per attention layer instead of 3 + 3, and all downstream consumers (RMSnorm × 2, ROPE-K, MLA-KV-gather) already use stride/offset to read their slice of the fused buffer.
