# MLA Decode Kernel — Tensor Parallelism (TP) Support Spec

Target file: `include/mirage/persistent_kernel/tasks/blackwell/mla_mtp_decode_sm100.cuh`

This document describes what the MLA decode (and reduce) kernel needs in order
to work correctly when the DeepSeek V3 Q heads are sharded across multiple
ranks (Tensor Parallelism, TP).

---

## 1. Background

DeepSeek V3 MLA (absorbed form) has these dimensions:

| symbol | value | meaning |
|---|---|---|
| `NUM_HEADS_GLOBAL` | 128 | total Q heads in the unsharded model |
| `D_K` | 576 | per-head Q/K width (= kv_lora_rank 512 + qk_rope_head_dim 64) |
| `D_V` | 512 | per-head output width (= kv_lora_rank) |
| `kv_len` | variable | current sequence length |
| `Q_LEN` | 1..4 | number of query tokens per MLA call (MTP multi-token decode) |

With TP world size `W ∈ {1, 2, 4, 8, 16}`, Q heads are split column-wise so each
rank owns `local_num_heads = NUM_HEADS_GLOBAL / W ∈ {128, 64, 32, 16, 8}`.
KV (c_latent + k_pe cache) is **replicated** across ranks — not sharded.

Per-rank attention is computed independently on each rank's local heads, and
the results are re-assembled into the hidden dim by a row-sharded `o_proj`
followed by AllReduce (handled outside this kernel).

---

## 2. What the current kernel assumes

The current kernel was designed for the single-GPU case and bakes in
`NUM_HEADS = 128` as a compile-time constant (`static constexpr int NUM_HEADS
= 128;` near the top of `mla_mtp_decode_sm100.cuh`).

It currently structures work as:

- Each thread block = one `(batch, head_group, split_idx)` triple.
- `hpb = NUM_HEADS / num_head_groups` heads processed per block.
- Block size: 128 threads. Each thread `tid` owns MMA output row `tid`.
- Thread→work mapping: `q_idx = tid / hpb`, `h_local = tid % hpb`.
- Q SMEM layout: `Q_LEN` batches of `hpb` rows each, stored contiguously.
- MMA tile M is **hardcoded to 128**, in both `idesc_qk` and `idesc_pv`:
  ```cpp
  constexpr uint32_t idesc_qk = ... | ((uint32_t)(128 >> 4) << 24);
  constexpr uint32_t idesc_pv = ... | ((uint32_t)(128 >> 4) << 24);
  ```

The output buffer `Oa` per block has shape `[D_V, 128]` (MMA row → slot
within block). The reduce kernel reads these slots back via `row = tid % 128`.

Semantically the kernel **relies on the invariant** that every MMA output row
0..127 corresponds to a valid Q row — either a real `(q, head)` pair or a
padding row that has been zeroed out so it contributes zero to the reduce.
When `hpb * Q_LEN < 128`, the invariant is maintained by:

- threads with `q_idx ≥ Q_LEN` setting `effective_len = 0` so their P row is
  written as all zeros, making their PV output row zero, and
- the reduce kernel filtering by `q >= Q_LEN` and `h_global >= local_num_heads`.

## 3. What's been patched so far (partial, verify correctness)

The kernel now takes an optional runtime parameter:

```cpp
int local_num_heads = mla_mtp::NUM_HEADS
```

which is threaded through:

- `hpb = local_num_heads / num_head_groups`
- Q indexing: `global_row = bi * Q_LEN * local_num_heads + q * local_num_heads + gi * hpb`
- Reduce bounds: `if (q >= Q_LEN || h_global >= local_num_heads) return;`
- Reduce output offset: `o_base = (bi*Q_LEN+q) * local_num_heads * D_V + h_global * D_V`

and the TMA descriptor in `tma.cuh` derives `num_heads` from the Q tensor's
shape instead of assuming 128.

**This alone still produces garbage output in TP=2** (MPK cosine vs
PyTorch-TP reference measured at roughly `-0.04`). The remaining issues are
described below.

## 4. What the kernel needs (correctness)

### 4.1 Decouple MMA tile M from `NUM_HEADS = 128`

`idesc_qk` / `idesc_pv` currently encode `M = 128`. In TP>1 the actual number
of valid Q rows per block is at most `Q_LEN * hpb`, which can be as small as
`1 * (local_num_heads / num_head_groups) = 16` or even less.

Acceptable solutions, in order of preference:

1. **Parameterize MMA M** to a supported tcgen05 size that is ≥ the valid row
   count and ≤ 128. The natural choices on SM100 are `M ∈ {64, 128}` for
   1-CTA tcgen05 MMA. Map e.g.:
   - `local_num_heads ≥ 128`, Q_LEN=1: M=128, hpb=128, num_head_groups=1
   - `local_num_heads == 64`, Q_LEN=1: M=64, hpb=64, num_head_groups=1
   - `local_num_heads == 32`, Q_LEN=1: M=32 if supported, else M=64 with padding
   - `local_num_heads == 16`, Q_LEN=1: M=16 if supported, else M=64 with padding

2. **Pad Q SMEM to 128 rows with zeros** before the MMA if M=128 must stay.
   Zero-filled rows → zero attention scores → softmax gives uniform attention
   which in turn makes `row_sum > 0` for those padding rows (non-zero output).
   To keep those rows from polluting reduce, the epilogue must explicitly
   **zero the output for rows with `q_idx * hpb + h_local ≥ local_num_heads`**,
   *before* dividing by `row_sum`.

3. **Run multiple smaller MMA passes** (e.g. two M=64 MMAs) and synthesize the
   128-row tmem region for reduce-compat. This is only worth doing if the
   existing post-MMA logic (softmax passes 1/2, PV, epilogue) can be reused
   without restructuring.

### 4.2 Preserve the "row tid → head tid" mapping

Everything downstream of the MMA (softmax, P SMEM store, PV, epilogue,
reduce) assumes that thread `tid`'s tmem row corresponds to the Q row with
index `tid`. If the MMA tile M changes, the thread-to-row mapping must keep
this invariant, or all three downstream passes need to be rewritten.

Specifically:

- Softmax pass 1 & 2 use `taddr + (tid << 16) + c` to load tmem. For M<128,
  only threads `tid < M` should be active; the rest must skip cleanly.
- P SMEM store writes to `p_base + tid * 128 + swizzled(g)`. Need to ensure
  this does not overlap between blocks when M<128.
- PV MMA uses the whole P SMEM (128 rows × 128 cols). If only M<128 rows are
  valid, rows `M..127` of P SMEM must be zero when the MMA reads them.
- Epilogue writes `Oout[(vc*BK+c)*128 + tid + i*128] = val`, so every `tid`
  ends up writing a slot in the `[D_V, 128]` per-block region. Reduce assumes
  slot `tid = q * hpb + h_local` — if M changes so that tid goes only up to
  M-1, reduce needs to know that only slots `0..M-1` are valid.

### 4.3 Update reduce

Reduce currently reads `La[... + row]` for `row = tid % 128` and `Oa[... + s
* D_V * 128 + d * 128 + row]` for all 128 rows, filtered at the end by
`q >= Q_LEN || h_global >= local_num_heads`.

If the decode kernel is changed to only populate M < 128 rows per block (and
either zeros or skips the rest), reduce should either:

- continue reading all 128 rows but ignore rows where h_global ≥ local_num_heads
  (works only if the decode writes zero in the invalid slots, which in turn
  requires the epilogue invariant in 4.1 option 2), or
- be changed to read only `hpb * Q_LEN` rows per block.

Either way: **reduce's output offset computation must use `local_num_heads`
rather than hardcoded 128** (already changed in the patched version).

### 4.4 Update task_register.cc / builder

The task registration in `src/kernel/task_register.cc` computes
`hpb = 128 / q_len` and `num_head_groups = 128 / hpb`. In TP mode these need
to be based on `local_num_heads`. Already patched:
```cpp
int hpb = std::min(128 / q_len, num_heads);
while (hpb > 0 && num_heads % hpb != 0) { hpb--; }
if (hpb <= 0) hpb = 1;
int num_head_groups = num_heads / hpb;
```
and `local_num_heads = num_heads` is passed through to the kernel.

The Python builder (`python/mirage/mpk/models/deepseek_v3/builder.py`) sizes
the Q tensor as `[mbt, local_num_heads * D_K]`, and sets
`grid_y_mla = num_head_groups_mla` consistent with the above.

## 5. Test harness (already built)

`demo/deepseek_v3/demo.py` has a function `run_tp_reference(args, state_dict,
layer_indices, tp_size, device, first_token_id)` that runs an explicit
per-rank PyTorch simulation of the TP attention path (sharded q_b_proj, local
MLA, sharded o_proj + allreduce). On single layer / single token this matches
the single-GPU PyTorch reference at cosine 0.999957, so it is a trustworthy
baseline.

With a correct kernel, the cosine between MPK's final logits (rank 0) and the
single-GPU PyTorch reference should be ≥ 0.99 for any supported TP size. The
existing test is `tests/runtime_python/blackwell/test_deepseek_v3_tp.sh`.

## 6. Non-goals

- **No Expert Parallelism change** for MoE: experts stay replicated in TP
  mode for now. Group-GEMM EP performance work is separate.
- **No change to KV cache layout**: kv_latent_pe cache is replicated per-rank
  and indexed by global position; MLA decode reads KV exactly as it does today.
- **No change to `o_proj` or MLP TP wiring**: those are column/row-parallel
  FP8 linears and are already correct (modulo a separate builder bug already
  fixed — see `project_tp_residual_bug.md`).
