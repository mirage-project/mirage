# FP8 Group GEMM Kernel — Row/Expert Assignment Analysis

Files:
- `common.cuh` = `include/mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_sm100_common.cuh`
- `wrapper.cu` = `tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/runtime_kernel_wrapper.cu`
- `test_wrapper.py` = `tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/test_wrapper.py`
- `smoke.py` = `tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/test_mpk_smoke.py`

---

## 1. BM=128 and M_total < 128

`BM=128` is a `constexpr` in the kernel body, **not** a template parameter:

```
common.cuh:114  constexpr int BM = 128, BK = 128, UK = 32;
```

The tile count `nm = (M_total + BM - 1) / BM` (common.cuh:121), so MPE=1, E=32, M_total=32 gives `nm=1` — exactly one BM block is launched.

**TMA behavior for OOB rows:** The A TMA descriptor is encoded with `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (wrapper.cu:76). For SM100a this means out-of-bounds tile coordinates zero-fill the shared memory tile — rows 32-127 in the single BM block receive all-zero FP8 data. The MMA accumulates zeros for those rows. The epilogue TMA store (common.cuh:383-385) issues a 2D bulk store at `(on, om)=(bn*BN, m_start)` — the descriptor extent is `[N, M_total]` (wrapper.cu:113), so the hardware only commits rows 0-31 to memory. Rows 32-127 in the SMEM staging buffer are written by the epilogue but fall outside the TMA store extent and are silently discarded.

Result: M_total < BM is handled correctly. The smoke test MPE=1 (`gate_up_M1`, `down_M1`) produces correct output for all 32 real rows.

---

## 2. Sparse layout — per-block expert selection

The TMA load warp reads a single expert ID for the entire BM=128 block:

```
common.cuh:212  int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
```

`m_start = bm * BM` — this is the **first row** of the block. The derived B column offset is:

```
common.cuh:214  int on = expert_id * N + bn * BN;
```

This sets the B-matrix TMA coordinate for the whole block. There is no per-row expert lookup anywhere in the kernel. All 128 rows in the block compute against the same expert's weight columns `[expert_id*N + bn*BN, expert_id*N + (bn+1)*BN)`.

For MPE=8 (8 tokens per expert), a BM=128 block spans `128/8 = 16` experts (e.g., rows 0-7 → expert 0, rows 8-15 → expert 1, …, rows 120-127 → expert 15). The kernel uses `expert_id = m_indices[bm*BM] = 0`, loading expert 0's B columns for all 128 rows. Rows 8-127 get multiplied against the wrong expert's weights. The epilogue stores these incorrect values to the correct output rows. **This is silent data corruption for any input where rows within a BM block span multiple experts.**

---

## 3. Reference impl consistency

Kernel line (common.cuh:212):
```cpp
int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
```

Reference line (test_wrapper.py:89):
```python
expert_id = int(m_indices[bm].item())
```

Both index `m_indices` at the block's first row (`bm = bm*BM` in Python iterates in steps of BM; `m_indices[bm]` = `m_indices[bm*BM]`). The `make_inputs` function constructs:

```python
test_wrapper.py:61  m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // MPE
```

This assigns row `i` to expert `i // MPE`. With uniform MPE, all rows in any given BM block share the same expert **if and only if MPE divides BM** — and the test only exercises MPE ∈ {1, 4, 8, 16}, all of which divide 128. So for every tested config, `m_indices[bm*BM : (bm+1)*BM]` is constant within each block: the constraint is satisfied, the single-index lookup is correct, and both kernel and reference agree. **The test validates the kernel only on layouts where the constraint holds. It does not detect the corruption described in section 2.**

---

## 4. Hidden layout constraints

The header comment (common.cuh:34-35) states the constraint explicitly:

```
// m_indices[M_total]  int32, expert id per row (rows in
//                     [bm*BM, (bm+1)*BM) must share expert)
```

No runtime assertion enforces this. `wrapper.cu:147` computes `int MPE = M_total / E;` (integer division, assumes uniform token counts), but does not assert `M_total % E == 0` or `MPE * BM % 1 == 0`. No check exists in any of the six files that would catch a sparse layout at runtime.

---

## 5. DSv3 usability verdict

DSv3 decode: batch=128, top_k=8, E=256. After routing, ~4 tokens/expert on average (highly non-uniform). Sorted by expert, M_total = 1024.

**(a) Padding to BM=128:** Each of 256 experts would need its rows padded to 128. Real tokens: 1024. Padded: 256×128 = 32768. Compute amplification: 32×. This is unusable.

**(b) Per-row fix feasibility:** The TMA load warp issues a single 2D bulk copy to SMEM for the entire BN×BK B-tile (common.cuh:224). The B address encodes one `expert_id * N + bn*BN` offset. To support mixed experts in one block, you would need either (i) 128 separate TMA loads (one per row), destroying the throughput model, or (ii) B laid out as `[M_total, K]` with rows pre-scattered by expert (i.e., pre-permuted), which eliminates the group GEMM structure. A small epilogue fix alone cannot compensate because the MMA tile itself was computed against the wrong B columns. **Per-row expert selection inside a BM block is architecturally incompatible with this kernel's TMA+tcgen05 design.**

**(c) Recommendation:** For DSv3 sparse decode (~4 tokens/expert), the old `moe_w13_fp8_sm100` kernel (or DeepGEMM's grouped GEMM with proper per-expert segment boundaries) is the right tool. This kernel is **correct and fast only when each expert's token count is a multiple of BM=128** — i.e., the dense pre-padded case. Keep this kernel for prefill batch GEMM where padding cost is amortized over large M. For sparse decode, do not use it without guaranteed uniform-MPE and MPE ≥ 1 with MPE | 128.

**(d) Not applicable** — the concern is confirmed, not based on a misread.
