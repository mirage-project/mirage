# DeepSeek V3 — Kernel Specification Category

A top-down, contract-first specification of the kernels needed to run the DeepSeek V3 MPK
demo. Each file specifies **one Python-level layer** — its semantics, exact I/O
(layout · dtype · meaning), `grid_dim`, supported shape variants, and how it maps to existing
kernels. These contracts are the debugging anchor: a wrong demo is diagnosed by checking each
layer against its contract, not by reverse-engineering the existing code.

> **Scope:** specification only — no kernel implementation. Each spec's `Reuse` line names the
> existing kernel(s) that fulfill it; when one layer fuses/dispatches multiple kernels, it
> carries a `dispatch`/`pipeline` table.
>
> **API convention:** signatures pass only `grid_dim`/`block_dim` + non-derivable selectors/config.
> **All dimension sizes and strides are derived from the `DTensor` arguments** (their
> shape / stride / dtype) — never passed explicitly.
>
> **Everything is compile-time.** `grid_dim` (and any selector that sizes it — e.g. a decode
> query-token *budget* or a `max_kv_len` capacity) is a **build-time constant** baked into the task
> graph; the number of tasks is fixed when the megakernel compiles. **Runtime** variation (the live
> sequence length, the number of accepted MTP tokens) is handled *inside* kernels via meta-tensors
> and masking — it never resizes the grid. So length-like params name **capacities/budgets**
> (`max_kv_len`, `decode_q_len`, `max_q_len`), not live lengths.

## Organizing rule
- **1 file = 1 Python-level layer.** If that layer dispatches/fuses multiple **kernels (tasks)**
  underneath, the file keeps one spec and **annotates the fusion** (a `variant → kernel` table).
- **Essentially-different Python layers** (different API/semantics — e.g. `+residual`, split-K,
  bf16-vs-fp8) live in **separate files**.

## Layer vs task
A **layer** is one Python-level API (one spec file). A **task** is a registered `__device__`
kernel (a `TaskType` + its codegen). A layer **registers/dispatches one or more tasks**:
- **single-task layer** — `Reuse:` names the one task it registers.
- **multi-task layer** — carries a **`condition → task`** table (`Impl: dispatch`); the layer
  picks one task **at build time** by that condition (M-size, `tp_degree`, scale-encoding, …).
  That table is the explicit, complete list of tasks the layer may dispatch.

(Task entries reference the existing `persistent_kernel.py` method that registers each task.
`rope` is a special case: one templated task, parameterized by the tensor's layout (no `role`
param — Q/K rope are the same op) — see its spec.)

## Target scenario (v1)
- **num_request = 1, B200 GPU.** TP=4 × EP=2, world_size=8. NEW grouped-GEMM **and** OLD per-expert MoE. Greedy.
- **Prefill once, then decode.** Prefill and decode are **separate forwards** — never mixed in one pass.
  Prefill can be long (~8K tokens); decode handles **1–8 tokens/step** (speculative / MTP shape).
- **Absorbed math for both (but two attention kernels):** prefill and decode both use the
  **absorbed** MLA math — they share BMM1 (q→latent), the compressed KV cache, and BMM2 (un-absorb);
  the **unabsorbed `kv_b` prefill path is dropped**. The attention *core* still forks by work
  distribution: **prefill** = q-tiled causal flash ([`mla_prefill_attn`](./mla_prefill_attn.md),
  single-pass), **decode** = kv-split + reduce ([`mla_decode_attn`](./mla_decode_attn.md) +
  [`mla_decode_reduce`](./mla_decode_reduce.md)). Different `grid_dim` → separate layers. Prefill is
  slower this way, but runs only once.
- **No paged KV cache:** num_request=1, single sequence → a **contiguous** KV cache; no `qo_indptr` / paging.
- **MTP layers NOT built** (main model only) — but the decode attention must be **MTP-compatible** (1–8 tokens).
- **Attention:** heads split across all 8 ranks → `Hd = 128/8 = 16` heads/rank (tp8).
- **Routed experts:** `routed_tp = world/ep = 4`, `ep = 2` → `E_loc = 256/2 = 128` experts/rank;
  routed intermediate `I_r = 2048/4 = 512`/rank. **Shared expert:** TP across all 8 → `I_s = 256`/rank.
- **Deferred:** MTP draft/verify/accept layers, probabilistic sampling, chunked prefill, paging / multi-request.

## DSv3 constants
`H=7168`, `n_heads=128`, `q_lora=1536`, `kv_lora=512`, `qk_nope=128`, `qk_rope=64`,
`qk_head=192`, `v_head=128`, `q_absorbed=576 (=512+64)`, `inter_dense=18432`,
`inter_moe=2048`, `E=256`, `Ep=8`, `n_group=8`, `topk_group=4`, `V=129280`,
`n_layers=61` (0–2 dense, 3–60 MoE). `T`=batched tokens (`mbt`, e.g. 128).

## FP8 scale convention
All FP8 GEMM / quantize scales are **`uint32` packed UE8M0** (4 exponent-only 8-bit scales per
word), K-major (contiguous along the K/contraction axis). **fp32 scales are optional and not
required** — a task may accept fp32, but the spec does not require fp32 support.

## Manifest (26 types)

| # | Spec | One-line semantics | Phase | Impl |
|---|---|---|---|---|
| 1 | [`embed`](./embed.md) | token-id → hidden lookup | both | reuse |
| 2 | [`rmsnorm`](./rmsnorm.md) | RMS normalize → bf16 | both | reuse |
| 3 | [`quantize_fp8`](./quantize_fp8.md) | bf16 → per-group FP8 + scale | both | reuse |
| 4 | [`linear`](./linear.md) | bf16 dense GEMM | both | reuse |
| 5 | [`splitk_linear`](./splitk_linear.md) | bf16 split-K GEMM | both | reuse |
| 6 | [`linear_fp8`](./linear_fp8.md) | dense FP8 GEMM | both | dispatch (M) |
| 7 | [`splitk_linear_fp8`](./splitk_linear_fp8.md) | split-K FP8 GEMM | decode | dispatch |
| 8 | [`bmm_fp8`](./bmm_fp8.md) | per-head batched FP8 matmul (BMM1/BMM2) | both | dispatch (scale) |
| 9 | [`rope`](./rope.md) | rotary on decoupled rope dims (Q & K, same op) | both | reuse (templated) |
| 10 | [`kv_cache_gather`](./kv_cache_gather.md) | append compressed KV to contiguous cache | both | reuse |
| 11 | [`mla_decode_attn`](./mla_decode_attn.md) | absorbed MLA attention, kv-split (decode) | decode | dispatch (tp) |
| 12 | [`mla_decode_reduce`](./mla_decode_reduce.md) | merge split-KV partials | decode | dispatch (tp) |
| 13 | [`mla_prefill_attn`](./mla_prefill_attn.md) | absorbed MLA attention, q-tiled (prefill) | prefill | reuse |
| 14 | [`silu_mul`](./silu_mul.md) | silu(gate)·up (dense) | both | reuse |
| 15 | [`moe_silu_mul`](./moe_silu_mul.md) | silu(gate)·up (MoE; both paths) | both | reuse |
| 16 | [`moe_router`](./moe_router.md) | sigmoid group top-k | both | reuse |
| 17 | [`moe_permute`](./moe_permute.md) | local token→expert permute | both | reuse |
| 18 | [`grouped_gemm_fp8`](./grouped_gemm_fp8.md) | grouped FP8 GEMM over local experts | both | dispatch (M) |
| 19 | [`moe_unpermute`](./moe_unpermute.md) | unpermute + weighted sum + residual | both | reuse |
| 20 | [`moe_w13_fp8`](./moe_w13_fp8.md) | OLD per-expert gate+up FP8 GEMM | both | reuse |
| 21 | [`moe_w2_fp8`](./moe_w2_fp8.md) | OLD per-expert down FP8 GEMM | both | reuse |
| 22 | [`moe_mul_sum_add`](./moe_mul_sum_add.md) | OLD weighted sum over experts + residual | both | reuse |
| 23 | [`all_reduce`](./all_reduce.md) | sum across TP×EP ranks | both | reuse |
| 24 | [`argmax_partial`](./argmax_partial.md) | per-worker partial argmax | both | reuse (pipeline 1/2) |
| 25 | [`argmax_reduce`](./argmax_reduce.md) | reduce partials → token | both | reuse (pipeline 2/2) |
| 26 | [`global_argmax`](./global_argmax.md) | cross-rank argmax (sharded vocab) | both | reuse (cond.) |

*MoE has two interchangeable expert-compute paths: **NEW** (`moe_permute`→`grouped_gemm_fp8`→`moe_silu_mul`→`grouped_gemm_fp8`→`moe_unpermute`) and **OLD** per-expert (`moe_w13_fp8`→`moe_silu_mul`→`moe_w2_fp8`→`moe_mul_sum_add`). Both share `moe_router`, `quantize_fp8`, `moe_silu_mul`.*

**Impl tags:** `reuse` (1 kernel) · `dispatch (X)` (one layer, pick a kernel by `X`) ·
`pipeline` (multi-stage, all run) · `new`. **Spec template:** `Semantics` · `Phase` · `grid_dim`
· `Inputs` · `Outputs` · `Params` · `Shape variants` · `Reuse` · `Open`.

## Forward pass (TP=4×EP=2, num_request=1)

Prefill and decode are **separate forwards** sharing the same **absorbed math** (BMM1 · compressed
KV · BMM2); only the **attention core forks** — prefill = q-tiled causal flash (`mla_prefill_attn`),
decode = kv-split + reduce (`mla_decode_attn`). o_proj also differs (dense for prefill's large-M,
split-K for decode's small-M).

```
token_ids → embed
[×61]
  rmsnorm(input)
  q:  linear_fp8(q_a) → rmsnorm(q_a) → linear_fp8(q_b)     # → q_nope[H,128] + q_pe[H,64]
  kv: linear_fp8(kv_a) → rmsnorm(kv_a)                     # → c_latent[512] + k_pe[64] (stays compressed)
  rope(Q: q_pe); rope(K: k_pe); kv_cache_gather            # append compressed KV to contiguous cache
  # ── absorbed MLA — shared BMM1/BMM2 + compressed KV; attention core forks by q_len ──
  quantize_fp8(q_nope) → bmm_fp8(BMM1) writes q[:,:,:512]; q_pe (roped) already in q[:,:,512:576]
  prefill: mla_prefill_attn          (q-tiled causal flash over kv[0:S], single-pass) → bmm_fp8(BMM2)
  decode:  mla_decode_attn(tp8) → mla_decode_reduce(tp8)  (kv-split, q_len 1–8 MTP) → bmm_fp8(BMM2)
  # the q[T,H,576] buffer is assembled in-place via tensor views (no assemble kernel)
  o_proj: linear_fp8 [prefill, large-M] | splitk_linear_fp8 [decode, small-M] → all_reduce(+res)
  rmsnorm(post-attn)
  dense(0-2): linear_fp8(gate_up) → silu_mul → linear_fp8(down) → all_reduce(+res)
  MoE(3-60) NEW:  moe_router → quantize_fp8 → moe_permute
              → grouped_gemm_fp8(w13) → moe_silu_mul → quantize_fp8 → grouped_gemm_fp8(w2)
              → moe_unpermute(+shared, +res) → all_reduce
  MoE(3-60) OLD:  moe_router → quantize_fp8
              → moe_w13_fp8 → moe_silu_mul(3D) → quantize_fp8 → moe_w2_fp8
              → moe_mul_sum_add(+shared, +res) → all_reduce
              [shared expert = linear_fp8 → silu_mul → linear_fp8, both paths;
               all_reduce sums routed across TP×EP=8 ranks]
rmsnorm(final) → linear(lm_head) → argmax_partial→argmax_reduce | global_argmax → token
```

## Reuse targets
- `python/mirage/mpk/models/deepseek_v3/builder.py` — call sites, TP/EP topology, grid_dim.
- `python/mirage/mpk/persistent_kernel.py` — existing layer methods.

## Resolved during design
- **Topology:** attention 16 heads/rank (decode=tp8); routed experts `E_loc=128`, `I_r=512`; shared `I_s=256`.
- **MoE:** local permute/grouped-GEMM/unpermute + single `all_reduce`; no EP dispatch/combine.
- **BMM dims:** BMM1 `128→512`, BMM2 `512→128`, per-head `Hd=16`.
- **Prefill = absorbed, but its own kernel (num_request=1):** prefill uses the absorbed math — it
  shares BMM1/BMM2 + the compressed KV cache with decode — but a **distinct q-tiled attention kernel**
  (`mla_prefill_attn` → `mla_prefill_absorbed_sm100`, single-pass), because its `grid_dim`/work
  distribution (q-tiled flash) differs from decode's (kv-split + reduce). The unabsorbed `kv_b`
  prefill path is dropped. Slower per prefill, but runs once.
- **MTP-compatible, MTP not built:** the decode attention must handle `q_len = 1–8` (speculative /
  MTP token counts) correctly; the MTP draft/verify/accept layers are **not built** (main model only).
- **No paging:** single sequence → contiguous KV cache; `kv_cache_gather` is a plain append, no page table.
- **Org:** 1 Python layer per file; multi-kernel layers carry a dispatch/pipeline table. **All 26
  types reuse existing kernels** (none new — prefill = `mla_prefill_absorbed`). `linear_with_residual`
  and `assemble_q_decode` removed (bf16 residual GEMM unused in DSv3; the `q[576]` buffer is assembled
  in-place via tensor views); the unabsorbed `mla_prefill_tp8_chunked` (`kv_b`) is dropped in favor of
  the absorbed prefill.
- **MoE: both paths kept** — NEW grouped-GEMM (`moe_permute`/`grouped_gemm_fp8`/`moe_unpermute`)
  and OLD per-expert (`moe_w13_fp8`/`moe_w2_fp8`/`moe_mul_sum_add`); shared `moe_silu_mul`.
```
