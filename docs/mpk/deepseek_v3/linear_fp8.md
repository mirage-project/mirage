# `linear_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `C = A·Wᵀ`, **dense** FP8 e4m3, block-scaled. (No fused residual — on the TP path
the residual is in [`all_reduce`](./all_reduce.md). Split-K is the separate
[`splitk_linear_fp8`](./splitk_linear_fp8.md); per-head batched is [`bmm_fp8`](./bmm_fp8.md).)

**Phase:** both.

**grid_dim:** `(N/128, 1, 1)` (e.g. N=1536→`(12,1,1)`, N=7168→`(56,1,1)`); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A_fp8` | `[M,K]` | e4m3 | row-major; quantized activation (M=T) |
| `A_scale` | — | uint32 (UE8M0) | K-major per-128 group (see [`quantize_fp8`](./quantize_fp8.md)) |
| `W_fp8` | `[N,K]` | e4m3 | row-major; quantized weight |
| `W_scale` | — | uint32 (UE8M0) | per-128-block scales |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[M,N]` | bf16 | row-major |

**Params:** none — sizes/scale are derived from the tensors. This layer is **TP-agnostic**: it
just computes `C = A·Wᵀ`. The `tp_role` column below is *caller context* (how the model sets up
the sharded tensors and places [`all_reduce`](./all_reduce.md)), **not** a layer parameter.

**Shape variants** (M=T; per-rank shapes at TP=4×EP=2):

| role | K | N (per-rank) | tp_role |
|---|---|---|---|
| q_a (down) | 7168 | 1536 | replicated |
| kv_a (down, +mqa) | 7168 | 576 | replicated |
| q_b (up) | 1536 | 3072 (=16·192) | column |
| o_proj (down) | 2048 (=16·128) | 7168 | row +AR |
| dense gate_up | 7168 | 4608 (=2·18432/8) | column |
| dense down | 2304 (=18432/8) | 7168 | row +AR |
| shared gate_up | 7168 | 512 (=2·256) | column |
| shared down | 256 | 7168 | row +AR |

*(decode o_proj uses split-K → [`splitk_linear_fp8`](./splitk_linear_fp8.md).)*

## Python API
```python
def linear_fp8_layer(
    self,
    input_fp8: DTensor,       # [M,K] e4m3, quantized activation (M=T); may be mpk.narrow slice
    input_scale: DTensor,     # uint32 UE8M0, K-major per-128-group
    weight_fp8: DTensor,      # [N,K] e4m3, quantized weight
    weight_scale: DTensor,    # uint32 UE8M0, per-128-block scales
    output: DTensor,          # [M,N] bf16, row-major (output)
    grid_dim: tuple,          # (N/128,1,1)
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Tasks dispatched (by M-size)** — the one `linear_fp8` layer registers one of these tasks at build time (same dense op, internal M-tiling; FP8-out epilogue via `*_fp8out`):

| M range | kernel |
|---|---|
| small-M (decode) | `fp8_gemm_dense_smallm_layer` |
| medium-M | `fp8_gemm_dense_mediumm_layer` |

*(v1: input is a standalone [`quantize_fp8`](./quantize_fp8.md), not the fused rmsnorm+quant.)*

**Runtime fusions (not separate contracts):**
- **`qkv_a` is one fused GEMM**: the `q_a`+`kv_a` rows are a single runtime GEMM
  `H → 2112 = [q_a 1536 | c_latent 512 | k_pe 64]`; consumers read `mpk.narrow` views.
- **`o_proj` residual is handled by splitk linear fp8.
- **Absorbed MLA for both prefill & decode** (absorbed math for both — no `kv_b` GEMM): `q_b` → `q_nope`+`q_pe`;
  `q_b_nope` → [`bmm_fp8`](./bmm_fp8.md) BMM1 → `q[:,:,:512]`; `q_b_pe` (+ `rope`) → `q[:,:,512:576]`
  (the `q[T,H,576]` buffer is assembled **in-place via views**, no separate kernel). **No `kv_b`
  GEMM** — kv_b is absorbed into BMM1 (W_UK, query side) / BMM2 (W_UV, output side); the KV cache
  stays compressed.

**Tensor-view requirement (MUST):** `A_fp8` is often a `mpk.narrow` slice — e.g. `q_b` reads the
post-norm `q_a` slice of `qkv_a [T,2112]` (stride `[2112,1]`); load A via `stride[0]` + offset.
Some outputs write into slices of a wider buffer (BMM1 → `q[:,:,:512]`, see [`bmm_fp8`](./bmm_fp8.md)).
