# `bmm_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** per-head batched matmul `C[t,h,:] = A[t,h,:]·W[h,:,:]ᵀ`; the absorbed-MLA
projections (BMM1 q→latent / BMM2 o-unabsorb). **Shared by both attention paths** — prefill
([`mla_prefill_attn`](./mla_prefill_attn.md)) and decode ([`mla_decode_attn`](./mla_decode_attn.md))
both feed through BMM1/BMM2.

**Phase:** both (prefill + decode).

**grid_dim:** `(Dout/128, Hd, 1)`; grid.y = one head/CTA, grid.x tiles `Dout`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A_fp8` | `[T,Hd,Kin]` | e4m3 | head-major; per-head activation |
| `A_scale` | `[T,Hd,*]` | uint32 | per-head UE8M0 |
| `W_fp8` | `[Hd,Dout,Kin]` | e4m3 | per-head absorbed weight |
| `W_scale` | `[Hd,*,*]` | uint32 | per-head block scales UE8M0 |

The scales all must be uint32 (4 packed UE8M0. Every 32 elements share one UE8M0 scale).

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[T,Hd,Dout]` | bf16 | head-major |

**Params:** none — sizes (`Hd`,`Dout`,`Kin`) and scale-encoding derived from the tensors.

**Shape variants** (`Hd=16` at this config):

| role | Kin | Dout | grid |
|---|---|---|---|
| BMM1 q→latent (W_UK) | 128 (qk_nope) | 512 (kv_lora) | (4,16,1) |
| BMM2 o un-absorb (W_UV) | 512 (kv_lora) | 128 (v_head) | (1,16,1) |

**Tensor-view requirement (MUST):** BMM1 writes its `[T,H,512]` latent output into the
**`[:,:,:512]` slice-view** of the decode `q[T,H,576]` buffer (whose `[:,:,512:576]` tail holds the
roped `q_pe`, written by `rope`) — so `q` is assembled **in-place**, no separate kernel. The store
must honor the parent per-head row stride (576), not 512. Inputs/outputs may also be reshaped
2D⇄3D views of the same bytes.

## Python API
```python
def bmm_fp8_layer(
    self,
    A_fp8: DTensor,           # [T,Hd,Kin] e4m3, head-major per-head activation
    A_scale: DTensor,         # [T,Hd,*] uint32, per-head UE8M0
    W_fp8: DTensor,           # [Hd,Dout,Kin] e4m3, per-head absorbed weight
    W_scale: DTensor,         # [Hd,*,*] uint32, per-head block scales UE8M0
    C: DTensor,               # [T,Hd,Dout] bf16, head-major (output; may write into a slice view)
    grid_dim: tuple,          # (Dout/128, Hd, 1); grid.y = one head/CTA
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Tasks dispatched (by scale-encoding)** — the one `bmm_fp8` layer registers one of these tasks (different partition maps / grid.x constraints → Python-level L1 select):

| scale encoding | kernel |
|---|---|
| UE8M0 (packed) | `linear_fp8_bmm_sm100_layer` |
| fp32 block scale | `linear_fp8_bmm_dense_sm100_layer` |

*`Hd` / `Dout` / `Kin` and the scale-encoding are **derived** from the tensor shapes/dtype
(`A_fp8`/`W_fp8` shape, `A_scale` dtype) — not passed explicitly (per the [API convention](./README.md)).*