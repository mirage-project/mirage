# `moe_topk_hash_routing_layer`

V4's MoE gate kernel for the **first `num_hash_layers` layers** (= 3 in V4-Flash-Base):
expert assignments come from a fixed table `tid2eid[input_ids]` rather than from per-token
scoring. Used for layers 0, 1, 2.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Class: `Gate.forward`, lines 564-584, hash branch (line 576-577 + the surrounding
  scoring/weight code)
- Constructor lines 556-560 register `tid2eid` as a non-trainable parameter
  `[vocab_size, n_activated_experts]` (INT32) and set `bias = None`

Snippet:
```python
scores         = sqrt(softplus(linear(x.float(), gate.weight.float())))   # still computed (for weights)
original_scores= scores
indices        = self.tid2eid[input_ids]                  # [B, K], INT32 (table lookup)
weights        = original_scores.gather(1, indices)       # [B, K]
weights       /= weights.sum(-1, keepdim=True)
weights       *= route_scale                              # 1.5
```

So `indices` are deterministic per input token id (no scoring used for selection), but
`weights` still come from the live `sqrtsoftplus` scoring of the gate logits.

## Math

```
indices = tid2eid[input_ids]                              # [B, K]  - precomputed lookup
weights = sqrtsoftplus(router_logits).gather(-1, indices) # [B, K]
weights = weights / weights.sum(-1, keepdim=True) * 1.5
```

Compared to [`moe_topk_sqrtsoftplus_routing_layer`](moe_topk_sqrtsoftplus_routing_layer.md)
the only difference is how `indices` are obtained:

| step | sqrtsoftplus layer | hash layer |
| --- | --- | --- |
| compute scores | `sqrt(softplus(x @ W_g))` | same |
| add bias | `scores + bias` | (no bias) |
| pick `indices` | `topk(scores+bias, K)` | `tid2eid[input_ids]` |
| compute `weights` | `gather(original_scores, indices)` | `gather(original_scores, indices)` |
| normalize, scale | same | same |

## Python API (proposed)

```python
pk.moe_topk_hash_routing_layer(
    input:    DTensor,        # BF16 [B, n_experts]   (router logits)
    tid2eid:  DTensor,        # INT32[vocab_size, num_experts_per_tok]
    input_ids:DTensor,        # INT32[B]              (per-token ids in the current batch)
    output: tuple[DTensor, DTensor, DTensor],         # (weights, routing_indices, masks)
    grid_dim: tuple,
    block_dim: tuple,
    routed_scaling_factor: float = 1.5,
)
```

Output triple matches V3 conventions, identical to
[`moe_topk_sqrtsoftplus_routing_layer`](moe_topk_sqrtsoftplus_routing_layer.md), so
downstream `moe_w13_fp8_layer` / `moe_w2_fp8_layer` / `moe_mul_sum_add_layer` can be reused.

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input` | BF16 | `[B, n_experts]` | router logits = `linear_layer(x, gate.weight)` |
| `tid2eid` | INT32 | `[vocab_size, num_experts_per_tok]` | `gate.tid2eid` (loaded from checkpoint, non-trainable) |
| `input_ids` | INT32 | `[B]` | the token ids being processed in this step (already broadcast/sliced for the current batch). MPK runtime exposes these via the meta-tensor pipeline (V3 already maintains an `input_ids` buffer for MTP). |

For V4-Flash-Base: `vocab_size = 129280`, `num_experts_per_tok = 6`, `n_experts = 256`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `weights` | BF16 | `[B, num_experts_per_tok]` | `moe_mul_sum_add_layer` |
| `routing_indices` | INT32 | `[n_experts, B]` | `moe_w13_fp8_layer`, `moe_w2_fp8_layer` |
| `masks` | INT32 | `[n_experts + 1]` | `moe_w13_fp8_layer`, `moe_w2_fp8_layer` |

## Builder usage

Used only for layers `L` where `L < num_hash_layers (=3)` (V4-Flash-Base layers 0, 1, 2):

```python
router_logits = pk.linear_layer(x, gate.weight, ...)                  # reuse V3
weights, idxs, mask = pk.moe_topk_hash_routing_layer(
    input=router_logits, tid2eid=gate.tid2eid, input_ids=input_ids,
    output=(weights, idxs, mask),
    grid_dim=..., block_dim=...,
    routed_scaling_factor=1.5,
)
```

V3 analog: none. V3 has no hash routing.

## Notes / risks

- `tid2eid` is a `[129280, 6]` INT32 table = ~3 MB. Persistent in device memory; loaded
  once at startup.
- The kernel still has to compute the `sqrtsoftplus` scoring to gather the routing
  weights. This is the same compute as
  [`moe_topk_sqrtsoftplus_routing_layer`](moe_topk_sqrtsoftplus_routing_layer.md) minus the
  topk; consider sharing the scoring/normalize/scale code path between the two kernels and
  branching only on the indices source.
- The "indices" tensor produced by `tid2eid[input_ids]` may contain the **same expert id
  multiple times** (since the table assignment is fixed per token id, not per token
  instance). The downstream MoE expert dispatch (`moe_w13_fp8_layer`) must tolerate
  duplicates. The V3 kernels do tolerate duplicates (the routing-indices tensor is
  expert-major and the mask is just a presence indicator), but verify before deploying.
- `input_ids` plumbing: V3 already pipes `input_ids` through MTP buffers (see
  `mtp_build_embed_input_layer` in `persistent_kernel.py:2064`). For non-MTP V4 we still
  need `input_ids` per token — extend the meta-tensor pipeline to expose them at MoE time
  for hash layers.
- Verification: golden test where the same `(router_logits, input_ids)` are run through
  both the reference `Gate.forward` (hash branch) and the MPK kernel, comparing all three
  outputs.
