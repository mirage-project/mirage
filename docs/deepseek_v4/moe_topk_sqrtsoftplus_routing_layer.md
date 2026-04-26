# `moe_topk_sqrtsoftplus_routing_layer`

V4's MoE gate kernel for **non-hash** layers (i.e., `layer_id >= num_hash_layers`).
Replaces V3's sigmoid scoring with **`sqrt(softplus(x))`** and applies a learnable bias
**only to expert selection**, not to routing weights. Drop-in replacement for V3's
`moe_topk_sigmoid_routing_layer` in `python/mirage/mpk/persistent_kernel.py:1318-1352`,
keeping the same `(weights, indices, mask)` output convention.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Class: `Gate.forward`, lines 564-584 (the non-hash branch)
- Snippet (with hash branch removed):
  ```python
  scores = linear(x.float(), self.weight.float())          # [B, n_experts]
  if score_func == "softmax":   scores = scores.softmax(-1)
  elif score_func == "sigmoid": scores = scores.sigmoid()
  else:                          scores = F.softplus(scores).sqrt()
  original_scores = scores
  if self.bias is not None:
      scores = scores + self.bias                          # bias shifts SELECTION only
  indices = scores.topk(topk, dim=-1)[1]                   # NO group-based selection
  weights = original_scores.gather(1, indices)             # weights from PRE-bias scores
  if score_func != "softmax":
      weights /= weights.sum(dim=-1, keepdim=True)         # L1 normalize
  weights *= route_scale                                   # route_scale = 1.5 in V4-Flash-Base
  return weights, indices
  ```

V4-Flash-Base config: `score_func = "sqrtsoftplus"`, `route_scale = 1.5`,
`topk = num_experts_per_tok = 6`, `n_experts = n_routed_experts = 256`.

Two important deltas vs V3:
1. **No group selection** in V4: V3's `moe_topk_sigmoid_routing_layer` does group-based
   top-k (`num_groups=8`, `topk_group=4`). V4 just runs flat `topk(6)` over 256 experts.
2. Scoring function is `sqrt(softplus(x))` instead of `sigmoid(x)`.

## Math

```
B = batch_size, E = n_routed_experts (=256), K = num_experts_per_tok (=6)

scores         = router_logits.float() (= x @ gate_weight.T)
softplus_scores= log(1 + exp(scores))              # numerically stable: max(0,x) + log(1 + exp(-|x|))
sp_sqrt        = sqrt(softplus_scores + tiny_eps)  # element-wise
selection      = sp_sqrt + bias                    # [B, E], bias is FP32 [E]
indices        = topk(selection, K, dim=-1).indices       # [B, K], INT32
weights_raw    = gather(sp_sqrt, dim=-1, index=indices)   # [B, K] from PRE-bias scores
weights        = weights_raw / weights_raw.sum(dim=-1, keepdim=True) * 1.5
mask           = build_routing_mask(indices)              # [E + 1], V3 convention
routing_idxs   = build_expert_major_indices(indices)      # [E, B], V3 convention
```

The `(weights, mask, routing_idxs)` output triple is the same shape and semantics as V3's
`moe_topk_sigmoid_routing_layer`'s output, so downstream `moe_w13_fp8_layer` /
`moe_w2_fp8_layer` / `moe_mul_sum_add_layer` can reuse without changes.

## Python API (proposed)

```python
pk.moe_topk_sqrtsoftplus_routing_layer(
    input: DTensor,           # FP32 or BF16 [B, n_experts]    (router logits)
    bias:  DTensor,           # FP32 [n_experts]
    output: tuple[DTensor, DTensor, DTensor],   # (weights, routing_indices, masks)
    grid_dim: tuple,
    block_dim: tuple,
    routed_scaling_factor: float = 1.5,
)
```

Output triple matches V3 exactly:
- `weights`: BF16 `[B, num_experts_per_tok]`
- `routing_indices`: INT32 `[n_experts, B]`  (expert-major)
- `masks`: INT32 `[n_experts + 1]`            (1-indexed mask, V3 convention)

This signature is intentionally identical to
`moe_topk_sigmoid_routing_layer(input, bias, output, grid_dim, block_dim,
num_groups, topk_group, routed_scaling_factor)` minus the `num_groups` and `topk_group`
args, since V4 does flat top-k.

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input` | BF16 | `[B, n_experts]` | output of `linear_layer(x, gate.weight)` (router logits) |
| `bias` | FP32 | `[n_experts]` | `gate.bias` (V4 names it `bias`; V3 named the analog `e_score_correction_bias`) |

For V4-Flash-Base: `B = batch * seqlen`, `n_experts = 256`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `weights` | BF16 | `[B, num_experts_per_tok]` | `moe_mul_sum_add_layer` |
| `routing_indices` | INT32 | `[n_experts, B]` | `moe_w13_fp8_layer`, `moe_w2_fp8_layer` |
| `masks` | INT32 | `[n_experts + 1]` | `moe_w13_fp8_layer`, `moe_w2_fp8_layer` |

## Builder usage

Used for layers `L` where `L >= num_hash_layers` (i.e., layers 3..42 in V4-Flash-Base):

```python
router_logits = pk.linear_layer(x, gate.weight, ...)         # reuse V3 BF16 linear
if L < num_hash_layers:
    weights, idxs, mask = pk.moe_topk_hash_routing_layer(...)   # see moe_topk_hash_routing_layer.md
else:
    weights, idxs, mask = pk.moe_topk_sqrtsoftplus_routing_layer(
        input=router_logits, bias=gate.bias,
        output=(weights, idxs, mask),
        grid_dim=..., block_dim=...,
        routed_scaling_factor=1.5,
    )
```

V3 analog: `moe_topk_sigmoid_routing_layer` (`persistent_kernel.py:1318`). The kernel
implementation borrows the top-k machinery; only the score function and the absence of
group selection differ.

## Notes / risks

- `softplus(x) = log(1 + exp(x))`. Use the numerically stable form
  `max(x, 0) + log(1 + exp(-|x|))` to avoid overflow for large positive `x`.
- The `sqrt` after softplus needs a small epsilon to avoid NaN at exact zero.
- Bias shifts the **selection** scores only — the returned weights come from
  `original_scores` via `gather`. Don't conflate.
- `route_scale = 1.5` in V4-Flash-Base. Pack as `int_bits` like V3 does
  (`persistent_kernel.py:1339`) to pass to the kernel via the `params` array.
- Verification: comparable test-mode harness to V3's MoE routing test; compare to the V4
  reference's `Gate.forward` non-hash branch.
