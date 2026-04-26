# DELTA: `silu_mul_layer` / `moe_silu_mul_layer` — add `swiglu_limit` clamp

V4 clamps the gate and up activations of every expert (and the shared expert) to
`±swiglu_limit` before the SiLU and the multiplication. V4-Flash-Base sets
`swiglu_limit = 10.0`. V3 has no clamp.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Class: `Expert.forward`, lines 596-606
- Snippet:
  ```python
  gate = self.w1(x).float()
  up   = self.w3(x).float()
  if self.swiglu_limit > 0:
      up   = torch.clamp(up,   min=-self.swiglu_limit, max=self.swiglu_limit)
      gate = torch.clamp(gate, max=self.swiglu_limit)        # NB: only upper-bounded
  x = F.silu(gate) * up
  ```

Note the asymmetry: `gate` is clamped only on the upper side; `up` is clamped on both
sides. The shared expert (constructed at line 628 with no `swiglu_limit`) uses
`swiglu_limit = 0`, which **disables** the clamp; only the routed experts apply it.

## V3 layers being modified

Both layers in `python/mirage/mpk/persistent_kernel.py` need an optional `swiglu_limit`
parameter:

| layer | line | role |
| --- | --- | --- |
| `silu_mul_layer` | 1736-1750 | dense / shared-expert SiLU+mul |
| `moe_silu_mul_layer` | 1546-1592 | routed-expert SiLU+mul (3D) |

## Proposed signature change

```python
pk.silu_mul_layer(
    input: DTensor,
    output: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    swiglu_limit: float = 0.0,         # NEW; 0 => disabled (V3 behavior)
)

pk.moe_silu_mul_layer(
    input: DTensor,
    output: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    swiglu_limit: float = 0.0,         # NEW
)
```

The kernel branches on `swiglu_limit > 0`:
- `swiglu_limit == 0`: identical to V3 (`silu(gate) * up`).
- `swiglu_limit > 0`: `silu(min(gate, limit)) * clamp(up, -limit, limit)`.

`swiglu_limit` is encoded into the kernel's `params` array (e.g., as the int-bits of the
fp32 value, mirroring how `routed_scaling_factor` is packed at
`persistent_kernel.py:1339`).

## Builder usage in V4

```python
# Routed experts
silu = pk.moe_silu_mul_layer(gate_up, ..., swiglu_limit=10.0)
# Shared expert (V4 constructs it without swiglu_limit -> defaults to 0)
shared_silu = pk.silu_mul_layer(shared_gate_up, ..., swiglu_limit=0.0)
```

## Notes / risks

- The clamps execute in fp32 in the reference (`gate = self.w1(x).float()`). In MPK we run
  the existing `silu_mul` in BF16; the clamp can be applied in BF16 with a small loss in
  precision. If accuracy regresses, do the clamp in fp32 inside the kernel before casting
  back.
- The asymmetric gate clamp (upper-only) is intentional — `silu(x)` for very negative `x`
  approaches 0 anyway, so a lower bound is unnecessary. Don't add a symmetric `gate` clamp
  by mistake.
- Shared expert behavior in V4-Flash-Base: `swiglu_limit = 0` (constructor line 628).
  Make sure the builder passes 0 there; otherwise the model will diverge from the reference.
- Verification: a small unit test that runs the V4 reference Expert and the MPK
  `moe_silu_mul_layer` with `swiglu_limit=10.0` on the same inputs and compares outputs to
  ~1e-3 BF16 tolerance.
